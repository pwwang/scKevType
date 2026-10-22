"""Derive the v2 tissue map's tissue lists by rule — never by hand.

    uv run python scripts/derive_tissue_map.py [--dry-run]

Run by hand, from the repo root, when the register or a corpus config moves. It is
not run by the build, and never by a test: it reads the atlases (their ``obs``
columns only) and the CellMarker resource.

**The rule.** For one tissue ``T`` of the middleware spec::

    tissues[T] = scope ∩ ( cellmarker(T) ∪ observed(T) )

* ``cellmarker(T)`` — the CL ids **CellMarker 2.0** files under the ``tissue_class``
  whose name is ``T`` (``Normal`` rows, matched on the resource's own
  ``cellontology_id`` column). This is the rule the seed already used for ``brain``;
  the class name is found by matching the register's tissue name against the
  resource's own class names, so there is no second hand-kept table here either.
* ``observed(T)`` — the CL ids an atlas wired to ``T`` by a corpus config actually
  carries, in cells whose tissue resolves to ``T``. This is the rule the seed already
  used for ``pancreas``.
* ``scope`` — the values of the corpus configs' ``label_map``: the terms the register
  can name. An id outside it is not a candidate anywhere and is not this file's
  business.

There is deliberately **no size cap and no hand-kept list**. A truncated, hand-kept
``blood`` list is exactly how ``CL:0001054`` — a type Step 7 added to the scope —
came to be missing from the tissue map while the build's own invariant required it
(``docs/SPEC-step9-tissue-map-sync.md``).

The two halves are not redundant. ``cellmarker(T)`` is the tissue's type inventory as
an external curation sees it, including types no atlas here samples; ``observed(T)``
is the closure that makes the build's invariant — *every atlas label has a candidate
for its tissue* — true by construction when a new atlas, or a new scope term, is
wired in later. An id the rule places in no tissue is printed, never guessed: the
build then fails loudly with the id and the candidates it did find.

``observed`` is measured, not assumed: each atlas's ``obs`` columns are read through
the registry's column names, and the tissue is resolved by the corpus config's own
``tissue_labels`` / ``tissue`` through the adapter's own :func:`resolve_tissue`, so
the measurement and the build see one corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from curate_register_v2 import EXTERNAL, atlas_file, read_xlsx

from sckevtype.contracts.freeze import file_digest
from sckevtype.data import registry as registry_module
from sckevtype.data.adapters import resolve_tissue
from sckevtype.data.adapters.cellxgene_h5ad import (
    _column,
    _obs_keys,
    resolve_column,
)
from sckevtype.data.build import (
    Artifacts,
    DatasetConfig,
    dataset_digest,
    load_config,
)
from sckevtype.data.pull import resolve_cache

V2_DIR = ROOT / "configs" / "seed" / "v2"
CORPUS_DIR = ROOT / "configs" / "corpus"
SEED = V2_DIR / "tissue_map.seed.yaml"
MEASURED = V2_DIR / "atlas_tissue_labels.json"
#: The resource whose ``tissue_class`` column the rule reads, and the file each
#: species' rows come from — the same two files the marker definitions use.
CELLMARKER = {"Human": EXTERNAL / "Cell_marker_Human.xlsx", "Mouse": EXTERNAL / "Cell_marker_Mouse.xlsx"}


def obs_histogram(path: Path, label_name: str | None, tissue_name: str | None) -> Counter:
    """``(raw tissue text, raw label) -> cells`` for one atlas, reading two obs columns.

    Two columns, not the expression matrix: a tissue map needs to know which types a
    tissue is said to contain, not what they express. The column names are the
    registry's, resolved by the adapter's own resolution so nothing is hard-coded here.
    """
    with h5py.File(path, "r") as handle:
        keys = _obs_keys(handle)
        label_column = resolve_column(keys, label_name, "label")
        tissue_column = resolve_column(keys, tissue_name, "tissue")
        if label_column is None:
            return Counter()
        labels = _column(handle, "obs", label_column)
        tissues = _column(handle, "obs", tissue_column) if tissue_column else [None] * len(labels)
    return Counter(zip(tissues, labels))


def corpus_wiring() -> dict[str, list[tuple[str, DatasetConfig, str]]]:
    """``artifacts version -> [(config path, dataset, mapping)]`` for every corpus config.

    Every config's wiring is a claim about a tissue, so every config is read: a
    dataset that one config leaves in ``global`` scope and another names a tissue for
    is a tissue requirement in the second, and the derivation has to see it.
    """
    wiring: dict[str, list[tuple[str, DatasetConfig, dict[str, str]]]] = defaultdict(list)
    for path in sorted(CORPUS_DIR.glob("*.yaml")):
        try:
            config = load_config(path)
        except ValueError:
            print(f"  {path.name:<20} not a corpus config: skipped (no wiring to read)", flush=True)
            continue
        artifacts = Artifacts(ROOT / "artifacts" / config.version)
        for dataset in config.datasets:
            if dataset.label_map is None:
                continue  # no table, no name for a type: nothing this dataset can require
            mapping = artifacts.label_map(dataset).mapping
            wiring[config.version].append((path.name, dataset, mapping))
    return wiring


def scope_of(wiring) -> tuple[set[str], str]:
    """The union of every corpus ``label_map``'s values — what the register can name."""
    scope: set[str] = set()
    for entries in wiring.values():
        for _, _, mapping in entries:
            scope |= set(mapping.values())
    return scope, ", ".join(sorted({name for entries in wiring.values() for name, _, _ in entries}))


def cellmarker_by_class() -> tuple[dict[str, list[str]], dict[str, Counter], dict[str, str]]:
    """``tissue_class -> CL ids``, ``tissue_class -> CL id -> rows``, and the file digests.

    Rows that are cancer are skipped — a tissue map is a map of *normal* tissue — and
    the CL id is the resource's own column, never a name match.
    """
    ids: dict[str, set[str]] = defaultdict(set)
    rows_seen: dict[str, Counter] = defaultdict(Counter)
    digests = {}
    for species, path in sorted(CELLMARKER.items()):
        digests[path.name] = file_digest(path)
        for row in read_xlsx(path):
            if row.get("cancer_type") != "Normal":
                continue
            cl_id = "CL:" + (row.get("cellontology_id") or "").replace("CL_", "")
            tissue_class = row.get("tissue_class")
            if not tissue_class or not cl_id.startswith("CL:") or cl_id == "CL:":
                continue
            ids[tissue_class].add(cl_id)
            rows_seen[tissue_class][cl_id] += 1
    return {name: sorted(found) for name, found in ids.items()}, rows_seen, digests


def class_for_tissue(tissue: str, classes) -> str | None:
    """The resource's ``tissue_class`` whose name is this tissue — an exact match, case-folded.

    Not a substring, not a similarity: ``blood`` is CellMarker's ``Blood`` and nothing
    else. A tissue the resource does not name gets no rows from it, which is reported
    rather than papered over.
    """
    matches = sorted(name for name in classes if name.casefold() == tissue.casefold())
    if len(matches) > 1:
        raise SystemExit(f"CellMarker has {len(matches)} tissue classes matching {tissue!r}: {matches}")
    return matches[0] if matches else None


def observed_by_tissue(wiring, cache_root: Path, *, digest: bool) -> tuple[dict[str, Counter], dict[str, dict]]:
    """``tissue -> CL id -> cells`` over every wired atlas whose bytes are on disk.

    One read per (atlas, config wiring): the histograms are keyed by the atlas's *raw*
    columns, and each config's own tissue resolution is applied afterwards, so two
    configs that read one atlas differently are both honoured without reading it twice.
    """
    registry = registry_module.load()
    observed: dict[str, Counter] = defaultdict(Counter)
    measured: dict[str, dict] = {}
    for version, entries in sorted(wiring.items()):
        for config_name, dataset, mapping in entries:
            if not dataset.registered:
                print(f"  {dataset.id:<20} synthetic fixture, not an atlas: not measured", flush=True)
                continue
            entry = registry.get(dataset.id)
            if dataset.access:
                entry = registry_module.with_access(registry, dataset.id, dataset.access).get(dataset.id)
            path = ROOT / dataset.path if dataset.path else atlas_file(dataset.id, entry, cache_root)
            if not path.exists():
                print(f"  {dataset.id:<20} not on disk, not measured, not invented", flush=True)
                continue
            histogram = obs_histogram(path, dataset.obs.label, dataset.obs.tissue)
            pairs: Counter = Counter()
            for (text, label), cells in histogram.items():
                cl_id = mapping.get(label) if label is not None else None
                if cl_id is None:
                    continue  # a label the register cannot name requires no candidate
                tissue = resolve_tissue(text, dataset.tissue_labels, dataset.tissue)
                if tissue is None:
                    continue  # resolves to no register tissue: its panels are the whole register
                observed[tissue][cl_id] += cells
                pairs[(tissue, cl_id)] += cells
            if digest:
                measured[f"{config_name}:{dataset.id}"] = {
                    "dataset_digest": dataset_digest(path),
                    "bytes": path.stat().st_size,
                    "cells": sum(pairs.values()),
                    "pairs": {f"{tissue}|{cl_id}": cells for (tissue, cl_id), cells in sorted(pairs.items())},
                }
    return observed, measured


def derive(tissues: list[str], scope: set[str], observed, cm_ids, cm_rows) -> tuple[dict[str, list[str]], dict[str, str]]:
    """The rule, applied once per tissue. Returns ``tissue -> ids`` and ``tissue -> class``."""
    derived, classes = {}, {}
    for tissue in tissues:
        tissue_class = class_for_tissue(tissue, cm_ids)
        classes[tissue] = tissue_class or ""
        counts: Counter = Counter()
        for cl_id in cm_ids.get(tissue_class, ()):  # the resource's own attribution
            if cl_id in scope:
                counts[cl_id] = max(counts[cl_id], 1)
        for cl_id in observed.get(tissue, ()):  # the wiring's closure
            counts[cl_id] = max(counts[cl_id], 1)
        evidence = cm_rows.get(tissue_class, Counter())
        derived[tissue] = sorted(counts, key=lambda cl_id: (-evidence.get(cl_id, 0), cl_id))
    return derived, classes


def source_string(classes, scope, measured, measured_digest, cm_digests, config_digests, unplaced) -> str:
    """The provenance the artifact carries: the rule, its inputs' digests, and the leftovers."""
    cm = ", ".join(f"{name} sha256 {digest[:12]}" for name, digest in sorted(cm_digests.items()))
    configs = ", ".join(f"{name} sha256 {digest[:12]}" for name, digest in sorted(config_digests.items()))
    atlases = ", ".join(f"{dataset} sha256 {entry['dataset_digest'][:12]}" for dataset, entry in sorted(measured.items()))
    atlases = f"configs/seed/v2/atlas_tissue_labels.json sha256 {measured_digest[:12]}" + (f" ({atlases})" if atlases else "")
    classes_seen = "; ".join(f"{tissue}={tissue_class or 'no CellMarker tissue_class'}" for tissue, tissue_class in sorted(classes.items()))
    return (
        "tissue -> candidate CL ids (paper_plan §4.5), DERIVED by rule, not hand-appended: "
        "tissues[T] = scope ∩ (cellmarker(T) ∪ observed(T)). cellmarker(T) is the CL ids CellMarker 2.0 files "
        f"under the tissue_class whose name is T, Normal rows only, via the resource's own cellontology_id column ({cm}); "
        "observed(T) is the CL ids an atlas wired to T by a corpus config carries in cells whose tissue resolves to T, "
        f"read from the atlases' own obs columns ({atlases}); "
        f"scope is the corpus label_map ({configs}); a term the rule places in no tissue is reported, never guessed. "
        f"Ranked by CellMarker rows, ties by CL id. Tissue classes: {classes_seen}. "
        f"Placed {len(scope) - len(unplaced)} of {len(scope)} scope ids"
        + (f"; no candidate anywhere: {', '.join(unplaced)}." if unplaced else ".")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive configs/seed/v2/tissue_map.seed.yaml by rule.")
    parser.add_argument("--dry-run", action="store_true", help="report the derivation; write nothing, digest no atlas")
    args = parser.parse_args(argv)

    wiring = corpus_wiring()
    scope, config_names = scope_of(wiring)
    cache_root = resolve_cache("data/raw")
    print(f"scope: {len(scope)} CL ids from {config_names}", flush=True)
    print("measured from disk:", flush=True)
    observed, measured = observed_by_tissue(wiring, cache_root, digest=not args.dry_run)

    cm_ids, cm_rows, cm_digests = cellmarker_by_class()
    seed = yaml.safe_load(SEED.read_text(encoding="utf-8"))
    named = {tissue for entries in wiring.values() for _, dataset, _ in entries for tissue in (dataset.tissue, *dataset.tissue_labels.values()) if tissue}
    tissues = sorted(set(seed["tissues"]) | named)
    derived, classes = derive(tissues, scope, observed, cm_ids, cm_rows)
    for tissue in [name for name, ids in derived.items() if not ids]:
        # No candidate at all: a panel for this tissue would have no options, so the
        # register does not carry it and a cell of that tissue draws on the register.
        print(f"  {tissue}: no candidate for any scope id — left out of the map", flush=True)
        del derived[tissue]

    print(f"\n{'tissue':<12}{'ids':>5}{'CellMarker':>12}{'atlas-only':>12}  class", flush=True)
    for tissue in sorted(derived):
        tissue_class = class_for_tissue(tissue, cm_ids)
        from_marker = {cl_id for cl_id in cm_ids.get(tissue_class, ()) if cl_id in scope}
        print(f"{tissue:<12}{len(derived[tissue]):>5}{len(from_marker):>12}{len(set(derived[tissue]) - from_marker):>12}  {tissue_class or '-'}", flush=True)
    placed = {cl_id for cl_ids in derived.values() for cl_id in cl_ids}
    unplaced = sorted(scope - placed)
    print(f"\nplaced {len(placed)} of {len(scope)} scope ids", flush=True)
    for cl_id in unplaced:
        print(f"  NOT PLACED: {cl_id} — no CellMarker tissue class, no wired atlas carries it in a named tissue", flush=True)
    if args.dry_run:
        print("\n--dry-run: nothing written", flush=True)
        return 0

    for tissue in sorted(derived):
        tissue_class = class_for_tissue(tissue, cm_ids)
        atlas_only = sorted(set(derived[tissue]) - set(cm_ids.get(tissue_class, ())))
        if atlas_only:
            cells = observed[tissue]
            print(f"  {tissue:<12} in the tissue only because an atlas carries it: " + ", ".join(f"{cl_id}({cells[cl_id]})" for cl_id in atlas_only), flush=True)
    config_digests = {name: file_digest(CORPUS_DIR / name) for name in config_names.split(", ")}
    MEASURED.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {MEASURED.relative_to(ROOT)}", flush=True)
    source = source_string(classes, scope, measured, file_digest(MEASURED), cm_digests, config_digests, unplaced)
    seed.update({"source": source, "version": "v2", "tissues": derived})
    SEED.write_text(yaml.safe_dump(seed, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print(f"wrote {SEED.relative_to(ROOT)}", flush=True)
    print("next: uv run python -m sckevtype.contracts.build --out artifacts/v2 --seed-dir configs/seed/v2 --force", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
