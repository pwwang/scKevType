#!/usr/bin/env python3
"""Lift the questions out of an existing raw ask tree, so the LLM is asked the same ones.

The rule this file exists to enforce: **the LLM baseline's input is not re-rendered.** Every prompt the
run builds comes from a `raw/` file that some earlier pass wrote, and the only thing this module does is
copy `request.state`, `request.questions[<id>].instructions` and `request.questions[<id>].criteria` out
of it, unchanged, beside the file's own sha256 and the request digest the raw file already carries.

Two raw layouts occur in this repo and both are read here:

``family``
    one family per file, named ``<dataset>__<cell>__<family>__<ask>.json`` and holding a single
    question (``reports/speed-cost/<engine>/raw/``).
``batched``
    several families per file, named ``<dataset>__<cell>__<batch>.json``, question ids prefixed
    ``<family>/<qid>`` (``reports/jev-full/raw-batched/``).

    python scripts/llm/build_asks.py --raw reports/speed-cost/jev-serial/raw \\
        --out reports/llm-asks-smoke.jsonl --manifest reports/llm-asks-smoke.manifest.json

``--include`` is the file-name pattern read out of the raw tree, ``*.json`` by default. A **singleton**
ask tree (one file per ask: ``<dataset>__<cell>__<family>__<ask>.json``) holds an ``__abstain.json``
beside each ``__base.json`` whenever a choice question was re-asked with an abstain option added, and
that second ask carries the *same question ids* with one extra option — so reading both is a duplicate
key (refused, loudly) and reading the abstain one instead of the base one would serve the LLM a question
the decision model was only asked as a follow-up. ``--include '*__base.json'`` is the ask set that
matches what the decision models were served. The batched layout has no such split: it carries the
variants inside one body under ``<family>__abstain__`` question ids.

The manifest records, per input file, its sha256 and the number of questions taken from it; the count of
questions written is asserted against the manifest at the end. A silent partial read would shrink the
population the baseline is scored on, which is the failure this pass is least able to notice later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(text: str) -> str:
    """The probe's own filename rule (``sckevtype.jev.probe.slug``), so the map is built forwards."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def dataset_of(group_id: str) -> str:
    return group_id.split("/", 1)[0]


def load_cell_map(path: Path) -> tuple[dict[str, tuple[str, str]], set[tuple[str, str]]]:
    """``slug(dataset)__slug(cell)`` -> ``(dataset, cell)``, built forwards from the population's own list.

    The raw file name is ``slug(dataset)__slug(cell)__…`` and the slug is **not invertible**
    (``tabula_muris_senis/3-M-5/6`` slugs its cell id into one field, and a siletti barcode's ``:``
    becomes ``_``), so the mapping cannot be parsed back out of the name. It is built in the forward
    direction from the draw that was asked.

    The second return value is the population's own cell set. It is what makes the mapping safe in both
    directions: a raw tree that is a *superset* of the population (the census tree, asked for the
    200-cell draw) contributes only the cells the population names -- a file outside it is skipped and
    counted -- while a population cell that no file supplies is a **failure**, because that would shrink
    the population the baseline is scored on.

    Accepts either a probe ``sample.json`` (``chosen`` list, or ``records`` as a list or a mapping) or a
    row dump (``dataset`` + ``cell`` per line).
    """
    text = path.read_text(encoding="utf-8")
    groups: list[str] = []
    if path.suffix == ".json":
        payload = json.loads(text)
        chosen = payload.get("chosen")
        if isinstance(chosen, list) and chosen:
            groups = [str(g) for g in chosen]
        else:
            records = payload.get("records")
            if isinstance(records, Mapping):
                groups = [str(g) for g in records]
            elif isinstance(records, list):
                groups = [str(g) for g in records]
    else:
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                groups.append(f"{row['dataset']}/{row['cell']}")
    if not groups:
        raise ValueError(f"{path}: no cell list found (looked for sample.json 'chosen'/'records' "
                         f"or a row dump's dataset+cell)")
    mapping: dict[str, tuple[str, str]] = {}
    population: set[tuple[str, str]] = set()
    for group in groups:
        dataset = dataset_of(group)
        cell = group.split("/", 1)[1]
        mapping[f"{slug(dataset)}__{slug(cell)}"] = (dataset, cell)
        population.add((dataset, cell))
    return mapping, population


class NotInPopulation(Exception):
    """A raw file that belongs to a cell the ``--cells-from`` population does not name."""


def split_name(stem: str, cell_map: Mapping[str, tuple[str, str]] | None = None) -> tuple[str, str, str | None]:
    """``<dataset>__<cell>__<rest>`` from a raw file's stem, resolved through the population's own list.

    With ``cell_map`` the dataset and cell are looked up by their *slugged* prefix, so a barcode's ``:``
    comes back as a colon and the key matches the row dumps'. A prefix the population does not name
    raises :class:`NotInPopulation` -- the caller counts it as skipped, never as a row. Without a map the
    two leading fields are taken as written, which is right for the atlases whose ids need no slugging
    and wrong for siletti, so a run without a map says so in its manifest.
    """
    parts = stem.split("__")
    if len(parts) < 3:
        raise ValueError(f"{stem!r} is not a <dataset>__<cell>__<rest> raw file name")
    if cell_map is not None:
        prefix = f"{parts[0]}__{parts[1]}"
        if prefix not in cell_map:
            raise NotInPopulation(prefix)
        dataset, cell = cell_map[prefix]
    else:
        dataset, cell = parts[0], parts[1]
    rest = parts[2] if len(parts) >= 4 else None
    return dataset, cell, rest


def asks_from_file(path: Path, cell_map: Mapping[str, tuple[str, str]] | None = None) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "request" not in raw:
        raise ValueError(f"{path}: no 'request' key -- this is not a raw ask file")
    request = raw["request"]
    state = request["state"]
    questions = request["questions"]
    dataset, cell, family_from_name = split_name(path.stem, cell_map)
    digest = sha256_file(path)
    out = []
    for qid, question in questions.items():
        if family_from_name is not None:
            family, clean_qid = family_from_name, qid
        else:
            if "/" not in qid:
                raise ValueError(f"{path}: a batched file's question id {qid!r} has no family prefix")
            family, clean_qid = qid.split("/", 1)
        if question.get("type") not in ("choice", "noul"):
            raise ValueError(f"{path}:{qid}: type {question.get('type')!r} is not served here")
        out.append({
            "key": f"{dataset}/{cell}/{family}/{clean_qid}",
            "dataset": dataset,
            "cell": cell,
            "family": family,
            "question": clean_qid,
            "type": question["type"],
            "state": state,
            "instructions": question["instructions"],
            "criteria": question["criteria"],
            "source_file": path.name,
            "source_sha256": digest,
            "request_sha256": raw.get("request_sha256"),
            "served_model": request.get("model"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="a raw ask tree (one directory of *.json)")
    ap.add_argument("--include", default="*.json",
                    help="file-name pattern read out of the raw tree (default *.json; '*__base.json' "
                         "skips the singleton layout's second, abstain-bearing ask of the same questions)")
    ap.add_argument("--out", required=True, help="the asks to write, one JSON object per question")
    ap.add_argument("--manifest", required=True, help="provenance: per-file sha256 and question counts")
    ap.add_argument("--families", default="", help="comma-separated family filter (default: all)")
    ap.add_argument("--cells-from", default="",
                    help="the population's own list (a probe sample.json or a row dump); the raw file "
                         "names are resolved through it because the slug is not invertible")
    ap.add_argument("--label", default="", help="a name for this ask set, recorded in the manifest")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.is_dir():
        raise SystemExit(f"{raw_dir} is not a directory")
    cell_map = population = None
    if args.cells_from:
        cell_map, population = load_cell_map(Path(args.cells_from))
    wanted = {f for f in args.families.split(",") if f}
    files = sorted(p for p in raw_dir.glob(args.include))
    if not files:
        raise SystemExit(f"{raw_dir}: no {args.include} -- refusing to write an empty ask set")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"raw_dir": str(raw_dir), "label": args.label, "families": sorted(wanted),
                "include": args.include,
                "cells_from": args.cells_from or None,
                "cell_map_size": len(cell_map) if cell_map else None,
                "files": [], "questions": 0, "files_read": 0, "files_failed": 0,
                "files_skipped_not_in_population": 0}
    per_family: dict[str, int] = {}
    per_dataset: dict[str, int] = {}
    seen_keys: set[str] = set()
    seen_cells: set[tuple[str, str]] = set()
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            try:
                asks = asks_from_file(path, cell_map)
            except NotInPopulation:
                manifest["files_skipped_not_in_population"] += 1
                continue
            except Exception as exc:  # named, counted, and the run is failed at the end
                print(f"  FAILED {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                manifest["files_failed"] += 1
                manifest["files"].append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            manifest["files_read"] += 1
            kept = 0
            for ask in asks:
                if wanted and ask["family"] not in wanted:
                    continue
                if ask["key"] in seen_keys:
                    raise SystemExit(f"{path.name}: two questions share the key {ask['key']!r}")
                seen_keys.add(ask["key"])
                out.write(json.dumps(ask, ensure_ascii=False) + "\n")
                kept += 1
                written += 1
                seen_cells.add((ask["dataset"], ask["cell"]))
                per_family[ask["family"]] = per_family.get(ask["family"], 0) + 1
                per_dataset[ask["dataset"]] = per_dataset.get(ask["dataset"], 0) + 1
            manifest["files"].append({"file": path.name, "sha256": sha256_file(path),
                                      "questions_total": len(asks), "questions_written": kept})
    manifest["questions"] = written
    manifest["cells_written"] = len(seen_cells)
    manifest["cells_expected"] = len(population) if population else None
    manifest["cells_missing"] = sorted(f"{d}/{c}" for d, c in (population - seen_cells)) if population else None
    manifest["per_family"] = per_family
    manifest["per_dataset"] = per_dataset
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"raw tree      : {raw_dir}")
    print(f"files read    : {manifest['files_read']} (failed: {manifest['files_failed']}, "
          f"skipped: {manifest['files_skipped_not_in_population']} outside the population)")
    print(f"questions out : {written}")
    print(f"cells out     : {len(seen_cells)} of {manifest['cells_expected']}")
    print(f"per family    : {per_family}")
    print(f"per dataset   : {per_dataset}")
    print(f"wrote {out_path} and {args.manifest}")
    if manifest["files_failed"]:
        raise SystemExit(f"{manifest['files_failed']} raw file(s) could not be read -- refusing to "
                         f"score a population that is not the whole tree")
    if population and seen_cells != population:
        raise SystemExit(f"{len(population - seen_cells)} cell(s) of the population the --cells-from list "
                         f"names were supplied by no raw file -- refusing to score a shrunken population")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
