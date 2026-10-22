"""Curate the v2 register: measure coverage, ground definitions, write the seeds.

Run by hand, from the repo root, when the register moves — not by the build, and
never by a test (it reads the atlases and the network):

    uv run python scripts/curate_register_v2.py [--no-download]

What it does, in order:

1. **Measures** every benchmark atlas whose bytes are on disk: its ``CL id ->
   cells`` histogram, read from the label column the *registry* names (never a
   column hard-coded here). Writes ``configs/seed/v2/atlas_labels.json`` with the
   digest of the bytes each histogram came from, so a coverage number is always
   attributable to a file.
2. **Grounds definitions** from two external curated resources — never from the
   benchmark atlases' own differential expression:
   * **Cell Ontology** (``cl-basic.json``, the release ``sckevtype.contracts.fetch``
     already pins for the id spine) supplies ``text``: the term's own definition.
   * **CellMarker 2.0** (NAR 2023) supplies the markers. The resource ships a
     ``cellontology_id`` column next to every marker row, so the match rule is
     *the resource's own* — no name similarity, no inference here:
     ``marker M is attributed to CL term T iff the row's cellontology_id is T``.
     Only ``Normal``, ``protein_coding`` rows are read. Terms CellMarker does not
     list fall back to **ScType** (Nat Commun 2022), whose ``geneSymbolmore1`` /
     ``geneSymbolmore2`` columns are matched by cell name against the CL label.
   * ``neg_markers`` are the markers CellMarker attributes to *other in-scope
     terms* and never to this one — the same idea the v1 seed hand-wrote ("T cell:
     negative MS4A1, LYZ"), computed instead of guessed.
3. **Applies the §2 rule** (``sckevtype.contracts.register.plan_scope``): the
   smallest term set that maps >=95% of every atlas, intersected with what step 2
   could ground. The shortfall is written out, not hidden.
4. **Writes the seeds** — ``configs/seed/v2/{cl_terms,definitions,tissue_map}.seed.yaml``
   plus copies of the three unchanged v1 seeds — and ``reports/step6-coverage.json``,
   the per-atlas table this step is measured by.

A term it cannot ground stays out of scope, and is counted in that table.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py

from sckevtype.contracts import paths
from sckevtype.contracts.freeze import file_digest, read_json
from sckevtype.contracts.register import AtlasCounts, plan_scope
from sckevtype.data import registry as registry_module
from sckevtype.data.adapters.cellxgene_h5ad import (
    _column,
    _obs_keys,
    resolve_column,
)
from sckevtype.data.build import dataset_digest, load_config
from sckevtype.data.pull import resolve_cache

# --- external resources (name, version, URL). The URL is where the *bytes* came
# --- from; the digest of those bytes is recorded in every provenance string.
CL_URL = "http://purl.obolibrary.org/obo/cl/cl-basic.json"
CELLMARKER_PAGE = "http://bio-bigdata.hrbmu.edu.cn/CellMarker"
CELLMARKER_VERSION = "CellMarker 2.0 (Nucleic Acids Research 2023, 51:D870-D876)"
CELLMARKER_URLS = {
    "Human": "http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_Human.xlsx",
    "Mouse": "http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_Mouse.xlsx",
}
SCTYPE_VERSION = "ScTypeDB_full.xlsx (Nature Communications 2022, 13:1246)"
SCTYPE_URL = "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"

# --- curation parameters. Every one of them is a budget or a policy, not a fact
# --- about biology, and each is stated in the seed's own header.
POS_MARKERS = 6           # markers kept per term (prompt budget: kev MAX_STATE = 384 tokens)
NEG_MARKERS = 3
TEXT_CHARS = 240          # CL definitions are longer than a panel option can afford
BRAIN_REGISTER = 12       # types in the brain tissue list (§4), taken from CellMarker's brain rows

#: The tissues the v2 tissue map carries beyond the atlas label sets, and the
#: CellMarker ``tissue_class`` each is drawn from (§4).
TISSUE_CLASSES = {"brain": "Brain"}

#: §6: the atlas whose fast copy is not where the registry's ``access[0].name``
#: says. Relative to the cache root; the digest is checked by the build's manifest.
CROSS_SOURCE = {"dominguez_conde": Path("dominguez_conde_cxg") / "global.h5ad"}

#: The measurement list is real-v1's nine atlases (the complete set of registry ids);
#: real-v2 is where an atlas's *reading* lives once it is wired (Step 7 §1).
MEASURE_CONFIG = ROOT / "configs" / "corpus" / "real-v1.yaml"
REAL_V2 = ROOT / "configs" / "corpus" / "real-v2.yaml"

V2_DIR = ROOT / "configs" / "seed" / "v2"
EXTERNAL = ROOT / "data" / "external"
ATLAS_LABELS = V2_DIR / "atlas_labels.json"
COVERAGE_REPORT = ROOT / "reports" / "step6-coverage.json"
#: The seeds v2 does not change: copied, and a test pins them byte-identical.
COPIED_SEEDS = ("genes.seed.txt", "filter.seed.yaml", "label_map.seed.yaml")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def download(url: str, target: Path) -> Path:
    """Fetch `url` to `target` unless the bytes are already there (then leave them)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    print(f"downloading {url} -> {target}", flush=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        target.write_bytes(response.read())
    return target


def read_xlsx(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet of an xlsx as a list of row dicts, keyed by header.

    ``openpyxl`` is not a dependency of this project and adding one to read two
    curated files would be a dependency the build never uses, so the sheet is read
    straight out of the OOXML zip: the file format *is* XML, and this is all of it
    that these two resources need.
    """
    archive = zipfile.ZipFile(path)
    shared = [
        "".join(text.text or "" for text in item.iter(f"{NS}t"))
        for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(f"{NS}si")
    ]
    sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[dict[str, str]] = []
    header: dict[str, str] = {}
    for row in sheet.iter(f"{NS}row"):
        cells = {}
        for cell in row.findall(f"{NS}c"):
            column = "".join(char for char in cell.get("r", "") if char.isalpha())
            value, inline = cell.find(f"{NS}v"), cell.find(f"{NS}is")
            if cell.get("t") == "s" and value is not None:
                cells[column] = shared[int(value.text)]
            elif inline is not None:
                cells[column] = "".join(text.text or "" for text in inline.iter(f"{NS}t"))
            elif value is not None:
                cells[column] = value.text or ""
        if not header:
            header = cells
        else:
            rows.append({header[column]: cells.get(column, "") for column in header})
    return rows


def cl_release(path: Path) -> dict[str, dict]:
    """``cl_id -> {name, definition, parents}`` from the CL release both builds read."""
    graph = json.loads(path.read_text(encoding="utf-8"))["graphs"][0]
    prefix = "http://purl.obolibrary.org/obo/CL_"
    terms = {}
    for node in graph.get("nodes", []):
        identifier = node.get("id", "")
        if not identifier.startswith(prefix) or node.get("meta", {}).get("deprecated"):
            continue
        terms["CL:" + identifier[len(prefix) :]] = {
            "name": node.get("lbl", ""),
            "definition": (node.get("meta", {}).get("definition") or {}).get("val", ""),
        }
    return terms


def first_sentence(text: str, limit: int = TEXT_CHARS) -> str:
    """The release's definition, trimmed to one option-sized sentence."""
    prose = " ".join(text.split())
    stop = prose.find(". ")
    if stop != -1:
        prose = prose[:stop]
    return prose[:limit].rstrip(" ,;") + "." if prose else ""


def label_histogram(path: Path, columns) -> dict[str, int]:
    """``CL id -> cells`` for one h5ad, reading the obs label column and nothing else.

    The adapter's own ``probe`` would answer this too, but it also materialises the
    expression matrix to report its shape — 2 GB for the pancreas atlas. A coverage
    table needs one column, so this reads one column, through the adapter's own
    column resolution so the name still comes from the registry and not from here.
    """
    with h5py.File(path, "r") as handle:
        name = resolve_column(_obs_keys(handle), columns.label, "label")
        if name is None:
            return {}
        return dict(Counter(label for label in _column(handle, "obs", name) if label is not None))


def atlas_file(dataset_id: str, entry, cache_root: Path) -> Path:
    """Where this atlas's bytes are, given both sources §6 records.

    The registry's ``access[0]`` is the primary source, and for every atlas but one
    that is the file on disk. ``dominguez_conde`` is the exception: its registry
    entry is a ``kind: file`` Sanger mirror that measured 61 KiB/s, so the corpus
    uses the CELLxGENE copy instead (3.0 GB, fetched in ~17 min) — see
    ``configs/corpus/real-v2.yaml``, which is where that switch is declared for the
    build. Reading it here too keeps the measurement and the build on one source.
    """
    primary = cache_root / dataset_id / entry.access[0].name
    if primary.exists():
        return primary
    return cache_root / CROSS_SOURCE[dataset_id] if dataset_id in CROSS_SOURCE else primary


def measure_atlases(datasets, cache_root: Path) -> tuple[list[AtlasCounts], list[str]]:
    """Per-atlas label histograms, read through the registry's own column names."""
    counts, missing = [], []
    registry = registry_module.load()
    # `datasets` is the full nine-atlas list (real-v1): the ones not on disk drop out by
    # file existence. Where real-v2 records how it reads an atlas — the probe-verified
    # access path §1 added, or the label column the registry leaves null — that is the
    # reading used here too, so the measurement and the build see one corpus.
    overrides = {dataset.id: dataset for dataset in load_config(REAL_V2).datasets}
    for dataset in datasets:
        corpus_dataset = overrides.get(dataset.id, dataset)
        entry = registry.get(dataset.id)
        if corpus_dataset.access:
            entry = registry_module.with_access(registry, dataset.id, corpus_dataset.access).get(dataset.id)
        primary = atlas_file(dataset.id, entry, cache_root)
        if not primary.exists():
            missing.append(dataset.id)
            continue
        labels = label_histogram(primary, corpus_dataset.obs if corpus_dataset.obs is not None else entry.obs)
        if not labels:
            missing.append(dataset.id)
            continue
        counts.append(
            AtlasCounts(
                atlas_id=dataset.id,
                dataset_digest=dataset_digest(primary),
                cells=sum(labels.values()),
                labels=labels,
            )
        )
        print(f"measured {dataset.id}: {sum(labels.values())} cells, {len(labels)} terms", flush=True)
    return counts, missing


def cellmarker_index(paths_by_species: dict[str, Path]) -> dict[str, list[str]]:
    """``CL id -> [marker symbol, ...]``, in the resource's own attribution.

    The match rule is the resource's: every CellMarker row carries the CL id its
    marker was curated for, so a marker joins a term when the row says so. Rows
    that are cancer, non-protein-coding, or carry no CL id are skipped — a marker
    list is a definition of a *normal* type, and a lncRNA is not a marker a
    transcript rank list can be read for.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for species, path in sorted(paths_by_species.items()):
        kept = 0
        for row in read_xlsx(path):
            cl_id = "CL:" + row.get("cellontology_id", "").replace("CL_", "")
            symbol = (row.get("Symbol") or "").strip()
            if not row.get("cellontology_id", "").startswith("CL_") or not symbol:
                continue
            if row.get("cancer_type") != "Normal" or row.get("Genetype") != "protein_coding":
                continue
            index[cl_id].append(symbol)
            kept += 1
        print(f"CellMarker {species}: {kept} normal protein-coding rows", flush=True)
    return index


def sctype_index(path: Path) -> dict[str, list[str]]:
    """``cell name (normalised) -> negative markers``, the fallback resource's view."""
    negatives = defaultdict(list)
    for row in read_xlsx(path):
        name = (row.get("cellName") or "").strip().casefold()
        for marker in (row.get("geneSymbolmore2") or "").split(","):
            if marker.strip():
                negatives[name].append(marker.strip())
    return negatives


def rank_markers(symbols, limit: int) -> tuple[str, ...]:
    """Most-attested first, ties broken by symbol: the order is a function of the rows."""
    ranked = sorted(Counter(symbols).items(), key=lambda item: (-item[1], item[0]))
    return tuple(symbol for symbol, _ in ranked[:limit])


def build_definitions(
    scope: list[str], terms: dict[str, dict], cellmarker: dict[str, list[str]], sctype: dict[str, list[str]],
    digests: dict[str, str],
) -> dict:
    """One definition per in-scope term: the release's prose, the resources' markers."""
    cm_digest = ",".join(f"{species}:{digest[:8]}" for species, digest in sorted(digests.items()) if species != "cl")
    definitions, fallbacks = [], []
    for cl_id in scope:
        term = terms[cl_id]
        positives = rank_markers(cellmarker.get(cl_id, ()), POS_MARKERS)
        provenance = f"cellmarker2:{cm_digest}:{cl_id.replace(':', '_')}"
        if not positives:
            alias = sctype.get(term["name"].strip().casefold(), ())
            positives = rank_markers(alias, POS_MARKERS)
            if positives:
                provenance = f"sc-type:{digests['sctype'][:8]}:{term['name']}"
                fallbacks.append(cl_id)
        # Negative markers: what the resource attributes to other in-scope terms and
        # never to this one. Computed from the same rows, so a negative is as
        # traceable as a positive — it is a marker CellMarker knows, pointed away.
        others = Counter(
            symbol
            for other in scope
            if other != cl_id
            for symbol in set(cellmarker.get(other, ())) - set(cellmarker.get(cl_id, ()))
        )
        definitions.append(
            {
                "cl_id": cl_id,
                "name": term["name"],
                "text": first_sentence(term["definition"]),
                "pos_markers": list(positives),
                "neg_markers": list(rank_markers(list(others.elements()), NEG_MARKERS)) if others else [],
                "provenance": provenance,
            }
        )
    print(f"definitions: {len(definitions)} terms, {len(fallbacks)} grounded by the ScType fallback", flush=True)
    return {"definitions": definitions, "fallbacks": fallbacks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate the v2 register seeds from measured coverage.")
    parser.add_argument("--no-download", action="store_true", help="use only resources already in data/external")
    args = parser.parse_args(argv)
    get = (lambda url, target: download(url, target)) if not args.no_download else _cached_or_die

    from sckevtype.data.build import load_config

    corpus = load_config(MEASURE_CONFIG)
    cache_root = resolve_cache(corpus.cache, root=ROOT)

    # --- 1. measure -----------------------------------------------------------
    counts, missing = measure_atlases(corpus.datasets, cache_root)
    if not counts:
        print("no atlas on disk: nothing to derive a scope from", file=sys.stderr)
        return 1

    # --- 2. external resources ------------------------------------------------
    cl_json = get(CL_URL, EXTERNAL / "cl-basic.json")
    terms = cl_release(cl_json)
    cm_paths = {species: get(url, EXTERNAL / f"Cell_marker_{species}.xlsx") for species, url in CELLMARKER_URLS.items()}
    sctype_path = get(SCTYPE_URL, EXTERNAL / "ScTypeDB_full.xlsx")
    digests = {species: file_digest(path) for species, path in cm_paths.items()}
    digests["cl"] = file_digest(cl_json)
    digests["sctype"] = file_digest(sctype_path)
    cellmarker = cellmarker_index(cm_paths)
    sctype = sctype_index(sctype_path)

    # --- 3. the scope rule, then the "scope >= defined" constraint -------------
    # The defined set is what step 2 can actually ground: a term whose CL release
    # entry is missing, or which neither resource has a marker for, is not a
    # definition and must not become a scope entry.
    groundable = {
        cl_id
        for cl_id in terms
        if terms[cl_id]["name"] and (cellmarker.get(cl_id) or sctype.get(terms[cl_id]["name"].strip().casefold()))
    }
    v1_register = {term["cl_id"] for term in read_json(paths.artifact("v1", "cl_terms.json"))["terms"]}
    plan = plan_scope(counts, groundable, required=v1_register)
    print(
        f"scope: {len(plan.demanded)} demanded, {len(plan.required)} required, "
        f"{len(plan.in_scope)} in scope, {len(plan.undefined)} undefined",
        flush=True,
    )

    # --- 4. tissue map (§4): v1's three tissues, plus what the atlases add ------
    v1_map = yaml.safe_load((paths.seed_dir() / "tissue_map.seed.yaml").read_text(encoding="utf-8"))
    tissues = {
        tissue: [cl_id for cl_id in cl_ids if cl_id in set(plan.in_scope)]
        for tissue, cl_ids in v1_map["tissues"].items()
    }
    pancreas = [cl_id for cl_id in sorted(plan.in_scope) if _atlas_tissue(counts, "muraro", cl_id)]
    if pancreas:
        tissues["pancreas"] = pancreas
    brain = _tissue_register(cm_paths, TISSUE_CLASSES["brain"], BRAIN_REGISTER, groundable, set(plan.in_scope))
    if brain:
        tissues["brain"] = brain

    # --- 5. write -------------------------------------------------------------
    payload = build_definitions(list(plan.in_scope), terms, cellmarker, sctype, digests)
    V2_DIR.mkdir(parents=True, exist_ok=True)
    for name in COPIED_SEEDS:
        shutil.copyfile(paths.seed_dir() / name, V2_DIR / name)
    shutil.copyfile(paths.seed_dir() / "cl_terms.v2.yaml", V2_DIR / "cl_terms.seed.yaml")
    _dump(V2_DIR / "definitions.seed.yaml", _definitions_seed(payload, digests))
    _dump(V2_DIR / "tissue_map.seed.yaml", _tissue_seed(tissues, digests))
    (V2_DIR / "atlas_labels.json").write_text(
        json.dumps({atlas.atlas_id: atlas.model_dump(mode="json") for atlas in counts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "rule": "smallest CL term set mapping >=95% of every atlas on disk (docs/SPEC-step6-register-v2.md §2)",
        "measured_from": str(cache_root.relative_to(ROOT)),
        "resources": {
            "cl": {"url": CL_URL, "digest": digests["cl"]},
            "cellmarker": {"version": CELLMARKER_VERSION, "page": CELLMARKER_PAGE, "digests": digests},
            "sc-type": {"version": SCTYPE_VERSION, "url": SCTYPE_URL, "digest": digests["sctype"]},
        },
        "missing_atlases": missing,
        # The terms a marker resource could ground — the input the rule's
        # "scope ⊆ defined" constraint is taken against. Committed so the scope can
        # be re-derived without the atlases or the network.
        "defined": sorted(groundable),
        **plan.payload(),
    }
    COVERAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_table(plan, missing)
    return 0


def _cached_or_die(url: str, target: Path) -> Path:
    if not target.exists():
        raise SystemExit(f"--no-download: {target} is not cached (would have fetched {url})")
    return target


def _atlas_tissue(counts, atlas_id: str, cl_id: str) -> bool:
    return any(atlas.atlas_id == atlas_id and cl_id in atlas.labels for atlas in counts)


def _tissue_register(cm_paths, tissue_class: str, size: int, groundable: set[str], in_scope: set[str]) -> list[str]:
    """A tissue list from the resource's own tissue scoping: the CL ids CellMarker
    files under that tissue, ranked by how much marker evidence it has for them."""
    evidence: Counter = Counter()
    for path in cm_paths.values():
        for row in read_xlsx(path):
            if row.get("tissue_class") != tissue_class or row.get("cancer_type") != "Normal":
                continue
            cl_id = "CL:" + row.get("cellontology_id", "").replace("CL_", "")
            if cl_id in in_scope and cl_id in groundable:
                evidence[cl_id] += 1
    return [cl_id for cl_id, _ in sorted(evidence.items(), key=lambda item: (-item[1], item[0]))[:size]]


def _definitions_seed(payload: dict, digests: dict) -> dict:
    return {
        "source": (
            f"curated from {CELLMARKER_VERSION} (markers, via the resource's own cellontology_id column) and "
            f"{SCTYPE_VERSION} (fallback); text is the Cell Ontology release's own definition "
            f"(cl-basic.json sha256 {digests['cl'][:12]}). Never derived from a benchmark atlas."
        ),
        "version": "v2",
        "synonyms": yaml.safe_load((paths.seed_dir() / "definitions.seed.yaml").read_text(encoding="utf-8"))["synonyms"],
        "definitions": payload["definitions"],
    }


def _tissue_seed(tissues: dict, digests: dict) -> dict:
    return {
        "source": (
            "tissue -> candidate CL ids (paper_plan §4.5). v1's three lists keep their scope; pancreas is the "
            "set the 95% rule demanded for muraro, brain is the CL terms CellMarker files under its Brain "
            f"tissue_class (sc-type sha256 {digests['sctype'][:12]}). Every id is drawn from the v2 scope."
        ),
        "version": "v2",
        "tissues": tissues,
    }


def _dump(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


def _print_table(plan, missing: list[str]) -> None:
    header = f"{'atlas':22s} {'cells':>9s} {'terms':>6s} {'v1 %':>8s} {'v2 %':>8s} {'dropped':>9s} {'ungrounded':>11s}"
    print(f"\n{header}")
    for atlas in plan.atlases:
        print(
            f"{atlas.atlas_id:22s} {atlas.cells:9d} {atlas.distinct_terms:6d} {atlas.required_rate:7.2%} "
            f"{atlas.in_scope_rate:7.2%} {atlas.dropped_cells:9d} {atlas.undefined_cells:11d}"
        )
    print(f"scope: {len(plan.in_scope)} terms; gate met by "
          f"{sum(1 for a in plan.atlases if a.meets())}/{len(plan.atlases)} atlases")
    print(f"missing from disk (not measured, not invented): {', '.join(missing) or 'none'}")


if __name__ == "__main__":
    raise SystemExit(main())
