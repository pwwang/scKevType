"""Shared helpers for the standard-annotator baselines (CellTypist, LR-on-HVGs).

Nothing here imports sckevtype: the baselines must be readable as *standard*
practice, not as a second copy of the project's own pipeline. The only inputs
are the frozen corpus (``runs/real-v2/*.jsonl``), the CL artefacts the corpus
itself was built against (``artifacts/v2/*.json``) and the raw atlases.

Reads ``raw/X`` (raw integer counts) from every atlas, not ``X``. ``X`` is not
comparable across these six files -- measured: muraro's ``X`` is a normalised
float matrix, siletti's ``X`` is raw integer counts -- whereas ``raw/X`` is
integer counts in all six. The corpus's own states are ranks of ``X``; ranks are
insensitive to the normalisation, and a count matrix is what both baselines
require. This is stated in baselines/README.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(os.environ.get("SCKEVTYPE_ROOT", "/mnt/f/E/hermes-workspace/scKevType"))
CORPUS = ROOT / "runs" / "real-v2"
ARTIFACTS = ROOT / "artifacts" / "v2"
QCOUT = ROOT / "baselines" / "out"
EXPR = QCOUT / "expr"

#: dataset id -> (h5ad path relative to ROOT, obs cell-id column, var gene-symbol column).
#: The obs column is the one the corpus's own adapter resolves (``_index`` unless the
#: registry overrides it; configs/corpus/real-v2.yaml sets siletti's ``cell_column: CellID``).
DATASETS = {
    "muraro": ("data/raw/muraro/muraro.h5ad", "_index", "feature_name"),
    "hao_cite": ("data/raw/hao_cite/hao_cite.h5ad", "_index", "feature_name"),
    "siletti": ("data/raw/siletti/siletti.h5ad", "CellID", "feature_name"),
    "tabula_sapiens": ("data/raw/tabula_sapiens/tabula_sapiens.h5ad", "_index", "feature_name"),
    "tabula_muris_senis": ("data/raw/tabula_muris_senis/tabula_muris_senis_10x.h5ad", "index", "feature_name"),
    "dominguez_conde": ("data/raw/dominguez_conde_cxg/global.h5ad", "_index", "feature_name"),
}

INDEX_KEYS = ("_index", "index")

#: Every arm this pipeline names, carrying the two tags that have to appear wherever its
#: number appears. ``training_regime``: **in-domain** means fitted on the same atlas it is
#: scored on (it uses that atlas's own labels, which no honest baseline may do, so it is a
#: reference ceiling rather than a competitor); **transfer** means fitted on hao_cite and
#: scored on atlases it has never seen, which is what our model does. ``input``: **full
#: matrix** means the arm reads the whole expression matrix, **fixed panel** means it sees
#: only the corpus's marker panel -- and the full-matrix arms are the ones with the easier job.
ARM_TAGS: dict[str, dict[str, str]] = {
    "splitexact": {
        "training_regime": "transfer",
        "input": "full matrix",
        "what": "logistic regression on 2000 HVGs, fitted on the corpus's hao_cite train split "
        "(our model's own training atlas), scored on unseen atlases",
    },
    "atlas": {
        "training_regime": "transfer",
        "input": "full matrix",
        "what": "the same fit on every in-scope hao_cite cell of the corpus's train tier "
        "(~70x the cells), still with no test atlas in the fit",
    },
    "cv_indomain": {
        "training_regime": "in-domain",
        "input": "full matrix",
        "what": "5-fold CV stratified over CELLS inside the scored atlas, HVGs re-selected inside "
        "each fold -- so cells from one donor sit on both sides of a fold. Reference ceiling only",
    },
    "cv_grouped": {
        "training_regime": "in-domain",
        "input": "full matrix",
        "what": "leave-one-donor-out CV inside the scored atlas, HVGs re-selected inside each "
        "training fold. Reference ceiling only, and the honest in-domain number",
    },
    "celltypist": {
        "training_regime": "transfer",
        "input": "full matrix",
        "what": "a pretrained CellTypist model, never fitted on any corpus cell, predicting from "
        "the full matrix",
    },
    "panel_lr": {
        "training_regime": "transfer",
        "input": "fixed panel",
        "what": "step 6: the corpus's own task shape (the record's rank-list state plus the "
        "panel's option definitions) fitted on the train split and scored on unseen atlases. "
        "Reads no expression matrix at all -- the only arm with both of our model's tags",
    },
    "ours": {
        "training_regime": "transfer",
        "input": "fixed panel",
        "what": "our model: trained on hao_cite, scored on four unseen atlases, reading the "
        "corpus's 201-token rank list plus the panel's option definitions",
    },
}

#: Quote this wherever a baseline number appears, not in a footnote.
ASYMMETRY = (
    "Every baseline arm here reads the FULL expression matrix; our model reads a fixed marker "
    "panel. That asymmetry favours the baselines and has to be stated next to their numbers."
)
TAG_LEGEND = (
    "Tags on every row: training_regime = in-domain (fitted on the same atlas it is scored on; "
    "uses that atlas's labels and is a reference ceiling only) or transfer (fitted on hao_cite, "
    "scored on unseen atlases -- our model's regime). input = full matrix (the arm may read every "
    "gene) or fixed panel (only the corpus's marker panel)."
)


def arm_kind(name: str) -> str:
    """``lr_cv_grouped/muraro`` -> ``cv_grouped``; ``celltypist_mtg/muraro`` -> ``celltypist``.

    The one resolver: every artifact names its arms with a ``<family>_<variant>/<dataset>`` key,
    and a second copy of this rule is how one artifact ends up tagging an arm differently from
    another. Names with no known prefix are returned unchanged so the raise below names them.
    """
    head = name.split("/", 1)[0]
    if head.startswith("lr_"):
        return head[3:]
    if head.startswith("celltypist_"):
        return "celltypist"
    return head


def arm_tags(name: str) -> dict[str, str]:
    """The two tags of one arm -- and a **hard error** when the arm is unregistered.

    A missing tag is the failure this exists to prevent, so it is not defaulted: an untagged
    number is one a reader cannot tell a transfer measurement from an in-domain ceiling, and
    every guard in this pipeline rests on the tags being present rather than on nobody having
    added an arm yet. ``ARM_TAGS.get(...) or {}`` is how a table acquires an empty tag cell
    that reads as a formatting glitch instead of a missing guarantee.
    """
    tags = ARM_TAGS.get(arm_kind(name))
    if not tags:
        raise KeyError(
            f"{name!r} (kind {arm_kind(name)!r}) has no entry in lib_common.ARM_TAGS -- register the arm "
            f"before any number from it is written into a table or a results file. Known arms: {sorted(ARM_TAGS)}"
        )
    return tags


def tag_string(name: str) -> str:
    """The two tags of one arm, for a table cell or a log line."""
    tags = arm_tags(name)
    return f"{tags['training_regime']}, {tags['input']}"



def decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "decode") and not isinstance(value, str):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _column_node(handle: h5py.File, group: str, name: str):
    """The node for one column: under ``group`` when it is there, else top level."""
    if group in handle and name in handle[group]:
        return handle[group][name]
    return handle[name]


def obs_column(handle: h5py.File, name: str, group: str = "obs") -> list:
    """One obs/var column, decoding an anndata categorical stored as a group."""
    node = _column_node(handle, group, name)
    if isinstance(node, h5py.Group):  # categorical
        categories = [decode(v) for v in node["categories"][:]]
        return [categories[c] if c >= 0 else None for c in node["codes"][:]]
    values = node[:]
    if getattr(values, "ndim", 0) == 0:
        return [decode(values.item())]
    return [None if v is None else decode(v) for v in values]


def cell_ids(handle: h5py.File, column: str) -> list[str]:
    names = [str(k) for k in handle["obs"]]
    for candidate in (column,) + INDEX_KEYS:
        if candidate in names:
            return [decode(v) for v in obs_column(handle, candidate)]
    raise ValueError(f"{handle.filename}: no cell-id column among {(column,)+INDEX_KEYS}")


def gene_names(handle: h5py.File, column: str) -> list[str]:
    for candidate in (column, "feature_name", "gene_symbols", "gene_symbol", "symbol"):
        if candidate in handle["var"]:
            return [decode(v) for v in obs_column(handle, candidate, "var")]
    raise ValueError(f"{handle.filename}: no gene-symbol column among var keys {list(handle['var'])}")


def count_matrix_key(handle: h5py.File) -> str:
    """The key of the raw *count* matrix: ``raw/X`` where the file has one, else ``X``.

    Measured across these six atlases: ``raw/X`` holds integer counts in the five
    files that have a ``raw`` group, and siletti (the one file without one) keeps
    integer counts in ``X`` -- while ``X`` elsewhere is normalised and not
    comparable. The caller asserts the chosen matrix really is integer-valued.
    """
    if "raw" in handle and isinstance(handle["raw"], h5py.Group) and "X" in handle["raw"]:
        return "raw/X"
    if "X" in handle:
        return "X"
    raise ValueError(f"{handle.filename}: no X and no raw/X")


def matrix_rows(handle: h5py.File, key: str, rows: np.ndarray, n_genes: int):
    """Read the given (unsorted, unique) row indices of a CSR matrix at ``key``.

    Contiguous runs of rows are read as single block slices of
    ``data``/``indices``, so the number of HDF5 reads is the number of runs, not
    the number of cells -- which is what makes reading 3,000 rows out of a 2.5M-row,
    31 GB file take seconds rather than minutes.
    """
    grp = handle[key]
    if not isinstance(grp, h5py.Group):
        return np.asarray(grp[:])[rows], None
    indptr = grp["indptr"][:]
    rows = np.sort(np.asarray(rows))
    out_data, out_indices = [], []
    run_start = 0
    while run_start < len(rows):
        run_end = run_start
        while run_end + 1 < len(rows) and rows[run_end + 1] == rows[run_end] + 1:
            run_end += 1
        lo, hi = int(rows[run_start]), int(rows[run_end])
        s, e = int(indptr[lo]), int(indptr[hi + 1])
        out_data.append(grp["data"][s:e])
        out_indices.append(grp["indices"][s:e])
        run_start = run_end + 1
    data = np.concatenate(out_data) if out_data else np.array([], dtype=np.float32)
    indices = np.concatenate(out_indices) if out_indices else np.array([], dtype=np.int32)
    return (data, indices, indptr, n_genes, rows)


def cl_names() -> dict[str, str]:
    """CL id -> CL name, from the snapshot the corpus was built against."""
    payload = json.loads((ARTIFACTS / "cl_terms.json").read_text(encoding="utf-8"))
    return {t["cl_id"]: t["name"] for t in payload["terms"]}


def cl_parents() -> dict[str, list[str]]:
    """CL id -> direct parent CL ids, from the same snapshot."""
    payload = json.loads((ARTIFACTS / "cl_terms.json").read_text(encoding="utf-8"))
    return {t["cl_id"]: list(t.get("parents") or []) for t in payload["terms"]}


def cl_ancestors(cl_id: str, parents: dict[str, list[str]] | None = None) -> set[str]:
    """Transitive CL ancestors of ``cl_id`` (cycle-safe; the id itself is not included)."""
    parents = parents if parents is not None else cl_parents()
    seen: set[str] = set()
    stack = list(parents.get(cl_id, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(parents.get(node, ()))
    return seen


def singularise(token: str) -> str:
    """Crude English singular for cell-type nouns: monocytes -> monocyte, erythrocytes -> erythrocyte."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalise_label(text: str) -> str:
    """Case-, punctuation- and plural-insensitive key for a cell-type name.

    Deliberately mechanical: no stemming of anything but plural 's', no synonym
    knowledge, so a match means the two strings are the same name. Any looser
    correspondence is a hand decision and lives in celltypist_label_map.json.
    """
    text = text.lower().replace("/", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(singularise(token) for token in text.split())


def base_cluster_label(label: str) -> str | None:
    """The label without a trailing cluster index: 'Pvalb_12' -> 'Pvalb', else None.

    Some CellTypist models (Adult_Human_MTG, Mouse_Whole_Brain) label per *cluster* rather
    than per cell type, e.g. 'Sst Chodl_2'. The curated table holds the base label once,
    and this rule applies it, so the mapping is stated once instead of four hundred times.
    """
    match = re.match(r"^(.*?)_\d+$", label)
    return match.group(1) if match else None


def resolve_celltypist_map(model: str) -> dict[str, dict]:
    """label -> {cl_id, basis, reason} for one CellTypist model.

    Curated entries (celltypist_label_map.json) win -- first an exact label, then the
    label's cluster base; everything else that resolves by exact normalised name is
    ``basis='name'``. Every cl_id is validated against the CL snapshot -- an id that is
    not a CL term is a hard error.
    """
    names = cl_names()
    defs = definitions()
    by_normalised: dict[str, list[str]] = {}
    for cl_id, d in defs.items():
        by_normalised.setdefault(normalise_label(d["name"]), []).append(cl_id)
    curated = json.loads((ROOT / "baselines" / "celltypist_label_map.json").read_text(encoding="utf-8"))
    table = curated["models"].get(model, {})
    for label, entry in table.items():
        if entry["cl_id"] not in names:
            raise SystemExit(f"celltypist_label_map.json: {label!r} -> {entry['cl_id']} is not a CL term")
    out: dict[str, dict] = {label: dict(entry, source="curated") for label, entry in table.items()}
    payload = json.loads((QCOUT / "celltypist_models.json").read_text(encoding="utf-8"))
    for label in payload[model]["labels"]:
        if label in out:
            continue
        base = base_cluster_label(label)
        if base is not None and base in table:
            out[label] = dict(table[base], source="curated-base", cluster_label=label, base_label=base)
            continue
        hit = by_normalised.get(normalise_label(label))
        if hit and len(hit) == 1:
            out[label] = {"cl_id": hit[0], "basis": "name", "reason": "exact normalised name match", "source": "mechanical"}
    return out


def definitions() -> dict[str, dict]:
    """CL id -> {name, text, pos_markers, neg_markers} as the panels render them."""
    payload = json.loads((ARTIFACTS / "definitions.json").read_text(encoding="utf-8"))
    return {d["cl_id"]: d for d in payload["definitions"]}


def name_to_cl() -> dict[str, str]:
    """CL name -> CL id. Measured unambiguous: 82 definitions, 82 distinct names."""
    out: dict[str, str] = {}
    for cl_id, d in definitions().items():
        out[d["name"]] = cl_id
    return out


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------- row-order certification
#
# The persisted matrices are `(rows = cells) x (columns = genes)`. Consumers index those
# rows **by position** and join the cell's label by the id at the same position, so exactly
# one thing has to be true of the pair on disk: the id at position i is the cell whose
# expression is row i. That is what "row order" means here -- the ids sorted by their
# position in the source atlas's own matrix, which is the order the readers emit rows in.
#
# This was violated once: `step2_expr.extract` assembled rows in ascending source-row order
# while saving the ids in *request* order, so 1,979 of muraro's 2,126 ids sat at the wrong
# index and every positional consumer attached its labels to the wrong cells. The repair is
# certified by content (`fix_cell_order.py`) and, from now on, by construction (`step2`),
# and every consumer asserts the certificate before using an array. A one-off probe proves
# it once; this proves it on every run.


def expr_paths(dataset: str, split: str) -> dict[str, Path]:
    """The four files one (dataset, split) extraction consists of, including its certificate."""
    stem = f"{dataset}_{split}"
    return {
        "matrix": EXPR / f"{stem}.npz",
        "cells": EXPR / f"{stem}_cells.npy",
        "genes": EXPR / f"{stem}_genes.npy",
        "order": EXPR / f"{stem}_order.json",
    }


def ids_sha256(ids) -> str:
    """A digest over the cell-id *sequence*: the array's identity, order included."""
    digest = hashlib.sha256()
    for cell in ids:
        digest.update(str(cell).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def positions_sha256(positions) -> str:
    """A digest over the source-matrix row indices the persisted rows were read from."""
    array = np.ascontiguousarray(np.asarray(positions, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_order_sidecar(
    dataset: str, split: str, ids, positions, verified_by: str, checks: dict, extra: dict | None = None
) -> dict:
    """Record what was verified about this matrix's row order, so consumers can assert it."""
    payload = {
        "dataset": dataset,
        "split": split,
        "rows": len(ids),
        "ids_sha256": ids_sha256(ids),
        "row_positions_sha256": positions_sha256(positions),
        "ids_in_source_row_order": True,
        "row_positions_strictly_ascending": True,
        "verified_by": verified_by,
        "checks": checks,
        "what_this_proves": "matrix row i holds source-matrix row `row_positions[i]`, and "
        "`_cells.npy[i]` is that row's own cell id -- so a consumer that labels row i with the "
        "label of `_cells.npy[i]` labels the right cell",
        "sample_rows": _sample_rows(ids, positions),
    }
    if extra:
        payload.update(extra)
    write_json(expr_paths(dataset, split)["order"], payload)
    return payload


def _sample_rows(ids, positions, k: int = 3) -> list[dict]:
    step = max(1, len(ids) // (k + 1))
    picks = list(range(0, len(ids), step))[:k]
    return [
        {"row": int(i), "source_row": int(positions[i]), "cell_id": str(ids[i])} for i in picks
    ]


def load_aligned_cells(dataset: str, split: str = "test") -> list[str]:
    """The persisted cell ids, only if their row order is certified -- else a hard error.

    A consumer must never index this matrix positionally on the strength of the file name:
    an array saved in request order looks identical (right length, right ids, wrong order)
    and silently mislabels every row. The certificate is written by whichever step wrote the
    matrix (``step2_expr.extract`` by construction, ``fix_cell_order.py`` by content check)
    and pins both the id sequence and the source rows it was read from.
    """
    paths = expr_paths(dataset, split)
    if not paths["cells"].exists():
        raise SystemExit(f"{dataset}/{split}: no persisted cell ids at {paths['cells']}")
    ids = [str(c) for c in np.load(paths["cells"], allow_pickle=True)]
    if not paths["order"].exists():
        raise SystemExit(
            f"{dataset}/{split}: {paths['cells'].name} has NO row-order certificate "
            f"({paths['order'].name} is missing), so nothing here can tell whether row i is cell i. "
            f"Repair/certify it with: uv run --no-sync python baselines/fix_cell_order.py {dataset}:{split}"
        )
    payload = json.loads(paths["order"].read_text(encoding="utf-8"))
    if payload.get("rows") != len(ids):
        raise SystemExit(
            f"{dataset}/{split}: certificate says {payload.get('rows')} rows, the file holds {len(ids)} ids "
            "-- the certificate is stale; re-run step2 or fix_cell_order"
        )
    if payload.get("ids_sha256") != ids_sha256(ids):
        raise SystemExit(
            f"{dataset}/{split}: {paths['cells'].name} is not the array the certificate covers "
            f"(ids_sha256 differs) -- the file was rewritten without re-certifying the row order"
        )
    if not payload.get("ids_in_source_row_order") or not payload.get("row_positions_strictly_ascending"):
        raise SystemExit(f"{dataset}/{split}: the certificate does not assert row order ({payload})")
    return ids
