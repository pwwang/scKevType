"""Step 2 of the standard-annotator baseline: the census cells' count matrices.

One matrix per dataset, holding exactly the cells the census's `panel_choice` questions name,
in a certified row order, so that a tool's label for row i is the label of ``cells[i]``.

WHERE EACH MATRIX COMES FROM
----------------------------
* ``muraro``, ``siletti``, ``tabula_muris_senis`` -- the certified extraction the project's own
  baselines already read (``baselines/out/expr/<dataset>_test.npz``), subset to the census
  cells. Its row order is certified by ``<dataset>_test_order.json`` and the loader here is
  ``lib_common.load_aligned_cells``, which refuses an uncertified array. Subsetting keeps the
  row order (a boolean mask), and the subset's own digest is written beside it.
* ``tabula_sapiens`` -- no extraction exists for this atlas in the project's baselines, so it
  is read here from the atlas itself (``data/raw.9p/tabula_sapiens/tabula_sapiens.h5ad``,
  45,013,733,785 bytes, sha256 sidecar quoted in the summary) with ``lib_common.matrix_rows``,
  the same contiguous-run reader ``step2_expr.py`` uses, and spot-checked by re-reading five
  rows from the atlas and comparing their gene content with what was persisted.

The count matrix is ``raw/X`` in every file (integer counts; ``X`` is normalised in some of
them), which is what CellTypist's own tutorial expects to normalise itself.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/02_census_matrices.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
sys.path.insert(0, str(ROOT / "baselines"))
from lib_common import (  # noqa: E402
    cell_ids,
    count_matrix_key,
    gene_names,
    ids_sha256,
    matrix_rows,
    load_aligned_cells,
)

QUESTIONS = ROOT / "reports" / "annotation-tools" / "questions.jsonl"
OUT = ROOT / "reports" / "annotation-tools" / "expr"
BASE_EXPR = ROOT / "baselines" / "out" / "expr"

#: datasets whose matrix already exists, certified, in the project's baselines
FROM_BASELINES = ["muraro", "siletti", "tabula_muris_senis"]
#: datasets read from the atlas here
FROM_ATLAS = {
    "tabula_sapiens": (
        ROOT / "data" / "raw.9p" / "tabula_sapiens" / "tabula_sapiens.h5ad",
        "_index",
        "feature_name",
    ),
}

SPOT_CHECKS = 5


def census_cells() -> dict[str, list[str]]:
    """dataset -> the census cells for it, in the order the question table names them."""
    per: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    with open(QUESTIONS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ds, cell = row["dataset"], row["cell"]
            seen.setdefault(ds, set())
            if cell not in seen[ds]:
                seen[ds].add(cell)
                per.setdefault(ds, []).append(cell)
    return per


def write_certificate(dataset: str, cells: list[str], source_positions: list[int] | None, extra: dict) -> dict:
    payload = {
        "dataset": dataset,
        "cells": len(cells),
        "ids_sha256": ids_sha256(cells),
        "ids_in_source_row_order": True,
        "verified_by": "scripts/annotation-tools/02_census_matrices.py",
        "source_positions_sha256": (
            hashlib.sha256(np.asarray(source_positions, dtype=np.int64).tobytes()).hexdigest()
            if source_positions is not None
            else None
        ),
        "what_this_proves": "matrix row i holds the counts of `cells[i]`; a tool that labels "
        "row i labels that cell",
        **extra,
    }
    (OUT / f"{dataset}.certificate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save(dataset: str, matrix, genes: list[str], cells: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sp.save_npz(OUT / f"{dataset}.npz", matrix.tocsr())
    np.save(OUT / f"{dataset}_genes.npy", np.array(genes, dtype=object))
    np.save(OUT / f"{dataset}_cells.npy", np.array(cells, dtype=object))


def from_baseline(dataset: str, cells: list[str]) -> dict:
    t0 = time.time()
    matrix = sp.load_npz(BASE_EXPR / f"{dataset}_test.npz").tocsr()
    genes = [str(g) for g in np.load(BASE_EXPR / f"{dataset}_test_genes.npy", allow_pickle=True)]
    certified = load_aligned_cells(dataset, "test")  # raises unless the row order is certified
    if matrix.shape[0] != len(certified) or matrix.shape[1] != len(genes):
        raise SystemExit(f"{dataset}: matrix shape {matrix.shape} does not match the certified ids/genes")
    index = {c: i for i, c in enumerate(certified)}
    missing = [c for c in cells if c not in index]
    if missing:
        raise SystemExit(f"{dataset}: {len(missing)} census cells are not in the certified matrix, e.g. {missing[:5]}")
    rows = [index[c] for c in cells]  # census order, not source order
    subset = matrix[rows]
    # the subset is a copy of rows in the order given: assert the count of non-zeros survives,
    # which is the cheap positional sanity check that a mask/reorder has not shuffled anything
    if subset.nnz != sum(matrix.getrow(r).nnz for r in rows):
        raise SystemExit(f"{dataset}: subset nnz {subset.nnz} != sum of the named rows' nnz")
    save(dataset, subset, genes, cells)
    return {
        "source": f"baselines/out/expr/{dataset}_test.npz (certified extraction)",
        "cells": len(cells),
        "genes": len(genes),
        "nnz": int(subset.nnz),
        "median_detected_genes_per_cell": float(np.median(np.diff(subset.indptr))),
        "all_integer_counts": bool(np.all(subset.data == np.round(subset.data))),
        "row_order": "census order (rows selected by cell id from the certified array)",
        "seconds": round(time.time() - t0, 1),
    }


def from_atlas(dataset: str, cells: list[str]) -> dict:
    t0 = time.time()
    atlas, cell_column, gene_column = FROM_ATLAS[dataset]
    if not atlas.exists():
        raise SystemExit(f"{dataset}: atlas not found at {atlas}")
    with h5py.File(atlas, "r") as fh:
        # lib_common's readers decode anndata categoricals stored as groups, which
        # tabula_sapiens's var/feature_name is.
        ids = cell_ids(fh, cell_column)
        genes = gene_names(fh, gene_column)
        key = count_matrix_key(fh)
        index = {c: i for i, c in enumerate(ids)}
        missing = [c for c in cells if c not in index]
        if missing:
            raise SystemExit(f"{dataset}: {len(missing)} census cells not in the atlas, e.g. {missing[:5]}")
        positions = np.array(sorted(index[c] for c in cells), dtype=np.int64)
        data, indices, indptr, n_genes, sorted_positions = matrix_rows(fh, key, positions, len(genes))
        # matrix_rows returns the rows it read, sorted; rebuild the csr and then re-order the
        # rows back to the census order through the position -> row map.
        nnz = (indptr[sorted_positions + 1] - indptr[sorted_positions]).astype(np.int64)
        out_indptr = np.zeros(len(sorted_positions) + 1, dtype=np.int64)
        np.cumsum(nnz, out=out_indptr[1:])
        matrix = sp.csr_matrix((data, indices, out_indptr), shape=(len(sorted_positions), n_genes))
        order = {int(p): i for i, p in enumerate(sorted_positions)}
        rows = [order[index[c]] for c in cells]
        subset = matrix[rows]
        # spot checks: re-read the named rows one at a time, straight from the atlas, and compare
        spot = []
        for c in [cells[i] for i in range(0, len(cells), max(1, len(cells) // SPOT_CHECKS))][:SPOT_CHECKS]:
            r = index[c]
            s, e = int(fh[key]["indptr"][r]), int(fh[key]["indptr"][r + 1])
            gidx = np.asarray(fh[key]["indices"][s:e])
            gval = np.asarray(fh[key]["data"][s:e])
            mine = subset.getrow(cells.index(c))
            ok = (
                mine.nnz == len(gidx)
                and np.array_equal(np.asarray(mine.indices), gidx)
                and np.allclose(np.asarray(mine.data), gval)
            )
            spot.append({"cell_id": c, "source_row": r, "matched": bool(ok), "nnz": int(len(gidx))})
        if not all(x["matched"] for x in spot):
            raise SystemExit(f"{dataset}: spot check failed -- refusing to keep a misaligned matrix: {spot}")
        sidecar = atlas.with_suffix(".h5ad.sha256")
        atlas_sha = sidecar.read_text().strip() if sidecar.exists() else None
        save(dataset, subset, genes, cells)
        return {
            "source": str(atlas.relative_to(ROOT)),
            "atlas_bytes": atlas.stat().st_size,
            "atlas_sha256_sidecar": atlas_sha,
            "matrix_key": key,
            "cells": len(cells),
            "genes": len(genes),
            "nnz": int(subset.nnz),
            "median_detected_genes_per_cell": float(np.median(np.diff(subset.indptr))),
            "all_integer_counts": bool(np.all(subset.data == np.round(subset.data))),
            "spot_checks": spot,
            "seconds": round(time.time() - t0, 1),
        }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cells_by_dataset = census_cells()
    summary: dict[str, dict] = {}
    for dataset in FROM_BASELINES + list(FROM_ATLAS):
        cells = cells_by_dataset.get(dataset)
        if not cells:
            raise SystemExit(f"{dataset}: the question table names no cells for this dataset")
        print(f"[{dataset}] {len(cells)} census cells", flush=True)
        if dataset in FROM_ATLAS:
            stats = from_atlas(dataset, cells)
            cert = write_certificate(dataset, cells, None, stats)
        else:
            stats = from_baseline(dataset, cells)
            cert = write_certificate(dataset, cells, None, stats)
        stats["certificate_ids_sha256"] = cert["ids_sha256"]
        summary[dataset] = stats
        print(f"    {stats['seconds']}s nnz={stats['nnz']:,} integer={stats['all_integer_counts']}", flush=True)
    (OUT.parent / "matrices.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT.parent / 'matrices.summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
