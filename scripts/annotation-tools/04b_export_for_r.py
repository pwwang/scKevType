"""Step 4b: export the population's matrices in a form R can read, for SingleR.

MatrixMarket (genes x cells, the orientation SingleR wants), one directory per dataset under
``<dir>/r-input/<dataset>/``:

    matrix.mtx    genes x cells, integer counts
    genes.txt     one gene symbol per line, in matrix row order
    cells.txt     one cell id per line, in matrix column order

The dataset list is read off the question table, so the export covers exactly the population's
datasets (four for the v2 population, six for the v4 one) rather than a name list kept here.

The ``_genes.npy`` / ``_cells.npy`` arrays are this pipeline's own persisted string lists under
the working directory, which are object-dtype and so need ``allow_pickle``; they are read by
this repo's own writer (``02_census_matrices.py``) and by nothing external.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/04b_export_for_r.py

    # the v4 population
    uv run --no-sync python scripts/annotation-tools/04b_export_for_r.py --dir reports/annotation-tools-v4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
DEFAULT_DIR = ROOT / "reports" / "annotation-tools"


def population_datasets(questions: Path) -> list[str]:
    names: list[str] = []
    with open(questions, encoding="utf-8") as fh:
        for line in fh:
            dataset = json.loads(line)["dataset"]
            if dataset not in names:
                names.append(dataset)
    if not names:
        raise SystemExit(f"{questions}: no datasets -- refusing to export nothing")
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the population's matrices for R.")
    parser.add_argument("--dir", default=str(DEFAULT_DIR),
                        help="the working directory: expr/ is read from it, r-input/ written to it")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    work = Path(args.dir).resolve()
    expr = work / "expr"
    out_root = work / "r-input"
    datasets = population_datasets(work / "questions.jsonl")

    summary = {}
    for dataset in datasets:
        matrix = sp.load_npz(expr / f"{dataset}.npz").tocsr()
        genes = [str(g) for g in np.load(expr / f"{dataset}_genes.npy", allow_pickle=True)]
        cells = [str(c) for c in np.load(expr / f"{dataset}_cells.npy", allow_pickle=True)]
        d = out_root / dataset
        d.mkdir(parents=True, exist_ok=True)
        # genes x cells: transpose, and store as integer counts (they are integer-valued)
        genes_by_cells = sp.csc_matrix(matrix.T.astype(np.int32))
        sio.mmwrite(d / "matrix.mtx", genes_by_cells, field="integer")
        (d / "genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
        (d / "cells.txt").write_text("\n".join(cells) + "\n", encoding="utf-8")
        summary[dataset] = {
            "genes": len(genes),
            "cells": len(cells),
            "nnz": int(matrix.nnz),
            "orientation": "genes x cells",
            "dir": str(d),
        }
        print(f"[{dataset}] {len(genes)} genes x {len(cells)} cells -> {d}")
    (out_root / "export.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
