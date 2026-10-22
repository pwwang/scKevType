"""Step 4b: export the census matrices in a form R can read, for SingleR.

MatrixMarket (genes x cells, the orientation SingleR wants), one directory per dataset under
``reports/annotation-tools/r-input/<dataset>/``:

    matrix.mtx    genes x cells, integer counts
    genes.txt     one gene symbol per line, in matrix row order
    cells.txt     one cell id per line, in matrix column order

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/04b_export_for_r.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
EXPR = ROOT / "reports" / "annotation-tools" / "expr"
OUT = ROOT / "reports" / "annotation-tools" / "r-input"

DATASETS = ["muraro", "siletti", "tabula_muris_senis", "tabula_sapiens"]


def main() -> int:
    summary = {}
    for dataset in DATASETS:
        matrix = sp.load_npz(EXPR / f"{dataset}.npz").tocsr()
        genes = [str(g) for g in np.load(EXPR / f"{dataset}_genes.npy", allow_pickle=True)]
        cells = [str(c) for c in np.load(EXPR / f"{dataset}_cells.npy", allow_pickle=True)]
        d = OUT / dataset
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
            "dir": str(d.relative_to(ROOT)),
        }
        print(f"[{dataset}] {len(genes)} genes x {len(cells)} cells -> {d.relative_to(ROOT)}")
    (OUT / "export.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
