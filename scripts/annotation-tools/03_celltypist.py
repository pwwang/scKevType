"""Step 3 of the standard-annotator baseline: CellTypist on the census cells.

WHAT IT DOES
------------
Runs CellTypist's pretrained logistic-regression models, one arm per (dataset, model), over
exactly the cells the census's `panel_choice` questions name, and persists each cell's
predicted label, the CL id that label maps to, and the full label-probability matrix -- so the
scoring step can ask both the unconstrained question ("is the tool's own top-1 the answer?")
and the panel-restricted one ("is its argmax over the question's options the answer?").

THE ARMS ARE PRE-DECLARED, NOT CHOSEN AFTER SEEING SCORES
---------------------------------------------------------
They are exactly the arms the project's own CellTypist baseline declared
(`baselines/step3_celltypist.py`, `ARMS`), extended to `tabula_sapiens` with the same two arms
that file declares for it, because the extraction those arms need did not exist until
`02_census_matrices.py` produced it:

    muraro              Adult_Human_PancreaticIslet (tissue-matched), Fetal_Human_Pancreas,
                        Immune_All_Low (CellTypist's documented default)
    siletti             Adult_Human_MTG (tissue-matched), Immune_All_Low
    tabula_muris_senis  Adult_Mouse_Gut (the only mouse tissue model), Immune_All_Low with
                        case-matched (uppercased) symbols
    tabula_sapiens      Immune_All_Low (its reference is immune; 311 of these 439 cells are
                        immune-compartment), Human_Lung_Atlas (the only human tissue model
                        whose tissue the corpus's tabula_sapiens cells cover)

The report quotes, per dataset, both the default-model arm and the **best arm** (the maximum
over these arms), because the referee's objection is strongest in its best form.

INPUT
-----
The count matrices from `02_census_matrices.py` (`raw/X`, integer counts), normalised here the
way CellTypist's own tutorial normalises: ``normalize_total(1e4)`` then ``log1p``. Duplicate
gene symbols are summed first (CellTypist requires unique ``var_names``) -- the same
aggregation the project's baseline uses, imported from it rather than re-implemented.

OUTPUT
------
``reports/annotation-tools/celltypist/<dataset>_<arm>.npz`` with ``cells``, ``predicted``,
``mapped_cl``, ``basis``, ``labels``, ``probabilities``; plus
``reports/annotation-tools/celltypist.summary.json``.

Run (WSL, from the repo root; celltypist is not in the venv, so uv's ephemeral layer supplies it):

    source scripts/wsl/env.sh
    uv run --with celltypist==1.7.1 --with scanpy==1.12.4 --no-sync \
        python scripts/annotation-tools/03_celltypist.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
sys.path.insert(0, str(ROOT / "baselines"))
from lib_common import resolve_celltypist_map  # noqa: E402
from step3_celltypist import collapse_duplicate_genes  # noqa: E402

EXPR = ROOT / "reports" / "annotation-tools" / "expr"
OUT = ROOT / "reports" / "annotation-tools" / "celltypist"

#: dataset -> [(arm, model file, symbol transform, why)]
ARMS: dict[str, list[tuple[str, str, str, str]]] = {
    "muraro": [
        ("pancreatic_islet", "Adult_Human_PancreaticIslet.pkl", "none",
         "tissue-matched: all 288 census muraro cells are pancreas"),
        ("fetal_pancreas", "Fetal_Human_Pancreas.pkl", "none",
         "developmental pancreas, the project baseline's secondary matched arm"),
        ("immune_low", "Immune_All_Low.pkl", "none",
         "CellTypist's documented default; no pancreas in its reference"),
    ],
    "siletti": [
        ("mtg", "Adult_Human_MTG.pkl", "none",
         "tissue-matched: all 465 census siletti cells are brain"),
        ("immune_low", "Immune_All_Low.pkl", "none",
         "CellTypist's documented default"),
    ],
    "tabula_muris_senis": [
        ("mouse_gut", "Adult_Mouse_Gut.pkl", "none",
         "species-matched (mouse), the only mouse tissue model applicable here"),
        ("immune_low_upper", "Immune_All_Low.pkl", "upper",
         "default human model, mouse symbols case-matched (Cd3e/CD3E)"),
    ],
    "tabula_sapiens": [
        ("immune_low", "Immune_All_Low.pkl", "none",
         "CellTypist's documented default; 311 of these 439 cells are immune-compartment"),
        ("lung_atlas", "Human_Lung_Atlas.pkl", "none",
         "the only human tissue model whose tissue the corpus's tabula_sapiens cells cover"),
    ],
}


def load(dataset: str):
    matrix = sp.load_npz(EXPR / f"{dataset}.npz").tocsr()
    genes = [str(g) for g in np.load(EXPR / f"{dataset}_genes.npy", allow_pickle=True)]
    cells = [str(c) for c in np.load(EXPR / f"{dataset}_cells.npy", allow_pickle=True)]
    cert = json.loads((EXPR / f"{dataset}.certificate.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for c in cells:
        digest.update(str(c).encode("utf-8"))
        digest.update(b"\n")
    if digest.hexdigest() != cert["ids_sha256"]:
        raise SystemExit(f"{dataset}: the cell array is not the one the certificate covers")
    if matrix.shape[0] != len(cells) or matrix.shape[1] != len(genes):
        raise SystemExit(f"{dataset}: matrix {matrix.shape} disagrees with {len(cells)} ids / {len(genes)} genes")
    return matrix, genes, cells


def run_arm(dataset: str, arm: str, model: str, transform: str, why: str) -> dict:
    import anndata as ad
    import scanpy as sc
    import celltypist
    from celltypist import models

    t0 = time.time()
    matrix, genes, cells = load(dataset)
    matrix, genes, n_collapsed = collapse_duplicate_genes(matrix, genes)
    if transform == "upper":
        genes = [g.upper() for g in genes]
        matrix, genes, extra = collapse_duplicate_genes(matrix, genes)
        n_collapsed += extra
    adata = ad.AnnData(X=matrix.astype(np.float32), obs={"cell_id": cells}, var={"gene": genes})
    adata.var_names = genes
    adata.obs_names = cells
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    instance = models.Model.load(str(Path(models.models_path) / model))
    model_features = set(instance.features)
    overlap = [g for g in genes if g in model_features]
    # majority_voting off: it is a cluster-level decision and this measurement is per cell.
    result = celltypist.annotate(adata, model=instance, majority_voting=False)
    predicted = np.asarray(result.predicted_labels.iloc[:, 0], dtype=object)
    mapping = resolve_celltypist_map(model)
    labels = list(result.probability_matrix.columns)
    probabilities = np.asarray(result.probability_matrix.to_numpy(), dtype=np.float32)

    mapped_cl = [mapping.get(p, {}).get("cl_id") for p in predicted]
    basis = [mapping.get(p, {}).get("basis") for p in predicted]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{dataset}_{arm}.npz",
        cells=np.array(cells, dtype=object),
        predicted=np.array(predicted, dtype=object),
        mapped_cl=np.array(mapped_cl, dtype=object),
        basis=np.array(basis, dtype=object),
        labels=np.array(labels, dtype=object),
        probabilities=probabilities,
    )
    digest = hashlib.sha256((Path(models.models_path) / model).read_bytes()).hexdigest()
    if len(predicted) == 0:
        raise SystemExit(f"{dataset}/{arm}: no prediction for any cell -- refusing to report")
    return {
        "dataset": dataset,
        "arm": arm,
        "model": model,
        "model_sha256": digest,
        "model_why": why,
        "symbol_transform": transform,
        "celltypist_version": celltypist.__version__,
        "scanpy_version": sc.__version__,
        "cells": len(cells),
        "genes_input": len(genes),
        "duplicate_symbols_collapsed": n_collapsed,
        "model_features": len(model_features),
        "model_features_present": len(overlap),
        "model_feature_coverage": round(len(overlap) / len(model_features), 4),
        "labels_in_model": len(labels),
        "labels_mapped_to_cl": len(mapping),
        "predictions_mapped_to_cl": int(sum(1 for c in mapped_cl if c is not None)),
        "predictions_mapped_rate": round(sum(1 for c in mapped_cl if c is not None) / len(predicted), 4),
        "distinct_predicted_labels": len(set(predicted)),
        "top_predicted_labels": _top(predicted),
        "mean_max_probability": float(np.mean(probabilities.max(axis=1))) if probabilities.size else None,
        "seconds": round(time.time() - t0, 1),
        "out": str((OUT / f"{dataset}_{arm}.npz").relative_to(ROOT)),
    }


def _top(predicted, k: int = 8) -> list:
    from collections import Counter

    return [f"{name} x{n}" for name, n in Counter(predicted).most_common(k)]


def main() -> int:
    only = None
    for arg in sys.argv[1:]:
        if arg.startswith("--datasets="):
            only = {x.strip() for x in arg.split("=", 1)[1].split(",") if x.strip()}
    selected = [d for d in ARMS if not only or d in only]
    if not selected:
        raise SystemExit(f"--datasets={sorted(only) if only else None} matched none of {sorted(ARMS)}")
    stats: dict[str, dict] = {}
    for dataset in selected:
        print(f"[{dataset}] {len(ARMS[dataset])} arms", flush=True)
        for arm, model, transform, why in ARMS[dataset]:
            key = f"{dataset}/{arm}"
            stats[key] = run_arm(dataset, arm, model, transform, why)
            s = stats[key]
            print(f"  {key}: cells={s['cells']} labels={s['labels_in_model']} "
                  f"mapped={s['predictions_mapped_rate']} cover={s['model_feature_coverage']} "
                  f"({s['seconds']}s)", flush=True)
            print(f"      top: {s['top_predicted_labels'][:5]}", flush=True)
    out = ROOT / "reports" / "annotation-tools" / "celltypist.summary.json"
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    payload.update(stats)
    payload["_provenance"] = {
        "argv": sys.argv[1:],
        "arms": sorted(stats),
        "note": "arms are the pre-declared set from baselines/step3_celltypist.py, extended to "
                "tabula_sapiens with the two arms that file declares for it",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(stats)} arms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
