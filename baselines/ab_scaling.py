"""Quick A/B: does standardising the LR's input help or hurt on this corpus?

Scaling is not a neutral choice here. ``StandardScaler(with_mean=False)`` on log1p data
divides each gene by its standard deviation, which is largely its mean expression level --
that downweights highly-expressed marker genes (exactly the ones that separate pancreatic
endocrine cell types) while leaving sparse noise genes relatively untouched. Centring is not
available without densifying the matrix. So the two candidates are "log1p as-is" and
"log1p then with_mean=False scaling", and the honest way to choose is to measure the
in-domain 5-fold CV arm with both, on the datasets whose labels are already extracted.

Run: uv run --no-sync python baselines/ab_scaling.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import QCOUT  # noqa: E402
from step3_celltypist import load  # noqa: E402
from step4_lr import CASE_INSENSITIVE_DATASETS, select_hvg, take_columns  # noqa: E402


def cv_accuracy(dataset: str, scaled: bool) -> dict:
    inventory = json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
    truth = {c["cell_id"]: c["cl_id"] for c in inventory.values() if c["dataset_id"] == dataset}
    counts, genes, cells = load(dataset, "test")
    y = np.array([truth[c] for c in cells], dtype=object)
    keep = np.array([not str(v).startswith("local:") for v in y])
    counts_kept, y_kept = counts[keep], y[keep]
    case_insensitive = dataset in CASE_INSENSITIVE_DATASETS
    predicted = np.array([None] * len(y), dtype=object)
    positions = np.where(keep)[0]
    coverage = []
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y_kept)), y_kept):
        index, _ = select_hvg(counts_kept[tr], genes)
        hvg = [genes[i].upper() if case_insensitive else genes[i] for i in index]
        x_tr, cov = take_columns(counts_kept[tr], genes, hvg, case_insensitive)
        x_te, _ = take_columns(counts_kept[te], genes, hvg, case_insensitive)
        coverage.append(cov["hvg_coverage"])
        steps = [StandardScaler(with_mean=False)] if scaled else []
        model = make_pipeline(*steps, LogisticRegression(max_iter=5000, C=1.0, random_state=0))
        model.fit(x_tr, y_kept[tr])
        predicted[positions[te]] = model.predict(x_te)
    filled = np.array([p is not None for p in predicted])
    accuracy = float(np.mean(np.array([str(p) for p in predicted[filled]]) == y[filled]))
    counts_true = Counter(y_kept)
    return {
        "dataset": dataset,
        "scaled": scaled,
        "cells": int(filled.sum()),
        "accuracy": round(accuracy, 4),
        "n_classes": len(counts_true),
        "majority_class_rate": round(max(counts_true.values()) / len(y_kept), 4),
        "mean_fold_hvg_coverage": round(float(np.mean(coverage)), 4),
    }


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["muraro", "tabula_muris_senis"]
    out = []
    for dataset in datasets:
        for scaled in (False, True):
            result = cv_accuracy(dataset, scaled)
            out.append(result)
            print(
                f"{dataset}: scaled={scaled} accuracy={result['accuracy']} "
                f"(majority floor {result['majority_class_rate']}, {result['n_classes']} classes, "
                f"hvg coverage {result['mean_fold_hvg_coverage']})",
                flush=True,
            )
    (QCOUT / "ab_scaling.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {QCOUT / 'ab_scaling.json'}")
