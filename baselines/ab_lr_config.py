"""Why is the in-domain LR below the majority-class floor, and can it be fixed?

Two facts forced this script:

* `cross_validate` already selects HVGs *inside each fold from that dataset's own cells*
  (see step4_lr.cross_validate), so "the features came from blood" cannot be the explanation
  for muraro's in-domain CV arm scoring 0.3085 against a 0.4283 majority floor.
* 7 classes x 2000 features fitted on ~1,500 cells per fold is ~14,000 parameters on 1,500
  samples. Over-fitting is a much better candidate than bad features.

So this sweeps the two knobs that control that: how many HVGs, and how strong the L2 penalty
is. Each configuration is scored by 5-fold CV on the dataset's own labels, with the fold's
HVGs re-selected inside the fold, and printed next to the majority floor -- which is the
number that decides whether the result is a baseline or a bug.

Run: uv run --no-sync python baselines/ab_lr_config.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import QCOUT  # noqa: E402
from step3_celltypist import load  # noqa: E402
from step4_lr import CASE_INSENSITIVE_DATASETS, select_hvg, take_columns  # noqa: E402


def cv(dataset: str, n_hvg: int, c_value: float, scaled: bool) -> dict:
    inventory = json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
    truth = {c["cell_id"]: c["cl_id"] for c in inventory.values() if c["dataset_id"] == dataset}
    counts, genes, cells = load(dataset, "test")
    y = np.array([truth[c] for c in cells], dtype=object)
    keep = np.array([not str(v).startswith("local:") for v in y])
    counts_kept, y_kept = counts[keep], y[keep]
    case_insensitive = dataset in CASE_INSENSITIVE_DATASETS
    predicted = np.array([None] * len(y), dtype=object)
    positions = np.where(keep)[0]
    for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y_kept)), y_kept):
        index, _ = select_hvg(counts_kept[tr], genes, n_top=n_hvg)
        hvg = [genes[i].upper() if case_insensitive else genes[i] for i in index]
        x_tr, _ = take_columns(counts_kept[tr], genes, hvg, case_insensitive)
        x_te, _ = take_columns(counts_kept[te], genes, hvg, case_insensitive)
        if scaled:
            scaler = StandardScaler(with_mean=False).fit(x_tr)
            x_tr, x_te = scaler.transform(x_tr), scaler.transform(x_te)
        model = LogisticRegression(C=c_value, max_iter=5000, random_state=0)
        model.fit(x_tr, y_kept[tr])
        predicted[positions[te]] = model.predict(x_te)
    filled = np.array([p is not None for p in predicted])
    counts_true = Counter(y_kept)
    return {
        "dataset": dataset,
        "n_hvg": n_hvg,
        "C": c_value,
        "scaled": scaled,
        "accuracy": round(float(np.mean(np.array([str(p) for p in predicted[filled]]) == y[filled])), 4),
        "n_classes": len(counts_true),
        "majority_floor": round(max(counts_true.values()) / len(y_kept), 4),
    }


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["muraro", "tabula_muris_senis"]
    out = []
    for dataset in datasets:
        for n_hvg in (50, 200, 500, 2000):
            for c_value in (0.01, 0.1, 1.0):
                for scaled in (True, False):
                    if (n_hvg, c_value, scaled) not in {(50, 0.01, True), (50, 0.1, True), (50, 1.0, True),
                                                       (200, 0.01, True), (200, 0.1, True), (200, 1.0, True),
                                                       (500, 0.01, True), (500, 0.1, True), (500, 0.1, False),
                                                       (500, 1.0, True), (2000, 0.01, True), (2000, 0.1, True),
                                                       (2000, 1.0, True), (2000, 1.0, False), (200, 1.0, False)}:
                        continue
                    result = cv(dataset, n_hvg, c_value, scaled)
                    out.append(result)
                    flag = "  <-- BELOW MAJORITY FLOOR" if result["accuracy"] < result["majority_floor"] else ""
                    print(f"{dataset}: n_hvg={result['n_hvg']:5d} C={result['C']:<5} scaled={result['scaled']} "
                          f"acc={result['accuracy']:.4f} floor={result['majority_floor']:.4f}{flag}", flush=True)
    (QCOUT / "ab_lr_config.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {QCOUT / 'ab_lr_config.json'}")
