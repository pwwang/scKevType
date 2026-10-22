"""Step 7: the two checks that make the CellTypist numbers in this report trustworthy.

CHECK 1 -- the census run reproduces the project's own CellTypist baseline.
``baselines/step3_celltypist.py`` ran CellTypist over every corpus test cell (2,126 muraro /
5,000 siletti / 3,500 tabula_muris_senis). This run scores a *subset* of those cells, extracted
through a different path (a subset of the persisted matrix, then re-normalised). If the two
disagree on a single cell, the subsetting or the row alignment is wrong and every number here
is suspect. The check is exact: same arm, same cell, same predicted label.

CHECK 2 -- SingleR recovers a type when the reference covers it.
The headline SingleR number is low, and the reason offered is reference coverage, not a broken
pipeline. The check that separates those: on the cells whose type the reference *does* contain,
SingleR should mostly be right. tabula_muris_senis's B cells (the reference has "B cells") and
tabula_sapiens's B cells ("B_cell") are the case study.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/07_validate.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from singler_label_map import singler_label_to_cl  # noqa: E402

TOOLS = ROOT / "reports" / "annotation-tools"
BASE = ROOT / "baselines" / "out" / "celltypist"

ARMS = {
    "muraro": ["pancreatic_islet", "fetal_pancreas", "immune_low"],
    "siletti": ["mtg", "immune_low"],
    "tabula_muris_senis": ["mouse_gut", "immune_low_upper"],
}


def check1() -> dict:
    out = {}
    for dataset, arms in ARMS.items():
        for arm in arms:
            old = np.load(BASE / f"{dataset}_{arm}.npz", allow_pickle=True)
            new = np.load(TOOLS / "celltypist" / f"{dataset}_{arm}.npz", allow_pickle=True)
            old_cells = [str(c) for c in old["cells"]]
            old_pred = [str(p) for p in old["predicted"]]
            index = {c: i for i, c in enumerate(old_cells)}
            new_cells = [str(c) for c in new["cells"]]
            new_pred = [str(p) for p in new["predicted"]]
            missing = [c for c in new_cells if c not in index]
            agree = sum(1 for c, p in zip(new_cells, new_pred) if c in index and old_pred[index[c]] == p)
            out[f"{dataset}/{arm}"] = {
                "baseline_cells": len(old_cells),
                "census_cells": len(new_cells),
                "cells_missing_from_baseline": len(missing),
                "agreeing": agree,
                "agreement_rate": round(agree / (len(new_cells) - len(missing)), 6) if new_cells else None,
                "verdict": "MATCH" if agree == len(new_cells) and not missing else "MISMATCH",
            }
            print(f"  {dataset}/{arm}: {agree}/{len(new_cells)} agree, "
                  f"{len(missing)} missing -> {out[f'{dataset}/{arm}']['verdict']}")
    return out


def check2() -> dict:
    questions = {}
    with open(TOOLS / "questions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            questions[(row["dataset"], row["cell"])] = row
    out = {}
    for dataset, target_cl, label, reference, level in (
        ("tabula_muris_senis", "CL:0000236", "B cells", "mouse", "main"),
        ("tabula_sapiens", "CL:0000236", "B_cell", "human", "main"),
    ):
        rows = list(csv.DictReader(open(TOOLS / "singler" / f"{dataset}.csv", encoding="utf-8")))
        cells_of_type = [r for r in rows
                         if (dataset, r["cell"]) in questions
                         and questions[(dataset, r["cell"])]["cl_id"] == target_cl]
        hit = sum(1 for r in cells_of_type
                  if singler_label_to_cl(r[f"label_{level}"], reference, level)[0] == target_cl)
        out[dataset] = {
            "type": target_cl,
            "reference_label": label,
            "cells_of_this_type": len(cells_of_type),
            "recalled": hit,
            "recall": round(hit / len(cells_of_type), 4) if cells_of_type else None,
        }
        print(f"  {dataset}: SingleR recalls {hit}/{len(cells_of_type)} of the cells whose "
              f"reference label exists ({label})")
    return out


def main() -> int:
    print("CHECK 1: census CellTypist run vs the project's own baseline")
    c1 = check1()
    print("CHECK 2: SingleR recall where the reference covers the type")
    c2 = check2()
    payload = {
        "check1_celltypist_reproduces_project_baseline": c1,
        "check2_singler_recall_where_reference_covers": c2,
        "verdict": "PASS" if all(v["verdict"] == "MATCH" for v in c1.values()) else "FAIL",
    }
    (TOOLS / "validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("verdict:", payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
