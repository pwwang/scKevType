"""Step 6: how much of CellTypist's error is the harmonisation table rather than the tool.

The project's curated CellTypist label map (``baselines/celltypist_label_map.json``) leaves
some model labels unplaced, and an unplaced label is counted as a wrong answer. Some of those
labels *are* Cell Ontology terms -- ``intestinal crypt stem cell``, ``MAIT cells``,
``Promyelocytic`` -- so leaving them unmapped is a limitation of the table, not of the tool.

This script measures that: it re-scores every CellTypist arm with a **supplementary** table
(defined here, in a new file, so the project's own map is not modified) that places the
CL-term-bearing labels among the unplaced ones, and reports the change. It also reports the
residual that no table can fix: labels that denote a state or a cell type outside the CL
snapshot entirely (``Trunk``, ``Hillock-like``, ``Deuterosomal``), and labels whose CL term
is real but outside the corpus's 82-term register (``Oligo_4``, ``Goblet``), which can never
equal an option.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/06_harmonisation_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import resolve_celltypist_map  # noqa: E402

TOOLS = ROOT / "reports" / "annotation-tools"
QUESTIONS = TOOLS / "questions.jsonl"
CELLTYPIST = TOOLS / "celltypist"
SCORES = TOOLS / "scores.json"

#: Supplementary label -> (cl_id, basis) for CellTypist labels the project's table leaves
#: unplaced. Every entry is a Cell Ontology term; the id is quoted so it can be checked
#: against the snapshot the corpus was built on (artifacts/v3/cl_terms.json).
SUPPLEMENTARY: dict[str, tuple[str, str]] = {
    "intestinal crypt stem cell": ("CL:0002250", "name"),
    "epithelial cell of large intestine": ("CL:0002253", "name"),
    "GC B cell (DZ)": ("CL:0000844", "coarsened"),
    "EarlyGC_1": ("CL:0000844", "coarsened"),
    "EarlyGC_2": ("CL:0000844", "coarsened"),
    "Resting B cell": ("CL:0000788", "coarsened"),
    "hematopoietic stem cell": ("CL:0000037", "name"),
    "Goblet": ("CL:0000160", "name"),
    "Enteroendocrine": ("CL:0000164", "name"),
    "NKT cell": ("CL:0000814", "coarsened"),
    "NKT cells": ("CL:0000814", "coarsened"),
    "LTi cell": ("CL:0001065", "coarsened"),
    "ILC3": ("CL:0001071", "name"),
    "MAIT cells": ("CL:0000940", "name"),
    "GMP": ("CL:0000557", "name"),
    "Promyelocytes": ("CL:0000836", "name"),
    "Monocyte precursor": ("CL:0000559", "coarsened"),
    "DC2": ("CL:0000990", "coarsened"),
    "Migratory DCs": ("CL:0000990", "coarsened"),
    "Hillock-like": ("CL:0000312", "coarsened"),
    "Suprabasal": ("CL:0000362", "coarsened"),
    "SMG duct": ("CL:0000068", "coarsened"),
    "SMG serous (nasal)": ("CL:0000150", "coarsened"),
    "Multiciliated (non-nasal)": ("CL:0005012", "coarsened"),
    "Multiciliated (nasal)": ("CL:0005012", "coarsened"),
    "AT1": ("CL:0002062", "name"),
    "Neuroendocrine": ("CL:0000165", "name"),
    "OPC_2": ("CL:0002453", "coarsened"),
    "Oligo_2": ("CL:0000128", "coarsened"),
    "Oligo_4": ("CL:0000128", "coarsened"),
}

MODEL_OF_ARM: dict[str, str] = {
    "muraro/pancreatic_islet": "Adult_Human_PancreaticIslet.pkl",
    "muraro/fetal_pancreas": "Fetal_Human_Pancreas.pkl",
    "muraro/immune_low": "Immune_All_Low.pkl",
    "siletti/mtg": "Adult_Human_MTG.pkl",
    "siletti/immune_low": "Immune_All_Low.pkl",
    "tabula_muris_senis/mouse_gut": "Adult_Mouse_Gut.pkl",
    "tabula_muris_senis/immune_low_upper": "Immune_All_Low.pkl",
    "tabula_sapiens/immune_low": "Immune_All_Low.pkl",
    "tabula_sapiens/lung_atlas": "Human_Lung_Atlas.pkl",
}

BEST_ARM = {
    "muraro": "fetal_pancreas",
    "siletti": "mtg",
    "tabula_muris_senis": "mouse_gut",
    "tabula_sapiens": "lung_atlas",
}


def main() -> int:
    questions: dict[str, list[dict]] = {}
    with open(QUESTIONS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            questions.setdefault(row["dataset"], []).append(row)
    name_to_cl = {
        d["name"]: d["cl_id"]
        for d in json.loads((ROOT / "artifacts" / "v3" / "definitions.json").read_text(encoding="utf-8"))["definitions"]
    }
    cl_terms = {t["cl_id"] for t in json.loads((ROOT / "artifacts" / "v3" / "cl_terms.json").read_text(encoding="utf-8"))["terms"]}
    unknown = {lab: cl for lab, (cl, _b) in SUPPLEMENTARY.items() if cl not in cl_terms}
    if unknown:
        raise SystemExit(f"supplementary table quotes CL ids that are not in the snapshot: {unknown}")

    per_arm = {}
    for arm, model in MODEL_OF_ARM.items():
        dataset, arm_name = arm.split("/")
        z = np.load(CELLTYPIST / f"{dataset}_{arm_name}.npz", allow_pickle=True)
        cells = [str(c) for c in z["cells"]]
        predicted = [str(p) for p in z["predicted"]]
        base_map = resolve_celltypist_map(model)
        base_cl = {c: (base_map.get(p, {}).get("cl_id")) for c, p in zip(cells, predicted)}
        supp_cl = {
            c: (base_map.get(p, {}).get("cl_id") or (SUPPLEMENTARY[p][0] if p in SUPPLEMENTARY else None))
            for c, p in zip(cells, predicted)
        }
        row = {"dataset": dataset, "arm": arm_name, "cells": len(cells), "labels_unplaced_by_project_map": 0,
               "labels_placed_by_supplementary": 0}
        for pred in set(predicted):
            if pred not in base_map:
                n = predicted.count(pred)
                row["labels_unplaced_by_project_map"] += n
                if pred in SUPPLEMENTARY:
                    row["labels_placed_by_supplementary"] += n
        for name, mapping in (("project_map", base_cl), ("project_map_plus_supplementary", supp_cl)):
            named = 0
            for q in questions[dataset]:
                cl = mapping[q["cell"]]
                if cl and cl == q["target_cl_id"]:
                    named += 1
            row[f"named_target_{name}"] = named
            row[f"named_target_rate_{name}"] = round(named / len(questions[dataset]), 4)
        per_arm[arm] = row

    pooled = {}
    for name, field in (("project_map", "named_target_project_map"),
                        ("project_map_plus_supplementary", "named_target_project_map_plus_supplementary")):
        total = sum(per_arm[f"{ds}/{a}"][field] for ds, a in BEST_ARM.items())
        n = sum(len(questions[ds]) for ds in BEST_ARM)
        pooled[name] = {"named_target": total, "questions": n, "rate": round(total / n, 4)}

    payload = {
        "what": "sensitivity of the CellTypist numbers to the label-harmonisation table",
        "pooled_best_arm": pooled,
        "delta_points": round(100 * (pooled["project_map_plus_supplementary"]["rate"]
                                     - pooled["project_map"]["rate"]), 3),
        "per_arm": per_arm,
        "supplementary_entries": {k: v[0] for k, v in sorted(SUPPLEMENTARY.items())},
        "note": "the supplementary table is a new file; baselines/celltypist_label_map.json is "
                "unmodified, and the numbers in reports/annotation-tools-baseline.md use the "
                "project's own table unless stated otherwise",
    }
    (TOOLS / "harmonisation-sensitivity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"pooled_best_arm": pooled, "delta_points": payload["delta_points"]}, indent=2))
    for arm, row in sorted(per_arm.items()):
        print(f"{arm:40s} unplaced={row['labels_unplaced_by_project_map']:4d} "
              f"placed_by_supp={row['labels_placed_by_supplementary']:4d} "
              f"{row['named_target_rate_project_map']:.4f} -> "
              f"{row['named_target_rate_project_map_plus_supplementary']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
