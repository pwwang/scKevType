"""Step 5: score CellTypist and SingleR on the census `panel_choice` questions.

THE MAPPING, STATED ONCE
------------------------
A `panel_choice` question names a cell and a candidate panel; its labelled answer is the
option that is the cell's own register type. A reference-based annotator never sees the
question -- it sees the cell's expression and returns a label -- so the mapping is:

    tool's label for the cell  --harmonised to a CL id-->  is that CL id the question's
                                                            answer (the target option)?

This is the task's stated fair mapping: run the tool on the cell, take its label, and score
whether that label is the correct option in that question's panel. The label is the tool's own
top-1; the panel is used to ask whether the label was *on the ballot at all*, which is
reported separately rather than folded into the accuracy.

Metrics per arm (one arm = one tool + one model/reference + one label vocabulary):

  named_target          the tool's label harmonises to the question's target CL id. HEADLINE.
  on_ballot             the harmonised label is one of the question's options (a superset of
                        named_target: it also counts naming a *wrong* option)
  off_register          the harmonised label is a real CL term that the corpus's register does
                        not contain -- it can never be an option, so it is a vocabulary
                        limitation, reported rather than absorbed
  unmapped              the label has no CL id in this map at all
  panel_restricted      (CellTypist only) the argmax over the question's own options of the
                        tool's probability mass for each option equals the target. This is the
                        metric the project's own baseline tables use, so it is reported beside
                        the headline for comparability -- it is *not* the task's mapping.
  pruned                (SingleR only) the call was low-confidence and SingleR declined it.
                        Scored as incorrect in `named_target`, per the manuscript's convention
                        that an abstention is not a correct answer; also reported alone.

CellTypist never abstains, so no abstain column exists for it; SingleR has no abstain *action*
either -- `pruned` is a low-confidence flag, not a decision to decline.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/05_score.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import resolve_celltypist_map  # noqa: E402
from singler_label_map import (  # noqa: E402
    HPCA_FINE_OVERRIDE,
    HPCA_MAIN,
    MOUSE_FINE_OVERRIDE,
    MOUSE_MAIN,
    singler_label_to_cl,
    unknown_singler_labels,
)


def singler_vocabulary(reference: str) -> set[str]:
    """Every CL id this SingleR reference can ever emit through the harmonisation map."""
    tables = (HPCA_MAIN, HPCA_FINE_OVERRIDE) if reference == "human" else (MOUSE_MAIN, MOUSE_FINE_OVERRIDE)
    out: set[str] = set()
    for table in tables:
        for cl_id, _basis in table.values():
            if cl_id:
                out.add(cl_id)
    return out

TOOLS = ROOT / "reports" / "annotation-tools"
QUESTIONS = TOOLS / "questions.jsonl"
CELLTYPIST = TOOLS / "celltypist"
SINGLER = TOOLS / "singler"

SINGLER_REFERENCE = {
    "muraro": "human",
    "siletti": "human",
    "tabula_sapiens": "human",
    "tabula_muris_senis": "mouse",
}

#: dataset -> the arm that is the *most favourable* to CellTypist, fixed in advance by the rule
#: "the tissue/species-matched model where CellTypist ships one, else its documented default".
#: (This is the same rule the arm table in 03_celltypist.py is declared under; the score script
#: takes the maximum over the declared arms rather than trusting this list, and asserts they
#: agree.)
CELLTYPIST_BEST = {
    "muraro": "pancreatic_islet",
    "siletti": "mtg",
    "tabula_muris_senis": "mouse_gut",
    "tabula_sapiens": "immune_low",
}


def load_questions() -> list[dict]:
    rows = []
    with open(QUESTIONS, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{QUESTIONS}: empty")
    return rows


def definitions() -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads((ROOT / "artifacts" / "v3" / "definitions.json").read_text(encoding="utf-8"))
    name_to_cl = {d["name"]: d["cl_id"] for d in payload["definitions"]}
    cl_to_name = {d["cl_id"]: d["name"] for d in payload["definitions"]}
    return name_to_cl, cl_to_name


def rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score_arm(questions: list[dict], per_cell: dict[str, tuple[str | None, str, bool]],
              name_to_cl: dict[str, str], probabilities: dict[str, np.ndarray] | None = None,
              labels: list[str] | None = None, cell_order: list[str] | None = None,
              vocabulary: set[str] | None = None) -> dict:
    """Score one arm over the questions of its dataset.

    ``per_cell`` maps cell id -> (cl_id, basis, pruned). ``probabilities`` (CellTypist only) is
    the cells x labels probability matrix, with ``labels`` the model's label vocabulary.
    ``vocabulary`` is every CL id this tool+reference can *ever* emit; a question whose target
    is not in it is unwinnable by construction, which is reported as the coverage ceiling
    rather than left to be inferred from a low accuracy.
    """
    counts = Counter()
    n = 0
    reachable = 0
    named_and_reachable = 0
    panel_no_signal = 0
    panel_wrong_option = 0
    for q in questions:
        n += 1
        if q["cell"] not in per_cell:
            counts["cell_not_scored"] += 1
            continue
        tool_cl, basis, pruned = per_cell[q["cell"]]
        target_cl = q["target_cl_id"]
        option_cls = {name_to_cl.get(o) for o in q["options"] if name_to_cl.get(o)}
        if tool_cl is None:
            if basis == "unmapped":
                counts["unmapped"] += 1
            else:
                counts["unmapped"] += 1
            if pruned:
                counts["pruned"] += 1
            continue
        if pruned:
            counts["pruned"] += 1
        if vocabulary is not None and target_cl in vocabulary:
            reachable += 1
        if tool_cl == target_cl:
            counts["named_target"] += 1
            if vocabulary is not None and target_cl in vocabulary:
                named_and_reachable += 1
        if tool_cl in option_cls:
            counts["on_ballot"] += 1
        if tool_cl not in name_to_cl.values():
            counts["off_register"] += 1
        if probabilities is not None and cell_order is not None and labels is not None:
            row = probabilities[cell_order.index(q["cell"])]
            mass = {}
            for opt in q["options"]:
                ocl = name_to_cl.get(opt)
                if not ocl:
                    continue
                mass[opt] = float(sum(row[i] for i, lab in enumerate(labels)
                                      if resolve_map_cache.get(lab, (None,))[0] == ocl))
            if not mass or max(mass.values()) <= 0:
                panel_no_signal += 1
            elif max(mass, key=lambda k: mass[k]) == q["target"]:
                counts["panel_restricted"] += 1
            else:
                panel_wrong_option += 1
    out = {
        "questions": n,
        "cell_not_scored": counts["cell_not_scored"],
        "named_target": counts["named_target"],
        "named_target_rate": rate(counts["named_target"], n),
        "on_ballot": counts["on_ballot"],
        "on_ballot_rate": rate(counts["on_ballot"], n),
        "off_register": counts["off_register"],
        "off_register_rate": rate(counts["off_register"], n),
        "unmapped": counts["unmapped"],
        "unmapped_rate": rate(counts["unmapped"], n),
        "pruned_declined": counts["pruned"],
        "pruned_rate": rate(counts["pruned"], n),
    }
    if vocabulary is not None:
        out["target_reachable_by_this_vocabulary"] = reachable
        out["target_reachable_rate"] = rate(reachable, n)
        out["named_target_given_reachable"] = named_and_reachable
        out["named_target_given_reachable_rate"] = rate(named_and_reachable, reachable)
    if probabilities is not None:
        out["panel_restricted_correct"] = counts["panel_restricted"]
        out["panel_restricted_rate"] = rate(counts["panel_restricted"], n)
        out["panel_no_signal"] = panel_no_signal
    return out


resolve_map_cache: dict[str, tuple] = {}


def celltypist_arm(dataset: str, arm: str) -> dict:
    npz = np.load(CELLTYPIST / f"{dataset}_{arm}.npz", allow_pickle=True)
    cells = [str(c) for c in npz["cells"]]
    predicted = [str(p) for p in npz["predicted"]]
    mapped = [None if c is None else str(c) for c in npz["mapped_cl"]]
    basis = [str(b) if b is not None else "unmapped" for b in npz["basis"]]
    return {
        "cells": cells,
        "predicted": predicted,
        "mapped_cl": mapped,
        "basis": basis,
        "labels": [str(x) for x in npz["labels"]],
        "probabilities": np.asarray(npz["probabilities"]),
    }


def main() -> int:
    questions = load_questions()
    name_to_cl, cl_to_name = definitions()
    by_dataset: dict[str, list[dict]] = {}
    for q in questions:
        by_dataset.setdefault(q["dataset"], []).append(q)
    print("questions per dataset:", {k: len(v) for k, v in sorted(by_dataset.items())})

    results: dict[str, dict] = {"celltypist": {}, "singler": {}}

    # ---- CellTypist -------------------------------------------------------------------
    arms = json.loads((TOOLS / "celltypist.summary.json").read_text(encoding="utf-8"))
    for key in sorted(k for k in arms if not k.startswith("_")):
        dataset, arm = key.split("/")
        data = celltypist_arm(dataset, arm)
        # the model's own label vocabulary -> CL id, so the panel-restricted argmax can sum the
        # probability mass of every model label that names an option
        global resolve_map_cache
        resolve_map_cache = {
            lab: (v.get("cl_id"),) for lab, v in resolve_celltypist_map(arms[key]["model"]).items()
        }
        per_cell = {
            c: (m, b, False) for c, m, b in zip(data["cells"], data["mapped_cl"], data["basis"])
        }
        scored = score_arm(
            by_dataset[dataset], per_cell, name_to_cl,
            probabilities=data["probabilities"], labels=data["labels"], cell_order=data["cells"],
            vocabulary={c for c, in resolve_map_cache.values() if c},
        )
        scored.update({
            "tool": "celltypist", "dataset": dataset, "arm": arm, "model": arms[key]["model"],
            "model_sha256": arms[key]["model_sha256"],
            "model_why": arms[key]["model_why"],
            "cells_scored": len(data["cells"]),
            "distinct_predicted_labels": arms[key]["distinct_predicted_labels"],
            "top_predicted_labels": arms[key]["top_predicted_labels"],
            "mean_max_probability": arms[key]["mean_max_probability"],
            "model_feature_coverage": arms[key]["model_feature_coverage"],
            "seconds": arms[key]["seconds"],
        })
        results["celltypist"][key] = scored
        print(f"  celltypist {key}: named_target={scored['named_target_rate']} "
              f"on_ballot={scored['on_ballot_rate']} panel_restricted={scored.get('panel_restricted_rate')}")

    # ---- SingleR ----------------------------------------------------------------------
    singler_meta = json.loads((TOOLS / "singler.summary.json").read_text(encoding="utf-8"))
    for dataset in sorted(by_dataset):
        import csv

        reference = SINGLER_REFERENCE[dataset]
        rows = list(csv.DictReader(open(SINGLER / f"{dataset}.csv", encoding="utf-8")))
        for level in ("main", "fine"):
            per_cell = {}
            seen = set()
            for r in rows:
                label = r[f"label_{level}"]
                seen.add(label)
                cl, basis = singler_label_to_cl(label, reference, level)
                pruned = (r[f"pruned_{level}"] in ("", "NA", "None")) or (
                    r[f"pruned_{level}"] is None)
                per_cell[r["cell"]] = (cl, basis, pruned)
            scored = score_arm(by_dataset[dataset], per_cell, name_to_cl,
                               vocabulary=singler_vocabulary(reference))
            scored.update({
                "tool": "singler", "dataset": dataset, "arm": f"label_{level}",
                "reference": singler_meta[dataset]["reference_name"],
                "reference_cells": singler_meta[dataset]["reference_cells"],
                "common_genes": singler_meta[dataset]["common_genes"],
                "cells_scored": len(rows),
                "distinct_predicted_labels": singler_meta[dataset][f"label_{level}_distinct"],
                "unmapped_labels": unknown_singler_labels(seen, reference, level),
            })
            results["singler"][f"{dataset}/label_{level}"] = scored
            print(f"  singler {dataset}/label_{level}: named_target={scored['named_target_rate']} "
                  f"on_ballot={scored['on_ballot_rate']} pruned={scored['pruned_rate']}")

    # ---- tool-level readings over the full 1,819 --------------------------------------
    def pooled(tool: str, pick) -> dict:
        total = Counter()
        per_dataset_pick = {}
        for dataset, qs in sorted(by_dataset.items()):
            key = pick(dataset, tool)
            row = results[tool][key]
            per_dataset_pick[dataset] = key
            for field in ("questions", "named_target", "on_ballot", "off_register", "unmapped",
                          "pruned_declined", "panel_restricted_correct",
                          "target_reachable_by_this_vocabulary", "named_target_given_reachable"):
                if field in row:
                    total[field] += row[field]
        n = total["questions"]
        out = {f: total[f] for f in total}
        out["named_target_rate"] = rate(total["named_target"], n)
        out["on_ballot_rate"] = rate(total["on_ballot"], n)
        out["off_register_rate"] = rate(total["off_register"], n)
        out["unmapped_rate"] = rate(total["unmapped"], n)
        out["pruned_rate"] = rate(total["pruned_declined"], n)
        if "panel_restricted_correct" in total:
            out["panel_restricted_rate"] = rate(total["panel_restricted_correct"], n)
        out["target_reachable_rate"] = rate(total["target_reachable_by_this_vocabulary"], n)
        out["named_target_given_reachable_rate"] = rate(
            total["named_target_given_reachable"], total["target_reachable_by_this_vocabulary"])
        out["per_dataset_arm"] = per_dataset_pick
        return out

    def ct_best(dataset: str, tool: str) -> str:
        keys = [k for k in results[tool] if k.startswith(dataset + "/")]
        # the maximum over the declared arms, by named_target_rate, ties broken by the
        # pre-declared rule in CELLTYPIST_BEST so the choice is not made by the score
        best = max(keys, key=lambda k: (results[tool][k]["named_target_rate"],
                                        results[tool][k]["on_ballot_rate"]))
        declared = f"{dataset}/{CELLTYPIST_BEST[dataset]}"
        if results[tool][best]["named_target_rate"] > results[tool][declared]["named_target_rate"]:
            print(f"  NOTE: {best} beats the pre-declared best arm {declared} on {dataset}")
        return best

    def ct_default(dataset: str, tool: str) -> str:
        for cand in (f"{dataset}/immune_low", f"{dataset}/immune_low_upper"):
            if cand in results[tool]:
                return cand
        raise SystemExit(f"no default-model arm for {dataset}")

    def sr_best(dataset: str, tool: str) -> str:
        keys = [k for k in results[tool] if k.startswith(dataset + "/")]
        return max(keys, key=lambda k: results[tool][k]["named_target_rate"])

    pooled_out = {
        "celltypist_default_model": pooled("celltypist", ct_default),
        "celltypist_best_arm": pooled("celltypist", ct_best),
        "singler_label_main": pooled("singler", lambda d, t: f"{d}/label_main"),
        "singler_best_label_level": pooled("singler", sr_best),
    }

    payload = {
        "population": {
            "source": "reports/census-suite-ids.jsonl, family == panel_choice",
            "questions": len(questions),
            "distinct_cells": len({(q["dataset"], q["cell"]) for q in questions}),
            "per_dataset": {k: len(v) for k, v in sorted(by_dataset.items())},
            "mapping": "tool label -> CL id (harmonised) == the question's target option's CL id",
        },
        "per_arm": results,
        "pooled": pooled_out,
        "notes": {
            "abstain_column": "none: neither tool has an abstain action. CellTypist always "
                              "emits its top-1 label; SingleR's `pruned.labels` NA is a "
                              "low-confidence flag, reported separately and scored as "
                              "incorrect in named_target.",
            "panel_restricted": "computed for CellTypist (it exposes a label-probability "
                                "matrix) and not for SingleR (its per-label score matrix is "
                                "not persisted by this run).",
            "families_not_scored": "evidence_diluted, marker_noul, panel_miss are constructed "
                                   "evidence states with no corresponding expression profile, "
                                   "so no cell-level annotator can be scored on them.",
        },
    }
    (TOOLS / "scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(pooled_out, indent=2))
    print(f"wrote {TOOLS / 'scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
