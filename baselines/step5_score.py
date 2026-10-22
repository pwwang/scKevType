"""Step 5 -- per-dataset, per-family scoring of every baseline arm.

Two levels, because the corpus's families ask different questions of the same cell:

Level A, cell level. One prediction per cell, scored against the cell's own CL id
(``_meta.cl_id``). Reported per dataset; the ``out_of_register`` cells are held out of
it, because their label is a ``local:`` id that no CL-trained annotator can emit, and
are reported separately as *how often a baseline named a register type for a cell whose
type is not in the register at all*.

Level B, family level. Each family is scored on its own target, with the baseline's
answer forced onto the record's option list where that is what the record asks:

  panel_choice      the true type is on the panel; the metric is top-1 *within the
                    panel* (argmax over the record's options), plus the unconstrained
                    top-1 for context.
  panel_miss        the true type is off the panel and the target is the abstain option.
                    No standard annotator has an abstain action, so its constrained
                    answer is a distractor by construction: the honest numbers are
                    "forced_answer_rate" (1.0 by construction) and "correct_rate" (0.0
                    by construction) -- and for the mouse/human arms, *which* distractor.
  evidence_removed  same shape as panel_miss: the deciding markers were masked out of
                    the state the model sees. Note the baselines do not see the state,
                    they see the full matrix, so this family is where the asymmetry is
                    sharpest: the intervention touches the corpus's state, not the
                    baseline's input.
  evidence_diluted  the target follows whether the definition still holds, so it is the
                    true type for some records and the abstain option for others: both
                    halves are reported, and the accuracy is the fraction the baseline
                    matches -- with the abstain half unwinnable for it.
  marker_noul       one yes/no question per option. A discriminative annotator's reading
                    is "yes iff this candidate is my argmax" -- a rule that answers the
                    same thing for every option of a cell, which is exactly why it can
                    only be right on one option per cell.
  out_of_register   the target is the abstain option because the cell's type is not a
                    register term at all.

Scored answers are probabilities, not just labels: the constrained argmax uses the
arm's own probability mass for each option (CellTypist: the sum over its labels that map
to that CL id; LR: the class probability, 0 for a class it was never trained on).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    ARM_TAGS,
    ASYMMETRY,
    QCOUT,
    ROOT,
    TAG_LEGEND,
    arm_tags,
    cl_names,
    cl_parents,
    definitions,
    load_aligned_cells,
    resolve_celltypist_map,
    write_json,
)

LR_ARMS = ["splitexact", "atlas", "cv_indomain", "cv_grouped"]
#: The CellTypist arm table is *not* duplicated here: main() imports ARMS from
#: step3_celltypist, so a new arm cannot be scored-but-unlisted or listed-but-unscored.
CHOICE_FAMILIES = ["panel_choice", "panel_miss", "evidence_removed", "evidence_diluted", "out_of_register"]

#: The columns of ``baselines/out/results.csv``, in order. ``training_regime`` and ``input`` sit
#: directly after ``arm`` because they are not decoration: they are the two facts a reader needs to
#: know whether a number may be read next to our model's, and a table cannot carry them in a
#: footnote the way a markdown table can. ``arm_kind``/``arm_tags`` (lib_common) resolve them, and
#: an unregistered arm is a hard error rather than an empty cell.
CSV_COLUMNS = ["arm", "training_regime", "input", "level", "family", "metric", "rate", "successes", "n",
               "wilson_lo", "wilson_hi", "thin"]


# --------------------------------------------------------------------------- helpers
def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval -- reported with every rate, and skipped when the cell is thin."""
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(successes: int, total: int, thin_at: int = 30) -> dict:
    return {
        "successes": successes,
        "n": total,
        "rate": round(successes / total, 4) if total else None,
        "wilson95": wilson(successes, total),
        "thin": total < thin_at,
    }


class Predictor:
    """One baseline arm's per-cell answers, in a common shape."""

    def __init__(self, name: str, kind: str, cells, predicted_cl, unmapped=False, probability_of=None,
                 label_space=None):
        self.name = name
        self.kind = kind
        self.index = {c: i for i, c in enumerate(cells)}
        self.predicted_cl = {c: predicted_cl[i] for i, c in enumerate(cells)}
        self.unmapped = set(c for c, v in self.predicted_cl.items() if v is None) if unmapped else set()
        self._probability_of = probability_of
        #: every CL id this arm could ever emit -- the thing a true label has to be in for the
        #: arm to be able to get the cell right at all
        self.label_space = set(label_space or ())

    def top(self, cell: str) -> str | None:
        return self.predicted_cl.get(cell)

    def probability(self, cell: str, cl_id: str) -> float:
        if cl_id is None or cell not in self.index:
            return 0.0
        if self._probability_of is None:
            return 1.0 if self.predicted_cl.get(cell) == cl_id else 0.0
        return self._probability_of(self.index[cell], cl_id)

    def argmax_of(self, cell: str, options: list[str]) -> str | None:
        """The option with the most probability mass, or None when it has none anywhere.

        Returning None matters: an arm whose class space does not contain *any* of the
        record's options has no answer to give, and silently naming the first option
        would read as a choice it never made. Those cells are counted separately as
        ``no_panel_signal``.
        """
        best, best_score = None, 0.0
        for option in options:
            score = self.probability(cell, option)
            if score > best_score:
                best, best_score = option, score
        return best


def assert_arm_cells_in_row_order(dataset: str, arm: str, cells: list[str]) -> None:
    """An arm is only scoreable if its own row order is the certified row order of its matrix.

    This is the guard that keeps a *stale* arm file from being scored: an npz written before the
    row-order repair carries the same ids in the same (request) order the matrix used to be
    labelled in, so it looks perfectly well-formed while every row of it belongs to another
    cell. Comparing the arm's own cell list with the certified one turns that into an error
    instead of a plausible number.
    """
    certified = load_aligned_cells(dataset)
    if list(cells) != certified:
        differing = sum(1 for a, b in zip(cells, certified) if a != b) if len(cells) == len(certified) else None
        raise SystemExit(
            f"{dataset}/{arm}: the arm's cell list is not the certified row order of "
            f"{dataset}_test_cells.npy ({differing} of {len(certified)} positions differ, "
            f"arm rows={len(cells)}) -- this arm was produced from a matrix whose ids and rows "
            f"disagreed, or from a different cell set. Re-run the step that writes it before scoring it."
        )


def load_lr_arm(name: str, dataset: str) -> Predictor:
    path = QCOUT / "lr" / f"{name}_{dataset}.npz"
    payload = np.load(path, allow_pickle=True)
    cells = [str(c) for c in payload["cells"]]
    assert_arm_cells_in_row_order(dataset, name, cells)
    if name.startswith("cv_"):
        # The in-domain arms save predictions only -- they have no single fitted model whose
        # probabilities could travel with them.
        predicted = [str(p) if str(p) else None for p in payload["predicted"]]
        return Predictor(name, "lr", cells, predicted, label_space={str(c) for c in payload["classes"]})
    classes = [str(c) for c in payload["classes"]]
    probabilities = payload["probabilities"]
    column = {cl: j for j, cl in enumerate(classes)}
    predicted = [str(p) for p in payload["predicted"]]
    return Predictor(
        name,
        "lr",
        cells,
        predicted,
        label_space=set(classes),
        probability_of=lambda row, cl, column=column, probabilities=probabilities: (
            float(probabilities[row, column[cl]]) if cl in column else 0.0
        ),
    )


def load_ct_arm(dataset: str, arm: str, model: str) -> Predictor:
    payload = np.load(QCOUT / "celltypist" / f"{dataset}_{arm}.npz", allow_pickle=True)
    cells = [str(c) for c in payload["cells"]]
    assert_arm_cells_in_row_order(dataset, arm, cells)
    mapped = [None if m in (None, "None") else str(m) for m in payload["mapped_cl"]]
    labels = [str(l) for l in payload["labels"]]
    probabilities = payload["probabilities"]
    by_cl: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(cells), dtype=np.float32))
    mapping = resolve_celltypist_map(model)
    for j, label in enumerate(labels):
        entry = mapping.get(label)
        if entry:
            by_cl[entry["cl_id"]] = by_cl[entry["cl_id"]] + probabilities[:, j]
    matrix = np.stack([by_cl[cl] for cl in sorted(by_cl)], axis=1) if by_cl else np.zeros((len(cells), 0), dtype=np.float32)
    columns = {cl: j for j, cl in enumerate(sorted(by_cl))}
    return Predictor(
        arm,
        "celltypist",
        cells,
        mapped,
        unmapped=True,
        # What this arm can name at all: every CL id any of the model's labels maps to.
        label_space={entry["cl_id"] for entry in mapping.values()},
        probability_of=lambda row, cl, columns=columns, matrix=matrix: (
            float(matrix[row, columns[cl]]) if cl in columns else 0.0
        ),
    )


# --------------------------------------------------------------------------- level A
def score_cells(predictor: Predictor, cells: list[str], truth: dict[str, str], ancestors_of, descendants_of) -> dict:
    n = scored = exact = consistent = unmapped = 0
    for cell in cells:
        if cell not in predictor.index:
            continue
        n += 1
        predicted = predictor.top(cell)
        if predicted is None:
            unmapped += 1
            continue
        scored += 1
        true = truth[cell]
        if predicted == true:
            exact += 1
            consistent += 1
        elif true in ancestors_of(predicted) or true in descendants_of(predicted):
            consistent += 1
    out = {
        "cells": n,
        "scored": scored,
        "unmapped": unmapped,
        "exact": rate(exact, scored),
        "hierarchy_consistent": rate(consistent, scored),
        "exact_over_all_cells": rate(exact, n),
    }
    if unmapped:
        out["unmapped_rate"] = rate(unmapped, n)
    return out


# --------------------------------------------------------------------------- level B
def score_family(family: str, records: list[dict], predictor: Predictor, truth: dict[str, str]) -> dict:
    if family == "marker_noul":
        return score_noul(records, predictor, truth)
    if family == "out_of_register":
        return score_out_of_register(records, predictor)
    correct = in_panel = forced = abstain_targets = type_targets = correct_on_type = no_signal = 0
    distractors: dict[str, int] = defaultdict(int)
    n = 0
    options_seen = options_mapped = 0
    for record in records:
        cell = record["cell_id"]
        if cell not in predictor.index:
            continue
        q = record["q"]
        options = [cl for cl in q["options_cl_ids"]]
        if not options:
            continue
        n += 1
        options_seen += q.get("n_options") or len(options)
        options_mapped += len(options)
        pick = predictor.argmax_of(cell, options)
        if pick is None:
            no_signal += 1
        target_cl = q.get("target_cl_id")
        if pick is not None and target_cl is not None and pick == target_cl:
            correct += 1
        if target_cl is not None and target_cl in options:
            in_panel += 1
        if q.get("target_kind") == "abstain":
            abstain_targets += 1
            distractors[pick or "no option carried any of this arm's probability mass"] += 1
        else:
            type_targets += 1
            if pick == target_cl:
                correct_on_type += 1
        if pick is not None:
            forced += 1
    out = {
        "cells": n,
        "constraint": "argmax over the record's option list, using the arm's own probability mass per option; "
        "a cell where no option carries any mass counts as no answer, not as a choice",
        "constrained_correct": rate(correct, n),
        "constrained_correct_when_answered": rate(correct, forced) if forced else None,
        "answers_given": rate(forced, n),
        "no_panel_signal": rate(no_signal, n),
        "options_contain_the_target_type": rate(in_panel, n),
        "option_names_resolved_to_cl": round(options_mapped / options_seen, 4) if options_seen else None,
        "target_abstain_cells": abstain_targets,
        "target_type_cells": type_targets,
        "correct_on_target_type_cells": rate(correct_on_type, type_targets) if type_targets else None,
        "has_abstain_action": False,
        "note": (
            "this family's target is the abstain option for every record: no standard annotator has an abstain "
            "action, so an answer here is a distractor by construction and 'correct' is unwinnable"
            if abstain_targets and not type_targets
            else None
        ),
        "most_common_forced_answers": dict(sorted(distractors.items(), key=lambda kv: -kv[1])[:8]) or None,
    }
    return out


def score_noul(records: list[dict], predictor: Predictor, truth: dict[str, str]) -> dict:
    n = correct = yes = target_yes = yes_and_yes = 0
    for record in records:
        cell = record["cell_id"]
        if cell not in predictor.index:
            continue
        q = record["q"]
        n += 1
        candidate = q["candidate_cl_id"]
        rule_says_yes = predictor.top(cell) == candidate
        target_says_yes = bool(q["target"])
        yes += bool(rule_says_yes)
        target_yes += target_says_yes
        yes_and_yes += bool(rule_says_yes) and target_says_yes
        if bool(rule_says_yes) == target_says_yes:
            correct += 1
    # Confusion counts, not derived from the marginals: the interesting failure here is a
    # 'yes' on a cell whose true type is something else.
    no_and_no = correct - yes_and_yes
    recall_yes = rate(yes_and_yes, target_yes)
    recall_no = rate(no_and_no, n - target_yes)
    return {
        "questions": n,
        "rule": "'yes' iff this candidate is the annotator's own top-1 label, and only when the label maps to a "
        "register term; one answer per cell, so at most one option per cell can be a 'yes'",
        "accuracy": rate(correct, n),
        "yes_rate": rate(yes, n),
        "target_yes_rate": rate(target_yes, n),
        # Accuracy alone is misleading here: the target is 'no' for most cells, so an
        # annotator that always says 'no' scores the negative-class prior. These four
        # numbers are the honest version, from the confusion counts above.
        "recall_on_yes_cells": recall_yes,
        "recall_on_no_cells": recall_no,
        "balanced_accuracy": round(
            (recall_yes["rate"] or 0.0) / 2 + (recall_no["rate"] or 0.0) / 2, 4
        ),
        "always_no_accuracy": rate(n - target_yes, n),
    }


def score_out_of_register(records: list[dict], predictor: Predictor) -> dict:
    n = named_a_register_type = 0
    for record in records:
        cell = record["cell_id"]
        if cell not in predictor.index:
            continue
        n += 1
        if predictor.top(cell) is not None:
            named_a_register_type += 1
    return {
        "cells": n,
        "cell_label": "not a CL term (a local:<dataset>:<slug> id): the corpus's own target for this family is the "
        "abstain option",
        "named_a_register_type": rate(named_a_register_type, n),
        "correct_rate": 0.0,
        "note": "a CL-trained annotator has no way to say 'this is not one of my types'; every answer it gives is "
        "scored as wrong by construction, so the informative number is how often it still names a type. Read it "
        "together with 'covered_cells': an arm that never emitted a prediction for these cells (e.g. the "
        "in-domain CV arm, which is fitted on register-labelled cells only) scores 0 on named_a_register_type "
        "for want of coverage, not for want of confidence.",
        "covered_cells": rate(n, len({r["cell_id"] for r in records})),
    }


# --------------------------------------------------------------------------- driver
def main() -> None:
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print("Usage: step5_score.py [--datasets=a,b] [--csv-only]   (default: every dataset the corpus has)")
        print("  --csv-only  rebuild baselines/out/results.csv from the existing results.json, re-scoring")
        print("              nothing. The CSV is a projection of the JSON, so this is how a change to the")
        print("              CSV's columns reaches the artifact on disk without a second scoring run.")
        return
    only = None
    csv_only = False
    for arg in argv:
        if arg.startswith("--datasets="):
            only = {name.strip() for name in arg.split("=", 1)[1].split(",") if name.strip()}
        elif arg == "--csv-only":
            csv_only = True
    if csv_only:
        path = QCOUT / "results.json"
        if not path.exists():
            raise SystemExit(f"--csv-only: no {path} to project -- run the scoring pass first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = write_results_csv(payload)
        tags = Counter(r["training_regime"] for r in rows_in(payload))
        print(f"--csv-only: re-scored nothing; wrote {QCOUT / 'results.csv'} from {path} ({rows} rows)")
        print(f"  every row carries its tags: {dict(sorted(tags.items()))}")
        print("  verify:  grep -c 'transfer' baselines/out/results.csv  # must be the row count above")
        return
    cells = json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
    truth = {c["cell_id"]: c["cl_id"] for c in cells.values() if c["registered"]}
    local = {c["cell_id"]: c["cl_id"] for c in cells.values() if not c["registered"]}
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for c in cells.values():
        by_dataset[c["dataset_id"]].append(c["cell_id"])

    # The arm table lives in step3_celltypist.ARMS, so this file cannot go stale silently:
    # an npz on disk whose (dataset, arm) is not in that table is a hard error rather than
    # a quietly skipped baseline.
    from step3_celltypist import ARMS as CT_ARMS

    datasets = [d for d in sorted(by_dataset) if not only or d in only]
    if not datasets:
        raise SystemExit(f"--datasets={sorted(only) if only else None} matched none of {sorted(by_dataset)}")
    print(f"parsed args: argv={argv} datasets={datasets}")

    parents = cl_parents()
    names = cl_names()
    descendants: dict[str, set[str]] = defaultdict(set)
    for cl_id, ps in parents.items():
        for p in ps:
            descendants[p].add(cl_id)
    ancestors_of = lru_cache(maxsize=None)(lambda cl_id: _closure(cl_id, parents))
    descendants_of = lru_cache(maxsize=None)(lambda cl_id: _closure(cl_id, descendants))

    records = [json.loads(line) for line in (QCOUT / "records_test.jsonl").read_text(encoding="utf-8").splitlines()]
    by_family: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_family[record["family"]][record["dataset_id"]].append(record)

    arms = {}
    for dataset in datasets:
        for name in LR_ARMS:
            if (QCOUT / "lr" / f"{name}_{dataset}.npz").exists():
                arms[f"lr_{name}/{dataset}"] = load_lr_arm(name, dataset)
        for arm, model, _transform, _why in CT_ARMS.get(dataset, []):
            path = QCOUT / "celltypist" / f"{dataset}_{arm}.npz"
            if not path.exists():
                print(f"  note: {dataset}/{arm} not on disk yet -- not scored")
                continue
            arms[f"celltypist_{arm}/{dataset}"] = load_ct_arm(dataset, arm, model)
    if not arms:
        raise SystemExit("no baseline predictions found on disk -- nothing to score")
    print(f"scoring {len(arms)} arms: {sorted(arms)}")

    results = {"level_a_cell": {}, "level_b_family": {}, "inventory": {}, "label_space_overlap": {}}
    for key, predictor in arms.items():
        _, dataset = key.split("/")
        predictor_name = key
        registered_cells = [c for c in by_dataset[dataset] if c in truth]
        results["level_a_cell"][predictor_name] = score_cells(
            predictor, registered_cells, truth, ancestors_of, descendants_of
        )
        # Two tags travel with every arm's number, everywhere it is quoted: is it in-domain or
        # transfer, and does it read the full matrix or only the panel. They are not decoration:
        # the in-domain arms use the scored atlas's own labels (reference ceilings, not
        # competitors) and the full-matrix arms have the easier job than our model.
        tags = arm_tags(predictor_name)
        results["level_a_cell"][predictor_name]["tags"] = tags
        # Can this arm even name the types this dataset contains? A disjoint answer space
        # makes 0.0 follow from the setup rather than from measurement, and a reviewer must
        # not read it as "the baseline is broken".
        true_types = {truth[c] for c in registered_cells}
        overlap = sorted(predictor.label_space & true_types)
        results["label_space_overlap"][predictor_name] = {
            "dataset": dataset,
            "arm_labels": len(predictor.label_space),
            "dataset_true_types": len(true_types),
            "overlapping_types": len(overlap),
            "structurally_unable": not overlap,
            "note": "no CL id this arm can emit is one of this dataset's true types, so every prediction on it is "
            "wrong by construction -- the arm cannot attempt the question, not merely fail it"
            if not overlap
            else f"{len(overlap)} of this dataset's {len(true_types)} true types are nameable by this arm",
        }
        family_block = {}
        if any(c in local for c in by_dataset[dataset]):
            family_block["out_of_register"] = score_family(
                "out_of_register", by_family["out_of_register"][dataset], predictor, truth
            )
        for family in CHOICE_FAMILIES:
            if family == "out_of_register":
                continue
            if by_family[family][dataset]:
                family_block[family] = score_family(family, by_family[family][dataset], predictor, truth)
        if by_family["marker_noul"][dataset]:
            family_block["marker_noul"] = score_family("marker_noul", by_family["marker_noul"][dataset], predictor, truth)
        results["level_b_family"][predictor_name] = family_block

    # A floor for reading every accuracy below: what a classifier that ignores the
    # expression matrix entirely would score on this dataset's register-labelled cells
    # (majority class) and what random guessing weighted by class prevalence gives. Without
    # these, an arm's accuracy on a 7-class dataset and a 56-class one look comparable.
    floors = {}
    for dataset, ids in by_dataset.items():
        counts = Counter(truth[c] for c in ids if c in truth)
        total = sum(counts.values())
        if not total:
            continue
        floors[dataset] = {
            "registered_cells": total,
            "distinct_true_types": len(counts),
            "majority_class_rate": round(max(counts.values()) / total, 4),
            "stratified_chance_rate": round(sum((v / total) ** 2 for v in counts.values()), 4),
            "top_true_types": {cl: n for cl, n in counts.most_common(3)},
        }

    results["inventory"] = {
        "test_cells": len(cells),
        "test_cells_registered": len(truth),
        "test_cells_local_label": len(local),
        "per_dataset_cells": {d: {"total": len(v), "registered": sum(1 for c in v if c in truth)}
                              for d, v in sorted(by_dataset.items())},
        "register_terms_in_test_labels": len(set(truth.values())),
        "arms": sorted(arms),
        "requested_datasets": datasets,
        "argv": argv,
        "floors": floors,
        "what_our_model_sees": "the corpus's 201-token rank list plus the panel's option definitions",
        "what_these_baselines_see": "the full expression matrix (all genes) -- the asymmetry this file exists to "
        "make explicit",
        "asymmetry": ASYMMETRY,
        "tag_legend": TAG_LEGEND,
        "arm_tags": {name: ARM_TAGS[name] for name in sorted(ARM_TAGS)},
    }
    write_json(QCOUT / "results.json", results)
    for key in sorted(arms):
        level_a = results["level_a_cell"][key]
        print(f"  {key}: cells={level_a['exact']['n']} exact={level_a['exact']['rate']} "
              f"hierarchy={level_a['hierarchy_consistent']['rate']} "
              f"families={sorted(results['level_b_family'][key])}", flush=True)

    rows = write_results_csv(results)
    print(f"wrote {QCOUT / 'results.json'} and {QCOUT / 'results.csv'} ({rows} rows)")


def rows_in(results: dict) -> list[dict]:
    """Every long-format row of ``results.json``, one per (arm, level, family, metric)."""
    rows = []
    for predictor_name, per_family in results["level_b_family"].items():
        for family, block in per_family.items():
            for metric, value in block.items():
                if isinstance(value, dict) and "rate" in value:
                    rows.append(_row(predictor_name, "family", family, metric, value))
                elif metric in ("correct_rate",) and isinstance(value, float):
                    rows.append(_row(predictor_name, "family", family, metric, {"rate": value, "n": None}))
        for metric in ("exact", "hierarchy_consistent", "exact_over_all_cells", "unmapped_rate"):
            block = results["level_a_cell"].get(predictor_name, {})
            if metric in block:
                rows.append(_row(predictor_name, "cell", "all_registered", metric, block[metric]))
    return rows


def write_results_csv(results: dict, path: Path | None = None) -> int:
    """Write ``results.csv`` from a ``results`` payload. Returns the row count.

    Reads the payload rather than any in-memory state, so the same function serves the scoring run
    and ``--csv-only``: a CSV rebuilt from ``results.json`` cannot disagree with the JSON it was
    rebuilt from. Every row carries its arm's two tags, resolved by ``lib_common`` -- and an arm
    with no registered tags raises there instead of writing a blank.
    """
    path = path or (QCOUT / "results.csv")
    rows = rows_in(results)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _row(arm: str, level: str, family: str, metric: str, block: dict) -> dict:
    ci = block.get("wilson95") or [None, None]
    # The arm's two tags ride on every row. arm_tags raises for an unregistered arm, so there is
    # no path here that writes a number without saying what it is.
    tags = arm_tags(arm)
    return {
        "arm": arm,
        "training_regime": tags["training_regime"],
        "input": tags["input"],
        "level": level,
        "family": family,
        "metric": metric,
        "rate": block.get("rate"),
        "successes": block.get("successes"),
        "n": block.get("n"),
        "wilson_lo": ci[0],
        "wilson_hi": ci[1],
        "thin": block.get("thin"),
    }


def _closure(cl_id: str, relation: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(relation.get(cl_id, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(relation.get(node, ()))
    return seen


if __name__ == "__main__":
    main()
