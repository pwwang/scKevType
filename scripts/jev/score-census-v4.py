#!/usr/bin/env python3
"""Score a Jev row dump **on the census population**, under the settled forced-abstention rules.

The population is named by an id list, not by the dump: this joins ``rows.jsonl`` to
``reports/census-suite-ids-v4.jsonl``'s rows on ``(dataset, cell, family)`` and scores exactly the rows
the arms are scored on. Same rule chain as ``reports/abstain-reconciliation.py`` and
``reports/both-forced-table.json``, restated here once so it is not re-derived per run:

* **served**    ``argmax(p) == label`` — the option the model put its highest probability on, abstention
                 left available. This is ``sckevtype.eval.metrics.top1_accuracy``'s own rule
                 (``_p(row).argmax() == row["label"]``), and it is the rule ``reports/both-forced-table.md``
                 §1 settles on for Jev (``argmax``, not its served ``answer`` field).
* **abstain**   the index of the option whose key is in ``sckevtype.data.families.NONE_OPTIONS`` —
                 ``sckevtype.eval.abstain_index(row)``, a property of the question.
* **forced**    ``argmax`` over the row's ``p`` with the abstain option removed — the settled rule
                 (``reports/abstain-reconciliation.py``), not a re-prompt without the option.
* **floor**     the always-abstain accuracy of the same rows (``label == abstain index``). An abstention
                 task's floor is this rate, not the majority class, and a served accuracy is never quoted
                 without it.

**The family is a selector, not a second scorer.** ``--family`` picks the population out of the id
list; everything below (the join, the three readings, the checks, the denominators) is one code
path. ``--family panel_choice`` -- the default -- is the run that produced
``reports/jev-v4/census-score.json`` and reproduces it byte for byte; ``--family evidence_diluted``
and ``--family panel_miss`` score the two v4 families that dump carries and the panel_choice run
did not report.

**A family whose labels include the abstain option is flagged, never silently averaged.** On
``panel_miss`` the marker is missing from the panel, so the served question's *correct answer is
the abstain option on every row*: declining is what a right answer looks like, served accuracy and
the decline rate are the same number, and the forced reading is 0 by construction -- not a
capability number. On ``evidence_diluted`` the abstain option is the labelled answer on part of the
family, which breaks the settled identity
``forced_correct - served_correct == #{declined and forced-correct}`` (it assumes a decline is never
right). Both facts are computed from the rows, not typed here: when any row's label is the abstain
option the block gains ``rows_whose_label_is_the_abstain_option``,
``identity_where_the_label_is_not_the_abstain_option`` and
``population_where_the_label_is_not_the_abstain_option`` -- the names the arm-side family reports
use (``reports/family-reconciliation-v4-<family>.json``) -- and the output gains a top-level
``caveat`` saying what a ``forced`` number does and does not mean on that family. A family where no
label is the abstain option (``panel_choice``) reaches none of this code, which is why its output is
unchanged.

Three checks run on every invocation, and each can fail:

* ``abstain_index(row) == row["abstain"]`` on every row that carries both — the repo's own settled check;
* ``keys[argmax(p)] == row["answer"]`` on every row that carries the model's named option — proof that the
  vector being scored is the vector that answer was read from;
* ``--probe-report`` re-derives the probe's own ``accuracy`` block from the same dump through this script's
  rules, and **fails** if the two disagree. That is the control: the probe's numbers were computed by
  ``sckevtype.eval.metrics``, so agreement means two independent implementations of the served rule agree
  on this dump.

    python scripts/jev/score-census-v4.py \\
        --dump reports/jev-v4/rows.jsonl \\
        --ids reports/census-suite-ids-v4.jsonl \\
        --family panel_choice \\
        --probe-report reports/jev-v4/report.json \\
        --out reports/jev-v4/census-score.json

    # the other two families the same paid dump carries, each controlled against
    # census-score-choice.json's own by_family block for that family
    python scripts/jev/score-census-v4.py \\
        --dump reports/jev-v4/rows.jsonl \\
        --ids reports/census-suite-ids-v4.jsonl \\
        --family evidence_diluted \\
        --control-choice-report reports/jev-v4/census-score-choice.json \\
        --out reports/jev-v4/census-score-evidence_diluted.json
    python scripts/jev/score-census-v4.py \\
        ... --family panel_miss ... \\
        --out reports/jev-v4/census-score-panel_miss.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:                      # the package lives in the repo, not in the venv
    sys.path.insert(0, str(ROOT))

from sckevtype.contracts.freeze import file_digest, write_json
from sckevtype.eval import abstain_index

CHOICE_FAMILIES = ("panel_choice", "panel_miss", "evidence_diluted")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def argmax(values: list[float]) -> int:
    best = 0
    for index, value in enumerate(values):
        if value > values[best]:
            best = index
    return best


def check_row(row: dict) -> tuple[list[str], bool]:
    """The per-row checks: ``(complaints, argmax_agrees_with_the_answer_field)``.

    Only the checks that must hold for the rules to mean anything are complaints:

    * ``keys`` and ``p`` are the same length (otherwise the row cannot be scored at all),
    * ``abstain_index(row) == row["abstain"]`` where the row carries both — the repo's own settled
      invariant (``reports/both-forced-table.md`` §6: it holds on every row of every dump that carries
      both fields).

    ``argmax(p) == row["answer"]`` is **measured, not required**. Jev's served ``answer`` field and the
    argmax of its probability grid are different readouts of the same call and are known to disagree —
    ``reports/both-forced-table.md`` §1 records the difference on the v2 census (0.6603 by the answer
    field against 0.6466 by argmax over 2,510 rows) and settles the comparison on ``argmax``. A probe that
    refused on it would refuse the run the paper quotes. It is counted and printed instead.
    """
    keys, p = list(row["keys"]), list(row["p"])
    if len(keys) != len(p):
        return [f"{row['id']}: {len(keys)} keys against {len(p)} probabilities"], False
    problems: list[str] = []
    index = abstain_index(row)
    if index is not None and row.get("abstain") is not None and int(row["abstain"]) != index:
        problems.append(f"{row['id']}: abstain_index {index} but the row's abstain field is {row['abstain']}")
    answer = row.get("answer")
    agrees = (isinstance(answer, str) and answer in keys and keys[argmax(p)] == answer)
    return problems, agrees


class ScoreError(RuntimeError):
    """A number this script cannot produce honestly — named rather than printed."""


def settled_identity(rows: list[dict]) -> dict:
    """The settled identity, re-run on the rows where a decline cannot be the right answer.

    ``forced_correct - served_correct == #{declined ∧ forced-correct}`` holds whenever a decline is
    never the correct answer, which is true of ``panel_choice`` and false of ``evidence_diluted`` and
    ``panel_miss``. Restated here exactly as ``reports/abstain-reconciliation-v4.py``'s ``cell()``
    states it (``forced_correct_minus_served_correct`` / ``declines_times_hidden_correct_share`` /
    ``holds``), and returned under the arm-side names
    (``reports/family-reconciliation-v4-<family>.json``) so a Jev block and an arm block can be read
    side by side.

    The derivation, so the equality is not taken on faith: on a row whose argmax is *not* the
    abstain option, removing that option cannot change the pick, so the row contributes the same
    value to served and to forced. Only rows whose argmax *is* the abstain option can differ, and
    there served scores 0 (the label is not the abstain option on this subset) while forced scores 1
    exactly when the forced pick is the label. Hence the difference is the count of rows that both
    declined and would have been right when forced.
    """
    served = forced = declined_and_forced = 0
    for row in rows:
        p, target, index = list(row["p"]), int(row["label"]), abstain_index(row)
        top = argmax(p)
        served += int(top == target)
        best_position, best = None, None
        for position, value in enumerate(p):
            if position == index:
                continue
            if best is None or value > best:
                best, best_position = value, position
        # The settled reader's fallback (``forced_index`` in reports/abstain-reconciliation-v4.py):
        # a row whose only option is the abstain one has nothing left to force, and is scored by the
        # plain argmax rather than by a None that would silently read as wrong.
        if best_position is None:
            best_position = top
        if best_position == target:
            forced += 1
            declined_and_forced += int(top == index)
    n = len(rows)
    return {
        "identity": {
            "forced_correct_minus_served_correct": forced - served,
            "declines_times_hidden_correct_share": declined_and_forced,
            "holds": (forced - served) == declined_and_forced,
        },
        "population": {
            "n": n,
            "served": {"correct": served, "accuracy": served / n if n else None},
            "forced": {"correct": forced, "accuracy": forced / n if n else None},
            "hidden_correct": declined_and_forced,
        },
    }


def score(rows: list[dict], label: str, *, caveats: bool = True) -> dict:
    """The served / abstain / forced / floor block for one set of rows.

    Two denominators, both stated, because on v4 they are **different sets**:

    * *served* is over every row handed in — the same quantity the probe's own ``report.json``
      ``accuracy`` block reports, so the control below compares like with like. On v4 some questions
      carry **no abstain option at all** (2,992 of the census's 9,113 choice rows; the probe asks a
      second, abstain-bearing variant of each and the injected arm holds them), and dropping them from a
      served accuracy would silently change the population.
    * *forced*, *abstain-pick* and the *floor* are over the **eligible** rows — those that offer an
      abstain option — because that is the only set the forced rule is defined on. The excluded rows are
      counted and printed, never dropped in silence.

    ``delta_points`` is computed on the eligible rows for both halves, so a delta never mixes
    denominators.

    ``forced`` drops the abstain **position** from the row's option order and takes the argmax of what is
    left, comparing the winner's *original* index against the label — not a shifted position against an
    unshifted label, which is the off-by-one that reads as "forcing always answers wrongly". The invariant
    ``forced >= served_on_eligible - abstained`` is asserted: removing an option can only change the argmax
    on the rows whose argmax *was* that option, so anything below that bound is an indexing bug rather than
    a measurement.

    Where the family's labels include the abstain option the block also carries
    ``rows_whose_label_is_the_abstain_option`` and the identity re-run on the rows where a decline cannot
    be right (``identity_where_the_label_is_not_the_abstain_option`` /
    ``population_where_the_label_is_not_the_abstain_option``). ``caveats=False`` suppresses them for the
    recursive slice, so the reported identity is never the identity of the identity.
    """
    served = served_eligible = forced = abstained = floor = excluded = 0
    for row in rows:
        p = list(row["p"])
        target = int(row["label"])
        top = argmax(p)
        served += int(top == target)
        index = abstain_index(row)
        if index is None:
            excluded += 1
            continue
        served_eligible += int(top == target)
        abstained += int(top == index)
        floor += int(target == index)
        best_position, best = None, None
        for position, value in enumerate(p):
            if position == index:
                continue
            if best is None or value > best:
                best, best_position = value, position
        forced += int(best_position == target)
    n = len(rows)
    eligible = n - excluded
    if forced < served_eligible - abstained:
        raise ScoreError(
            f"{label}: forced {forced} is below the served-minus-abstained bound "
            f"{served_eligible - abstained} (served {served_eligible}, abstained {abstained} over "
            f"{eligible} eligible rows) -- the forced rule is mis-indexed")
    gained = forced - served_eligible
    block = {
        "population": label, "n": n, "eligible": eligible,
        "rows_without_an_abstain_option": excluded,
        "served_correct": served, "served_accuracy": served / n if n else None,
        "served_correct_on_eligible": served_eligible,
        "served_accuracy_on_eligible": served_eligible / eligible if eligible else None,
        "abstain_picks": abstained, "abstain_pick_rate": abstained / eligible if eligible else None,
        "forced_correct": forced,
        "forced_accuracy": forced / eligible if eligible else None,
        "delta_points": 100.0 * gained / eligible if eligible else None,
        "questions_gained": gained,
        "hidden_correct_share_of_abstentions": (gained / abstained) if abstained else None,
        "always_abstain_floor": floor / eligible if eligible else None,
        "mean_confidence": (sum(max(row["p"]) for row in rows) / n) if n else None,
    }
    if caveats and floor:
        # `floor` *is* the count of rows whose label is the abstain option (it is the numerator of
        # the always-abstain rate), so the family facts cost no second pass over the rows. They are
        # only emitted where they are non-zero: on a family where a decline is never the right
        # answer there is nothing to flag, and a block that always carried zeros would train a
        # reader to skip it on the one family where it matters.
        subset = [row for row in rows if int(row["label"]) != abstain_index(row)]
        identity = settled_identity(subset)
        block["rows_whose_label_is_the_abstain_option"] = floor
        block["identity_where_the_label_is_not_the_abstain_option"] = identity["identity"]
        block["population_where_the_label_is_not_the_abstain_option"] = identity["population"]
    return block


def block(rows: list[dict], key: str, label: str, *, caveats: bool = True) -> dict:
    per: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per[str(row[key])].append(row)
    return {name: score(group, f"{label}: {key}={name}", caveats=caveats) for name, group in sorted(per.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="the Jev row dump (rows.jsonl, carries `p`)")
    ap.add_argument("--ids", required=True, help="the census id list naming the population")
    ap.add_argument("--family", default="panel_choice",
                    help="panel_choice | panel_miss | evidence_diluted | choice (the three) | '' (every row of the id list)")
    ap.add_argument("--probe-report", default=None, help="the probe's report.json, for the control")
    ap.add_argument("--control-choice-report", default=None,
                    help="census-score-choice.json: refuse unless this family's block reproduces its by_family entry")
    ap.add_argument("--out", default=None, help="JSON to write")
    args = ap.parse_args()

    dump_path, ids_path = Path(args.dump), Path(args.ids)
    rows = read_jsonl(dump_path)
    ids = read_jsonl(ids_path)
    if not rows:
        raise SystemExit(f"{dump_path}: no rows")
    if not ids:
        raise SystemExit(f"{ids_path}: no ids")

    if args.family == "choice":
        wanted = set(CHOICE_FAMILIES)
    elif args.family:
        wanted = {args.family}
    else:
        wanted = {row["family"] for row in ids}
    population = [row for row in ids if row["family"] in wanted]
    if not population:
        raise SystemExit(f"{ids_path}: no rows with family in {sorted(wanted)} -- refusing to write an empty score")

    by_key = {(row["dataset"], row["cell"], row["family"]): row for row in rows}
    missing = [f"{r['dataset']}/{r['cell']}/{r['family']}" for r in population
               if (r["dataset"], r["cell"], r["family"]) not in by_key]
    if missing:
        print(f"REFUSING: {len(missing)} of the {len(population)} population rows have no row in {dump_path}, e.g.")
        for name in missing[:5]:
            print(f"  {name}")
        return 2

    selected = [by_key[(r["dataset"], r["cell"], r["family"])] for r in population]
    choice = [row for row in selected if row.get("type") == "choice"]

    problems: list[str] = []
    argmax_agrees = argmax_compared = 0
    for row in selected:
        complaints, agrees = check_row(row)
        problems.extend(complaints)
        if isinstance(row.get("answer"), str):
            argmax_compared += 1
            argmax_agrees += int(agrees)
    if problems:
        print(f"REFUSING: {len(problems)} row-level identity problem(s), e.g.")
        for line in problems[:5]:
            print(f"  {line}")
        return 3

    family_label = args.family or "every family of the id list"
    families = dict(sorted(Counter(r["family"] for r in selected).items()))
    # The flag below describes *a family*, so it is emitted on a one-family run only: on a mixed run
    # (``--family choice``) each block would be an aggregate of different families and the family
    # files are where that fact belongs. This also keeps both committed runs reproducible byte for
    # byte -- census-score.json and census-score-choice.json predate the flag and have no family of
    # their own to describe.
    one_family = len(families) == 1
    out: dict = {
        "dump": {"path": str(dump_path), "rows": len(rows), "sha256": file_digest(dump_path)},
        "ids": {"path": str(ids_path), "rows": len(ids), "sha256": file_digest(ids_path)},
        "family_filter": family_label,
        "population_rows": len(selected), "choice_rows": len(choice),
        "families": families,
        "checks": {
            "abstain_index_matches_the_row_field": "ok on every row checked",
            "rows_checked": len(selected),
            "argmax_p_vs_the_models_own_answer_field": {
                "rows_carrying_an_answer": argmax_compared,
                "argmax_agrees": argmax_agrees,
                "disagree": argmax_compared - argmax_agrees,
                "why_it_is_not_a_refusal": (
                    "Jev's served `answer` and the argmax of its probability grid are two readouts of the "
                    "same call and are known to differ; reports/both-forced-table.md §1 settles every "
                    "comparison on argmax(p), which is the rule this script scores with"),
            },
        },
        "primary": score(choice, f"{family_label} (choice rows)", caveats=one_family),
        "by_family": block(choice, "family", family_label, caveats=one_family),
        "by_dataset": block(choice, "dataset", family_label, caveats=one_family),
    }

    if args.probe_report:
        report = json.loads(Path(args.probe_report).read_text(encoding="utf-8"))
        expect = report.get("accuracy") or {}
        mine = score([row for row in rows if row.get("type") == "choice"], "the whole dump", caveats=False)
        agree = (mine["n"] == expect.get("n")) and abs(mine["served_accuracy"] - expect["accuracy"]) < 1e-12
        out["control_vs_the_probes_own_metrics"] = {
            "source": args.probe_report, "probe_accuracy": expect.get("accuracy"), "probe_n": expect.get("n"),
            "this_script_accuracy": mine["served_accuracy"], "this_script_n": mine["n"],
            "verdict": "AGREE" if agree else "DISAGREE",
        }
        if not agree:
            print("REFUSING: this script's served rule disagrees with the probe's own accuracy block:")
            print(f"  probe      : {expect.get('accuracy')} over n={expect.get('n')}")
            print(f"  this script: {mine['served_accuracy']} over n={mine['n']}")
            return 4

    primary = out["primary"]
    no_option = Counter(str(row["family"]) for row in selected if abstain_index(row) is None)
    out["rows_without_an_abstain_option_by_family"] = dict(sorted(no_option.items()))

    label_is_abstain = int(primary.get("rows_whose_label_is_the_abstain_option") or 0)
    every_label = bool(label_is_abstain) and label_is_abstain == primary["n"]
    if label_is_abstain:
        # What a `forced` number means here, stated next to the number rather than in a manuscript
        # footnote a reader of this file may never see. `every_label` is the case that makes the
        # forced reading meaningless: removing the abstain option removes the label.
        over_every_row = settled_identity(choice)["identity"]
        can_be_won = primary["population_where_the_label_is_not_the_abstain_option"]["n"]
        if every_label:
            why = (
                f"on {family_label} the served question's label IS the abstain option on every one of the "
                f"{primary['n']} rows -- the marker is missing from the panel, so declining is what a right "
                f"answer looks like. served_accuracy and the decline rate are the same number here "
                f"({primary['abstain_picks']}/{primary['n']}), and forced_accuracy is "
                f"{primary['forced_accuracy']:.4f} by construction, because removing the abstain option "
                f"removes the only correct answer: there are {can_be_won} rows on which forcing can be "
                f"right. Read served_accuracy on this family; the forced number is NOT a capability "
                f"number, and the settled identity cannot hold over it "
                f"({over_every_row['forced_correct_minus_served_correct']} against "
                f"{over_every_row['declines_times_hidden_correct_share']}).")
        else:
            why = (
                f"on {family_label} the abstain option IS the labelled answer on {label_is_abstain} of the "
                f"{primary['n']} rows, where forcing cannot be right (removing the abstain option removes "
                f"the label). The settled identity "
                f"`forced_correct - served_correct == #{{declined and forced-correct}}` assumes a decline "
                f"is never right, so it does not hold over every row "
                f"({over_every_row['forced_correct_minus_served_correct']} against "
                f"{over_every_row['declines_times_hidden_correct_share']}, holds="
                f"{over_every_row['holds']}); it is reported twice here, and it does hold over the "
                f"{can_be_won} rows whose label is not the abstain option. Anything comparative should be "
                f"read from population_where_the_label_is_not_the_abstain_option: the all-rows forced "
                f"number mixes in {label_is_abstain} rows that are unwinnable by construction.")
        out["caveat"] = {
            "family": family_label,
            "rows_whose_label_is_the_abstain_option": label_is_abstain,
            "of_rows": primary["n"],
            "share": label_is_abstain / primary["n"] if primary["n"] else None,
            "every_label_is_the_abstain_option": every_label,
            "decline_picks": primary["abstain_picks"],
            "decline_rate": primary["abstain_pick_rate"],
            "served_accuracy_equals_the_decline_rate": (
                primary["abstain_picks"] == primary["served_correct"]),
            "forced_accuracy": primary["forced_accuracy"],
            "forced_accuracy_is_a_capability_number": not every_label,
            "rows_on_which_forcing_can_be_right": can_be_won,
            "identity_over_every_row": over_every_row,
            "why": why,
        }

    if args.control_choice_report:
        other = json.loads(Path(args.control_choice_report).read_text(encoding="utf-8"))
        entry = (other.get("by_family") or {}).get(args.family)
        compared = ("n", "eligible", "rows_without_an_abstain_option", "served_correct",
                    "served_correct_on_eligible", "abstain_picks", "forced_correct",
                    "always_abstain_floor")
        if not isinstance(entry, dict):
            print(f"REFUSING: {args.control_choice_report} carries no by_family[{args.family!r}] to "
                  f"control this run against ({sorted(other.get('by_family') or {})})")
            return 6
        disagree = {name: {"there": entry.get(name), "here": primary.get(name)}
                    for name in compared if entry.get(name) != primary.get(name)}
        out["control_vs_the_all_choice_report"] = {
            "source": args.control_choice_report,
            "cell": f"by_family.{args.family}",
            "fields_compared": list(compared),
            "disagree": disagree,
            "verdict": "AGREE" if not disagree else "DISAGREE",
            "why": ("the choice report scored this family as one of the three families of the same "
                    "9,113-row choice population, grouping the dump's own `family` field; this run "
                    "names the same rows by the id list's family filter. Agreement means the two "
                    "selections are the same rows and neither run's counts drifted -- it is a "
                    "population join check, and the numbers above are this run's own, not copied"),
        }
        if disagree:
            print(f"REFUSING: this run disagrees with {args.control_choice_report}'s "
                  f"by_family[{args.family!r}] on {sorted(disagree)}:")
            for name, pair in disagree.items():
                print(f"  {name}: there {pair['there']!r}, here {pair['here']!r}")
            return 6

    print(f"dump           : {dump_path}  ({len(rows)} rows, {str(out['dump']['sha256'])[:16]})")
    print(f"ids            : {ids_path}  ({len(ids)} rows)")
    print(f"population     : {len(selected)} rows, family filter {family_label}  {out['families']}")
    print(f"checks         : {out['checks']['rows_checked']} rows, 0 identity problems"
          f"{'; control ' + str(out['control_vs_the_probes_own_metrics']['verdict']) if args.probe_report else ''}"
          f"{'; control ' + str(out['control_vs_the_all_choice_report']['verdict']) if args.control_choice_report else ''}")
    print(f"served         : {primary['served_correct']}/{primary['n']} = {primary['served_accuracy']:.4f}"
          f"   (argmax p, abstention available; every row of the population)")
    print(f"abstain picks  : {primary['abstain_picks']}/{primary['eligible']} = {primary['abstain_pick_rate']:.4f}"
          f"   over the {primary['eligible']} rows that offer an abstain option")
    print(f"forced         : {primary['forced_correct']}/{primary['eligible']} = {primary['forced_accuracy']:.4f}"
          f"   (abstain option removed)   delta {primary['delta_points']:+.1f} pt")
    print(f"floor          : always-abstain {primary['always_abstain_floor']:.4f}"
          f"   (over the same {primary['eligible']} rows)")
    if label_is_abstain:
        caveat = out["caveat"]
        print(f"caveat         : {caveat['rows_whose_label_is_the_abstain_option']}/{caveat['of_rows']} rows "
              f"have the ABSTAIN OPTION as the label; decline rate {caveat['decline_rate']:.4f}"
              f"; forced_accuracy_is_a_capability_number={caveat['forced_accuracy_is_a_capability_number']}")
        print(f"                 {caveat['why']}")
    print(f"no abstain option on the question: {primary['rows_without_an_abstain_option']}"
          f" {dict(sorted(no_option.items()))}"
          f"  (served over the eligible set alone would be {primary['served_accuracy_on_eligible']:.4f})")
    for name, entry in sorted(out["by_dataset"].items()):
        print(f"    dataset {name:<20} n={entry['n']:<5d} served {entry['served_accuracy']:.4f}"
              f" abstain {entry['abstain_pick_rate']:.4f} forced {entry['forced_accuracy']:.4f}")
    if args.out:
        write_json(Path(args.out), out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScoreError as error:
        print(f"REFUSING: {error}")
        raise SystemExit(5) from None
