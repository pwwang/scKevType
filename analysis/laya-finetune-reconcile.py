#!/usr/bin/env python3
"""Re-aggregate a laya rows dump onto the settled census-matched population.

The population and the rule are **not** re-derived here: they are
``reports/census-suite-ids.jsonl``'s ``panel_choice`` rows (1,819 suite rows) and
``reports/abstain-reconciliation.py``'s three rules — ``served`` = ``argmax(p)``,
``abstain-pick`` = the answer is the option whose key is in
``sckevtype.data.families.NONE_OPTIONS``, ``forced`` = ``argmax`` over ``p`` with that
option removed.  This script only joins a laya dump to those rows and counts.

The join is on the row's own ``(id, family)`` — the suite-level join
``reports/both-forced-table.json`` uses for engines that were served the suite's own
records.  A census-level join is impossible for laya: it was never served the census's
2,510 records, which is why ``census_2510`` is reported as not measured.

Running it against ``scratch/runs/laya-pilot-score/rows`` must reproduce
``reports/both-forced-table.json``'s ``laya-pilot`` cell to the integer — that is the
check that this join is the settled one and not a new one.

Usage::

    source scripts/wsl/env.sh && uv run --no-sync python reports/laya-finetune-reconcile.py \\
        --rows scratch/runs/laya-pilot-score/rows --label "Laya pilot (322M, fine-tuned)" \\
        --out reports/laya-finetune-reconcile-pilot.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sckevtype.eval import abstain_index

CENSUS_SUITE_IDS = "reports/census-suite-ids.jsonl"
BOTH_FORCED = "reports/both-forced-table.json"


def read_jsonl(path: Path):
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def argmax(values) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def forced(row) -> int | None:
    """``argmax`` over ``p`` with the abstain option removed — abstention not offered."""
    index = abstain_index(row)
    best, best_value = None, None
    for i, value in enumerate(row["p"]):
        if i == index:
            continue
        if best_value is None or value > best_value:
            best_value, best = value, i
    return best


def sha256_of(paths) -> dict[str, str]:
    digests = {}
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digests[str(path)] = digest.hexdigest()
    return digests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True,
                        help="a laya rows directory (or a glob of .jsonl files)")
    parser.add_argument("--label", default="laya")
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-missing", action="store_true",
                        help="report coverage below 1,819 instead of failing")
    args = parser.parse_args(argv)

    if os.path.isdir(args.rows):
        files = sorted(glob.glob(os.path.join(args.rows, "*.jsonl")))
    else:
        files = sorted(glob.glob(args.rows))
    if not files:
        print(f"no rows files under {args.rows}")
        return 2

    # ---------------------------------------------------------------- the population
    population = [r for r in read_jsonl(ROOT / CENSUS_SUITE_IDS) if r["family"] == "panel_choice"]
    wanted = {(r["id"], r["family"]) for r in population}
    if len(population) != 1819:
        print(f"the census-matched panel_choice population is {len(population)} rows, not 1,819 — "
              "the settled population has moved; stop and re-read reports/census-suite-ids.jsonl")
        return 2

    # ---------------------------------------------------------------- the dump
    answers: dict[tuple[str, str], dict] = {}
    base_rows = 0
    for path in files:
        for row in read_jsonl(Path(path)):
            if row.get("ask") != "base":
                continue
            base_rows += 1
            key = (row["id"], row["family"])
            if key in wanted:
                answers[key] = row

    n = len(answers)
    missing = len(population) - n
    if missing and not args.allow_missing:
        print(f"the dump covers {n} of {len(population)} census-matched panel_choice rows "
              f"({missing} missing); pass --allow-missing to report a partial population")
        return 2

    served_correct = abstain_picks = forced_correct = 0
    no_abstain_option = 0
    hidden_correct = 0
    none_ps: list[float] = []
    label_ps: list[float] = []
    served_picks: dict[str, int] = {}
    abstain_keys: dict[str, int] = {}
    option_counts: list[int] = []
    chance_levels: list[float] = []
    for key in sorted(answers):
        row = answers[key]
        index = abstain_index(row)
        if index is None:
            no_abstain_option += 1
        pick = argmax(row["p"])
        label = row["label"]
        if pick == label:
            served_correct += 1
        served_picks[row["keys"][pick]] = served_picks.get(row["keys"][pick], 0) + 1
        option_counts.append(len(row["keys"]))
        # The abstain option is never the correct answer on `panel_choice`, so a policy that always
        # abstains scores 0 here; the other parameter-free reference is the uniform one, 1/K.
        chance_levels.append(1.0 / len(row["keys"]))
        if index is not None:
            none_ps.append(float(row["p"][index]))
            abstain_keys[row["keys"][index]] = abstain_keys.get(row["keys"][index], 0) + 1
            if pick == index:
                abstain_picks += 1
        label_ps.append(float(row["p"][label]))
        forced_pick = forced(row)
        if forced_pick is not None and forced_pick == label:
            forced_correct += 1
            if pick == index:
                hidden_correct += 1

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    result = {
        "what_this_is": "A laya rows dump re-aggregated onto reports/census-suite-ids.jsonl's "
                        "1,819 panel_choice suite rows, with reports/abstain-reconciliation.py's "
                        "served / abstain-pick / forced rules unchanged.",
        "label": args.label,
        "rows_source": {"glob": args.rows, "files": len(files), "base_rows": base_rows},
        "population": {"source": CENSUS_SUITE_IDS, "n": len(population)},
        "coverage": {"matched": n, "missing": missing,
                     "rows_without_an_abstain_option": no_abstain_option},
        "census_matched_1819": {
            "n": n,
            "served": {"correct": served_correct, "accuracy": served_correct / n if n else None},
            "abstain_pick": {"picks": abstain_picks, "rate": abstain_picks / n if n else None},
            "forced": {"correct": forced_correct, "accuracy": forced_correct / n if n else None},
            "delta_points": (forced_correct - served_correct) / n * 100 if n else None,
            "questions_gained": forced_correct - served_correct,
            "hidden_correct_share_of_abstentions": hidden_correct / abstain_picks if abstain_picks else None,
            "diagnostics": {
                "mean_uniform_chance_1_over_K": sum(chance_levels) / len(chance_levels) if chance_levels else None,
                "mean_options_per_question": sum(option_counts) / len(option_counts) if option_counts else None,
                "min_options_per_question": min(option_counts) if option_counts else None,
                "max_options_per_question": max(option_counts) if option_counts else None,
                "mean_p_abstain": sum(none_ps) / len(none_ps) if none_ps else None,
                "median_p_abstain": percentile(none_ps, 0.5),
                "p90_p_abstain": percentile(none_ps, 0.9),
                "max_p_abstain": max(none_ps) if none_ps else None,
                "rows_with_p_abstain_over_0.5": sum(1 for value in none_ps if value > 0.5),
                "mean_p_of_the_correct_option": sum(label_ps) / len(label_ps) if label_ps else None,
                "median_p_of_the_correct_option": percentile(label_ps, 0.5),
                "abstain_option_keys": abstain_keys,
                "served_pick_distribution_top10": sorted(served_picks.items(),
                                                         key=lambda kv: -kv[1])[:10],
            },
        },
        "sha256": sha256_of(files),
    }

    # ---------------------------------------------------------------- the check
    reference = None
    try:
        both = json.loads((ROOT / BOTH_FORCED).read_text())
        for engine in both["engines"]:
            if engine["key"] == "laya-pilot":
                reference = engine["census_matched_1819"]
    except (OSError, KeyError):
        reference = None
    if reference is not None and args.label.startswith("Laya pilot"):
        mine = result["census_matched_1819"]
        agree = (mine["served"]["correct"] == reference["served"]["correct"]
                 and mine["abstain_pick"]["picks"] == reference["abstain_pick"]["picks"]
                 and mine["forced"]["correct"] == reference["forced"]["correct"])
        result["cross_check_both_forced_table"] = {
            "reference": {"served_correct": reference["served"]["correct"],
                          "abstain_picks": reference["abstain_pick"]["picks"],
                          "forced_correct": reference["forced"]["correct"]},
            "measured": {"served_correct": mine["served"]["correct"],
                         "abstain_picks": mine["abstain_pick"]["picks"],
                         "forced_correct": mine["forced"]["correct"]},
            "agrees": bool(agree),
        }

    cell = result["census_matched_1819"]
    diagnostics = cell["diagnostics"]
    print(f"[{args.label}]  census-matched panel_choice  n={cell['n']} "
          f"(missing {missing}, no-abstain-option {no_abstain_option})")
    print(f"  served   {cell['served']['correct']:5d}/{cell['n']} = {cell['served']['accuracy']:.4f}")
    print(f"  abstain  {cell['abstain_pick']['picks']:5d}/{cell['n']} = {cell['abstain_pick']['rate']:.4f}")
    print(f"  forced   {cell['forced']['correct']:5d}/{cell['n']} = {cell['forced']['accuracy']:.4f}")
    print(f"  delta    {cell['delta_points']:+.2f} pt   hidden-correct "
          f"{cell['hidden_correct_share_of_abstentions']:.4f}")
    print(f"  reference: uniform 1/K {diagnostics['mean_uniform_chance_1_over_K']:.4f} over "
          f"{diagnostics['mean_options_per_question']:.1f} options/question "
          f"({diagnostics['min_options_per_question']}-{diagnostics['max_options_per_question']}); "
          f"always-abstain scores 0 on this family")
    print(f"  p(abstain): mean {diagnostics['mean_p_abstain']:.4f} "
          f"median {diagnostics['median_p_abstain']:.4f} p90 {diagnostics['p90_p_abstain']:.4f} "
          f"max {diagnostics['max_p_abstain']:.4f}  over 0.5: "
          f"{diagnostics['rows_with_p_abstain_over_0.5']}/{cell['n']}")
    print(f"  p(correct option): mean {diagnostics['mean_p_of_the_correct_option']:.4f} "
          f"median {diagnostics['median_p_of_the_correct_option']:.4f}")
    print(f"  abstain option keys: {diagnostics['abstain_option_keys']}")
    print("  top served picks: " + ", ".join(f"{name} {count}"
          for name, count in diagnostics["served_pick_distribution_top10"][:6]))
    if "cross_check_both_forced_table" in result:
        check = result["cross_check_both_forced_table"]
        print(f"  cross-check vs reports/both-forced-table.json: {check['measured']} vs "
              f"{check['reference']} -> {'AGREES' if check['agrees'] else 'DISAGREES'}")

    if args.out:
        (ROOT / args.out).write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
