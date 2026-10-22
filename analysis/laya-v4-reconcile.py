#!/usr/bin/env python3
"""Re-aggregate a laya rows dump onto the **v4 census-matched population**.

Same rule, named population. ``reports/laya-finetune-reconcile.py`` is the settled join and it is
imported here rather than re-implemented -- ``argmax``, ``forced`` and ``read_jsonl`` are that file's
own functions, loaded by path, so the served / abstain-pick / forced rules cannot drift between the
v3 and the v4 table. What differs is only the thing that file hard-codes:

* its population is ``reports/census-suite-ids.jsonl`` (1,819 ``panel_choice`` rows) and it refuses to
  run when that file holds anything else -- correct for the v2/v3 table, useless for the v4 one;
* its counts are reported under ``census_matched_1819`` and cross-checked against
  ``reports/both-forced-table.json``'s ``laya-pilot`` cell (the v3 self-check).

This script takes the population as an argument (``--ids``, ``--family``), derives its own count from
the file it was handed -- there is no number in this file that a suite could move -- and reports the
counts under ``census_matched_<n>``. The v3 file is left untouched, so the settled cell it reproduces
still reproduces.

Usage::

    uv run python reports/laya-v4-reconcile.py --rows runs/laya-v4-census-spec/rows \\
        --label "Laya v4 (spec)" --out runs/laya-v4-census-spec/reconcile.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sckevtype.eval import abstain_index                      # noqa: E402

SETTLED = ROOT / "analysis" / "laya-finetune-reconcile.py"
DEFAULT_IDS = "reports/census-suite-ids-v4.jsonl"


def _settled_module():
    """``reports/laya-finetune-reconcile.py`` as a module (its name has dashes, so not by import)."""
    spec = importlib.util.spec_from_file_location("laya_finetune_reconcile", SETTLED)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the settled reconcile rules from {SETTLED}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The three names this script reuses. If the settled file is refactored, this refuses by name
    # rather than silently re-implementing a rule that has moved.
    for name in ("argmax", "forced", "read_jsonl", "sha256_of"):
        if not hasattr(module, name):
            raise SystemExit(f"{SETTLED} no longer defines {name}(); the v4 join would have to "
                             f"re-derive the rule -- re-read the settled file before continuing")
    return module


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="a laya rows directory (or a glob of .jsonl files)")
    parser.add_argument("--ids", default=DEFAULT_IDS,
                        help=f"the population file (default {DEFAULT_IDS})")
    parser.add_argument("--family", default="panel_choice",
                        help="only rows of this family are the population ('' = every row)")
    parser.add_argument("--label", default="laya")
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-missing", action="store_true",
                        help="report coverage below the population instead of failing")
    args = parser.parse_args(argv)

    settled = _settled_module()
    read_jsonl, argmax, forced, sha256_of = (settled.read_jsonl, settled.argmax, settled.forced,
                                             settled.sha256_of)

    import glob

    if os.path.isdir(args.rows):
        files = sorted(glob.glob(os.path.join(args.rows, "*.jsonl")))
    else:
        files = sorted(glob.glob(args.rows))
    if not files:
        print(f"no rows files under {args.rows}")
        return 2

    # ---------------------------------------------------------------- the population
    ids_path = Path(args.ids)
    if not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    if not ids_path.is_file():
        print(f"no population file at {ids_path}")
        return 2
    population = [r for r in read_jsonl(ids_path) if args.family in (None, "", r.get("family"))]
    if not population:
        print(f"{ids_path}: no {args.family!r} rows -- refusing to report a join onto nothing")
        return 2
    wanted = {(r["id"], r["family"]) for r in population}
    print(f"population: {len(population)} rows from {ids_path} (family {args.family!r}), "
          f"{len(wanted)} distinct (id, family) keys")
    if len(wanted) != len(population):
        print(f"{ids_path}: {len(population) - len(wanted)} repeated (id, family) keys -- a repeated key "
              f"would be counted twice; refusing")
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
        print(f"the dump covers {n} of {len(population)} census-matched {args.family} rows "
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

    cell_key = f"census_matched_{len(population)}"
    result = {
        "what_this_is": f"A laya rows dump re-aggregated onto {args.ids}'s {len(population)} "
                        f"{args.family} rows, with reports/abstain-reconciliation.py's served / "
                        f"abstain-pick / forced rules unchanged. The rules themselves are imported "
                        f"from reports/laya-finetune-reconcile.py (the settled join); only the "
                        f"population is a parameter here.",
        "label": args.label,
        "rows_source": {"glob": args.rows, "files": len(files), "base_rows": base_rows},
        "population": {"source": args.ids, "family": args.family, "n": len(population),
                       "distinct_keys": len(wanted)},
        "coverage": {"matched": n, "missing": missing,
                     "rows_without_an_abstain_option": no_abstain_option},
        cell_key: {
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
                "served_pick_distribution_top10": sorted(served_picks.items(), key=lambda kv: -kv[1])[:10],
            },
        },
        "settled_rules_from": {"path": str(SETTLED.relative_to(ROOT)),
                               "sha256": sha256_of([SETTLED])[str(SETTLED)]},
        "sha256": sha256_of(files),
    }

    cell = result[cell_key]
    diagnostics = cell["diagnostics"]
    print(f"[{args.label}]  census-matched {args.family}  n={cell['n']} "
          f"(missing {missing}, no-abstain-option {no_abstain_option})")
    print(f"  served   {cell['served']['correct']:5d}/{cell['n']} = {cell['served']['accuracy']:.4f}")
    print(f"  abstain  {cell['abstain_pick']['picks']:5d}/{cell['n']} = {cell['abstain_pick']['rate']:.4f}")
    print(f"  forced   {cell['forced']['correct']:5d}/{cell['n']} = {cell['forced']['accuracy']:.4f}")
    print(f"  delta    {cell['delta_points']:+.2f} pt")
    if diagnostics["mean_uniform_chance_1_over_K"] is not None:
        print(f"  reference: uniform 1/K {diagnostics['mean_uniform_chance_1_over_K']:.4f} over "
              f"{diagnostics['mean_options_per_question']:.1f} options/question "
              f"({diagnostics['min_options_per_question']}-{diagnostics['max_options_per_question']})")
    if diagnostics["mean_p_abstain"] is not None:
        print(f"  p(abstain): mean {diagnostics['mean_p_abstain']:.4f} "
              f"median {diagnostics['median_p_abstain']:.4f} p90 {diagnostics['p90_p_abstain']:.4f}")
    print(f"  abstain option keys: {diagnostics['abstain_option_keys']}")
    print("  top served picks: " + ", ".join(f"{name} {count}"
          for name, count in diagnostics["served_pick_distribution_top10"][:6]))

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
