#!/usr/bin/env python3
"""Score a learning-curve arm on the settled census-matched panel_choice population.

The population and the rule are NOT re-derived here: they are the ones reports/abstain-reconciliation.py
and reports/both-forced-table.json already fix.

  population   the suite rows of reports/census-suite-ids.jsonl with family panel_choice (1,819 ids,
               joined on the row's own id -- the rule both-forced-table.json uses for suite-level
               engines, and the population reports/abstain-reconciliation.py calls census_matched_1819)
  naming       row["top1"] == row["label"]  (top1 is the eval's argmax over the row's own p vector at
               T = 1, written by sckevtype.eval.run.write_full_row_dump)
  abstain-pick row["top1"] == row["abstain"]  (the option index whose key is the abstain wording; a
               property of the question, carried by the dump)

The fine-tuned arms' dumps carry top1/top1_p and no p vector, so the forced rule (argmax with the
abstain option removed) is not recoverable from them -- both-forced-table.json says so in the same
words for the same arms, and this script therefore reports the two quantities the dump supports.

Run it with no arguments to check itself against the numbers already on disk: it re-scores the five
existing 100% runs and must reproduce reports/both-forced-table.json's cells to the integer.

    source scripts/wsl/env.sh && uv run --no-sync python reports/learning-curve-score.py --self-test
    ... --arm delta-kev-0.8b-lc25-s1 --fraction 0.25 --records 1398 --seed 1 \
        --dump reports/rows-full-delta-kev-0.8b-lc25-s1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = "reports/census-suite-ids.jsonl"
POPULATION = "census_matched_1819"

#: The five completed 100% runs and the two numbers reports/both-forced-table.json records for each.
#: ``abstain`` is read off the dump's own field, exactly as the table does.
KNOWN_100 = {
    "delta-kev-0.8b-s1": {"correct": 218, "abstain_pick": 1569},
    "delta-kev-0.8b-s2": {"correct": 83, "abstain_pick": 1691},
    "delta-kev-0.8b-s3": {"correct": 101, "abstain_pick": 1675},
    "delta-kev-0.8b-s4": {"correct": 191, "abstain_pick": 1582},
    "delta-kev-0.8b-s5": {"correct": 108, "abstain_pick": 1654},
}


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def population_ids() -> list[str]:
    rows = read_jsonl(IDS)
    ids = sorted({row["id"] for row in rows if row["family"] == "panel_choice"})
    if len(ids) != 1819:
        raise SystemExit(f"REFUSED: {IDS} carries {len(ids)} panel_choice ids, not the settled 1,819")
    return ids


def score(dump_path: str, ids: list[str]) -> dict:
    """The two measured quantities, on exactly those ids, with the missing-row check."""
    rows = {}
    for row in read_jsonl(dump_path):
        if row.get("family") == "panel_choice":
            rows[row["id"]] = row
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"REFUSED: {dump_path} carries no row for {len(missing)} of the 1,819 ids "
                         f"(first: {missing[:3]}); a partial population is not this population")
    n = len(ids)
    correct = sum(1 for i in ids if rows[i]["top1"] == rows[i]["label"])
    abstained = sum(1 for i in ids if rows[i]["top1"] == rows[i]["abstain"])
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "abstain_pick": abstained, "abstain_pick_rate": abstained / n}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def self_test(ids: list[str]) -> int:
    bad = 0
    print("[self-test: the five 100% runs, against reports/both-forced-table.json]")
    for arm, want in KNOWN_100.items():
        dump = f"reports/rows-full-{arm}.jsonl"
        got = score(dump, ids)
        ok = got["correct"] == want["correct"] and got["abstain_pick"] == want["abstain_pick"]
        bad += 0 if ok else 1
        print(f"  {arm:20s} correct {got['correct']:4d} (want {want['correct']:4d})  "
              f"abstain-pick {got['abstain_pick']:4d} (want {want['abstain_pick']:4d})  "
              f"{'OK' if ok else 'MISMATCH'}")
    print(f"  {'SELF-TEST OK' if not bad else 'SELF-TEST FAILED'}: {len(KNOWN_100) - bad}/{len(KNOWN_100)} reproduce")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a learning-curve arm on the census-matched population.")
    parser.add_argument("--self-test", action="store_true", help="re-score the five 100% runs and compare to both-forced-table.json")
    parser.add_argument("--arm", action="append", default=[], help="arm name (repeatable)")
    parser.add_argument("--dump", action="append", default=[], help="rows-full dump for the arm (repeatable, same order)")
    parser.add_argument("--fraction", action="append", default=[], help="training fraction of the arm (repeatable)")
    parser.add_argument("--records", action="append", default=[], help="training records of the arm (repeatable)")
    parser.add_argument("--seed", action="append", default=[], help="training seed of the arm (repeatable)")
    parser.add_argument("--out", default="reports/learning-curve.json")
    args = parser.parse_args()

    ids = population_ids()
    print(f"population      : {IDS} -> {len(ids)} panel_choice ids ({POPULATION})")

    if args.self_test and not args.arm:
        return self_test(ids)

    if not args.arm:
        parser.error("no --arm given; use --self-test for the on-disk check")
    if not (len(args.arm) == len(args.dump) == len(args.fraction) == len(args.records) == len(args.seed)):
        parser.error("--arm/--dump/--fraction/--records/--seed must be given the same number of times")

    arms = []
    for arm, dump, frac, recs, seed in zip(args.arm, args.dump, args.fraction, args.records, args.seed):
        result = score(dump, ids)
        entry = {"arm": arm, "dump": dump, "dump_sha256": sha256_file(dump),
                 "training_fraction": float(frac), "training_records": int(recs), "seed": int(seed), **result}
        arms.append(entry)
        print(f"  {arm:26s} frac {float(frac):.2f}  n {result['n']}  correct {result['correct']:4d} "
              f"= {result['accuracy']:.4f}  abstain-pick {result['abstain_pick']:4d} = {result['abstain_pick_rate']:.4f}")

    out = {
        "what_this_is": ("the learning curve of the 0.8B delta recipe on the census-matched panel_choice "
                         "population: naming accuracy (top1 == label) and abstain-pick rate (top1 == abstain) "
                         "per training-set size"),
        "population": {"name": POPULATION, "n": len(ids), "ids_from": IDS, "ids_sha256": sha256_file(IDS),
                       "join": "on the row's own id, the rule reports/both-forced-table.json uses for suite-level engines",
                       "rule": "naming: top1 == label; abstain-pick: top1 == abstain; both read from the "
                               "rows-full dump (reports/abstain-reconciliation.py's served quantities for a "
                               "top1-only dump; forced is not recoverable from these dumps)"},
        "arms": arms,
    }
    path = os.path.join(ROOT, args.out)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
