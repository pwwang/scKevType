#!/usr/bin/env python3
"""Re-measure a suite's state tokens with the real tokenizer and compare them to its manifest.

The budget a run must train against is the one the suite manifest records
(``context.max_state``), measured from the data at build time by
``sckevtype.train.suite``. This re-measures the written partitions with kev's own
counting rule (:mod:`sckevtype.contracts.budget`) and reports whether the recorded
budget holds every record — a second opinion on the number, on the bytes on disk.

Usage: python scripts/measure-state-tokens.py <suite-dir> [--sample N]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

from sckevtype.contracts import budget as state_budget


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("suite", nargs="?", default="artifacts/suites/real-v2", help="suite directory")
    parser.add_argument("--sample", type=int, default=0, help="records per split (0 = every record)")
    args = parser.parse_args()

    suite = Path(args.suite)
    manifest = json.loads((suite / "manifest.json").read_text(encoding="utf-8"))
    context = manifest.get("context") or {}
    if "max_state" not in context:
        print(f"{suite}/manifest.json records no context.max_state — nothing to compare against", file=sys.stderr)
        return 1
    recorded = context["max_state"]
    pin = context.get("tokenizer")
    tokenizer = state_budget.pinned_tokenizer(pin) if pin else None
    print(f"suite: {suite}")
    print(f"manifest records: max_state {recorded} (source: {context.get('source')}), "
          f"max_branch {context.get('max_branch')}, max_packed {context.get('max_packed')}")
    print(f"tokenizer: {pin['id'] + ' @ ' + pin['revision'][:12] if pin else 'kev default'}")

    per_split: dict[str, list[int]] = {}
    for split in manifest.get("splits", ()):
        path = suite / f"{split}.jsonl"
        if not path.exists():
            continue
        states = (json.loads(line)["state"] for line in itertools.islice(path.open(), args.sample or None))
        per_split[split] = state_budget.state_lengths(tokenizer, states)

    print()
    print(f"{'split':14s} {'n':>7s} {'min':>6s} {'median':>7s} {'p95':>6s} {'max':>6s}  over {recorded}")
    worst = 0
    over_total = 0
    for split, lengths in per_split.items():
        summary = state_budget.summarize(lengths)
        over = state_budget.over_budget(lengths, recorded)
        worst = max(worst, summary["max"])
        over_total += over
        print(f"{split:14s} {summary['n']:7d} {summary['min']:6d} {summary['median']:7d} "
              f"{summary['p95']:6d} {summary['max']:6d}  {over}/{summary['n']}")

    required = state_budget.budget_for(worst)
    print()
    print(f"observed max: {worst} tokens    required budget (round up to {state_budget.BUDGET_UNIT}): {required}")
    print(f"manifest records {recorded}; {over_total} record(s) over it"
          f"{' (sampled)' if args.sample else ''}")
    if over_total or recorded < required:
        print(f"MISMATCH: rebuild the suite with --max_state {required}, or with --truncate to cut these to {recorded}")
        return 1
    print(f"AGREES: every measured state fits the recorded {recorded}-token budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
