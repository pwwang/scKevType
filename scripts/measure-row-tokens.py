#!/usr/bin/env python3
"""Measure the training-row token requirement of a suite, per family.

kev couples the budgets: `training_context(max_state)` gives

    branch budget = 1024 + (max_state - 384)
    packed budget = 2048 + (max_state - 384)

and a record must satisfy state + branch <= packed. So the budget that decides whether a suite can
train is the **row** (state + one branch), not the state alone — measuring the state alone is the
bug this script exists to expose.

Usage: python scripts/measure-branch-tokens.py [suite-dir] [sample] [split]
"""
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

suite = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/suites/real-v2")
sample = int(sys.argv[2]) if len(sys.argv) > 2 else 200
split = sys.argv[3] if len(sys.argv) > 3 else "train"
revision = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", revision=revision)


def ntok(text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _branch_text(question: object) -> str:
    """The option text a branch carries: its instructions plus every option's rendered body."""
    if isinstance(question, str):
        return question
    if not isinstance(question, dict):
        return str(question)
    parts = []
    for value in question.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for option, body in value.items():
                parts.append(f"{option}: {body}")
    return "\n".join(parts)


rows: dict[str, list[tuple[int, int, int]]] = defaultdict(list)  # family -> [(state, branch, row)]
example: dict[str, tuple[int, int, str]] = {}

with (suite / f"{split}.jsonl").open() as fh:
    for line in itertools.islice(fh, sample):
        rec = json.loads(line)
        family = rec["_meta"].get("family", "?")
        s = ntok(rec["state"])
        items = [_branch_text(q) for q in (
            rec["questions"].values() if isinstance(rec["questions"], dict) else rec["questions"])]
        for text in items:
            b = ntok(text)
            rows[family].append((s, b, s + b))
            if family not in example or (s + b) > example[family][2]:
                example[family] = (s, b, s + b, text)

print(f"suite: {suite}   split: {split}   records sampled: {sample}")
print()
print(f"{'family':20s} {'n':>5s} {'state':>7s} {'branch':>7s} {'ROW':>7s} {'rows>3712':>10s}")
worst_row = 0
for family, vals in sorted(rows.items()):
    st = [v[0] for v in vals]
    br = [v[1] for v in vals]
    rw = [v[2] for v in vals]
    worst_row = max(worst_row, max(rw))
    over = sum(1 for v in rw if v > 3712)   # kev's packed budget at max_state=2048
    print(f"{family:20s} {len(vals):5d} {max(st):7d} {max(br):7d} {max(rw):7d} {over:5d}/{len(vals)}")

all_rows = [v[2] for vals in rows.values() for v in vals]
print()
print(f"max row (state + branch) across the sample: {worst_row}")
print(f"median row: {int(statistics.median(all_rows))}   p95: {sorted(all_rows)[int(0.95*(len(all_rows)-1))]}")
print()
print("kev's pack budget at max_state=N is N + 1664, so the smallest N that fits this data is")
print(f"    N >= {worst_row} - 1664 = {worst_row - 1664}   (rounded up to 64: {-(-(worst_row - 1664) // 64) * 64})")
print("    ceiling MAX_TRAIN_STATE = 7552")
print()
for family, (s, b, r, text) in sorted(example.items(), key=lambda kv: -kv[1][2]):
    print(f"--- {family}: worst row {r} (state {s} + branch {b})")
    print(text[:400].replace("\n", " "))
    print()
