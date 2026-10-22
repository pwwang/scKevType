#!/usr/bin/env python3
"""Measure branch (question) token lengths per family, with the pinned tokenizer.

The training row budget is state + one branch. States are ~1,570 tokens; if branches are also
thousands, the row limit — not the state limit — is what decides whether anything can train.

Usage: python scripts/measure-branch-tokens.py [suite-dir] [sample]
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
revision = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", revision=revision)

per_fam: dict[str, list[int]] = defaultdict(list)
example: dict[str, str] = {}
keys_seen: set[str] = set()

with (suite / "train.jsonl").open() as fh:
    for line in itertools.islice(fh, sample):
        rec = json.loads(line)
        state_len = len(tok(rec["state"], add_special_tokens=False)["input_ids"])
        for q in rec["questions"]:
            keys_seen |= set(q.keys())
            fam = q.get("family") or rec.get("_meta", {}).get("family") or "?"
            # the branch is whatever text the question carries; try the likely field names
            text = ""
            for field in ("text", "question", "prompt", "branch", "body"):
                if isinstance(q.get(field), str):
                    text = q[field]
                    break
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            per_fam[fam].append(n)
            example.setdefault(fam, text)

print(f"suite: {suite}  sample: {sample} records")
print(f"question keys seen: {sorted(keys_seen)}")
print()
print(f"{'family':22s} {'n':>6s} {'min':>7s} {'median':>8s} {'p95':>7s} {'max':>7s}")
for fam, lens in sorted(per_fam.items()):
    lens_sorted = sorted(lens)
    p95 = lens_sorted[int(0.95 * (len(lens_sorted) - 1))]
    print(f"{fam:22s} {len(lens):6d} {min(lens):7d} {int(statistics.median(lens)):8d} {p95:7d} {max(lens):7d}")

print()
print("state tokens for the same sample: median",
      int(statistics.median([len(tok(json.loads(l)["state"], add_special_tokens=False)["input_ids"])
                             for l in list(itertools.islice((suite / 'train.jsonl').open(), sample))])))

print()
for fam, text in sorted(example.items()):
    print(f"--- {fam}: first 300 chars of a branch")
    print(text[:300].replace("\n", " "))
    print()
