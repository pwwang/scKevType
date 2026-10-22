#!/usr/bin/env python3
"""Prove an ask set is the decision models' own questions, and that two ask sets are the same questions.

Two questions this answers, both of which the LLM baseline's comparability rests on:

**Is this ask set the population it claims to be?** ``--reference`` joins it, key for key, against the row
dump the decision models were scored on. Every ask must find its row and every row must find its ask
(reported both ways), and for a ``choice`` question the option list must match the row's ``keys`` **in
order** -- option order is part of the corpus bytes and this class of model is order-sensitive, so a
re-ordered ballot is a different question wearing the same name.

**Are two ask sets the same questions?** ``--compare`` reduces both to a digest over the canonical
content of every ask (state, instructions, criteria, key, sorted by key) and prints it. Two trees built
from different raw dumps -- Jev's and Kev's -- must reduce to the same digest, because that equality is
exactly what the probe's own ``plan`` step asserts when it says ``0 mismatch(es)``.

    python scripts/llm/verify-asks.py --asks reports/llm-asks-smoke.jsonl \\
        --reference reports/jev-smoke/rows.jsonl --compare reports/llm-asks-smoke.local.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def canonical(ask: dict) -> str:
    """The ask's content, without its provenance fields: what a model would actually be shown."""
    return json.dumps({"key": ask["key"], "type": ask["type"], "state": ask["state"],
                       "instructions": ask["instructions"], "criteria": ask["criteria"]},
                      sort_keys=True, ensure_ascii=False)


def digest(asks: list[dict]) -> str:
    h = hashlib.sha256()
    for ask in sorted(asks, key=lambda a: a["key"]):
        h.update(canonical(ask).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asks", required=True)
    ap.add_argument("--reference", default="", help="a row dump to join against, both ways")
    ap.add_argument("--compare", default="", help="a second ask set; the two digests must agree")
    args = ap.parse_args()

    asks = load_jsonl(Path(args.asks))
    keys = [(a["dataset"], a["cell"], a["family"], a["question"]) for a in asks]
    print(f"asks          : {len(asks)}")
    print(f"records       : {len({a['dataset'] + '/' + a['cell'] + '/' + a['family'] for a in asks})}")
    print(f"per family    : {dict(sorted(Counter(a['family'] for a in asks).items()))}")
    print(f"per dataset   : {dict(sorted(Counter(a['dataset'] for a in asks).items()))}")
    print(f"content digest: {digest(asks)}")
    rc = 0

    if args.reference:
        ref = {(r["dataset"], r["cell"], r["family"], r["question"]): r
               for r in load_jsonl(Path(args.reference))}
        missing = [k for k in keys if k not in ref]
        extra = [k for k in ref if k not in set(keys)]
        order_bad = []
        for ask in asks:
            row = ref.get((ask["dataset"], ask["cell"], ask["family"], ask["question"]))
            if row is None:
                continue
            ref_keys = [str(k) for k in row["keys"]]
            ask_keys = [str(k) for k in ask["criteria"]]
            if ask["type"] == "choice":
                if ask_keys != ref_keys:
                    order_bad.append((ask["key"], ask_keys, ref_keys))
            elif set(ask_keys) != set(ref_keys):
                order_bad.append((ask["key"], ask_keys, ref_keys))
        print(f"reference     : {args.reference}  ({len(ref)} rows)")
        print(f"  asks with no reference row : {len(missing)}")
        print(f"  reference rows with no ask : {len(extra)}")
        print(f"  option-list mismatches     : {len(order_bad)}")
        for bad in order_bad[:3]:
            print(f"    {bad}")
        if missing or extra or order_bad:
            rc = 1

    if args.compare:
        other = load_jsonl(Path(args.compare))
        a, b = digest(asks), digest(other)
        print(f"compare       : {args.compare}  ({len(other)} asks)")
        print(f"  digests equal: {a == b}")
        if a != b:
            sa = {x["key"]: canonical(x) for x in asks}
            sb = {x["key"]: canonical(x) for x in other}
            only_a = sorted(set(sa) - set(sb))[:5]
            only_b = sorted(set(sb) - set(sa))[:5]
            differ = sorted(k for k in set(sa) & set(sb) if sa[k] != sb[k])[:5]
            print(f"  keys only in --asks   : {only_a}")
            print(f"  keys only in --compare: {only_b}")
            print(f"  keys that differ      : {differ}")
            rc = 1

    print("VERIFY_ASKS " + ("OK" if rc == 0 else "MISMATCH"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
