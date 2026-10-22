#!/usr/bin/env python3
"""What one optimizer step of real-v2 actually feeds the backbone, per record and per microbatch.

`docs/compute.md` records the probe's unit as "the token the backbone actually sees" and
prices the peak at MB per **row token** — `sum(len(state) + len(row["ids"]))` over a record's
questions (`sckevtype.train.train.forward_tokens`). That number is what the 0.32 MB/row-token
slope is quoted against, and it is not what the kernel pays for.

Two other counts exist, and neither appears in the probe's table:

* **padded slots** — `kev.model.DecisionModel._rows_hidden` puts *every* question of a record
  into **one** right-padded batch of length `max(len(state) + len(branch))`
  (`_pad_rows`). The kernel therefore runs `n_questions x L_max` token-slots, not
  `sum(state + branch)`, and the padding is real work.
* **row length** `L_max` — the quantity the full-attention layers are quadratic in. The probe
  was measured on the smoke suite, whose rows are ~160 tokens; real-v2's longest row is 4,845.

This prints both next to the probe's count, for the split, for the worst record, and for the
microbatches the trainer actually draws under its own seed — so the scope of the slope is on
the page instead of in a footnote.

It needs the tokenizer only: no model, no GPU, no torch device.

    python scripts/measure-row-form.py [suite] [split] [seed] [batch] [accum]
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from kev.data import materialize
    from kev.model import (
        SERVE_MAX_BRANCH,
        encode,
        load_tokenizer,
        rows_of,
        training_context,
    )
    from kev.suite import read_manifest
    from kev.train import load_split, training_requests

    from sckevtype.train import train as T

    suite = sys.argv[1] if len(sys.argv) > 1 else "artifacts/suites/real-v2"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    accum = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    manifest = read_manifest(suite)
    context = T.suite_budget(suite)
    budget = T.recorded_max_state(context)
    pin = context["tokenizer"]
    tokenizer = load_tokenizer(pin["id"], revision=pin["revision"])
    limits = training_context(budget)
    print(f"suite {suite}  split {split}  max_state {budget}  "
          f"row limit {limits['max_branch']}  packed limit {limits['max_packed']}")

    def shapes(record: dict) -> tuple[int, int, int, int]:
        """(state, questions, row_tokens, padded_slots) for one record, through kev's own encoder."""
        enc = encode(tokenizer, materialize(record), strict=True,
                     max_state=limits["max_state"], max_branch=limits["max_branch"])
        state, _, rows = rows_of(enc)
        lengths = [len(state) + len(row["ids"]) for row in rows]
        return len(state), len(rows), sum(lengths), len(rows) * max(lengths)

    records = load_split(suite, split)
    measured = [shapes(record) for record in records]
    states = [m[0] for m in measured]
    questions = [m[1] for m in measured]
    row_tokens = [m[2] for m in measured]
    slots = [m[3] for m in measured]

    def dist(name: str, values: list[int]) -> None:
        ordered = sorted(values)
        print(f"  {name:16s} min {ordered[0]:7,d}  median {int(statistics.median(ordered)):7,d}  "
              f"p95 {ordered[int(0.95 * (len(ordered) - 1))]:7,d}  max {ordered[-1]:7,d}")

    print(f"\n{len(records)} records in {split}:")
    dist("state", states)
    dist("questions/record", questions)
    dist("row tokens", row_tokens)
    dist("padded slots", slots)

    total_rows = sum(row_tokens)
    total_slots = sum(slots)
    print("\nthe whole split, as the trainer's own counter sees it:")
    print(f"  row tokens   {total_rows:12,d}   (forward_tokens — the probe's unit, and its token/s denominator)")
    print(f"  padded slots {total_slots:12,d}   ({total_slots / total_rows:.2f}x row tokens — what the kernel runs)")
    packed = sum(len(encode(tokenizer, materialize(record), strict=True, max_state=limits["max_state"],
                            max_branch=SERVE_MAX_BRANCH)["ids"]) for record in records)
    print(f"  packed       {packed:12,d}   (state + every branch once — the form kev does *not* use here)")

    # The trainer's own draw: random.Random(seed).shuffle(requests), then requests[i*b:(i+1)*b].
    args = T.parse_args(["--suite", suite, "--base", manifest["context"]["tokenizer"]["id"],
                         "--base_revision", manifest["context"]["tokenizer"]["revision"],
                         "--steps", "1", "--batch", str(batch), "--accum", str(accum),
                         "--dtype", "bf16", "--device", "cuda", "--out", "runs/row-form-probe"])
    namespace = T.kev_namespace(args, "runs/row-form-probe")
    requests = training_requests(namespace, tokenizer, manifest, manifest["holdout_sources"])
    rng = random.Random(seed)
    rng.shuffle(requests)
    print(f"\nfirst {accum} microbatches the trainer draws (seed {seed}, batch {batch}) — "
          f"the config that OOM'd on the A100:")
    print(f"  {'microbatch':>10s} {'records':>7s} {'rows':>5s} {'row tokens':>11s} {'padded slots':>13s} {'L_max':>7s}")
    step_rows, step_slots = 0, 0
    for microbatch in range(accum):
        chunk = requests[microbatch * batch:(microbatch + 1) * batch]
        shaped = [shapes(record) for record in chunk]
        rows = sum(m[1] for m in shaped)
        tokens = sum(m[2] for m in shaped)
        slots_ = sum(m[3] for m in shaped)
        l_max = max(m[3] // m[1] for m in shaped)
        step_rows += tokens
        step_slots += slots_
        print(f"  {microbatch:10d} {len(chunk):7d} {rows:5d} {tokens:11,d} {slots_:13,d} {l_max:7d}")
    print(f"  {'per step':>10s} {'':7s} {'':5s} {step_rows:11,d} {step_slots:13,d}")
    print("\n  a microbatch's peak is one microbatch, not the step: gradients accumulate across the")
    print("  eight, but each backward frees its own graph, so the high-water mark is the widest one.")

    worst = max(range(len(records)), key=lambda i: measured[i][3])
    print(f"\nwidest record in the split: {records[worst]['_meta']['id']}  "
          f"state {measured[worst][0]:,}  questions {measured[worst][1]}  "
          f"row tokens {measured[worst][2]:,}  padded slots {measured[worst][3]:,}  "
          f"L_max {measured[worst][3] // measured[worst][1]:,}")
    print(f"  predicted at the probe's 0.32 MB/row-token: "
          f"{measured[worst][2] * 0.32 / 1024:.2f} GB — the A100 needed more than 39.5 GB for one of these.")
    print(json.dumps({"n": len(records), "row_tokens": total_rows, "padded_slots": total_slots,
                      "max_row_tokens": max(row_tokens), "max_padded_slots": max(slots),
                      "max_l": max(m[3] // m[1] for m in measured)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
