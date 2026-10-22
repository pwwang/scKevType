#!/usr/bin/env python3
"""Fold an LLM-baseline pass into one record shaped exactly like `reports/speed-cost/<engine>.json`.

The point of the shape is that the row can join the speed/cost table without anyone re-typing a number:
`scripts/speed-cost/table.py` reads a directory of these records and prints median/p95 latency, tokens
per ask and per row, asks/s and GPU-seconds. This writes the same keys from the same measured inputs --
the per-ask dump the pass wrote, and the shard run files that carry the pass's own wall clock.

Two token counts are reported per ask, and the difference between them is the finding, not a defect:

``tokens_in_per_ask.processed``
    what the model actually read: the state repeated for every question of the record, summed over the
    record's questions. This is what a serving bill would charge.
``tokens_in_per_ask.state_once``
    state once plus every question's own text, counted once each -- the analogue of a Kev ask's
    ``input_tokens``, which is the encoder's packed count (state + every option branch, encoded once).

Everything here is either measured (a value some process recorded) or arithmetic on measured values,
labelled as such. Nothing is estimated from a spec sheet.

    python scripts/llm/make_row.py --runs reports/llm-baseline/smoke-varA \\
        --out reports/llm-baseline-speedcost-row.json --gpus 4 \\
        --population reports/llm-asks-smoke.manifest.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def stats(values: list[float], seconds: bool = False) -> dict:
    if not values:
        return {"n": 0}
    out = {"n": len(values), "median": statistics.median(values),
           "p90": quantile(values, 0.90), "p95": quantile(values, 0.95),
           "p99": quantile(values, 0.99), "mean": statistics.fmean(values),
           "min": min(values), "max": max(values), "total": sum(values)}
    if seconds:
        # Only a latency has a "over a second" tail; on a token count the column would be a tautology.
        out["count_over_1s"] = sum(1 for v in values if v > 1000)
        out["share_over_1s"] = sum(1 for v in values if v > 1000) / len(values)
    return out


def divided(block: dict, rows_per_ask: float, note: str) -> dict:
    out = {"per_row_note": note}
    for key in ("median", "p90", "p95", "p99", "mean", "total"):
        if key in block:
            out[key] = block[key] / rows_per_ask
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--engine", default="llm-baseline")
    ap.add_argument("--engine-id", default="")
    ap.add_argument("--population", default="", help="the ask set's manifest, for the population block")
    ap.add_argument("--caveats", action="append", default=[])
    args = ap.parse_args()

    runs = Path(args.runs)
    asks = []
    with (runs / "ask-log.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                asks.append(json.loads(line))
    if not asks:
        raise SystemExit(f"{runs}/ask-log.jsonl: no asks -- refusing to write a record for a pass that "
                         f"answered nothing")

    shard_files = sorted(runs.glob("run-shard*.json"))
    if not shard_files:
        raise SystemExit(f"{runs}: no run-shard*.json -- the pass's own wall clock is not on disk, and "
                         f"it will not be guessed from the asks' latencies")
    shards: dict[int, dict] = {}
    for path in shard_files:
        history = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(history, dict):
            history = [history]
        shard = history[0]["shard"]
        shards[shard] = {
            "invocations": len(history),
            "ask_wall_s": sum(h["ask_wall_s"] for h in history),
            "load_s": sum(h["load_s"] for h in history),
            "asks": sum(h["asks"] for h in history),
            "node": history[-1].get("node"),
            "job_ids": [h.get("slurm_job_id") for h in history],
            "variant": history[-1]["variant"], "thinking": history[-1]["thinking"],
            "model": history[-1]["model"], "revision": history[-1]["revision"],
            "max_new_tokens": history[-1]["max_new_tokens"],
        }

    n_asks = len(asks)
    rows = sum(a["n_questions"] for a in asks)
    latency = [a["latency_ms"] for a in asks]
    tok_in = [a["input_tokens_processed"] for a in asks]
    tok_state = [a["input_tokens_state_once"] for a in asks]
    tok_out = [a["output_tokens"] for a in asks]
    rows_per_ask = rows / n_asks
    # GPU-seconds: the pass's ask clock per shard is that shard's own card, so the reserved total is the
    # sum over shards (each held exactly one card) and the pass wall clock is the largest shard's.
    gpu_seconds = sum(s["ask_wall_s"] for s in shards.values())
    pass_wall_s = max(s["ask_wall_s"] for s in shards.values())

    per_family = {}
    for family in sorted({a["family"] for a in asks}):
        group = [a for a in asks if a["family"] == family]
        g_rows = sum(a["n_questions"] for a in group)
        per_family[family] = {
            "asks": len(group), "questions": g_rows,
            "latency_ms_per_ask": stats([a["latency_ms"] for a in group], seconds=True),
            "tokens_in_per_ask_processed": stats([a["input_tokens_processed"] for a in group]),
            "tokens_in_per_ask_state_once": stats([a["input_tokens_state_once"] for a in group]),
            "tokens_out_per_ask": stats([a["output_tokens"] for a in group]),
            "latency_ms_per_question": stats([a["latency_ms"] / a["n_questions"] for a in group]),
            "gpu_seconds_per_question_compute": (
                sum(a["latency_ms"] for a in group) / 1000.0 / g_rows) if g_rows else None,
        }

    population = None
    if args.population:
        manifest = json.loads(Path(args.population).read_text(encoding="utf-8"))
        population = {"asks_manifest": args.population, "raw_dir": manifest.get("raw_dir"),
                      "label": manifest.get("label"), "cells": manifest.get("cells_written"),
                      "questions": manifest.get("questions"), "per_family": manifest.get("per_family"),
                      "per_dataset": manifest.get("per_dataset")}

    record = {
        "engine": args.engine,
        "kind": "llm",
        "engine_id": args.engine_id or next(iter(shards.values()))["model"],
        "gpus": args.gpus,
        "device": "NVIDIA A100-SXM4-40GB",
        "node": ", ".join(sorted({str(s["node"]) for s in shards.values()})),
        "job_id": ", ".join(str(j) for s in shards.values() for j in s["job_ids"]),
        "caveats": args.caveats,
        "sources": {
            "asks": {"raw": str(runs / "raw"), "files_on_disk": len(list((runs / "raw").glob("*.json"))),
                     "files_with_timing": n_asks},
            "ask_log": {"path": str(runs / "ask-log.jsonl"), "asks": n_asks},
            "shards": {str(k): v for k, v in sorted(shards.items())},
        },
        "measured": {
            "asks": n_asks,
            "questions": rows,
            "rows_per_ask": rows_per_ask,
            "questions_per_ask": {"median": statistics.median([a["n_questions"] for a in asks]),
                                  "mean": rows_per_ask,
                                  "note": "the median ask here is a one-question `choice` record; the "
                                          "mean is the rows-per-ask the per-question columns divide by"},
            "latency_ms_per_ask": stats(latency, seconds=True),
            "tokens_in_per_ask": dict(stats(tok_in), state_once=stats(tok_state)),
            "tokens_out_per_ask": stats(tok_out),
            "gpus": args.gpus,
            "ask_wall_s": pass_wall_s,
            "ask_wall_s_all_shards": gpu_seconds,
            "asks_per_s": n_asks / pass_wall_s if pass_wall_s else None,
            "gpu_seconds_per_ask_reserved": gpu_seconds / n_asks,
            "gpu_seconds_per_ask_compute": sum(latency) / 1000.0 / n_asks,
            "latency_ms_per_row": divided(stats(latency), rows_per_ask,
                                          f"the ask's own number divided by {rows_per_ask:.4f} rows per ask"),
            "tokens_in_per_row": divided(stats(tok_in), rows_per_ask,
                                         f"the ask's own number divided by {rows_per_ask:.4f} rows per ask"),
            "tokens_in_per_row_state_once": divided(stats(tok_state), rows_per_ask,
                                                    f"state once + each question's own text, divided by "
                                                    f"{rows_per_ask:.4f} rows per ask"),
            "tokens_out_per_row": divided(stats(tok_out), rows_per_ask,
                                          f"the ask's own number divided by {rows_per_ask:.4f} rows per ask"),
            "rows_per_s": rows / pass_wall_s if pass_wall_s else None,
            "gpu_seconds_per_row_reserved": gpu_seconds / rows,
            "gpu_seconds_per_row_compute": sum(latency) / 1000.0 / rows,
            "per_family": per_family,
        },
        "population": population,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    m = record["measured"]
    print(f"engine {record['engine']}  asks {n_asks}  questions {rows}  rows/ask {rows_per_ask:.3f}")
    print(f"  latency/ask median {m['latency_ms_per_ask']['median']:.1f} ms  "
          f"p95 {m['latency_ms_per_ask']['p95']:.1f} ms")
    print(f"  tokens in/ask (processed) median {m['tokens_in_per_ask']['median']:.0f}  "
          f"(state once) median {m['tokens_in_per_ask']['state_once']['median']:.0f}  "
          f"out/ask median {m['tokens_out_per_ask']['median']:.0f}")
    print(f"  asks/s {m['asks_per_s']:.3f}  GPU-s/ask reserved {m['gpu_seconds_per_ask_reserved']:.4f}  "
          f"compute {m['gpu_seconds_per_ask_compute']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
