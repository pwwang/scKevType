#!/usr/bin/env python3
"""The Laya arm of the speed/cost pass: answer the 200-cell smoke draw with the 322M encoder and time it.

The other engines of the ladder are reached through one ask per record (Jev over HTTP, Kev in
process); Laya is reached through :func:`sckevtype.laya.score.load_checkpoint` +
:func:`~sckevtype.laya.score.answer`, which is what ``sckevtype.laya.score infer`` calls. This script
drives exactly those two functions — never a reimplementation — so the thing being timed is the thing
that produces Laya's rows, and the population is the same 200 cells: the records come from
``sckevtype.jev.probe.collect_records`` on the same ``sample.json`` the Jev and Kev passes read.

Two passes, both measured, because they answer different questions:

``per-item``   one item per forward. This is the per-**question** latency a reader multiplies by
               their own $/GPU-hour; it holds no batching.
``batched``    ``--batch`` items per forward, the serving default (``sckevtype.laya.score``'s own 4).
               This is the throughput number, and the one that makes Laya's per-question cost look
               the way it does in a real deployment.

Token counts are the render's own: an item's ``ids`` is a packed ``array('i')`` (four bytes per token,
:func:`sckevtype.laya.score._tensors` unpacks it), so ``len(ids) // 4`` is the token count of the
state+question text Laya actually reads. Laya never decodes, so its output tokens are 0 by
construction — the same statement Kev's row makes, and for the same reason.

    python3 scripts/speed-cost/laya-timing.py --checkpoint $W/runs/laya-pilot-ft \\
        --sample reports/jev-smoke/sample.json --corpus runs/real-v2 --batch 4 \\
        --out reports/speed-cost/laya/timing.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from sckevtype.jev.probe import collect_records
from sckevtype.laya.score import answer, asks_for, checkpoint_render_config, load_checkpoint


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def stats(values: Sequence[float]) -> dict[str, Any]:
    import math

    if not values:
        return {"n": 0, "median": None, "p95": None, "mean": None, "min": None, "max": None, "total": None}
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return {"n": len(ordered), "median": ordered[len(ordered) // 2], "p95": ordered[index],
            "mean": sum(ordered) / len(ordered), "min": ordered[0], "max": ordered[-1], "total": sum(ordered)}


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(prog="laya-timing.py", description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", default="reports/jev-smoke/sample.json")
    parser.add_argument("--corpus", default="runs/real-v2")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="records to score (0 = the whole sample)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    partition = Path(args.corpus) / f"{sample['split']}.jsonl"
    cfg = checkpoint_render_config(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(str(Path(args.checkpoint) / "tokenizer"))
    model, _tok, _cfg = load_checkpoint(args.checkpoint, device)
    print(f"laya {args.checkpoint} on {device}; render {cfg.as_dict()}; torch {torch.__version__}", flush=True)

    tasks = collect_records(partition, sample)
    if args.limit:
        tasks = tasks[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    dropped = 0
    started = time.time()
    for index, (group, family, record) in enumerate(tasks, start=1):
        # `render.build_record` reads `_meta["id"]`, and this corpus's records do not carry one (0 of
        # 4,000 in `runs/real-v2/test.jsonl`: the field is filled in by `kev.data.load_records` when a
        # suite is built, not by the corpus). The frozen suite's own ids are `"<split>/<source>/<line>"`,
        # which names no cell; the id this harness sets is the `"<split>/<dataset>/<cell>/<family>"`
        # form every dump in this repo keys a record by (the Jev and Kev rows both carry it). Nothing
        # about the timing depends on the value — it is carried into the item so a reader can say which
        # record each measured item answered — and `sckevtype.laya.score.rows_from_answers` is never
        # called here, so no row is scored against it.
        record = {**record, "_meta": {**record["_meta"], "id": f"{sample['split']}/{group}/{family}"}}
        made, lost = asks_for(record, tok, cfg, sample["split"])
        dropped += len(lost)
        for item in made:
            item["tokens"] = len(item["ids"]) // 4
            items.append(item)
        if index % 100 == 0 or index == len(tasks):
            print(f"  [{index:4d}/{len(tasks)}] items={len(items)} dropped={dropped} "
                  f"{index / max(time.time() - started, 1e-9):.2f} records/s", flush=True)
    render_s = time.time() - started
    base_items = [item for item in items if item["ask"] == "base"]
    print(f"rendered {len(items)} items ({len(base_items)} base, {len(items) - len(base_items)} abstain) "
          f"from {len(tasks)} records in {render_s:.1f}s", flush=True)

    # --- per-item: one forward per item, the per-question latency -------------------------------
    per_item_ms: list[float] = []
    timing: list[dict[str, Any]] = []
    t0 = time.time()
    for index, item in enumerate(base_items, start=1):
        sync(device)
        start = time.perf_counter()
        answer(model, [item], tok.pad_token_id, device)
        sync(device)
        elapsed = 1000.0 * (time.perf_counter() - start)
        per_item_ms.append(elapsed)
        timing.append({"id": item["id"], "question": item["question"], "family": item["family"],
                       "ask": item["ask"], "type": item["type"], "tokens_in": item["tokens"],
                       "latency_ms": elapsed})
        if index % 500 == 0 or index == len(base_items):
            print(f"  per-item [{index:4d}/{len(base_items)}] "
                  f"{index / max(time.time() - t0, 1e-9):.2f} items/s", flush=True)
    per_item_wall = time.time() - t0

    # --- batched: the serving default, for throughput --------------------------------------------
    batch = max(1, args.batch)
    batch_ms: list[float] = []
    batch_tokens: list[int] = []
    t0 = time.time()
    for start in range(0, len(base_items), batch):
        part = base_items[start: start + batch]
        sync(device)
        begin = time.perf_counter()
        answer(model, part, tok.pad_token_id, device)
        sync(device)
        batch_ms.append(1000.0 * (time.perf_counter() - begin))
        batch_tokens.append(sum(item["tokens"] for item in part))
    batch_wall = time.time() - t0
    print(f"  batched: {len(batch_ms)} forwards of <= {batch} items in {batch_wall:.1f}s", flush=True)

    def block(latencies: list[float], tokens: list[int], asks: int, wall: float, kind: str) -> dict[str, Any]:
        return {
            "asks": asks, "asks_base": asks, "asks_abstain": 0, "unit": kind,
            "latency_ms_per_ask": stats(latencies),
            "tokens_in_per_ask": stats([float(value) for value in tokens]),
            "tokens_out_per_ask": stats([0.0] * asks),
            "gpus": 1 if device.startswith("cuda") else 0,
            "ask_wall_s": wall,
            "asks_per_s": asks / wall if wall else None,
            "gpu_seconds_per_ask_reserved": (1.0 * wall / asks) if (asks and wall and device.startswith("cuda")) else None,
            "gpu_seconds_per_ask_compute": (sum(latencies) / 1000.0 / asks) if asks else None,
        }

    measured = block(batch_ms, batch_tokens, len(base_items), batch_wall, f"batched forward, {batch} items")
    payload = {
        "checkpoint": args.checkpoint, "device": device, "render_config": cfg.as_dict(), "batch": batch,
        "records": len(tasks), "items": len(items), "base_items": len(base_items),
        "abstain_items": len(items) - len(base_items), "dropped": dropped, "render_s": render_s,
        "rows": len(base_items), "rows_detail": {"base_items": len(base_items), "abstain_items": len(items) - len(base_items),
                                                 "dropped": dropped, "records": len(tasks)},
        "measured": measured,
        "per_item_measured": {"unit": "one forward per item", "ask_wall_s": per_item_wall,
                              "latency_ms_per_ask": stats(per_item_ms),
                              "tokens_in_per_ask": stats([float(item["tokens"]) for item in base_items])},
        "caveats": [
            "tokens_in is the rendered text Laya reads (len(ids)//4), which is a different and larger "
            "string than Kev's packed state+branches count: the two are not the same tokenizer's answer to "
            "the same question, they are two engines' own unit of input",
            "output tokens are 0 by construction: Laya's head is a classifier, nothing is decoded",
        ],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out.parent / "per-item.jsonl").open("w", encoding="utf-8") as handle:
        for line in timing:
            handle.write(json.dumps(line) + "\n")
    print(f"wrote {out} and {out.parent / 'per-item.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
