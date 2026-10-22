#!/usr/bin/env python3
"""One engine's per-ask dumps in, one timing record out — the speed/cost pass's measurement step.

Why this exists. Every engine in the ladder already writes, per ask, the answer it produced: the Jev
probe writes ``raw/<dataset>__<cell>__<family>__<ask>.json`` with the request, the response body and
the prediction row derived from it, and the Kev probe writes the same shape under ``runs/kev-probe-*/raw``.
Three of the fields those files carry are timing fields — ``prediction.latency_ms``,
``prediction.input_tokens`` and (Jev only) ``prediction.output_tokens`` — and nothing in ``reports/``
has ever read them, because the *row* dumps the rest of the repo projects them into drop exactly those
fields. This script reads the raw dumps instead of the row dumps, and it is the only thing that does.

Units, stated once because the whole table depends on them:

* an **ask** is one serving call: one record's questions answered in one request (Jev over HTTP) or
  one in-process forward (Kev, Laya). Latency and tokens are measured per ask.
* a **question** (row) is one scored answer. A choice record asks one question; a ``marker_noul``
  record asks one yes/no question per marker (~26 of them), which is why the 200-cell draw is 800
  asks and 3,966 rows. Per-question latency and tokens are the ask's divided by the rows that ask
  produced: derived, and written here as ``*_per_row`` so a reader can see which is which.
* ``latency_ms`` for a Kev/Laya arm is timed **inside** the engine's own predictor, around the forward
  pass (``kev.predictors.LocalPredictor.__call__`` starts its clock after the state has been encoded),
  so it is compute, not plumbing. ``latency_ms`` for Jev is the full round trip of the HTTP call.

The collector never invents a number. An engine whose dump is absent or empty is reported with
``measured: null`` and a ``caveats`` entry naming what is missing; a field the engine's own dump does
not carry (Kev records no output tokens, because its readout is a pointer over option boundary tokens
rather than decoding) is reported as ``0`` with the reason in ``caveats``, never as a plausible value.

    python3 scripts/speed-cost/collect_timing.py kev --raw runs/speed-cost-0.8b/raw \\
        --engine kev-0.8b --engine-id jaredpalmer/kev-0.8b --gpus 1 \\
        --rows runs/speed-cost-0.8b/rows.jsonl --ask-log runs/speed-cost-0.8b/ask-log.json \\
        --groups-from reports/jev-smoke/sample.json --out reports/speed-cost/kev-0.8b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

#: The ask kinds a raw file name can end in. ``base`` is the ask that answers the record's questions;
#: ``abstain`` is the second ask the axis-1 convention adds for a choice question that lacks an
#: abstain option. Both are serving cost and both are counted; they are also counted separately,
#: because only the base ask maps one-to-one onto a question.
ASKS = ("base", "abstain")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile: the smallest value at or above ``fraction`` of the sorted sample.

    Stated rather than assumed because the two conventions differ on a 800-sample p95 (index 759 vs
    760), and a table whose p95 moves when the collector is rewritten is not a measurement. With no
    interpolation there is also nothing to explain about a latency that was never served.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "p90": None, "p95": None, "p99": None, "mean": None,
                "min": None, "max": None, "total": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": ordered[len(ordered) // 2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "mean": sum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "total": sum(ordered),
    }


def ask_kind(name: str) -> str:
    """``muraro__D28.1_43__panel_choice__base.json`` -> ``base``."""
    stem = name[:-5] if name.endswith(".json") else name
    return stem.rsplit("__", 1)[-1]


def slug_like(text: str) -> str:
    """``sckevtype.jev.probe.slug``'s rule, kept here only as a fallback for a box without the package.

    The import below is the authority; this copy exists so the collector can still restrict a census
    dump on a box where ``sckevtype`` is not importable, and the record says which of the two ran.
    ``tabula_muris_senis/3-M-5/6`` really does have two slashes in it, which is why the name is slugged
    rather than split naively.
    """
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def groups_from_sample(path: Path | None) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """The smoke draw's ask-file prefixes, so a census dump can be restricted to the same 200 cells.

    The 200-cell draw is a subset of the census (``reports/jev-full/sample.json`` holds every eligible
    cell), so a census dump carries the smoke draw's asks *plus* 9,240 others. The restriction cannot
    be done by parsing the file name back into a cell id — the name is ``slug(dataset)__slug(cell)__…``
    and the slug is not invertible (``tabula_muris_senis/3-M-5/6`` slugged its cell id into one field) —
    so the prefixes are built *forwards* with the probe's own ``slug``/``dataset_of``, and a file is in
    scope when its name starts with one of them. The returned mapping is prefix -> group id, so every
    kept file can name the cell it belongs to.
    """
    if path is None:
        return None, {}
    try:                                                   # the authority, when the repo is importable
        from sckevtype.jev.probe import dataset_of, slug

        origin = "sckevtype.jev.probe.slug"
    except Exception:                                      # noqa: BLE001 — a box without the package
        def dataset_of(group_id: str) -> str:
            return group_id.split("/", 1)[0]

        slug, origin = slug_like, "a local copy of the slug rule (sckevtype was not importable)"
    sample = read_json(path)
    prefixes = {}
    for group in sample["records"]:
        group = str(group)
        prefixes[f"{slug(dataset_of(group))}__{slug(group.split('/', 1)[1])}__"] = group
    return prefixes, {"sample": str(path), "cells": sample.get("cells"), "records": sample.get("records"),
                      "questions": sample.get("questions"), "seed": sample.get("seed"),
                      "sha256": sha256_of(path), "ask_name_rule": origin, "prefixes": len(prefixes)}


def col_stats(values: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [float(v[key]) for v in values if v.get(key) is not None]


def collect_asks(raw: Path, prefixes: dict[str, str] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every ask file under ``raw``, as ``{ask, latency_ms, in_tokens, out_tokens}``.

    A file that carries no ``prediction`` block, or a prediction with no ``latency_ms``, is counted
    as unmeasured rather than skipped silently: the count is what separates "this engine's dump has
    no timing in it" from "this engine was fast".
    """
    files = sorted(path for path in raw.glob("*.json"))
    rows, unmeasured, kept = [], 0, 0
    for path in files:
        group = None
        if prefixes is not None:
            fields = path.name.split("__")
            group = prefixes.get("__".join(fields[:2]) + "__") if len(fields) >= 3 else None
            if group is None:
                continue
        kept += 1
        try:
            saved = read_json(path)
        except (ValueError, OSError):
            unmeasured += 1
            continue
        prediction = saved.get("prediction") or {}
        if prediction.get("latency_ms") is None:
            unmeasured += 1
            continue
        rows.append({"ask": ask_kind(path.name), "cell": group or path.name,
                     "latency_ms": float(prediction["latency_ms"]),
                     "in_tokens": int(prediction.get("input_tokens") or 0),
                     "out_tokens": int(prediction.get("output_tokens") or 0),
                     "served_model": prediction.get("served_model")})
    provenance = {"raw": str(raw), "files_on_disk": len(files), "files_in_scope": kept,
                  "files_with_timing": len(rows), "files_without_timing": unmeasured,
                  "restricted_to_groups": prefixes is not None}
    return rows, provenance


def rows_breakdown(rows_path: Path | None) -> dict[str, Any]:
    """The rows an engine's ask pass produced, by question type — the denominator of ``per_row``."""
    if rows_path is None or not rows_path.exists():
        return {"rows": None, "choice": None, "noul": None, "path": None if rows_path is None else str(rows_path)}
    by_type: dict[str, int] = {}
    total = 0
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        qtype = json.loads(line).get("type") or "unknown"
        by_type[qtype] = by_type.get(qtype, 0) + 1
    return {"rows": total, "choice": by_type.get("choice"), "noul": by_type.get("noul"),
            "by_type": by_type, "path": str(rows_path)}


def timing_block(rows: list[dict[str, Any]], gpus: int, ask_wall_s: float | None) -> dict[str, Any]:
    """The measured block: per-ask latency and tokens, plus everything derived from them."""
    latencies = col_stats(rows, "latency_ms")
    ins = col_stats(rows, "in_tokens")
    outs = col_stats(rows, "out_tokens")
    block: dict[str, Any] = {
        "asks": len(rows),
        "asks_base": sum(row["ask"] == "base" for row in rows),
        "asks_abstain": sum(row["ask"] == "abstain" for row in rows),
        "latency_ms_per_ask": stats(latencies),
        "tokens_in_per_ask": stats(ins),
        "tokens_out_per_ask": stats(outs),
        "gpus": gpus,
    }
    # How much of the tail is warmup. The first asks of an in-process pass pay the kernel-compile and
    # allocator-pool cost, and that lands in the p95 rather than in the median, so the count is
    # reported beside the percentiles instead of being left for a reader to wonder about.
    if latencies:
        block["latency_ms_per_ask"]["share_over_1s"] = sum(value > 1000.0 for value in latencies) / len(latencies)
        block["latency_ms_per_ask"]["count_over_1s"] = sum(value > 1000.0 for value in latencies)
    if ask_wall_s:
        block["ask_wall_s"] = ask_wall_s
        block["asks_per_s"] = len(rows) / ask_wall_s
        block["gpu_seconds_per_ask_reserved"] = gpus * ask_wall_s / len(rows) if rows else None
        block["gpu_seconds_per_ask_compute"] = gpus * sum(latencies) / 1000.0 / len(rows) if rows else None
    return block


def derive_per_row(block: dict[str, Any], rows_block: dict[str, Any]) -> None:
    """``*_per_row`` from ``*_per_ask`` and the row count, guarded on the row count being known.

    ``rows_per_ask`` above 1 is the whole reason the per-question column exists: a marker_noul ask
    answers ~26 questions in one forward pass, so its per-ask cost is a per-26-questions cost. The
    division is the only arithmetic here, and it is recorded as a derived field.
    """
    rows = rows_block.get("rows")
    asks = block.get("asks") or 0
    if not rows or not asks:
        block["rows"] = rows
        return
    per_ask = rows / asks
    block["rows"] = rows
    block["rows_per_ask"] = per_ask
    for name in ("latency_ms", "tokens_in", "tokens_out"):
        source = block[f"{name}_per_ask"]
        block[f"{name}_per_row"] = {
            "median": None if source["median"] is None else source["median"] / per_ask,
            "p90": None if source.get("p90") is None else source["p90"] / per_ask,
            "p95": None if source["p95"] is None else source["p95"] / per_ask,
            "p99": None if source.get("p99") is None else source["p99"] / per_ask,
            "mean": None if source["mean"] is None else source["mean"] / per_ask,
            "total": None if source["total"] is None else source["total"] / per_ask,
            "per_row_note": f"the ask's own number divided by {per_ask:.2f} rows per ask",
        }
    if block.get("ask_wall_s"):
        block["rows_per_s"] = rows / block["ask_wall_s"]
        block["gpu_seconds_per_row_reserved"] = block["gpu_seconds_per_ask_reserved"] / per_ask
        block["gpu_seconds_per_row_compute"] = block["gpu_seconds_per_ask_compute"] / per_ask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collect_timing.py", description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("kev", "jev", "laya"))
    parser.add_argument("--raw", help="directory of per-ask JSON dumps (kev, jev)")
    parser.add_argument("--rows", default=None, help="the ask pass's rows.jsonl (kev, jev)")
    parser.add_argument("--ask-log", default=None, help="the ask pass's ask-log.json (kev)")
    parser.add_argument("--wall-s", type=float, default=None, help="the answering pass's own wall clock, seconds")
    parser.add_argument("--engine", required=True, help="the name this row carries in the table")
    parser.add_argument("--engine-id", default=None, help="what the engine is: a Hub id, a run dir, a served model id")
    parser.add_argument("--gpus", type=int, default=0, help="GPUs the engine's own pass held (0 = no local GPU, a hosted API)")
    parser.add_argument("--device-name", default=None, help="the card, as the run reported it")
    parser.add_argument("--node", default=None, help="the host the pass ran on")
    parser.add_argument("--job-id", default=None, help="the scheduler job, when there was one")
    parser.add_argument("--groups-from", default=None, help="a sample.json whose cell ids restrict the dump")
    parser.add_argument("--caveats", action="append", default=[], help="a stated limitation, repeated per line")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    groups, population = groups_from_sample(Path(args.groups_from)) if args.groups_from else (None, {})
    caveats = list(args.caveats)
    record: dict[str, Any] = {"engine": args.engine, "kind": args.kind, "engine_id": args.engine_id,
                              "gpus": args.gpus, "device": args.device_name, "node": args.node,
                              "job_id": args.job_id, "population": population, "caveats": caveats,
                              "sources": {}}

    if args.kind in ("kev", "jev"):
        if not args.raw:
            print("--raw is required for a kev or jev engine", file=sys.stderr)
            return 2
        rows, provenance = collect_asks(Path(args.raw), groups)
        record["sources"]["asks"] = provenance
        rows_block = rows_breakdown(Path(args.rows) if args.rows else None)
        record["sources"]["rows"] = rows_block
        ask_wall = args.wall_s
        if args.ask_log and Path(args.ask_log).exists():
            # Read the ask-log whether or not a wall clock was passed on the command line: it carries
            # the run's own placement, dtype, temperature and context, which is the only record that
            # the sharding happened and which card the numbers came off.
            log = read_json(Path(args.ask_log))
            record["sources"]["ask_log"] = {
                "path": args.ask_log, "elapsed_s": log.get("elapsed_s"), "phase": log.get("phase"),
                "checkpoint_kind": log.get("checkpoint_kind"), "placement": log.get("placement"),
                "temperature": log.get("temperature"), "context": log.get("context"),
                "gpus": log.get("gpus"), "dtype": log.get("dtype"), "run": log.get("run"),
                "model": log.get("model"),
            }
            if ask_wall is None:
                ask_wall = log.get("elapsed_s")
            if log.get("rows"):
                rows_block.setdefault("rows_from_ask_log", log["rows"])
        if not rows:
            record["measured"] = None
            caveats.append(f"{args.raw} carries no ask with a latency_ms field; nothing was measured")
        else:
            block = timing_block(rows, args.gpus, ask_wall)
            derive_per_row(block, rows_block)
            record["measured"] = block
            if args.kind == "kev" and not any(v["out_tokens"] for v in rows):
                caveats.append("input tokens are the encoder's own packed count (state + option branches); "
                               "output tokens are 0 by construction, not by measurement failure: Kev answers with a "
                               "pointer readout over option boundary tokens and never decodes text")
    else:
        timing = Path(args.raw) if args.raw else None
        if timing is None or not timing.exists():
            record["measured"] = None
            caveats.append("no laya timing file was produced; the Laya pass did not run or wrote nothing")
        else:
            payload = read_json(timing)
            record["sources"]["laya_timing"] = {"path": str(timing), "sha256": sha256_of(timing)}
            # Laya's unit is the item, and one item is one question: `asks_for` renders one item per
            # question (so a marker_noul record is ~26 items, where a Kev ask answers all 26 in one
            # forward). The headline row is therefore the **one-item-per-forward** pass — the only
            # latency in this engine's payload that is one question's own — and the batched pass is
            # carried whole in `batched` because it is the serving default its own CLI uses (batch 4)
            # and it is where the throughput number comes from. Mixing them silently would let a
            # 4-question forward be read as a per-question latency, which is the whole reason both
            # passes were measured.
            per_item = dict(payload.get("per_item_measured") or {})
            # The per-item pass is what the headline row is built from, and the harness records only
            # what that pass could know: its latencies and its token counts. The two fields it cannot
            # know are filled from measured inputs rather than guessed — output tokens are 0 by
            # construction (nothing is decoded), and the GPU-second and throughput figures are the
            # pass's own wall clock and latency sum over its own ask count.
            asks = per_item.get("asks") or payload.get("base_items") or 0
            per_item.setdefault("tokens_out_per_ask", stats([0.0] * asks))
            latency_total = (per_item.get("latency_ms_per_ask") or {}).get("total")
            if per_item.get("gpu_seconds_per_ask_compute") is None and latency_total and asks:
                per_item["gpu_seconds_per_ask_compute"] = args.gpus * latency_total / 1000.0 / asks
            if per_item.get("gpu_seconds_per_ask_reserved") is None and per_item.get("ask_wall_s") and asks:
                per_item["gpu_seconds_per_ask_reserved"] = args.gpus * per_item["ask_wall_s"] / asks
            if per_item.get("asks_per_s") is None and per_item.get("ask_wall_s") and asks:
                per_item["asks_per_s"] = asks / per_item["ask_wall_s"]
            block = per_item
            block["unit"] = "one forward per item (one item = one question)"
            block["asks"] = per_item.get("asks") or payload.get("base_items")
            block["asks_base"] = block["asks"]
            block["asks_abstain"] = payload.get("abstain_items") or 0
            block["gpus"] = args.gpus
            if payload.get("measured"):
                batched = payload["measured"]
                block["batched"] = batched
                block["batched_note"] = (f"the serving default, {batched.get('unit')}: "
                                         f"{batched.get('asks')} items in {round(batched.get('ask_wall_s') or 0, 1)}s, "
                                         f"{round(batched.get('asks_per_s') or 0, 2)} items/s, median batch latency "
                                         f"{round(batched.get('latency_ms_per_ask', {}).get('median') or 0, 1)} ms")
                block["asks_per_s_batched"] = batched.get("asks_per_s")
                block["gpu_seconds_per_ask_reserved_batched"] = batched.get("gpu_seconds_per_ask_reserved")
            derive_per_row(block, {"rows": payload.get("rows")})
            record["measured"] = block
            record["rows_detail"] = payload.get("rows_detail")
            caveats.extend(payload.get("caveats", []))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    measured = record["measured"]
    summary = "not measured" if not measured else (
        f"{measured['asks']} asks, median {measured['latency_ms_per_ask']['median']:.1f} ms, "
        f"p95 {measured['latency_ms_per_ask']['p95']:.1f} ms, "
        f"{measured['tokens_in_per_ask']['mean']:.0f} in-tokens/ask")
    print(f"{args.engine}: {summary} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
