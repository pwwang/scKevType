#!/usr/bin/env python3
"""Analysis for the 27B feasibility walk: one table, plus the peak-vs-padded-tokens line.

Reads only the runs' own records: `run.log.jsonl` (per optimizer step) and `provenance.json`.
Prints markdown. Nothing here is re-measured; every number is a field the trainer wrote.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

W = Path("/path/to/sckevtype")
GiB = 2 ** 30


def rows_of(run: Path) -> list[dict]:
    log = run / "run.log.jsonl"
    if not log.is_file():
        return []
    out = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def new_peak_points(rows: list[dict]) -> list[tuple[int, float]]:
    """(dearest pass, peak GiB) for the steps that RAISED the epoch peak.

    `peak_device_bytes` is the epoch's peak so far, so a step's own peak is only readable at a step that
    raised it — those are the points the line is fit through, and every other step's value is inherited.
    """
    points, seen = [], 0.0
    for row in rows:
        peak = row.get("peak_device_bytes", 0)
        if peak > seen:
            seen = peak
            points.append((int(row.get("slots_max", 0)), peak / GiB))
    return points


def fit(points: list[tuple[int, float]]) -> tuple[float, float, int] | None:
    """Least squares peak = intercept + slope x padded_tokens, over >= 2 distinct points."""
    pairs = sorted(set(points))
    if len(pairs) < 2:
        return None
    n = len(pairs)
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return intercept, slope, n


def budget_of(run: Path) -> int | None:
    """The token budget a run's directory name carries (`runs/smoke-27b-tb8192`), or None."""
    name = run.name
    tail = name.rsplit("tb", 1)[-1] if "tb" in name else ""
    return int(tail) if tail.isdigit() else None


def budget_summary(run: Path) -> dict:
    rows = rows_of(run)
    prov = json.loads((run / "provenance.json").read_text()) if (run / "provenance.json").is_file() else None
    warm = json.loads((run / "warm_start.json").read_text()) if (run / "warm_start.json").is_file() else None
    summary = {
        "run": run.name, "steps": len(rows), "rows": rows, "provenance": prov,
        "token_budget": budget_of(run),
        "dearest_pass": max((r.get("slots_max", 0) for r in rows), default=0),
        "max_passes": max((r.get("passes", 0) for r in rows), default=0),
        "max_window_tokens": max((r.get("window_tokens", 0) for r in rows), default=0),
        "peak_gib": max((r.get("peak_device_bytes", 0) for r in rows), default=0) / GiB,
        "per_card": {}, "split": None, "oversized": None, "load_s": None, "wall_s": None,
        "first_loss": rows[0]["loss"] if rows else None, "last_loss": rows[-1]["loss"] if rows else None,
    }
    if len(rows) > 1:
        summary["s_per_step"] = (rows[-1]["elapsed_s"] - rows[0]["elapsed_s"]) / (len(rows) - 1)
    else:
        summary["s_per_step"] = None
    if prov:
        per = (prov["epochs"] or [{}])[-1].get("peak_bytes_per_device") or {}
        summary["per_card"] = {k: v / GiB for k, v in per.items()}
        if per:
            summary["peak_gib"] = max(per.values()) / GiB
        stats = prov.get("token_budget") or {}
        summary["split"] = stats.get("split_records")
        summary["oversized"] = stats.get("oversized")
        summary["passes_total"] = stats.get("passes")
        summary["stopped_early"] = prov.get("stopped_early")
        summary["epoch_seconds"] = (prov["epochs"] or [{}])[-1].get("seconds")
        summary["forward_tokens"] = (prov["epochs"] or [{}])[-1].get("forward_tokens")
    if warm:
        summary["wall_s"] = warm.get("trainer_wall_s")
        if rows:
            summary["load_s"] = (warm.get("trainer_wall_s") or 0) - rows[-1]["elapsed_s"]
    return summary


def main() -> int:
    budgets = sys.argv[1:] or ["4096", "8192", "16384", "32768"]
    summaries = [budget_summary(W / f"runs/smoke-27b-tb{b}") for b in budgets]

    print("## the walk\n")
    print("| token budget | steps | s/step | dearest pass (padded tok) | passes in that step | "
          "peak GiB/card | per card cuda:0..3 (GiB) | records cut | passes over budget | load s |")
    print("|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    for s in summaries:
        if not s["steps"]:
            print(f"| {s['token_budget']} | — | — | — | — | — | (no rows) | — | — | — |")
            continue
        per = "/".join(f"{s['per_card'][k]:.2f}" for k in sorted(s["per_card"], key=lambda d: int(d.split(':')[1]))) or "—"
        rate = f"{s['s_per_step']:.1f}" if s["s_per_step"] else "—"
        load = f"{s['load_s']:.0f}" if s["load_s"] is not None else "—"
        print(f"| {s['token_budget']} | {s['steps']} | {rate} | {s['dearest_pass']} | {s['max_passes']} | "
              f"{s['peak_gib']:.2f} | {per} | {s['split'] if s['split'] is not None else '—'} | "
              f"{s['oversized'] if s['oversized'] is not None else '—'} | {load} |")

    print("\n## the peak against the dearest pass, per budget\n")
    for s in summaries:
        if not s["steps"]:
            continue
        points = new_peak_points(s["rows"])
        f = fit(points)
        line = (f"peak = {f[0]:.3f} + {f[1] * 1e6 / 1024:.4f} MB x padded_tokens  (n={f[2]} raising steps)"
                if f else "fewer than 2 raising steps: no line")
        print(f"* budget {s['token_budget']}: {line}")
        print(f"  - raising steps (dearest pass -> peak GiB): "
              + ", ".join(f"{p[0]}->{p[1]:.2f}" for p in points))

    print("\n## the same line for the 9B epoch this probe is compared against\n")
    nine = budget_summary(W / "runs/delta-kev-9b-s1")
    if nine["steps"]:
        points = new_peak_points(nine["rows"])
        f = fit(points)
        nine_budget = (nine["provenance"] or {}).get("token_budget", {}).get("budget")
        print(f"* 9B (runs/delta-kev-9b-s1, {nine['steps']} steps, budget {nine_budget}): "
              + (f"peak = {f[0]:.3f} + {f[1] * 1e6 / 1024:.4f} MB x padded_tokens (n={f[2]})" if f
                 else "no line"))
        print(f"  - raising steps: " + ", ".join(f"{p[0]}->{p[1]:.2f}" for p in points))
        print(f"  - dearest pass {nine['dearest_pass']}, worst card {nine['peak_gib']:.3f} GiB, "
              f"{nine['s_per_step']:.2f} s/step, epoch {nine['epoch_seconds']:.0f} s, "
              f"forward tokens {nine['forward_tokens']}")

    print("\n## the first steps of each budget (the long tail is drawn by step 4)\n")
    for s in summaries:
        if not s["steps"]:
            continue
        print(f"* budget {s['token_budget']}:")
        for r in s["rows"][:6]:
            print(f"  - step {r['step']}: window_tokens {r['window_tokens']}, slots_max {r['slots_max']}, "
                  f"passes {r['passes']}, peak {r['peak_device_bytes'] / GiB:.3f} GiB, loss {r['loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
