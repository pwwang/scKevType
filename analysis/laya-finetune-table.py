#!/usr/bin/env python3
"""Emit the census-matched comparison table from the settled JSONs, so no number is hand-typed.

Reads ``reports/both-forced-table.json`` (every engine already on the population) and, when present,
the two Laya reconcile outputs this step produced
(``reports/laya-finetune-reconcile-base.json`` and ``...-pilot.json``), and prints one markdown table
sorted by served accuracy. The report pastes this output verbatim.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    path = os.path.join(ROOT, name)
    return json.load(open(path)) if os.path.exists(path) else None


def row(label, cell, klass):
    forced = cell.get("forced") or {}
    return {
        "label": label, "class": klass,
        "served": cell["served"]["accuracy"],
        "abstain": cell["abstain_pick"]["rate"],
        "forced": forced.get("accuracy"),
        "delta": cell.get("delta_points"),
        "hidden": cell.get("hidden_correct_share_of_abstentions"),
    }


def main() -> int:
    both = load("reports/both-forced-table.json")
    rows = [row(e["label"], e["census_matched_1819"], e["class"]) for e in both["engines"]
            if e["census_matched_1819"].get("n", 0) > 0 and e["key"] != "laya-pilot"]

    # The Laya rows this step measured. `both-forced-table.json` already carries the pilot's cell;
    # it is re-read from the dump so the table and the report cannot disagree, and the zero-shot cell
    # has no other home.
    for name, label, klass in (("reports/laya-finetune-reconcile-base.json",
                                "Laya zero-shot (shipped base, its own 1024/256)", "non-LLM encoder"),
                               ("reports/laya-finetune-reconcile-pilot.json",
                                "Laya pilot (322M, fine-tuned)", "non-LLM encoder"),
                               ("reports/laya-finetune-reconcile-spec.json",
                                "Laya fine-tuned on the training split (this step)", "non-LLM encoder")):
        payload = load(name)
        if payload is None:
            print(f"# (no {name} on this box -- that row is omitted)", file=sys.stderr)
            continue
        rows.append(row(label, payload["census_matched_1819"], klass))

    rows.sort(key=lambda r: -r["served"])
    print("| engine | served | abstain-pick | forced | gain (pt) | hidden correct |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        forced = "not recoverable" if r["forced"] is None else f"{r['forced']:.4f}"
        delta = "" if r["delta"] is None else f"{r['delta']:+.1f}"
        hidden = "" if r["hidden"] is None else f"{r['hidden']:.3f}"
        print(f"| {r['label']} | {r['served']:.4f} | {r['abstain']:.4f} | {forced} | {delta} | {hidden} |")
    print()
    print(f"{len(rows)} engines with a row on the 1,819 census-matched panel_choice questions.")
    print("Served = argmax(p); abstain-pick = the answer is the abstain option; forced = argmax over p")
    print("with the abstain option removed (reports/abstain-reconciliation.py's rules, unchanged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
