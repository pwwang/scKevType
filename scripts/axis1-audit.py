#!/usr/bin/env python3
"""Axis-1 audit: what the pooled AUROC 0.611 is made of, and how the statistic is served.

Two things this prints, both of which are measurements of the *inputs* to the axis-1 number rather
than of the model:

* **the support** — how many axis-1 positive and negative rows sit in each atlas, family and
  panel-size cell. AUROC needs both classes, so a cell with one class has no AUROC at all and a
  cell with a handful of one class has a number that is arithmetic rather than evidence. Those
  cells are flagged here rather than averaged into a pooled figure that reads as if they carried
  their weight.
* **the serve shape** — the axis-1 label is read off the row's own option keys (``metrics
  .panel_adequacy``), so a row whose question was served *without* an abstain option can only ever
  be a negative. ``eval.run.with_abstain`` re-asks those rows with one appended, which is what gives
  axis 1 a statistic on every choice row; it also means the positive and negative classes are drawn
  from different prompt shapes. This section measures how different, per class, without the model.

Run locally (no GPU, no checkpoint):

    uv run python scripts/axis1-audit.py                       # report + the suite that is present
    uv run python scripts/axis1-audit.py --suite artifacts/suites/real-v2-eval --split test

Once a cluster probe has dumped per-row scores (``--rows``), the same script computes the
per-atlas / per-family / per-size AUROC that only a scoring pass can produce:

    uv run python scripts/axis1-audit.py --rows reports/axis1-rows-transfer.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sckevtype.eval import abstain_key
from sckevtype.eval import metrics as M

#: A cell with fewer than this many rows of either class is reported as a count, not an AUROC.
THIN = 25

SUITES = ("artifacts/suites/real-v2-eval", "artifacts/suites/real-v2")


def read_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            meta = record["_meta"]
            for qid, question in record["questions"].items():
                if question["type"] != "choice":
                    continue
                keys = list(question["criteria"])
                key = abstain_key(keys)
                label = question.get("label")
                rows.append(
                    {
                        "id": meta["id"],
                        "question": qid,
                        "split": meta["split"],
                        "dataset": meta["dataset_id"],
                        "donor": meta.get("donor_id"),
                        "family": meta["family"],
                        "variant": meta.get("variant"),
                        "size": len(keys),
                        "factory_abstain": key is not None,
                        # The axis-1 label of metrics.panel_adequacy, restated here because it is a
                        # property of the question alone: a row with no abstain option cannot be a
                        # positive, however the serving path fills its none_p in.
                        "positive": key is not None and label == key,
                        # Where the abstain option sits in the prompt *as served*: the factory's
                        # position when it gave one, and appended-last when eval.run.with_abstain is
                        # what put it there.
                        "served_abstain_last": key is None or keys.index(key) == len(keys) - 1,
                    }
                )
    return rows


def cells(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: defaultdict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return dict(out)


def support(group: list[dict]) -> str:
    positive = sum(row["positive"] for row in group)
    negative = len(group) - positive
    n = len(group)
    if positive == 0 or negative == 0:
        return f"n={n:6d}  pos={positive:6d}  neg={negative:6d}  prev={positive / n:.3f}  NO AUROC (one class only)"
    flag = "  THIN" if min(positive, negative) < THIN else ""
    return (f"n={n:6d}  pos={positive:6d}  neg={negative:6d}  prev={positive / n:.3f}"
            f"  min_class={min(positive, negative):5d}{flag}")


def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def report_side(path: str) -> None:
    with open(path) as handle:
        report = json.load(handle)
    axis1 = report["axis1_panel_adequacy"]
    section(f"1. WHAT THE REPORT ITSELF BREAKS OUT  ({path})")
    print(f"  suite scored      : {report['suite']['path']}  (name {report['suite']['name']!r})")
    print(f"  rows scored       : {report['partitions']['rows']}")
    print(f"  axis 1 pooled     : n={axis1['n']}  pos={axis1['n_positive']}  neg={axis1['n_negative']}  "
          f"auroc={axis1['auroc']:.4f}  detection@FPR{axis1['target_fpr']}={axis1['detection_rate']:.4f} "
          f"(achieved FPR {axis1['achieved_fpr']:.4f})")
    print("  axis 1 per family (the ONLY per-group axis-1 AUROC the report carries):")
    for family, cell in axis1["per_family"].items():
        if cell.get("auroc") is None:
            print(f"    {family:20s} n={cell['n']:6d}   {cell.get('note', '')}")
        else:
            print(f"    {family:20s} n={cell['n']:6d}  pos={cell['n_positive']:6d}  neg={cell['n_negative']:6d}  "
                  f"auroc={cell['auroc']:.4f}  detection={cell['detection_rate']:.4f}")
    print("  per-atlas and per-panel-size breakdowns the report carries are ACCURACY, not axis-1 AUROC:")
    print("    per dataset :", {k: round(v["accuracy"], 4) for k, v in report["breakdowns"]["dataset"].items()})
    size = report["breakdowns"]["size"]
    thin = sorted((int(k) for k, v in size.items() if v["n"] < THIN))
    print(f"    per size    : {len(size)} bins, sizes {min(int(k) for k in size)}..{max(int(k) for k in size)}, "
          f"bins with n<{THIN}: {thin}")
    print("    (no per-atlas axis-1 AUROC exists in the report: it carries no per-row scores — see section 4)")


def support_side(rows: list[dict], suite: str, split: str) -> None:
    section(f"2. AXIS-1 SUPPORT BY CELL  (suite {suite}, split {split}, local uncapped copy where noted)")
    print(f"  {len(rows)} choice rows")
    for keys in (("dataset",), ("family",), ("size",)):
        label = keys[0]
        print(f"\n  --- per {label} " + "-" * (90 - len(label)))
        grouped = cells(rows, keys)
        order = sorted(grouped, key=lambda k: (str(k[0]).zfill(3) if label == "size" else str(k[0])))
        for key in order:
            print(f"    {key[0]!s:24s} {support(grouped[key])}")
    print("\n  --- per (dataset, family) " + "-" * 74)
    grouped = cells(rows, ("dataset", "family"))
    for key in sorted(grouped):
        print(f"    {key[0]:20s} {key[1]:18s} {support(grouped[key])}")
    print("\n  --- per (size, family) " + "-" * 78)
    grouped = cells(rows, ("size", "family"))
    for key in sorted(grouped, key=lambda k: (k[0], k[1])):
        if not grouped[key][0]["positive"] and len(grouped[key]) < THIN:
            continue
        print(f"    size {key[0]:<3d} {key[1]:18s} {support(grouped[key])}")


def serve_shape(rows: list[dict]) -> None:
    section("3. THE SERVE SHAPE OF EACH CLASS  (positives vs negatives are not drawn alike)")
    positive = [row for row in rows if row["positive"]]
    negative = [row for row in rows if not row["positive"]]
    for name, group in (("POSITIVE (target is the abstain option)", positive),
                        ("NEGATIVE (target is a real option)", negative)):
        if not group:
            continue
        injected = sum(not row["factory_abstain"] for row in group)
        last = sum(row["served_abstain_last"] for row in group)
        print(f"\n  {name}: n={len(group)}")
        print(f"    abstain option INJECTED by the serving path  : {injected:6d}  ({injected / len(group):.3f})")
        print(f"    abstain option LAST in the prompt as served  : {last:6d}  ({last / len(group):.3f})")
        print(f"    by variant                                   : "
              f"{dict(Counter(row['variant'] for row in group))}")
        print(f"    factory-given rows, abstain position         : "
              f"{dict(Counter('last' if row['served_abstain_last'] else 'other' for row in group if row['factory_abstain']))}")
    if positive and negative:
        pin = sum(not row["factory_abstain"] for row in positive) / len(positive)
        nin = sum(not row["factory_abstain"] for row in negative) / len(negative)
        plast = sum(row["served_abstain_last"] for row in positive) / len(positive)
        nlast = sum(row["served_abstain_last"] for row in negative) / len(negative)
        print("\n  THE CONFOUND, in two numbers:")
        print(f"    injected share   positive {pin:.3f}   negative {nin:.3f}   (difference {nin - pin:+.3f})")
        print(f"    abstain-last     positive {plast:.3f}   negative {nlast:.3f}   (difference {nlast - plast:+.3f})")
        print("    An injected row's None option is appended, so it is ALWAYS last; a positive is never injected.")
        print("    Any position dependence in P(abstain) therefore reaches the AUROC as a class-correlated shift.")


def row_side(path: str) -> None:
    section(f"4. AUROC FROM A SCORED ROW DUMP  ({path})")
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    print(f"  {len(rows)} scored rows")
    for keys in (("dataset",), ("family",), ("size",)):
        label = keys[0]
        print(f"\n  --- axis-1 AUROC per {label} " + "-" * (70 - len(label)))
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[label]].append(row)
        for key in sorted(grouped, key=lambda k: (str(k).zfill(3) if label == "size" else str(k))):
            group = grouped[key]
            scores = [row["none_p"] for row in group]
            labels = [bool(row["positive"]) for row in group]
            try:
                print(f"    {key!s:24s} {support(group)}  auroc={M.auroc(scores, labels):.4f}")
            except ValueError:
                print(f"    {key!s:24s} {support(group)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default="reports/eval-real-v2-eval.json")
    parser.add_argument("--suite", default=None, help="defaults to the capped eval suite when it is present")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", default=None, help="per-row score dump from a cluster probe")
    args = parser.parse_args()

    if Path(args.report).exists():
        report_side(args.report)
    else:
        print(f"no report at {args.report}")

    if args.rows:
        row_side(args.rows)

    if args.suite:
        suite = args.suite
    else:
        suite = next((s for s in SUITES if Path(s, "manifest.json").exists()), None)
    if suite is None:
        print("\nno suite found locally: expected one of", SUITES)
        return 1
    path = Path(suite, f"{args.split}.jsonl")
    if not path.exists():
        print(f"\nno {path}")
        return 1
    print(f"\nreading {path} ({path.stat().st_size / 1e6:.0f} MB) ...")
    rows = [row for row in read_rows(str(path)) if row["split"] == args.split]
    capped = "real-v2-eval" in suite
    support_side(rows, suite, args.split)
    if capped:
        print("\n  (this IS the capped suite the report scored -- section 2's n should match the report)")
    else:
        print("\n  (this is the UNCAPPED suite: the report scored a 36% stratified cap of this test split,"
              "\n   so the counts here are the cap's population, not the numbers the report reports)")
    serve_shape(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
