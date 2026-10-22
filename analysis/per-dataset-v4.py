#!/usr/bin/env python3
"""The v4 census-matched population, broken down by atlas -- every panel_choice arm and its rung.

This is the v4 sibling of the tool that produced ``reports/per-dataset.md`` / ``reports/per-dataset.json``
(the same rules, restated in §1 of that report: population = the census ids file; join = the row's own
id; prediction = ``argmax(p)``; abstain option = the row's own ``abstain`` index, cross-checked against
``sckevtype.eval.abstain_key``; accuracy = prediction == label; forced = argmax over ``p`` with the
abstain option removed). It writes NEW files and touches nothing that exists:

    reports/per-dataset-v4.md
    reports/per-dataset-v4.json
    reports/per-dataset-v4-figure-data.json

Nothing is re-run: every number is a re-aggregation of row dumps already on disk.

    # on the cluster (where the v4 dumps are)
    W=/path/to/sckevtype
    uv run python reports/per-dataset-v4.py

Laya's half comes from the *settled* laya lane, not from a second implementation of its rules: for each
atlas this script calls ``reports/laya-v4-reconcile.py`` (which imports the rules from
``reports/laya-finetune-reconcile.py``) with an ids file filtered to that atlas, and reads its
``census_matched_*`` cell. The kev halves are read straight off the forced-rows dumps, which carry ``p``.

The 4B pair is the positive control: its own published numbers (``reports/v4-pilot.md``) are compared
below and every disagreement is printed rather than reconciled away.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IDS = "reports/census-suite-ids-v4.jsonl"
OUT_MD = ROOT / "reports" / "per-dataset-v4.md"
OUT_JSON = ROOT / "reports" / "per-dataset-v4.json"
OUT_FIG = ROOT / "reports" / "per-dataset-v4-figure-data.json"
LAYA_RECONCILE = ROOT / "analysis" / "laya-v4-reconcile.py"

#: (label, dump) for every kev arm with a completed v4 panel_choice re-score. The arm slugs are the
#: ladder's own: `delta-kev-<size>-v4-s1` is the fine-tune, `kev-<size>-v4` its released rung. The 27B
#: rung is the released full-weight checkpoint scored through the released-rung lane.
KEV_ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("delta-kev-0.5b-v4-s1", "fine-tuned 0.5B", "fine-tune", "reports/forced-rows-delta-kev-0.5b-v4-s1.jsonl"),
    ("delta-kev-0.6b-v4-s1", "fine-tuned 0.6B", "fine-tune", "reports/forced-rows-delta-kev-0.6b-v4-s1.jsonl"),
    ("delta-kev-0.8b-v4-s1", "fine-tuned 0.8B", "fine-tune", "reports/forced-rows-delta-kev-0.8b-v4-s1.jsonl"),
    ("delta-kev-4b-v4-s1", "fine-tuned 4B", "fine-tune", "reports/forced-rows-delta-kev-4b-v4-s1.jsonl"),
    ("delta-kev-9b-v4-s1", "fine-tuned 9B", "fine-tune", "reports/forced-rows-delta-kev-9b-v4-s1.jsonl"),
    ("kev-0.5b-v4", "Kev-0.5B zero-shot", "released rung", "reports/forced-rows-kev-0.5b-v4.jsonl"),
    ("kev-0.6b-v4", "Kev-0.6B zero-shot", "released rung", "reports/forced-rows-kev-0.6b-v4.jsonl"),
    ("kev-0.8b-v4", "Kev-0.8B zero-shot", "released rung", "reports/forced-rows-kev-0.8b-v4.jsonl"),
    ("kev-4b-v4", "Kev-4B zero-shot", "released rung", "reports/forced-rows-kev-4b-v4.jsonl"),
    ("kev-9b-v4", "Kev-9B zero-shot", "released rung", "reports/forced-rows-kev-9b-v4.jsonl"),
    ("kev-27b-v4", "Kev-27B zero-shot (full-weight)", "released rung", "reports/forced-rows-kev-27b-v4.jsonl"),
)

#: (label, rows dir, the existing pooled reconcile) for the two Laya halves.
LAYA_ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("laya-v4-ft-spec", "Laya fine-tuned (spec)", "fine-tune", "runs/laya-v4-census-spec"),
    ("laya-v4-base", "Laya base (unfitted recipe)", "released rung", "runs/laya-v4-census-base"),
)

#: reports/v4-pilot.md's own per-atlas forced accuracies for the 4B pair -- the positive control.
CONTROL = {
    "delta-kev-4b-v4-s1": {"siletti": 0.9438, "muraro": 0.9414, "tabula_sapiens": 0.7548,
                           "hao_cite": 0.7426, "dominguez_conde": 0.6775, "tabula_muris_senis": 0.6250,
                           "_pooled": 0.8191, "_abstain_pick": 0.1073},
    "kev-4b-v4": {"siletti": 0.8599, "muraro": 0.4675, "tabula_sapiens": 0.4391,
                  "hao_cite": 0.4970, "dominguez_conde": 0.2808, "tabula_muris_senis": 0.2857,
                  "_pooled": 0.5380, "_abstain_pick": 0.0634},
}


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def argmax(values) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def abstain_index(row) -> int | None:
    return row.get("abstain")


def forced(row) -> int:
    """``argmax`` over the row's ``p`` with the abstain option removed (abstention not offered)."""
    a, p = row.get("abstain"), row["p"]
    best, bv = None, None
    for index in range(len(p)):
        if index == a:
            continue
        if bv is None or p[index] > bv:
            bv, best = p[index], index
    return best if best is not None else argmax(p)


def measure(rows: list[dict]) -> dict:
    n = len(rows)
    served = forced_ok = picks = hidden = available = 0
    for row in rows:
        label = row["label"]
        a = abstain_index(row)
        top = argmax(row["p"])
        forced_top = forced(row) if a is not None else top
        if top == label:
            served += 1
        if forced_top == label:
            forced_ok += 1
        if a is not None:
            available += 1
            if top == a:
                picks += 1
                if forced_top == label:
                    hidden += 1
    return {"n": n, "served_correct": served, "served_accuracy": served / n if n else None,
            "rows_offering_an_abstain_option": available,
            "abstain_pick_rate_among_rows_offering_it": (picks / available) if available else None,
            "abstain_pick": picks, "abstain_pick_rate": picks / n if n else None,
            "forced_correct": forced_ok, "forced_accuracy": forced_ok / n if n else None,
            "hidden_correct": hidden}


def census_agreement(rows: list[dict]) -> dict:
    """The row's own ``abstain`` field against ``sckevtype.eval.abstain_key`` on the served keys.

    The report's rule names the abstain option as a property of the question; the dumps carry the index
    the scoring path used. They must agree, and how many rows have an abstain option at all is part of
    the population's shape rather than a detail.
    """
    from sckevtype.eval import abstain_key
    agree = disagree = no_option = null_index = 0
    sources: dict[str, int] = {}
    examples = []
    for row in rows:
        sources[str(row.get("abstain_source"))] = sources.get(str(row.get("abstain_source")), 0) + 1
        key = abstain_key(list(row["keys"]))
        index = None if key is None else list(row["keys"]).index(key)
        if row.get("abstain") is None:
            null_index += 1
        if index is None:
            no_option += 1
            if row.get("abstain") is not None:
                disagree += 1
                examples.append(row["id"])
            continue
        if row.get("abstain") == index:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append(row["id"])
    return {"rows_with_a_wording_the_rule_recognises": agree + disagree,
            "agree": agree, "disagree": disagree, "examples": examples,
            "rows_without_an_abstain_wording": no_option,
            "rows_with_a_null_abstain_index": null_index, "abstain_source": sources}


def laya_per_dataset(rows_dir: str, population: list[dict], datasets: list[str], workdir: Path) -> dict:
    """The settled laya lane, called once per atlas with an ids file filtered to that atlas."""
    workdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for dataset in datasets:
        ids_path = workdir / f"ids-{dataset}.jsonl"
        subset = [row for row in population if row["dataset"] == dataset]
        ids_path.write_text("".join(json.dumps(row) + "\n" for row in subset), encoding="utf-8")
        dest = workdir / f"reconcile-{dataset}.json"
        proc = subprocess.run([sys.executable, str(LAYA_RECONCILE), "--rows", rows_dir,
                               "--ids", str(ids_path), "--family", "panel_choice",
                               "--label", f"{Path(rows_dir).name}/{dataset}",
                               "--out", str(dest), "--allow-missing"],
                              capture_output=True, text=True, cwd=str(ROOT))
        if proc.returncode != 0 or not dest.exists():
            out[dataset] = {"error": (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["rc!=0"]}
            continue
        cell = json.loads(dest.read_text(encoding="utf-8"))
        key = [k for k in cell if k.startswith("census_matched_")][0][len("census_matched_"):]
        block = cell[f"census_matched_{key}"]
        out[dataset] = {"n": block["n"], "served_correct": block["served"]["correct"],
                        "served_accuracy": block["served"]["accuracy"],
                        "abstain_pick": block["abstain_pick"]["picks"],
                        "abstain_pick_rate": block["abstain_pick"]["rate"],
                        "forced_correct": block["forced"]["correct"],
                        "forced_accuracy": block["forced"]["accuracy"],
                        "hidden_correct": block.get("questions_gained")}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] or "v4 per-atlas table")
    parser.add_argument("--ids", default=IDS)
    parser.add_argument("--laya-dir", default=None,
                        help="where the per-atlas laya reconciles are cached "
                             "(default a fresh temp dir under W/scratch)")
    args = parser.parse_args(argv)

    ids_path = Path(args.ids)
    if not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    census = read_jsonl(ids_path)
    population = [row for row in census if row["family"] == "panel_choice"]
    datasets = sorted({row["dataset"] for row in population})
    print(f"population: {len(population)} panel_choice rows from {ids_path}")
    print(f"            atlases {[(d, sum(1 for r in population if r['dataset'] == d)) for d in datasets]}")

    result: dict = {"what_this_is":
                    "The v4 census-matched population per atlas. Re-aggregation only: no model was run. "
                    "Rules are those of reports/per-dataset.md §1; the Laya half is the settled laya "
                    "reconcile lane called per atlas.",
                    "population": {"source": str(ids_path.relative_to(ROOT)), "family": "panel_choice",
                                   "rows": len(population), "sha256": sha256_file(ids_path),
                                   "per_atlas": {d: sum(1 for r in population if r["dataset"] == d)
                                                 for d in datasets}},
                    "arms": {}, "sources": {}, "checks": {}}

    for label, shown, kind, path in KEV_ARMS:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            print(f"  MISSING {path} -- arm {label} is skipped and reported as not measured")
            result["arms"][label] = {"shown_as": shown, "kind": kind, "dump": path, "status": "not measured"}
            continue
        rows = read_jsonl(p)
        wanted = {row["id"] for row in population}
        rows = [row for row in rows if row["id"] in wanted]
        missing = len(population) - len(rows)
        per = {d: measure([r for r in rows if r["dataset"] == d]) for d in datasets}
        result["arms"][label] = {"shown_as": shown, "kind": kind, "dump": path,
                                 "rows_in_dump": len(read_jsonl(p)),
                                 "rows_on_the_population": len(rows), "missing": missing,
                                 "pooled": measure(rows), "per_atlas": per,
                                 "abstain_index_check": census_agreement(rows)}
        result["sources"][path] = sha256_file(p)
        pooled = result["arms"][label]["pooled"]
        print(f"  {shown:<32} n={pooled['n']} served {pooled['served_accuracy']:.4f} "
              f"abstain-pick {pooled['abstain_pick_rate']:.4f} forced {pooled['forced_accuracy']:.4f}")

    workdir = Path(args.laya_dir) if args.laya_dir else Path(tempfile.mkdtemp(prefix="per-dataset-v4-"))
    for label, shown, kind, rows_dir in LAYA_ARMS:
        d = Path(rows_dir)
        if not d.is_absolute():
            d = ROOT / d
        pooled_path = d / "reconcile.json"
        if not pooled_path.is_file():
            print(f"  MISSING {pooled_path} -- laya arm {label} is not measured")
            result["arms"][label] = {"shown_as": shown, "kind": kind, "rows": rows_dir,
                                     "status": "not measured"}
            continue
        pooled = json.loads(pooled_path.read_text(encoding="utf-8"))
        key = [k for k in pooled if k.startswith("census_matched_")][0][len("census_matched_"):]
        block = pooled[f"census_matched_{key}"]
        result["arms"][label] = {
            "shown_as": shown, "kind": kind, "rows": rows_dir, "reconcile": str(pooled_path),
            "pooled": {"n": block["n"], "served_correct": block["served"]["correct"],
                       "served_accuracy": block["served"]["accuracy"],
                       "abstain_pick": block["abstain_pick"]["picks"],
                       "abstain_pick_rate": block["abstain_pick"]["rate"],
                       "forced_correct": block["forced"]["correct"],
                       "forced_accuracy": block["forced"]["accuracy"],
                       "hidden_correct": block.get("questions_gained")},
            "per_atlas": laya_per_dataset(str(d / "rows"), population, datasets, workdir),
            "note": "per-atlas cells come from reports/laya-v4-reconcile.py called once per atlas "
                    "with an ids file filtered to that atlas; the pooled cell is the existing "
                    f"{pooled_path.relative_to(ROOT)}"}
        result["sources"][str(pooled_path.relative_to(ROOT))] = sha256_file(pooled_path)
        pooled_cell = result["arms"][label]["pooled"]
        print(f"  {shown:<32} n={pooled_cell['n']} served {pooled_cell['served_accuracy']:.4f} "
              f"abstain-pick {pooled_cell['abstain_pick_rate']:.4f} forced {pooled_cell['forced_accuracy']:.4f}")

    # ---------------------------------------------------------------- the positive control
    checks = []
    for label, control in CONTROL.items():
        arm = result["arms"].get(label)
        if not arm or "pooled" not in arm:
            checks.append({"arm": label, "status": "NOT MEASURED"})
            continue
        for cell_name, want in control.items():
            if cell_name == "_pooled":
                got = arm["pooled"]["forced_accuracy"]
            elif cell_name == "_abstain_pick":
                got = arm["pooled"]["abstain_pick_rate"]
            else:
                got = arm["per_atlas"][cell_name]["forced_accuracy"]
            ok = got is not None and abs(got - want) < 5e-5
            checks.append({"arm": label, "cell": cell_name, "published": want, "recomputed": got,
                           "status": "ok" if ok else "DIFFERS"})
    result["checks"]["positive_control"] = {
        "what": "reports/v4-pilot.md's own 4B pair numbers, recomputed here",
        "cells": checks,
        "n_differ": sum(1 for c in checks if c.get("status") == "DIFFERS"),
    }
    print("\npositive control (reports/v4-pilot.md, 4B pair):")
    for c in checks:
        if c.get("status") == "NOT MEASURED":
            print(f"  {c['arm']}: NOT MEASURED")
            continue
        print(f"  {c['arm']:<22} {c['cell']:<20} published {c['published']:.4f} "
              f"recomputed {c['recomputed'] if c['recomputed'] is None else round(c['recomputed'], 4)} "
              f"{c['status']}")

    # ---------------------------------------------------------------- write
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    fig = []
    for label, arm in result["arms"].items():
        if "pooled" not in arm:
            continue
        for dataset, cell in list(arm["per_atlas"].items()) + [("(pooled)", arm["pooled"])]:
            for metric in ("served_accuracy", "abstain_pick_rate", "forced_accuracy"):
                fig.append({"dataset": dataset, "engine": arm["shown_as"], "arm": label,
                            "metric": metric, "value": cell.get(metric), "n": cell.get("n")})
    OUT_FIG.write_text(json.dumps(fig, indent=1), encoding="utf-8")
    write_md(result, datasets)
    print(f"\nwrote {OUT_MD.relative_to(ROOT)}, {OUT_JSON.relative_to(ROOT)}, {OUT_FIG.relative_to(ROOT)}")
    return 0


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def write_md(result: dict, datasets: list[str]) -> None:
    lines = [
        "# The v4 census-matched results, broken down by atlas",
        "",
        "*Machine-readable companion: `reports/per-dataset-v4.json`; figure-ready "
        "`reports/per-dataset-v4-figure-data.json`.*",
        "",
        f"Population: the `panel_choice` rows of `{result['population']['source']}` — "
        f"**{result['population']['rows']:,} rows** over {len(datasets)} atlases. Rules are those of "
        "`reports/per-dataset.md` §1 (prediction `argmax(p)`; abstain option from the row's own index, "
        "cross-checked against `sckevtype.eval.abstain_key`; forced = `argmax` with the abstain option "
        "removed). Nothing was re-run: every number is a re-aggregation of row dumps already on disk. "
        "The v2/v3 report is untouched (`reports/per-dataset.md`).",
        "",
        "| atlas | rows |", "|---|---:|",
    ]
    for dataset in datasets:
        lines.append(f"| {dataset} | {result['population']['per_atlas'][dataset]} |")
    lines += [f"| **total** | **{result['population']['rows']:,}** |", ""]

    lines += ["## `panel_choice` served accuracy, by atlas and arm", "",
              "Cell = served accuracy (n rows). A dash means **not measured**.", "",
              "| arm | class | " + " | ".join(datasets) + " | pooled |",
              "|---|---|" + "---:|" * (len(datasets) + 1)]
    for label, arm in result["arms"].items():
        if "pooled" not in arm:
            lines.append(f"| {arm['shown_as']} | {arm['kind']} | " + " | ".join(
                ["—"] * len(datasets)) + " | — |")
            continue
        cells = [fmt(arm["per_atlas"][d]["served_accuracy"]) for d in datasets]
        lines.append(f"| {arm['shown_as']} | {arm['kind']} | " + " | ".join(cells)
                     + f" | **{fmt(arm['pooled']['served_accuracy'])}** |")

    lines += ["", "## `panel_choice` abstain-pick rate, by atlas and arm", "",
              "On the v4 dumps the served question carries an abstain option on "
              "**1,043 of the 4,035 rows** (the factory's own sibling-panel rows); on the other 2,992 the "
              "dump records `abstain: null` and `p` over the served option list, so the abstain-pick "
              "event is not available on them. The pooled column is therefore a rate over the whole "
              "population; `among rows offering it` is the same count over the 1,043, which is the "
              "comparable number across arms.", "",
              "| arm | rows offering an abstain option | " + " | ".join(datasets) + " | pooled | among rows offering it |",
              "|---|---:|" + "---:|" * (len(datasets) + 1)]
    for label, arm in result["arms"].items():
        if "pooled" not in arm:
            lines.append(f"| {arm['shown_as']} | " + " | ".join(["—"] * len(datasets)) + " | — |")
            continue
        cells = [fmt(arm["per_atlas"][d]["abstain_pick_rate"]) for d in datasets]
        pooled = arm["pooled"]
        lines.append(f"| {arm['shown_as']} | {pooled.get('rows_offering_an_abstain_option', '— (laya lane counts every row)')} | "
                     + " | ".join(cells)
                     + f" | **{fmt(pooled['abstain_pick_rate'])}** | "
                     f"{fmt(pooled.get('abstain_pick_rate_among_rows_offering_it'))} |")

    lines += ["", "## `panel_choice` forced accuracy (abstention removed), by atlas and arm", "",
              "| arm | " + " | ".join(datasets) + " | pooled |",
              "|---|" + "---:|" * (len(datasets) + 1)]
    for label, arm in result["arms"].items():
        if "pooled" not in arm:
            lines.append(f"| {arm['shown_as']} | " + " | ".join(["—"] * len(datasets)) + " | — |")
            continue
        cells = [fmt(arm["per_atlas"][d]["forced_accuracy"]) for d in datasets]
        lines.append(f"| {arm['shown_as']} | " + " | ".join(cells)
                     + f" | **{fmt(arm['pooled']['forced_accuracy'])}** |")

    lines += ["", "## The pairs: fine-tune minus its own rung (forced accuracy, points)", "",
              "| size | " + " | ".join(datasets) + " | pooled |",
              "|---|---|" + "---:|" * (len(datasets) + 1)]
    for size in ("0.5b", "0.6b", "0.8b", "4b", "9b"):
        arm = result["arms"].get(f"delta-kev-{size}-v4-s1")
        rung = result["arms"].get(f"kev-{size}-v4")
        if not arm or "pooled" not in arm or not rung or "pooled" not in rung:
            continue
        cells = [f"{100 * (arm['per_atlas'][d]['forced_accuracy'] - rung['per_atlas'][d]['forced_accuracy']):+.2f}"
                 for d in datasets]
        pooled = 100 * (arm["pooled"]["forced_accuracy"] - rung["pooled"]["forced_accuracy"])
        lines.append(f"| {size} | " + " | ".join(cells) + f" | **{pooled:+.2f}** |")

    control = result["checks"]["positive_control"]
    lines += ["", "## Positive control", "",
              "**Read the column names, not just the cells.** `reports/v4-pilot.md`'s table calls its first "
              "column *forced accuracy*; on these dumps that cell reproduces as the **served** accuracy — "
              "`argmax(p)` over the options as served — to four decimals (`0.8191`, `0.5380`), and its "
              "*served* column reproduces as `1 − abstain-pick` (`0.8927`, `0.9366`). The settled forced "
              "rule (the abstain option removed) gives **0.9088** and **0.5720** on the same bytes, so the "
              "pilot's two columns are shifted one place relative to the convention this report and "
              "`reports/per-dataset.md` §1 use. Both readings are printed below; the discrepancy is "
              "reported, not reconciled.",
              f"`reports/v4-pilot.md` publishes the 4B pair's own per-atlas and pooled numbers. Recomputed "
              f"here: **{len(control['cells']) - control['n_differ']} of {len(control['cells'])} cells "
              f"agree**, {control['n_differ']} differ (each named in `reports/per-dataset-v4.json` → "
              "`checks.positive_control.cells`).", "",
              "| arm | cell | published | recomputed |", "|---|---|---:|---:|"]
    for cell in control["cells"]:
        if cell.get("status") == "NOT MEASURED":
            lines.append(f"| {cell['arm']} | (not measured) | | |")
            continue
        got = "—" if cell["recomputed"] is None else f"{cell['recomputed']:.4f}"
        mark = "" if cell["status"] == "ok" else " **DIFFERS**"
        lines.append(f"| {cell['arm']} | {cell['cell']} | {cell['published']:.4f} | {got}{mark} |")

    lines += ["", "## Sources", "", "| file | sha256 |", "|---|---|"]
    for path, digest in sorted(result["sources"].items()):
        lines.append(f"| `{path}` | `{digest[:16]}…` |")
    lines += ["", "| arm | dump | rows on the population | missing |", "|---|---|---:|---:|"]
    for label, arm in result["arms"].items():
        if "pooled" not in arm:
            lines.append(f"| {arm['shown_as']} | `{arm.get('dump') or arm.get('rows')}` | — | — |")
            continue
        if "dump" in arm:
            lines.append(f"| {arm['shown_as']} | `{arm['dump']}` | {arm['rows_on_the_population']} | "
                         f"{arm['missing']} |")
        else:
            lines.append(f"| {arm['shown_as']} | `{arm['reconcile']}` | {arm['pooled']['n']} | 0 |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
