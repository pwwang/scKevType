#!/usr/bin/env python3
"""The v4 census-matched `panel_choice` population under the three readings, for every arm.

The v4 sibling of ``reports/abstain-reconciliation.py`` (which is untouched, as is its output): it
re-applies that report's own three rules to row dumps already on disk and adds nothing to what they
mean --

  * **served**      ``argmax(row["p"])`` over every option the question offered;
  * **abstain-pick** the served pick *is* the abstain option (the model declined);
  * **forced**      ``argmax`` over ``row["p"]`` with the abstain option removed (abstention not offered).

and the identity that ties them together:

  **forced_correct − served_correct = #{declined ∧ forced-correct}** ( = declines × hidden-correct share )

i.e. every answer the model would have named correctly and instead declined is exactly the accuracy the
abstention costs, and the three readings are consistent by construction. Each arm's identity is checked
and printed rather than assumed.

It writes new files and overwrites nothing:

    reports/abstain-reconciliation-v4.md
    reports/abstain-reconciliation-v4.json

    # on the cluster (where the v4 dumps are)
    W=/path/to/sckevtype
    uv run python reports/abstain-reconciliation-v4.py

The Laya halves are read off their own settled reconciles (``runs/laya-v4-census-*/reconcile.json``),
whose rules are imported from ``reports/laya-finetune-reconcile.py`` -- the same three readings, so the
column is comparable arm for arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IDS = "reports/census-suite-ids-v4.jsonl"
OUT_MD = ROOT / "reports" / "abstain-reconciliation-v4.md"
OUT_JSON = ROOT / "reports" / "abstain-reconciliation-v4.json"

KEV_ARMS: tuple[tuple[str, str, str], ...] = (
    ("delta-kev-0.5b-v4-s1", "fine-tuned 0.5B", "reports/forced-rows-delta-kev-0.5b-v4-s1.jsonl"),
    ("delta-kev-0.6b-v4-s1", "fine-tuned 0.6B", "reports/forced-rows-delta-kev-0.6b-v4-s1.jsonl"),
    ("delta-kev-0.8b-v4-s1", "fine-tuned 0.8B", "reports/forced-rows-delta-kev-0.8b-v4-s1.jsonl"),
    ("delta-kev-4b-v4-s1", "fine-tuned 4B", "reports/forced-rows-delta-kev-4b-v4-s1.jsonl"),
    ("delta-kev-9b-v4-s1", "fine-tuned 9B", "reports/forced-rows-delta-kev-9b-v4-s1.jsonl"),
    ("delta-kev-27b-v4-s1", "fine-tuned 27B", "reports/forced-rows-delta-kev-27b-v4-s1.jsonl"),
    ("kev-0.5b-v4", "Kev-0.5B zero-shot", "reports/forced-rows-kev-0.5b-v4.jsonl"),
    ("kev-0.6b-v4", "Kev-0.6B zero-shot", "reports/forced-rows-kev-0.6b-v4.jsonl"),
    ("kev-0.8b-v4", "Kev-0.8B zero-shot", "reports/forced-rows-kev-0.8b-v4.jsonl"),
    ("kev-4b-v4", "Kev-4B zero-shot", "reports/forced-rows-kev-4b-v4.jsonl"),
    ("kev-9b-v4", "Kev-9B zero-shot", "reports/forced-rows-kev-9b-v4.jsonl"),
    ("kev-27b-v4", "Kev-27B zero-shot (full-weight)", "reports/forced-rows-kev-27b-v4.jsonl"),
)
LAYA_ARMS: tuple[tuple[str, str, str], ...] = (
    ("laya-v4-ft-spec", "Laya fine-tuned (spec)", "runs/laya-v4-census-spec/reconcile.json"),
    ("laya-v4-base", "Laya base (unfitted recipe)", "runs/laya-v4-census-base/reconcile.json"),
)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(path: Path) -> str:
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


def forced_index(row) -> int:
    a, p = row.get("abstain"), row["p"]
    best, bv = None, None
    for index in range(len(p)):
        if index == a:
            continue
        if bv is None or p[index] > bv:
            bv, best = p[index], index
    return best if best is not None else argmax(p)


def cell(rows: list[dict]) -> dict:
    n = len(rows)
    served = picks = forced_ok = hidden = no_abstain_index = 0
    available = 0
    for row in rows:
        label = row["label"]
        a = row.get("abstain")
        top = argmax(row["p"])
        f = forced_index(row) if a is not None else top
        if a is None:
            no_abstain_index += 1
        else:
            available += 1
        if top == label:
            served += 1
        if f == label:
            forced_ok += 1
        if a is not None and top == a:
            picks += 1
            if f == label:
                hidden += 1
    identity_holds = (forced_ok - served) == hidden
    return {
        "n": n,
        "served": {"correct": served, "accuracy": served / n if n else None},
        "abstain_pick": {"picks": picks, "rate": picks / n if n else None,
                         "rows_offering_an_abstain_option": available,
                         "rate_among_rows_offering_it": (picks / available) if available else None},
        "forced": {"correct": forced_ok, "accuracy": forced_ok / n if n else None},
        "hidden_correct": hidden,
        "hidden_correct_share_of_abstentions": (hidden / picks) if picks else None,
        "delta_points": 100 * ((forced_ok - served) / n) if n else None,
        "rows_whose_serving_offered_no_abstain_option": no_abstain_index,
        "identity": {"forced_correct_minus_served_correct": forced_ok - served,
                     "declines_times_hidden_correct_share": hidden,
                     "holds": identity_holds},
    }


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] or "v4 abstention reconciliation")
    parser.add_argument("--ids", default=IDS)
    args = parser.parse_args(argv)

    ids_path = Path(args.ids)
    if not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    census = read_jsonl(ids_path)
    population = [row for row in census if row["family"] == "panel_choice"]
    wanted = {row["id"] for row in population}
    print(f"population: {len(population)} panel_choice rows from {ids_path}")

    result: dict = {"what_this_is":
                    "The v4 census-matched panel_choice population under the served / abstain-pick / "
                    "forced readings, for every arm with a v4 dump, plus the identity that ties them "
                    "together. Re-aggregation only: no model was run.",
                    "rules_from": "reports/abstain-reconciliation.py (served / abstain-pick / forced)",
                    "population": {"source": str(ids_path.relative_to(ROOT)), "family": "panel_choice",
                                   "rows": len(population), "sha256": sha256_file(ids_path)},
                    "arms": {}, "sources": {}, "checks": {}}

    for label, shown, path in KEV_ARMS:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            print(f"  MISSING {path} -- {label} is not measured")
            result["arms"][label] = {"shown_as": shown, "dump": path, "status": "not measured"}
            continue
        rows = [row for row in read_jsonl(p) if row["id"] in wanted]
        block = cell(rows)
        block.update({"shown_as": shown, "kind": "kev", "dump": path,
                      "rows_in_dump": sum(1 for _ in open(p, encoding="utf-8")),
                      "missing": len(population) - len(rows)})
        result["arms"][label] = block
        result["sources"][path] = sha256_file(p)
        print(f"  {shown:<32} n={block['n']} served {fmt(block['served']['accuracy'])} "
              f"abstain-pick {fmt(block['abstain_pick']['rate'])} forced {fmt(block['forced']['accuracy'])} "
              f"hidden {block['hidden_correct']} identity "
              f"{'ok' if block['identity']['holds'] else 'DIFFERS'}")

    for label, shown, path in LAYA_ARMS:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            print(f"  MISSING {path} -- {label} is not measured")
            result["arms"][label] = {"shown_as": shown, "reconcile": path, "status": "not measured"}
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        key = [k for k in raw if k.startswith("census_matched_")][0]
        b = raw[key]
        served, picks, forced_ok = b["served"]["correct"], b["abstain_pick"]["picks"], b["forced"]["correct"]
        hidden = b.get("questions_gained")
        n = b["n"]
        block = {"shown_as": shown, "kind": "laya", "reconcile": str(p.relative_to(ROOT)),
                 "n": n,
                 "served": {"correct": served, "accuracy": b["served"]["accuracy"]},
                 "abstain_pick": {"picks": picks, "rate": b["abstain_pick"]["rate"]},
                 "forced": {"correct": forced_ok, "accuracy": b["forced"]["accuracy"]},
                 "hidden_correct": hidden,
                 "hidden_correct_share_of_abstentions": (hidden / picks) if picks and hidden is not None
                 else None,
                 "delta_points": 100 * ((forced_ok - served) / n) if n else None,
                 "identity": {"forced_correct_minus_served_correct": forced_ok - served,
                              "declines_times_hidden_correct_share": hidden,
                              "holds": hidden is None or (forced_ok - served) == hidden},
                 "note": "read off the settled laya reconcile (rules imported from "
                         "reports/laya-finetune-reconcile.py)"}
        result["arms"][label] = block
        result["sources"][str(p.relative_to(ROOT))] = sha256_file(p)
        print(f"  {shown:<32} n={n} served {fmt(b['served']['accuracy'])} "
              f"abstain-pick {fmt(b['abstain_pick']['rate'])} forced {fmt(b['forced']['accuracy'])} "
              f"hidden {hidden} identity {'ok' if block['identity']['holds'] else 'DIFFERS'}")

    measured = [a for a in result["arms"].values() if "served" in a]
    result["checks"]["identity"] = {
        "what": "forced_correct - served_correct == #{declined and forced-correct}, per arm",
        "n_arms": len(measured),
        "n_violating": sum(1 for a in measured if not a["identity"]["holds"]),
    }
    # A population-wide shape fact the readings hang on: how many rows were served without an abstain
    # option at all, and what the dumps record as the source of the abstain option they did carry.
    kev = [a for a in measured if a["kind"] == "kev"]
    if kev:
        result["checks"]["serving_convention"] = {
            "rows_whose_serving_offered_no_abstain_option": sorted(
                {a["rows_whose_serving_offered_no_abstain_option"] for a in kev}),
            "note": "0 means every row of the population was served with an abstain option",
        }
    print("\nidentity (forced_correct - served_correct == hidden_correct): "
          f"{result['checks']['identity']['n_arms']} arms checked, "
          f"{result['checks']['identity']['n_violating']} violate it")

    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_md(result)
    print(f"\nwrote {OUT_MD.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    return 0


def write_md(result: dict) -> None:
    lines = [
        "# The v4 census-matched population under the three readings",
        "",
        "*Machine-readable companion: `reports/abstain-reconciliation-v4.json`.*",
        "",
        f"Population: the `panel_choice` rows of `{result['population']['source']}` — "
        f"**{result['population']['rows']:,} rows**, one row of the v4 eval suite each. The readings are "
        "`reports/abstain-reconciliation.py`'s own: **served** `argmax(p)`; **abstain-pick** the served "
        "pick is the abstain option (the model declined); **forced** `argmax(p)` with the abstain option "
        "removed. Every arm's fine-tuned / untuned pair is on the same population, the same suite and the "
        "same T = 1 rows, so the halves are comparable cell for cell.",
        "",
        "Not in this table: `panel_choice` numbers from any other lane (Jev, the general-LLM baseline) — "
        "they answer a different prompt and are not on this population. The v2/v3 companion "
        "(`reports/abstain-reconciliation.{md,json}`) is a different population and is untouched.",
        "",
        "## The readings, per arm",
        "",
        "| arm | class | n | served | abstain-pick | forced | forced − served (pt) | declines that were actually right | identity |",
        "|---|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for label, arm in result["arms"].items():
        if "served" not in arm:
            lines.append(f"| {arm['shown_as']} | {arm['kind']} | — | — | — | — | — | — | *not measured* |")
            continue
        hidden = arm["hidden_correct"]
        share = arm["hidden_correct_share_of_abstentions"]
        hidden_cell = (f"{hidden} ({fmt(share)})" if hidden is not None else "—")
        of_which = arm["abstain_pick"].get("rate_among_rows_offering_it")
        pick_cell = fmt(arm["abstain_pick"]["rate"]) + (f" ({fmt(of_which)} of the "
                                                       f"{arm['abstain_pick'].get('rows_offering_an_abstain_option')} "
                                                       f"rows offering one)" if of_which is not None else "")
        lines.append(
            f"| {arm['shown_as']} | {arm['kind']} | {arm['n']:,} | "
            f"**{fmt(arm['served']['accuracy'])}** | {pick_cell} | "
            f"**{fmt(arm['forced']['accuracy'])}** | {arm['delta_points']:+.2f} | {hidden_cell} | "
            f"{'ok' if arm['identity']['holds'] else '**DIFFERS**'} |")

    lines += ["", "## The same rows as counts", "",
              "| arm | served correct | abstain picks | forced correct | forced − served | declines × hidden share |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, arm in result["arms"].items():
        if "served" not in arm:
            continue
        ident = arm["identity"]
        lines.append(f"| {arm['shown_as']} | {arm['served']['correct']:,} | {arm['abstain_pick']['picks']:,} | "
                     f"{arm['forced']['correct']:,} | {ident['forced_correct_minus_served_correct']:,} | "
                     f"{ident['declines_times_hidden_correct_share']:,} |")

    lines += ["", "## The pairs: fine-tune minus its own rung, under each reading", "",
              "| size | served (pt) | abstain-pick (pt) | forced (pt) |", "|---|---:|---:|---:|"]
    for size in ("0.5b", "0.6b", "0.8b", "4b", "9b"):
        arm = result["arms"].get(f"delta-kev-{size}-v4-s1")
        rung = result["arms"].get(f"kev-{size}-v4")
        if not arm or "served" not in arm or not rung or "served" not in rung:
            continue
        lines.append(
            f"| {size} | {100 * (arm['served']['accuracy'] - rung['served']['accuracy']):+.2f} | "
            f"{100 * (arm['abstain_pick']['rate'] - rung['abstain_pick']['rate']):+.2f} | "
            f"{100 * (arm['forced']['accuracy'] - rung['forced']['accuracy']):+.2f} |")

    ident = result["checks"]["identity"]
    lines += ["", "## Checks", "",
              f"* **The identity holds on {ident['n_arms'] - ident['n_violating']} of {ident['n_arms']} "
              f"arms** (`forced_correct − served_correct == #{'{'}declined ∧ forced-correct{'}'}`); "
              f"{ident['n_violating']} violate it.",
              "* Every cell is a re-aggregation of dumps already on disk; no model was run for this report.",
              "* A dash means **not measured** — never a zero and never an estimate.", "",
              "## Sources", "", "| file | sha256 |", "|---|---|"]
    for path, digest in sorted(result["sources"].items()):
        lines.append(f"| `{path}` | `{digest[:16]}…` |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
