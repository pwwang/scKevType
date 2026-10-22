#!/usr/bin/env python3
"""The v4 census's other two choice families, arm for arm: `evidence_diluted` and `panel_miss`.

`reports/abstain-reconciliation-v4.py` is the settled table for the `panel_choice` half of the v4
census. This is its sibling for the two families the v4 family fill scored afterwards
(`reports/v4-family-fill.md` §1), and it adds nothing to what the readings mean: the rules --
**served** `argmax(row["p"])` over every option the question offered, **abstain-pick** the served
pick *is* the abstain option, **forced** `argmax` with the abstain option removed -- are imported by
path from that file's own ``cell()``/``read_jsonl()``/``fmt()``, so a family cell and a
``panel_choice`` cell cannot drift apart. What is new here is only the population and the arm list:

* the ids come from ``--ids`` (default ``reports/census-suite-ids-v4.jsonl``) filtered to
  ``--family``; the population count is derived from that file, never typed in;
* the KeV dumps are the ``panel_choice`` dump names with the family as a trailing tag, exactly the
  convention ``<cluster-job>`` chose
  (``reports/forced-rows-<slug>-<family>.jsonl``). The arm list therefore has one source -- the
  settled script's own ``KEV_ARMS`` -- with the tag inserted, so an arm cannot be added to one
  family table and forgotten in the other;
* the Laya halves are read off their own family reconciles
  (``runs/laya-v4-census-<spec|base>-<family>/reconcile.json``), the lane that already scores Laya
  on ``panel_choice``: same three readings, the rules imported from
  ``reports/laya-finetune-reconcile.py`` by ``reports/laya-v4-reconcile.py``. Each of those cells is
  **recomputed here from that lane's own row dump** (``.../rows/*.jsonl``, ``ask == "base"``, the
  abstain option located by ``sckevtype.eval.abstain_index``) through the settled ``cell()`` and
  compared against the reconcile before it is written down, so a Laya number in this table is one
  this script has reproduced from bytes; a mismatch is a refusal, not a quieter row.

**The abstain-pick denominator is quoted, not assumed.** These two families differ from
``panel_choice`` in how the abstain option was served. On ``panel_choice`` 2,992 of the 4,035
census rows were served with no abstain option at all (``abstain: null``, ``abstain_source:
injected``) and a decline is not an event there; on ``evidence_diluted`` and ``panel_miss`` every
row of every dump checked here carries one (``abstain_source: factory``). So every abstain-pick rate
is reported over the rows that actually offered the option *and* over the population, with the
denominator printed beside it.

**Two family facts are stated here rather than left to be inferred.** (1) On ``panel_miss`` the
abstain option *is* the correct answer -- the marker is missing from the panel, so declining is what
a right answer looks like -- which makes served accuracy and the abstain-pick rate the same number
for every arm and the forced reading 0 by construction; the row count that shows it is printed. (2)
On ``evidence_diluted`` the abstain option is the *labelled* answer on part of the family (1,338 of
4,035 rows; the count is read from the dumps, never typed here). The settled identity
``forced_correct - served_correct == #{declined and forced-correct}`` assumes a decline is never
right -- true on ``panel_choice`` and stated as such in ``reports/laya-finetune-reconcile.py``, false
here -- so this table reports that identity twice: over every row, and over the rows whose label is
not the abstain option, where it must hold.

Refused here, by name:

* ``panel_choice`` -- the settled table owns it (``reports/abstain-reconciliation-v4.py``);
* a family the ids file does not carry, or a repeated id;
* a dump whose ``.meta.json`` names another family or another population size, or whose recorded
  ``rows.sha256`` is not the file's own digest -- a copy of the wrong family's dump is the failure
  mode this catches;
* a dump that covers fewer than the population's rows (``--allow-missing`` reports the partial
  population instead of failing);
* a Laya reconcile whose own row dump does not reproduce its cell.

Writes new files and overwrites nothing:

    reports/family-reconciliation-v4-<family>.md
    reports/family-reconciliation-v4-<family>.json

    # on the cluster, where the dumps and the Laya runs are
    W=/path/to/sckevtype
    uv run python reports/family-reconciliation-v4.py --family evidence_diluted
    uv run python reports/family-reconciliation-v4.py --family panel_miss
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sckevtype.eval import abstain_index                      # noqa: E402

SETTLED = ROOT / "analysis" / "abstain-reconciliation-v4.py"
DEFAULT_IDS = "reports/census-suite-ids-v4.jsonl"
PANEL_CHOICE = "panel_choice"
# (arm label, shown as, the half's slug in runs/laya-v4-census-<slug>-<family>)
LAYA_HALVES: tuple[tuple[str, str, str], ...] = (
    ("laya-v4-ft-spec", "Laya fine-tuned (spec)", "spec"),
    ("laya-v4-base", "Laya base (unfitted recipe)", "base"),
)


def _settled():
    """``reports/abstain-reconciliation-v4.py`` as a module -- the readings, not a copy of them."""
    spec = importlib.util.spec_from_file_location("abstain_reconciliation_v4", SETTLED)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the settled readings from {SETTLED}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("cell", "read_jsonl", "sha256_file", "fmt", "KEV_ARMS"):
        if not hasattr(module, name):
            raise SystemExit(f"{SETTLED} no longer defines {name}; this table would have to "
                             f"re-derive the readings -- re-read the settled file before continuing")
    return module


def family_dump_path(panel_choice_path: str, family: str) -> str:
    """The family dump name, from the settled script's own panel_choice path (the tag convention)."""
    if not panel_choice_path.endswith(".jsonl"):
        raise SystemExit(f"{panel_choice_path} is not a .jsonl dump path")
    return panel_choice_path[: -len(".jsonl")] + f"-{family}.jsonl"


def refuse(reason: str) -> int:
    print(f"REFUSED: {reason}")
    return 2


def laya_served_rows(rows_dir: Path, wanted: set[str]) -> list[dict] | None:
    """The served question of each population row, from a Laya lane's own row dump (or None).

    ``ask == "base"`` is the served question (that lane's reconcile filters the same way). The
    abstain option is located with ``sckevtype.eval.abstain_index`` -- the helper that lane itself
    reconciles with -- and carried as ``abstain`` so the settled KeV ``cell()`` can count the rows,
    instead of a second counting rule having to exist for Laya.
    """
    files = sorted(glob.glob(str(rows_dir / "*.jsonl")))
    if not files:
        return None
    kept: dict[str, dict] = {}
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("ask") != "base" or row.get("id") not in wanted:
                    continue
                kept[row["id"]] = dict(row, abstain=abstain_index(row), keys=list(row["keys"]),
                                       p=list(row["p"]))
    return [kept[key] for key in sorted(kept)]


def identity_without_abstain_labels(cell_fn, rows: list[dict]) -> dict:
    """The settled identity, re-run on the rows where a decline cannot be the right answer."""
    subset = [row for row in rows if row["label"] != row.get("abstain")]
    block = cell_fn(subset)
    return {"n": block["n"], "served": block["served"], "forced": block["forced"],
            "hidden_correct": block["hidden_correct"], "identity": block["identity"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--family", required=True,
                        help="evidence_diluted or panel_miss (panel_choice is the settled table's)")
    parser.add_argument("--ids", default=DEFAULT_IDS)
    parser.add_argument("--allow-missing", action="store_true",
                        help="report a partial population instead of failing on the coverage")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)

    if args.family == PANEL_CHOICE:
        return refuse("panel_choice is not this table's family -- the settled readings for it are "
                      "reports/abstain-reconciliation-v4.py's, and re-tabulating it here would "
                      "give two sources for one population")

    settled = _settled()
    read_jsonl, cell, sha256_file, fmt = (settled.read_jsonl, settled.cell, settled.sha256_file,
                                          settled.fmt)

    ids_path = Path(args.ids)
    if not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    if not ids_path.is_file():
        return refuse(f"no population file at {ids_path}")
    census = read_jsonl(ids_path)
    population = [row for row in census if row.get("family") == args.family]
    if not population:
        return refuse(f"{ids_path} carries no {args.family!r} rows")
    wanted = {row["id"] for row in population}
    if len(wanted) != len(population):
        return refuse(f"{ids_path}: {len(population) - len(wanted)} repeated ids in "
                      f"{args.family} -- a repeat would be counted twice")
    print(f"population: {len(population)} {args.family} rows from {ids_path}")

    try:
        family_key = str(ids_path.relative_to(ROOT))
    except ValueError:
        family_key = str(ids_path)
    ids_digest = sha256_file(ids_path)
    sources: dict[str, str] = {family_key: ids_digest}
    result: dict = {
        "what_this_is": f"The v4 census's {args.family} population under the settled served / "
                        f"abstain-pick / forced readings, for every arm the family fill scored. "
                        f"Re-aggregation only: no model was run. The readings are "
                        f"reports/abstain-reconciliation-v4.py's own cell() (imported, not "
                        f"re-implemented); the Laya halves are their own lane's family reconciles, "
                        f"recomputed here from that lane's row dump.",
        "rules_from": {"path": "reports/abstain-reconciliation-v4.py",
                       "sha256": sha256_file(SETTLED)},
        "family": args.family,
        "population": {"source": family_key, "family": args.family, "rows": len(population),
                       "distinct_ids": len(wanted), "sha256": ids_digest},
        "arms": {}, "sources": {}, "best_worst": {}, "checks": {},
    }

    labels_that_are_the_abstain_option: set[int] = set()

    # ------------------------------------------------------------------ the KeV arms
    for label, shown, pc_path in settled.KEV_ARMS:
        path = family_dump_path(pc_path, args.family)
        p = ROOT / path
        if not p.is_file():
            print(f"  MISSING {path} -- {shown} is not measured on {args.family}")
            result["arms"][label] = {"shown_as": shown, "kind": "kev", "dump": path,
                                     "status": "not measured"}
            continue
        meta_path = Path(str(p) + ".meta.json")
        meta = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("ids", {}).get("family") != args.family:
                return refuse(f"{meta_path} says it scored family "
                              f"{meta['ids'].get('family')!r}, not {args.family!r} -- this is "
                              f"another family's dump under this name")
            if meta.get("rows", {}).get("written") != len(population):
                return refuse(f"{meta_path} says {meta['rows'].get('written')} rows were written, "
                              f"the population is {len(population)} -- not this family's dump")
            digest = sha256_file(p)
            if meta.get("rows", {}).get("sha256") not in (None, digest):
                return refuse(f"{p}'s meta.json records rows.sha256 "
                              f"{str(meta['rows']['sha256'])[:16]}... but the file's own digest is "
                              f"{digest[:16]}... -- the bytes are not the ones that were scored")
            sources[str(meta_path.relative_to(ROOT))] = sha256_file(meta_path)

        rows = [row for row in read_jsonl(p) if row.get("id") in wanted]
        families_in_dump = sorted({row.get("family") for row in rows})
        if families_in_dump != [args.family]:
            return refuse(f"{path} holds rows of {families_in_dump}, not [{args.family!r}]")
        missing = len(population) - len(rows)
        if missing and not args.allow_missing:
            return refuse(f"{path} covers {len(rows)} of {len(population)} {args.family} rows "
                          f"({missing} missing); pass --allow-missing to report a partial population")
        block = cell(rows)
        without_abstain_labels = identity_without_abstain_labels(cell, rows)
        label_is_abstain = sum(1 for row in rows if row["label"] == row.get("abstain"))
        labels_that_are_the_abstain_option.add(label_is_abstain)
        block.update({
            "shown_as": shown, "kind": "kev", "dump": path,
            "rows_in_dump": sum(1 for _ in open(p, encoding="utf-8")),
            "missing": missing, "meta": str(meta_path.relative_to(ROOT)) if meta else None,
            "abstain_source_in_dump": sorted({str(row.get("abstain_source")) for row in rows}),
            "rows_whose_label_is_the_abstain_option": label_is_abstain,
            "identity_where_the_label_is_not_the_abstain_option":
                without_abstain_labels["identity"],
            "population_where_the_label_is_not_the_abstain_option": {
                k: without_abstain_labels[k] for k in ("n", "served", "forced", "hidden_correct")},
        })
        result["arms"][label] = block
        sources[path] = sha256_file(p)
        pick = block["abstain_pick"]
        print(f"  {shown:<26} n={block['n']:<5} served {fmt(block['served']['accuracy'])} "
              f"abstain-pick {fmt(pick['rate'])} of {block['n']} "
              f"({fmt(pick['rate_among_rows_offering_it'])} of the "
              f"{pick['rows_offering_an_abstain_option']} offering one) "
              f"forced {fmt(block['forced']['accuracy'])} identity "
              f"{'ok' if block['identity']['holds'] else 'DIFFERS'} "
              f"(label-not-abstain: "
              f"{'ok' if without_abstain_labels['identity']['holds'] else 'DIFFERS'})")

    # ------------------------------------------------------------------ the Laya halves
    for label, shown, slug in LAYA_HALVES:
        path = f"runs/laya-v4-census-{slug}-{args.family}/reconcile.json"
        p = ROOT / path
        if not p.is_file():
            print(f"  MISSING {path} -- {shown} is not measured on {args.family}")
            result["arms"][label] = {"shown_as": shown, "kind": "laya", "reconcile": path,
                                     "status": "not measured"}
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("population", {}).get("family") != args.family:
            return refuse(f"{path} was reconciled onto {raw.get('population', {}).get('family')!r}, "
                          f"not {args.family!r}")
        keys = [k for k in raw if k.startswith("census_matched_")]
        if len(keys) != 1:
            return refuse(f"{path} has {len(keys)} census_matched_* cells; expected exactly one")
        b = raw[keys[0]]
        n, coverage = b["n"], raw.get("coverage", {})
        if coverage.get("missing") and not args.allow_missing:
            return refuse(f"{path} is missing {coverage['missing']} of the population's rows; "
                          f"pass --allow-missing to report it")
        if n != len(population) - int(coverage.get("missing", 0)):
            return refuse(f"{path}: cell n={n} does not match the population ({len(population)} "
                          f"rows, {coverage.get('missing')} missing)")
        served, picks, forced_ok = (b["served"]["correct"], b["abstain_pick"]["picks"],
                                    b["forced"]["correct"])

        # ---- reproduce this cell from the lane's own rows, through the settled counting rule
        rows_dir = p.parent / "rows"
        row_files = sorted(glob.glob(str(rows_dir / "*.jsonl")))
        rows = laya_served_rows(rows_dir, wanted) if row_files else None
        if row_files:
            if rows is None or len(rows) != n:
                return refuse(f"{rows_dir} holds {0 if rows is None else len(rows)} served rows for "
                              f"this population, the reconcile reports n={n} -- the cell cannot be "
                              f"reproduced from bytes, so it does not go in this table")
            row_cell = cell(rows)
            got = (row_cell["served"]["correct"], row_cell["abstain_pick"]["picks"],
                   row_cell["forced"]["correct"])
            if got != (served, picks, forced_ok):
                return refuse(f"{path}'s cell (served {served}, abstain-picks {picks}, forced "
                              f"{forced_ok}) is not what its own rows give ({got}) -- the reconcile "
                              f"and its rows disagree, so neither number goes in this table")
            for row_file in row_files:
                sources[str(Path(row_file).relative_to(ROOT))] = sha256_file(Path(row_file))
            without_abstain_labels = identity_without_abstain_labels(cell, rows)
            label_is_abstain = sum(1 for row in rows if row["label"] == row.get("abstain"))
            labels_that_are_the_abstain_option.add(label_is_abstain)
        else:
            row_cell, without_abstain_labels, label_is_abstain = None, None, None

        offering = (row_cell["abstain_pick"]["rows_offering_an_abstain_option"] if row_cell
                    else (None if coverage.get("rows_without_an_abstain_option") is None
                          else n - coverage["rows_without_an_abstain_option"]))
        hidden_from_share = b.get("hidden_correct_share_of_abstentions")
        block = {
            "shown_as": shown, "kind": "laya", "reconcile": path, "n": n,
            "served": {"correct": served, "accuracy": b["served"]["accuracy"]},
            "abstain_pick": {"picks": picks, "rate": picks / n if n else None,
                             "rows_offering_an_abstain_option": offering,
                             "rate_among_rows_offering_it": (picks / offering) if offering else None},
            "forced": {"correct": forced_ok, "accuracy": b["forced"]["accuracy"]},
            "hidden_correct": (b.get("questions_gained") if hidden_from_share is None
                               else hidden_from_share * picks),
            "delta_points": 100 * ((forced_ok - served) / n) if n else None,
            "identity": {"forced_correct_minus_served_correct": forced_ok - served,
                         "declines_times_hidden_correct_share":
                             None if hidden_from_share is None else hidden_from_share * picks,
                         "holds": (hidden_from_share is None
                                   or (forced_ok - served) == hidden_from_share * picks)},
            "rows_whose_serving_offered_no_abstain_option":
                coverage.get("rows_without_an_abstain_option"),
            "cell_reproduced_from_the_lane_rows": bool(row_cell),
            "reconcile_rows_source": raw.get("rows_source"),
            "note": "read off this half's own family reconcile "
                    "(runs/laya-v4-census-<half>-<family>/reconcile.json); its rules are imported "
                    "from reports/laya-finetune-reconcile.py by reports/laya-v4-reconcile.py",
        }
        if row_cell is not None:
            block["reproduced_cell"] = {
                k: row_cell[k] for k in ("n", "served", "abstain_pick", "forced", "hidden_correct")}
            block["rows_whose_label_is_the_abstain_option"] = label_is_abstain
            block["identity_where_the_label_is_not_the_abstain_option"] = \
                without_abstain_labels["identity"]
            block["population_where_the_label_is_not_the_abstain_option"] = {
                k: without_abstain_labels[k] for k in ("n", "served", "forced", "hidden_correct")}
        result["arms"][label] = block
        sources[path] = sha256_file(p)
        print(f"  {shown:<26} n={block['n']:<5} served {fmt(block['served']['accuracy'])} "
              f"abstain-pick {fmt(block['abstain_pick']['rate'])} of {n} "
              f"({fmt(block['abstain_pick']['rate_among_rows_offering_it'])} of the "
              f"{offering} offering one) forced {fmt(block['forced']['accuracy'])}"
              + (f" cells_from_rows ok identity {'ok' if block['identity']['holds'] else 'DIFFERS'} "
                 f"(label-not-abstain: "
                 f"{'ok' if without_abstain_labels['identity']['holds'] else 'DIFFERS'})"
                 if row_cell else " cells_from_rows NOT CHECKED"))

    # ------------------------------------------------------------------ checks
    measured = {k: v for k, v in result["arms"].items() if "served" in v}
    kev_measured = {k: v for k, v in measured.items() if v["kind"] == "kev"}
    if measured:
        result["checks"]["identity"] = {
            "what": "forced_correct - served_correct == #{declined and forced-correct}, per arm",
            "n_arms": len(measured),
            "n_violating": sum(1 for a in measured.values() if not a["identity"]["holds"]),
        }
        restricted = [a for a in measured.values()
                      if a.get("identity_where_the_label_is_not_the_abstain_option")]
        non_empty = [a for a in restricted
                     if a.get("population_where_the_label_is_not_the_abstain_option", {}).get("n", 0) > 0]
        result["checks"]["identity_where_a_decline_cannot_be_right"] = {
            "what": "the same identity, on the rows whose label is not the abstain option",
            "n_arms": len(non_empty),
            "n_arms_with_an_empty_subpopulation": len(restricted) - len(non_empty),
            "n_violating": sum(1 for a in non_empty
                               if not a["identity_where_the_label_is_not_the_abstain_option"]["holds"]),
            "note": "an arm whose subpopulation is empty is counted here as empty, never as a pass -- "
                    "the identity would hold vacuously on no rows at all",
        }
        if len(labels_that_are_the_abstain_option) != 1:
            return refuse(f"the measured arms disagree on how many rows have the abstain option as "
                          f"their label ({sorted(labels_that_are_the_abstain_option)}); the arms "
                          f"would not be answering the same questions, so this table would compare "
                          f"different things")
        count = labels_that_are_the_abstain_option.pop()
        result["checks"]["the_label_is_the_abstain_option_on"] = {
            "rows": count, "share": count / len(population), "agrees_across_all_measured_arms": True,
            "note": "where this is non-zero the settled identity cannot hold as stated (a decline "
                    "can be a right answer) and the forced reading is not a capability reading",
        }
        result["checks"]["serving"] = {
            "rows": sorted({a["n"] for a in measured.values()}),
            "rows_without_an_abstain_option": sorted(
                {a["rows_whose_serving_offered_no_abstain_option"] for a in measured.values()}),
            "abstain_source_in_dump": sorted({s for a in kev_measured.values()
                                              for s in a.get("abstain_source_in_dump", [])}),
            "note": "0 rows without an abstain option means this family's abstain-pick denominator "
                    "is the whole population; the column above is quoted both ways regardless",
        }
    if kev_measured:
        order = sorted(kev_measured, key=lambda k: -kev_measured[k]["served"]["accuracy"])
        best, worst = order[0], order[-1]
        result["best_worst"] = {"by": "served accuracy among the KeV arms",
                                "best": {"arm": best, "shown_as": kev_measured[best]["shown_as"],
                                         "n": kev_measured[best]["n"],
                                         "served": kev_measured[best]["served"]["accuracy"],
                                         "abstain_pick": kev_measured[best]["abstain_pick"]},
                                "worst": {"arm": worst, "shown_as": kev_measured[worst]["shown_as"],
                                          "n": kev_measured[worst]["n"],
                                          "served": kev_measured[worst]["served"]["accuracy"],
                                          "abstain_pick": kev_measured[worst]["abstain_pick"]},
                                "order": order}
        print(f"\nKeV served-accuracy order ({args.family}): " + " > ".join(
            f"{kev_measured[k]['shown_as']} {fmt(kev_measured[k]['served']['accuracy'])}"
            for k in order))
    result["sources"] = dict(sorted(sources.items()))

    out_md = Path(args.out_md) if args.out_md else ROOT / "reports" / \
        f"family-reconciliation-v4-{args.family}.md"
    out_json = Path(args.out_json) if args.out_json else ROOT / "reports" / \
        f"family-reconciliation-v4-{args.family}.json"
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    write_md(result, out_md, fmt)
    print(f"\nwrote {out_md} and {out_json}")
    return 0


def write_md(result: dict, path: Path, fmt) -> None:
    family = result["family"]
    pop = result["population"]
    checks = result["checks"]
    label_abstain = checks.get("the_label_is_the_abstain_option_on", {})
    restricted = checks.get("identity_where_a_decline_cannot_be_right", {})
    lines = [
        f"# The v4 census's `{family}` family under the three readings",
        "",
        f"*Machine-readable companion: `{path.with_suffix('.json').name}`.*",
        "",
        f"Population: the `{family}` rows of `{pop['source']}` -- **{pop['rows']:,} rows** "
        f"({pop['distinct_ids']:,} distinct ids), one row of the v4 eval suite each, scored by the "
        f"v4 family fill (`reports/v4-family-fill.md`). The readings are `reports/"
        f"abstain-reconciliation-v4.py`'s own `cell()` -- imported by path, not re-implemented: "
        "**served** `argmax(p)` over every option the question offered; **abstain-pick** the served "
        "pick *is* the abstain option; **forced** `argmax(p)` with the abstain option removed.",
        "",
        "The two lanes this population carries are the same two the `panel_choice` table has: the "
        f"KeV arms, re-scored onto this family's rows (`reports/forced-rows-<slug>-{family}.jsonl`), "
        f"and Laya's two halves, whose scoring lane renders an item tree and reconciles onto the "
        f"family's ids (`runs/laya-v4-census-<spec|base>-{family}/reconcile.json`). Each Laya row "
        "here was **recomputed from that lane's own row dump** through the same `cell()` and "
        "compared against its reconcile before it was written down.",
        "",
        f"## The readings, per arm ({pop['rows']:,} {family} rows each)",
        "",
        "| arm | class | n | served | abstain-pick (of the rows offering one) | forced | "
        "forced − served (pt) | identity: all rows / label≠abstain |",
        "|---|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for label, arm in result["arms"].items():
        if "served" not in arm:
            lines.append(f"| {arm['shown_as']} | {arm['kind']} | — | — | — | — | — | "
                         "*not measured* |")
            continue
        pick = arm["abstain_pick"]
        cell_txt = fmt(pick["rate"])
        if pick.get("rows_offering_an_abstain_option") is not None:
            cell_txt += (f" ({fmt(pick.get('rate_among_rows_offering_it'))} of the "
                         f"{pick['rows_offering_an_abstain_option']} offering one)")
        ident = "ok" if arm["identity"]["holds"] else "**DIFFERS**"
        other = arm.get("identity_where_the_label_is_not_the_abstain_option")
        empty = (arm.get("population_where_the_label_is_not_the_abstain_option", {}).get("n") == 0)
        ident += " / " + ("— (no such row)" if empty else
                          (("ok" if other["holds"] else "**DIFFERS**") if other else "—"))
        lines.append(
            f"| {arm['shown_as']} | {arm['kind']} | {arm['n']:,} | "
            f"**{fmt(arm['served']['accuracy'])}** | {cell_txt} | {fmt(arm['forced']['accuracy'])} | "
            f"{arm['delta_points']:+.2f} | {ident} |")

    bw = result.get("best_worst") or {}
    serving = checks.get("serving", {})
    lines += ["", "## The abstain-pick denominator", "",
              f"* Rows offering an abstain option, this family: **{pop['rows']:,} of "
              f"{pop['rows']:,}** (rows without one: {serving.get('rows_without_an_abstain_option')}"
              f"; the KeV dumps' own `abstain_source`: {serving.get('abstain_source_in_dump')}). "
              f"Every abstain-pick rate above is therefore the same number over the population and "
              f"over the rows that offered the option, and both are printed. On `panel_choice` the "
              f"same column is defined on 1,043 of 4,035 rows -- the denominator is quoted here "
              f"because it is *not* the same denominator.",
              "", "## Where the KeV arms land", ""]
    if bw:
        for name, block in (("Best, served", bw["best"]), ("Worst, served", bw["worst"])):
            pick = block["abstain_pick"]
            lines.append(f"* **{name}**: {block['shown_as']} {fmt(block['served'])} served, "
                         f"abstain-pick {pick['picks']:,}/{block['n']:,} ({fmt(pick['rate'])}; "
                         f"{fmt(pick.get('rate_among_rows_offering_it'))} of the "
                         f"{pick['rows_offering_an_abstain_option']:,} rows offering one).")
        lines += ["", "Served accuracy, best to worst:", "",
                  "| rank | arm | served | abstain-pick |", "|---:|---|---:|---:|"]
        for rank, key in enumerate(bw["order"], start=1):
            arm = result["arms"][key]
            lines.append(f"| {rank} | {arm['shown_as']} | {fmt(arm['served']['accuracy'])} | "
                         f"{arm['abstain_pick']['picks']:,}/{arm['n']:,} "
                         f"({fmt(arm['abstain_pick']['rate'])}) |")

    lines += ["", "## The identity column, on this family", ""]
    if label_abstain.get("rows"):
        holds = restricted.get("n_arms", 0) - restricted.get("n_violating", 0)
        empty_arms = restricted.get("n_arms_with_an_empty_subpopulation", 0)
        lines.append(
            f"* On **{label_abstain['rows']:,} of {pop['rows']:,} rows** "
            f"({label_abstain['share']:.4f}) the served question's **label is the abstain option** -- "
            f"the count agrees across every measured arm. The settled identity "
            f"`forced_correct − served_correct == #{{declined ∧ forced-correct}}` assumes a decline "
            f"is never the right answer (true on `panel_choice`, and `reports/"
            f"laya-finetune-reconcile.py` says so in a comment). Where that assumption fails the raw "
            f"identity cannot hold, so it is reported twice. On the rows whose label is not the "
            f"abstain option the identity "
            + (f"holds on **{holds} of {restricted.get('n_arms', 0)} measured arms**, which is where "
               f"the served / forced readings are the confidence-versus-evidence comparison the "
               f"`panel_choice` table makes."
               if restricted.get("n_arms", 0) else
               f"is reported as **no such row on any of the {empty_arms} measured arms** -- nothing "
               f"is claimed from an empty subpopulation, and the raw column above is the only one "
               f"that exists on this family."))
        if empty_arms and restricted.get("n_arms", 0):
            lines.append(f"* {empty_arms} measured arm(s) have no such row at all (their subpopulation "
                         f"is empty) and are counted as empty, never as a pass.")
    else:
        lines.append("* The label is never the abstain option on this family, so the settled "
                     "identity is the whole story: it holds on every measured arm.")
    if family == "panel_miss":
        lines += ["", "**On this family that count is the whole population, by construction.** The "
                      "marker is missing from the panel, so the served question's correct answer "
                      "*is* the abstain option: an arm that declines is an arm that is right. Served "
                      "accuracy and the abstain-pick rate are therefore the same number for every "
                      "arm, and the forced column is 0 for every arm because removing the abstain "
                      "option removes the only correct answer. Read the served column here, not the "
                      "forced one."]
    ident = checks.get("identity", {})
    lines += ["", "## Checks", "",
              "* Every KeV cell is a re-aggregation of a dump already on disk; every Laya cell was "
              "reproduced from that lane's own rows before it was written here. No model was run "
              "for this table.",
              f"* Identity over every row: holds on "
              f"{ident.get('n_arms', 0) - ident.get('n_violating', 0)} of {ident.get('n_arms', 0)} "
              f"arms. Over the rows where a decline cannot be right: holds on "
              f"{restricted.get('n_arms', 0) - restricted.get('n_violating', 0)} of "
              f"{restricted.get('n_arms', 0)} measured arms"
              + (f" ({restricted['n_arms_with_an_empty_subpopulation']} arm(s) have no such row and "
                 f"are counted as empty, not as a pass)." if restricted.get("n_arms_with_an_empty_subpopulation") else "."),
              "* A dash means **not measured** -- never a zero and never an estimate.", "",
              "## Sources", "", "| file | sha256 |", "|---|---:|"]
    for src, digest in result["sources"].items():
        lines.append(f"| `{src}` | `{digest[:16]}…` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
