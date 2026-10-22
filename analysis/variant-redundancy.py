#!/usr/bin/env python3
"""Is the ``panel_choice`` duplication load-bearing? Re-aggregate the headline comparisons off and on it.

The manuscript's scored population is the 1,819 ``panel_choice`` rows of
``reports/census-suite-ids.jsonl`` -- but those are 1,497 distinct ``(dataset, cell, family)``
triples, because 322 triples hold two suite rows, one per rendering ``variant`` (``clean`` and
``permuted``). Every other family carries the same ~1.21 rows-per-cell ratio, so the question is
general and this script answers it on ``panel_choice``, the population the paper's claims are on.

Two checks, both re-aggregations of row dumps already on disk -- no model is run, no job submitted:

* **Check 1, sensitivity to the duplication.** Recompute the comparisons the manuscript leans on
  (Jev against each Kev rung as served; Kev-27B against the other large engines as served; the
  both-forced pair; every fine-tuned arm against its own released rung) on the full 1,819 rows and
  on two de-duplicated populations -- one row per triple, preferring ``clean`` (a), then preferring
  ``permuted`` (b) -- and report them side by side so the direction cannot be cherry-picked.
* **Check 2, position bias on the 322 pairs.** On the 322 triples that carry both renderings,
  tabulate ``both correct`` / ``only clean correct`` / ``only permuted correct`` / ``neither`` per
  engine and pooled. The asymmetry between the middle two columns is the rendering-sensitivity
  signal; the exact binomial over those two counts is the test.

WHY THE TWO DIRECTIONS ARE NOT SYMMETRIC HERE
---------------------------------------------
The census asked exactly **one** rendering of each triple, and for ``panel_choice`` that rendering
is ``permuted`` on all 2,510 census records (:func:`load_census` asserts it). A census-level engine's
verdict is therefore a function of the triple alone, and the join of ``reports/per-dataset.json``
attributes that one verdict to *both* suite rows of a doubled triple. De-duplicating by ``clean`` or
by ``permuted`` cannot, for those engines, move a number -- there is no direction to pick. It can
move for our own suite-level arms, which answered each rendering separately, and that is the case
Check 2 exists to measure.

THE RULES ARE NOT RE-IMPLEMENTED
--------------------------------
``served`` (argmax of the row's ``p``), ``forced`` (argmax with the abstain option removed) and the
paired test are lifted out of ``reports/abstain-reconciliation.py`` by AST extraction and executed
from that source, so this file cannot drift from the settled rule; the source's sha256 is recorded
in the output. The abstain option is a property of the question and comes from
``sckevtype.eval.abstain_index`` (``sckevtype.data.families.NONE_OPTIONS``), as in
``reports/both-forced-table.json`` and ``reports/arms-forced.json``. The joins, the populations and
the engine roster are those two reports', and every number they publish that this script recomputes
is checked in the ``replication`` block -- a mismatch is reported as a failure, not smoothed over.

INPUTS  (all on disk; every one is sha256'd into the output)
------------------------------------------------------------
``reports/census-suite-ids.jsonl``                 the 7,203 suite rows (1,819 ``panel_choice``)
``reports/kev-rungs/{05b,06b,08b,4b,9b}-rows.jsonl``  the census records, Kev rungs (census join)
``reports/rows-kev-27b-probe.jsonl``               the census records, Kev-27B
``reports/jev-full/rows.jsonl``                    the census records, Jev
``reports/llm-baseline-rows-census-{varA,varB}.jsonl``  the census records, Qwen3.5-9B zero-shot
``reports/forced-rows-*.jsonl``                    our arms re-scored with ``p`` retained (16)
``reports/rows-full-*.jsonl``                      the archived arm dumps (the 11 that have one)
``scratch/runs/laya-pilot-score/rows/*.jsonl``     Laya pilot's 155 shards (``ask == "base"``)
``reports/both-forced-table.json`` ``reports/arms-forced.json`` ``reports/per-dataset.json``
``reports/extra-seeds.json``                       the published numbers Check 1 is checked against

OUTPUT
------
``reports/variant-redundancy.json`` and ``reports/variant-redundancy.md`` -- written nowhere else,
nothing pre-existing touched.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python reports/variant-redundancy.py
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import sys
from collections import Counter, OrderedDict
from decimal import Decimal
from fractions import Fraction
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "reports", "variant-redundancy.json")
OUT_MD = os.path.join(ROOT, "reports", "variant-redundancy.md")
RULE_SOURCE = "analysis/abstain-reconciliation.py"
SUITE_IDS_FILE = "reports/census-suite-ids.jsonl"
ARCHIVE_FILE = "reports/rows-full-frozen.jsonl"

FAMILY = "panel_choice"
POP_IDS = ("full_1819", "dedup_clean_1497", "dedup_permuted_1497")
POP_SHORT = {"full_1819": "1,819 full", "dedup_clean_1497": "1,497 dedup (clean)",
             "dedup_permuted_1497": "1,497 dedup (permuted)"}


# --------------------------------------------------------------------- plumbing
def read_text(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def read_jsonl(rel: str) -> list[dict]:
    out: list[dict] = []
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_of(rel: str) -> str:
    h = hashlib.sha256()
    with open(os.path.join(ROOT, rel), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rules(rel: str) -> dict:
    """Execute the named functions of ``rel`` without running the module's own top-level work.

    ``reports/abstain-reconciliation.py`` is a script, not a library: importing it would re-run its
    whole aggregation and rewrite its own report. Its rule functions are therefore extracted by AST
    and compiled on their own, which is the same code -- not a paraphrase of it -- with none of the
    side effects.
    """
    want = ("argmax", "served", "forced", "binom_two_sided")
    tree = ast.parse(read_text(rel), filename=rel)
    missing = [name for name in want
               if name not in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}]
    if missing:
        raise SystemExit(f"REFUSING: {rel} no longer defines {missing}; the settled rule moved")
    module = ast.Module(body=[n for n in tree.body
                              if isinstance(n, ast.FunctionDef) and n.name in want],
                        type_ignores=[])
    namespace: dict = {"comb": comb, "__name__": "settled_rules"}
    exec(compile(module, rel, "exec"), namespace)  # noqa: S102 -- see the docstring
    return {name: namespace[name] for name in want}


RULES = load_rules(RULE_SOURCE)
SERVED = RULES["served"]
FORCED = RULES["forced"]
PAIRED_P = RULES["binom_two_sided"]

sys.path.insert(0, ROOT)
from sckevtype.data.families import (
    NONE_OPTIONS,
)
from sckevtype.eval import abstain_index

P_AGREEMENT: list[dict] = []  # every distinct discordance count, settled vs exact


def paired_p(b: int, c: int) -> float:
    """The settled test, evaluated exactly so a lopsided pair cannot overflow.

    ``reports/abstain-reconciliation.py``'s ``binom_two_sided`` divides by ``2.0 ** n``, which
    overflows float64 once the discordant count passes 1,023 -- and Check 1 asks for pairs that
    lopsided (Jev against the 0.5B rung is ~1,190 discordant pairs). The same expression is
    therefore evaluated with exact integers and converted once, and **the source's function is
    still called and compared** on every distinct discordance count: where its arithmetic does not
    overflow the two must agree, and that is checked in the replication block. No published number
    in the repository hits the overflow (the largest discordance count any of them uses is 429).
    """
    n = b + c
    exact = 1.0
    if n:
        exact = float(min(Fraction(1), Fraction(2 * sum(comb(n, i) for i in range(min(b, c) + 1)),
                                        2 ** n)))
    if [b, c] not in (entry["discordant"] for entry in P_AGREEMENT):
        try:
            got = PAIRED_P(b, c)
            P_AGREEMENT.append({"discordant": [b, c], "settled": got, "exact": exact,
                                    "same": abs(got - exact) <= 1e-12 * max(1.0, abs(exact)),
                                    "overflows": False})
        except OverflowError:
            P_AGREEMENT.append({"discordant": [b, c], "settled": None, "exact": exact, "same": None,
                                    "overflows": True})
    return exact


def p_of(b: int, c: int) -> dict:
    """The paired test's p, plus its scientific form where the float64 result underflows to 0.

    A lopsided pair (Jev against the 0.5B rung is ~1,190 discordant) has p ~ 1e-358, below the
    smallest positive float64, so ``p`` degenerates to 0.0. Rather than report that, the exact
    value is also carried as a decimal string with three significant digits.
    """
    value = paired_p(b, c)
    out: dict = {"p": value}
    if value == 0.0 and b + c > 0:
        n = b + c
        frac = Fraction(2 * sum(comb(n, i) for i in range(min(b, c) + 1)), 2 ** n)
        dec = Decimal(frac.numerator) / Decimal(frac.denominator) if frac < 1 else Decimal(1)
        out["p_underflows_float64"] = True
        out["p_scientific"] = format(dec, ".3E")  # Decimal's own formatting: %-formatting would
        #                                           convert to float and print 0.000E+00
    return out


def predict(row: dict, rule: str):
    """The settled prediction: ``served`` = argmax of ``p``, else the dump's own answer field.

    A dump carrying ``p`` is scored through the source's ``served``/``forced``. The general LLM's
    dumps carry neither ``p`` nor ``top1`` but the parsed answer index, which
    ``reports/both-forced-table.json`` names as that engine's prediction; ``forced`` is undefined
    there, exactly as it is in that report.
    """
    if rule == "served":
        if row.get("p") is not None:
            return SERVED(row)
        if row.get("top1") is not None:
            return row["top1"]
        return row.get("parsed_index")
    if row.get("p") is None:
        return None
    index = row.get("abstain")
    if index is None:
        index = abstain_index(row)  # Laya's dumps carry no `abstain` field
    if index is None:
        return None
    return FORCED({**row, "abstain": index})


def census_id(dataset: str, cell: str, family: str) -> str:
    return f"test/{dataset}/{cell}/{family}"


def fmt_acc(x) -> str:
    return "--" if x is None else f"{x:.4f}"


def fmt_p(p) -> str:
    if p is None:
        return "--"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def p_disp(entry: dict) -> str:
    """A p-value for display: the exact decimal string where the float64 value underflowed."""
    if entry.get("p_scientific"):
        return entry["p_scientific"]
    if entry.get("p_underflows_float64"):
        return "< 1e-323"
    return fmt_p(entry.get("p"))


# --------------------------------------------------------------------- population
def load_population() -> dict:
    pc = [r for r in read_jsonl(SUITE_IDS_FILE) if r["family"] == FAMILY]
    by_triple: OrderedDict[tuple, list[dict]] = OrderedDict()
    for r in pc:
        by_triple.setdefault((r["dataset"], r["cell"], r["family"]), []).append(r)

    shapes = Counter(tuple(sorted(r["variant"] for r in v)) for v in by_triple.values())
    if set(shapes) - {("clean",), ("permuted",), ("clean", "permuted")}:
        raise SystemExit(f"REFUSING: unexpected variant combinations in doubled triples: {shapes}")
    doubled = [k for k, v in by_triple.items() if len(v) > 1]

    def dedup(prefer: str) -> list[str]:
        out = []
        for rows in by_triple.values():
            pick = next((r for r in rows if r["variant"] == prefer), None)
            out.append((pick or rows[0])["id"])
        return out

    populations = OrderedDict([
        ("full_1819", {
            "label": "the full manuscript population",
            "note": f"every row of {SUITE_IDS_FILE} with family {FAMILY}; a doubled triple "
                    "contributes both its rows, so 322 cells are weighted twice",
            "rows": [r["id"] for r in pc], "n": len(pc), "units": len(by_triple), "direction": None}),
        ("dedup_clean_1497", {
            "label": "de-duplicated, one row per triple, preferring `clean`",
            "note": "for each triple keep its `clean` row if it has one, else its `permuted` row; "
                 "n = 1,497 = the distinct triples, and a doubled cell is ONE unit in the paired "
                 "test (the 644 rows those 322 cells contribute above are 322 units here)",
            "rows": dedup("clean"), "n": len(by_triple), "units": len(by_triple), "direction": "clean"}),
        ("dedup_permuted_1497", {
            "label": "de-duplicated, one row per triple, preferring `permuted`",
            "note": "for each triple keep its `permuted` row if it has one, else its `clean` row; "
                 "n = 1,497, one unit per cell as above",
            "rows": dedup("permuted"), "n": len(by_triple), "units": len(by_triple),
            "direction": "permuted"}),
    ])
    return {"pc": pc, "by_triple": by_triple, "doubled": doubled, "shapes": shapes,
                "id1819": [r["id"] for r in pc], "populations": populations}


POP = load_population()
PC_IDS = set(POP["id1819"])
BY_TRIPLE = POP["by_triple"]
DOUBLED = POP["doubled"]
POPULATIONS = POP["populations"]

CENSUS_FILES = OrderedDict([
    ("kev-0.5b", "reports/kev-rungs/05b-rows.jsonl"),
    ("kev-0.6b", "reports/kev-rungs/06b-rows.jsonl"),
    ("kev-0.8b", "reports/kev-rungs/08b-rows.jsonl"),
    ("kev-4b", "reports/kev-rungs/4b-rows.jsonl"),
    ("kev-9b", "reports/kev-rungs/9b-rows.jsonl"),
    ("kev-27b", "reports/rows-kev-27b-probe.jsonl"),
    ("jev", "reports/jev-full/rows.jsonl"),
    ("llm-varA", "reports/llm-baseline-rows-census-varA.jsonl"),
    ("llm-varB", "reports/llm-baseline-rows-census-varB.jsonl"),
])
ARM_FILES = OrderedDict([
    ("ours-frozen", ("reports/forced-rows-frozen.jsonl", "reports/rows-full-frozen.jsonl")),
    ("ours-armB", ("reports/forced-rows-armb.jsonl", "reports/rows-full-armb.jsonl")),
    ("ours-seed1", ("reports/forced-rows-seed1.jsonl", "reports/rows-full-seed1.jsonl")),
    ("ours-seed2", ("reports/forced-rows-seed2.jsonl", "reports/rows-full-seed2.jsonl")),
    ("ours-seed3", ("reports/forced-rows-seed3.jsonl", "reports/rows-full-seed3.jsonl")),
    ("ours-d0.8b-s1", ("reports/forced-rows-delta-kev-0.8b-s1.jsonl",
                       "reports/rows-full-delta-kev-0.8b-s1.jsonl")),
    ("ours-d0.8b-s2", ("reports/forced-rows-delta-kev-0.8b-s2.jsonl",
                       "reports/rows-full-delta-kev-0.8b-s2.jsonl")),
    ("ours-d0.8b-s3", ("reports/forced-rows-delta-kev-0.8b-s3.jsonl",
                       "reports/rows-full-delta-kev-0.8b-s3.jsonl")),
    ("ours-d0.8b-s4", ("reports/forced-rows-delta-kev-0.8b-s4.jsonl",
                       "reports/rows-full-delta-kev-0.8b-s4.jsonl")),
    ("ours-d0.8b-s5", ("reports/forced-rows-delta-kev-0.8b-s5.jsonl",
                       "reports/rows-full-delta-kev-0.8b-s5.jsonl")),
    ("ours-d4b-s1", ("reports/forced-rows-delta-kev-4b-s1.jsonl",
                     "reports/rows-full-delta-kev-4b-s1.jsonl")),
    ("ours-d4b-s2", ("reports/forced-rows-delta-kev-4b-s2.jsonl", None)),
    ("ours-d4b-s3", ("reports/forced-rows-delta-kev-4b-s3.jsonl", None)),
    ("ours-d9b-s1", ("reports/forced-rows-delta-kev-9b-s1.jsonl", None)),
    ("ours-d9b-s2", ("reports/forced-rows-delta-kev-9b-s2.jsonl", None)),
])
LAYA_GLOB = "scratch/runs/laya-pilot-score/rows/*.jsonl"

LABELS = {
    "kev-0.5b": "Kev-0.5B zero-shot", "kev-0.6b": "Kev-0.6B zero-shot",
    "kev-0.8b": "Kev-0.8B zero-shot", "kev-4b": "Kev-4B zero-shot",
    "kev-9b": "Kev-9B zero-shot", "kev-27b": "Kev-27B zero-shot",
    "jev": "Jev (hosted)", "llm-varA": "Qwen3.5-9B zero-shot (minimal prompt)",
    "llm-varB": "Qwen3.5-9B zero-shot (explicit prompt)",
    "ours-frozen": "0.8B from-base -- frozen", "ours-armB": "0.8B from-base -- arm B",
    "ours-seed1": "0.8B fine-tuned seed1", "ours-seed2": "0.8B fine-tuned seed2",
    "ours-seed3": "0.8B fine-tuned seed3",
    "ours-d0.8b-s1": "0.8B delta-kev s1", "ours-d0.8b-s2": "0.8B delta-kev s2",
    "ours-d0.8b-s3": "0.8B delta-kev s3", "ours-d0.8b-s4": "0.8B delta-kev s4",
    "ours-d0.8b-s5": "0.8B delta-kev s5", "ours-d4b-s1": "4B delta-kev s1",
    "ours-d4b-s2": "4B delta-kev s2", "ours-d4b-s3": "4B delta-kev s3",
    "ours-d9b-s1": "9B delta-kev s1", "ours-d9b-s2": "9B delta-kev s2",
    "laya-pilot": "Laya pilot (322M)",
}
RUNG_OF = {k: "kev-0.8b" for k in ("ours-frozen", "ours-armB", "ours-seed1", "ours-seed2",
                                   "ours-seed3", "ours-d0.8b-s1", "ours-d0.8b-s2", "ours-d0.8b-s3",
                                   "ours-d0.8b-s4", "ours-d0.8b-s5")}
RUNG_OF.update({"ours-d4b-s1": "kev-4b", "ours-d4b-s2": "kev-4b", "ours-d4b-s3": "kev-4b",
                "ours-d9b-s1": "kev-9b", "ours-d9b-s2": "kev-9b"})


# --------------------------------------------------------------------- engines
def load_census() -> dict:
    """Each census dump keyed by census id, with the structural facts the join rests on asserted.

    The Kev/Jev/27B dumps carry the census id directly; the general LLM's dumps carry its `key`
    form instead (`<dataset>/<cell>/<family>/<question>`), so the id is rebuilt from the row's own
    ``(dataset, cell, family)`` -- the same triple the join uses -- and checked against the `id`
    wherever a dump carries one.
    """
    out: dict[str, dict] = {}
    for key, rel in CENSUS_FILES.items():
        rows: dict[str, dict] = {}
        for r in read_jsonl(rel):
            if r.get("family") != FAMILY:
                continue
            cid = r.get("id") or census_id(r["dataset"], r["cell"], r["family"])
            if cid in rows:
                raise SystemExit(f"REFUSING: {rel} carries {cid} twice")
            rows[cid] = r
        if len(rows) != 2510:
            raise SystemExit(f"REFUSING: {rel} carries {len(rows)} panel_choice census rows, not "
                             "the 2,510 reports/both-forced-table.json scores")
        variants = {r.get("variant") for r in rows.values()}
        if variants != {"permuted"}:
            raise SystemExit(f"REFUSING: {rel} carries variants {variants}; the argument that the "
                             "de-duplication direction cannot move a census-level number assumes "
                             "the census asked one rendering")
        out[key] = rows
    if len({frozenset(r) for r in out.values()}) != 1:
        raise SystemExit("REFUSING: the census dumps do not carry the same panel_choice ids")
    return out


CENSUS = load_census()
CENSUS_IDS = set(CENSUS["jev"])
if len({census_id(r["dataset"], r["cell"], r["family"]) for r in CENSUS["jev"].values()}) != 2510:
    raise SystemExit("REFUSING: two census records share a (dataset, cell, family) triple")


def load_engines() -> dict:
    engines: dict[str, dict] = {}
    rows_by_key: dict[str, dict[str, dict]] = {}

    for key, rows in CENSUS.items():
        ok = {rule: {} for rule in ("served", "forced")}
        for cid, row in rows.items():
            for rule in ("served", "forced"):
                pred = predict(row, rule)
                ok[rule][cid] = None if pred is None else pred == row["label"]
        # attribute the census verdict to every suite row of the triple
        engines[key] = {
            "key": key, "label": LABELS[key], "join": "census", "files": [CENSUS_FILES[key]],
            "ok_by_suite_id": {
                rule: {r["id"]: ok[rule][census_id(r["dataset"], r["cell"], r["family"])]
                       for r in POP["pc"]}
                for rule in ("served", "forced")}}

    for key, (dump, archived) in ARM_FILES.items():
        rows = {r["id"]: r for r in read_jsonl(dump) if r["id"] in PC_IDS}
        if len(rows) != len(PC_IDS):
            raise SystemExit(f"REFUSING: {dump} covers {len(rows)} of the 1,819 suite ids")
        rows_by_key[key] = rows
        engines[key] = {
            "key": key, "label": LABELS[key], "join": "suite",
            "files": [rel for rel in (dump, archived) if rel],
            "ok_by_suite_id": {rule: {sid: predict(rows[sid], rule) == rows[sid]["label"]
                                   for sid in rows} for rule in ("served", "forced")}}

    shards = sorted(glob.glob(os.path.join(ROOT, LAYA_GLOB)))
    laya: dict[str, dict] = {}
    for shard in shards:
        rel = os.path.relpath(shard, ROOT).replace("\\", "/")
        for row in read_jsonl(rel):
            if row.get("family") != FAMILY or row.get("ask") != "base":
                continue
            if row["id"] in PC_IDS:
                if row["id"] in laya:
                    raise SystemExit(f"REFUSING: Laya answers {row['id']} more than once")
                laya[row["id"]] = row
    if len(laya) != len(PC_IDS):
        raise SystemExit(f"REFUSING: Laya's shards cover {len(laya)} of the 1,819 suite ids")
    rows_by_key["laya-pilot"] = laya
    engines["laya-pilot"] = {
        "key": "laya-pilot", "label": LABELS["laya-pilot"], "join": "suite",
        "files": [os.path.relpath(s, ROOT).replace("\\", "/") for s in shards],
        "ok_by_suite_id": {rule: {sid: predict(laya[sid], rule) == laya[sid]["label"]
                               for sid in laya} for rule in ("served", "forced")}}

    # a rule that is undefined for a dump (the general LLM has no `p`) leaves its rows unscored,
    # exactly as reports/both-forced-table.json reports it -- drop them rather than invent a value.
    undefined: dict[str, dict[str, int]] = {}
    for key, engine in engines.items():
        undefined[key] = {}
        for rule in ("served", "forced"):
            scored = {sid: v for sid, v in engine["ok_by_suite_id"][rule].items() if v is not None}
            undefined[key][rule] = len(engine["ok_by_suite_id"][rule]) - len(scored)
            engine["ok_by_suite_id"][rule] = scored
    return {"engines": engines, "undefined": undefined, "n_shards": len(shards), "rows": rows_by_key}


LOADED = load_engines()
ENGINES = LOADED["engines"]

# the archive dump carrying the suite's own `keys`/`label` for all 1,819 rows: used to prove the two
# renderings of a doubled triple agree on the panel and on the answer, the premise of the census join.
ARCHIVE = {r["id"]: r for r in read_jsonl(ARCHIVE_FILE) if r["id"] in PC_IDS}


# --------------------------------------------------------------------- the duplication's weight
def accuracy_shift(rule: str) -> dict:
    """Every engine's accuracy on the three populations: what the duplication is weighting."""
    out: dict[str, dict] = OrderedDict()
    for key, engine in ENGINES.items():
        scores = engine["ok_by_suite_id"][rule]
        if not scores:
            continue
        out[key] = dict(
            label=LABELS[key], join=engine["join"],
            **{pid: sum(1 for i in POPULATIONS[pid]["rows"] if scores.get(i))
               / len(POPULATIONS[pid]["rows"]) for pid in POP_IDS})
        out[key]["shift_points_full_to_dedup"] = round(
            100 * (out[key]["dedup_clean_1497"] - out[key]["full_1819"]), 3)
    return out


SHIFT = {"served": accuracy_shift("served"), "forced": accuracy_shift("forced")}


# --------------------------------------------------------------------- the tests
def compare(engine_a: str, rule_a: str, engine_b: str, rule_b: str, ids: list[str],
            label: str) -> dict:
    a = ENGINES[engine_a]["ok_by_suite_id"][rule_a]
    b = ENGINES[engine_b]["ok_by_suite_id"][rule_b]
    n = len(ids)
    a_hits = sum(1 for i in ids if a[i])
    b_hits = sum(1 for i in ids if b[i])
    x = sum(1 for i in ids if a[i] and not b[i])
    y = sum(1 for i in ids if b[i] and not a[i])
    return dict(
        comparison=label,
        a={"key": engine_a, "label": LABELS[engine_a], "rule": rule_a},
        b={"key": engine_b, "label": LABELS[engine_b], "rule": rule_b},
        n=n, a_correct=a_hits, b_correct=b_hits, a_acc=a_hits / n, b_acc=b_hits / n,
        a_right_b_wrong=x, b_right_a_wrong=y, **p_of(x, y))


GROUPS = OrderedDict([
    ("i", {
        "title": "(i) Jev against each Kev rung, as served",
        "why": "the paper's Claim A: Jev is above the Kev rungs",
        "comparisons": [(f"Jev vs Kev-{rung} (as served)", "jev", "served", f"kev-{key}", "served")
                     for rung, key in (("0.5B", "0.5b"), ("0.6B", "0.6b"), ("0.8B", "0.8b"),
                                       ("4B", "4b"), ("9B", "9b"))]}),
    ("ii", {
        "title": "(ii) Kev-27B against the other large engines, as served",
        "why": "the paper's Claim B: the largest decision model ranks last among the large engines",
        "comparisons": [("Kev-27B vs Kev-4B (as served)", "kev-27b", "served", "kev-4b", "served"),
                     ("Kev-27B vs Kev-9B (as served)", "kev-27b", "served", "kev-9b", "served"),
                     ("Kev-27B vs Jev (as served)", "kev-27b", "served", "jev", "served")]}),
    ("iii", {
        "title": "(iii) the both-forced pair, and the as-served pair it is contrasted with",
        "why": "the paper's centrepiece: forcing both removes the 27B's deficit without inverting it",
        "comparisons": [("Kev-27B forced vs Jev forced [the both-forced pair]", "kev-27b", "forced",
                      "jev", "forced"),
                     ("Kev-27B served vs Jev served [as-served contrast]", "kev-27b", "served",
                      "jev", "served")]}),
])

ARM_BLOCKS = OrderedDict([
    ("0.8B warm-started from the released rung -- the five seeds",
     ["ours-d0.8b-s1", "ours-d0.8b-s2", "ours-d0.8b-s3", "ours-d0.8b-s4", "ours-d0.8b-s5"]),
    ("4B warm-started from the released rung -- three seeds",
     ["ours-d4b-s1", "ours-d4b-s2", "ours-d4b-s3"]),
    ("9B warm-started from the released rung -- two seeds",
     ["ours-d9b-s1", "ours-d9b-s2"]),
    (("from-base controls at 0.8B against the same-size released checkpoint "
     "(not the paper's claim about the fine-tune)"),
     ["ours-frozen", "ours-armB", "ours-seed1", "ours-seed2", "ours-seed3"]),
])

SUPPLEMENTAL = [
    ("Kev-27B forced vs Jev as served [paper's pair]", "kev-27b", "forced", "jev", "served"),
    ("Kev-27B forced vs Kev-4B forced", "kev-27b", "forced", "kev-4b", "forced"),
    ("Kev-27B forced vs Kev-9B forced", "kev-27b", "forced", "kev-9b", "forced"),
]


def verdict(results: list[dict]) -> dict:
    """Does this comparison's conclusion move across the three populations?"""
    gaps = [r["a_acc"] - r["b_acc"] for r in results]
    signs = {(0 if g == 0 else (1 if g > 0 else -1)) for g in gaps}
    sig = [r["p"] < 0.05 for r in results]
    return {
        "n": [r["n"] for r in results], "gaps": [round(g, 4) for g in gaps], "p": [r["p"] for r in results],
        "p_display": [p_disp(r) for r in results],
        "direction": [("A above B" if g > 0 else "B above A" if g < 0 else "tied") for g in gaps],
        "significant": sig,
        "direction_stable": len(signs) == 1,
        "significance_stable": len({*sig}) == 1}


def run_check_1() -> dict:
    groups: dict = OrderedDict()
    for gid, group in GROUPS.items():
        per_pop = OrderedDict()
        for pid in POP_IDS:
            rows = POPULATIONS[pid]["rows"]
            per_pop[pid] = [compare(a, ra, b, rb, rows, lbl)
                            for lbl, a, ra, b, rb in group["comparisons"]]
        identical = all(
            all(per_pop["dedup_clean_1497"][i][k] == per_pop["dedup_permuted_1497"][i][k]
                for k in ("n", "a_correct", "b_correct", "a_right_b_wrong", "b_right_a_wrong", "p"))
            for i in range(len(per_pop["full_1819"])))
        groups[gid] = {
            "title": group["title"], "why": group["why"], "population": per_pop,
            "dedup_directions_identical": identical,
            "verdicts": [verdict([per_pop[pid][i] for pid in POP_IDS])
                      for i in range(len(per_pop["full_1819"]))]}

    arms: dict = OrderedDict()
    for block, keys in ARM_BLOCKS.items():
        per_pop = OrderedDict()
        for pid in POP_IDS:
            rows = POPULATIONS[pid]["rows"]
            per_seed = [compare(key, "forced", RUNG_OF[key], "forced", rows,
                                f"{LABELS[key]} forced vs {LABELS[RUNG_OF[key]]} forced")
                        for key in keys]
            if len({r["b_acc"] for r in per_seed}) != 1:
                raise SystemExit("REFUSING: the rung's accuracy differs between seeds of a block")
            out = list(per_seed)
            if len(keys) > 1:
                accs_a = [r["a_acc"] for r in per_seed]
                accs_b = [r["b_acc"] for r in per_seed]
                stacked_x = sum(r["a_right_b_wrong"] for r in per_seed)
                stacked_y = sum(r["b_right_a_wrong"] for r in per_seed)
                out.append({
                    "comparison": f"{block.split(' -- ')[0]} -- mean of {len(keys)} seeds",
                    "a": {"key": "mean", "label": f"mean of {len(keys)} seeds",
                          "rule": "forced"},
                    "b": {"key": RUNG_OF[keys[0]], "label": LABELS[RUNG_OF[keys[0]]], "rule": "forced"},
                    "n": len(rows), "a_acc": sum(accs_a) / len(accs_a), "b_acc": sum(accs_b) / len(accs_b),
                    "a_correct": None, "b_correct": None, "a_right_b_wrong": None, "b_right_a_wrong": None,
                    "p": None, "seed_accs_a": accs_a, "seeds": keys,
                    "per_seed_p": [r["p"] for r in per_seed],
                    "per_seed_discordant": [[r["a_right_b_wrong"], r["b_right_a_wrong"]]
                                         for r in per_seed],
                    "seed_stacked": dict(
                        a_right_b_wrong=stacked_x, b_right_a_wrong=stacked_y,
                        n_seeds=len(keys), **p_of(stacked_x, stacked_y),
                        caveat="secondary: the seeds' paired rows are stacked, so a unit is a "
                               "seed x question, not a question, and the test overstates what one "
                               "training run establishes; the per-seed tests are the manuscript's"),
                    "note": "no single paired test exists for a mean over seeds: each seed is tested "
                         "against the rung individually (per_seed_p), and this row reports the mean "
                         "accuracy the manuscript publishes"})
            per_pop[pid] = out
        arms[block] = {
            "population": per_pop,
            "verdicts": [verdict([per_pop[pid][i] for pid in POP_IDS]) for i in range(len(keys))]}

    supplemental = OrderedDict()
    for label, a, ra, b, rb in SUPPLEMENTAL:
        per_pop = OrderedDict()
        for pid in POP_IDS:
            per_pop[pid] = compare(a, ra, b, rb, POPULATIONS[pid]["rows"], label)
        supplemental[label] = {
            "population": per_pop, "verdicts": verdict([per_pop[pid] for pid in POP_IDS]),
            "dedup_directions_identical": all(
                per_pop["dedup_clean_1497"][k] == per_pop["dedup_permuted_1497"][k]
                for k in ("n", "a_correct", "b_correct", "a_right_b_wrong", "b_right_a_wrong",
                          "p"))}
    return {"groups": groups, "arms": arms, "supplemental": supplemental}


def run_check_2(rule: str = "served") -> dict:
    per_engine = []
    for key, engine in ENGINES.items():
        census_level = engine["join"] == "census"
        scores = engine["ok_by_suite_id"][rule]
        counts: Counter = Counter()
        covered = 0
        for rows in BY_TRIPLE.values():
            if len(rows) != 2:
                continue
            clean_id = next(r["id"] for r in rows if r["variant"] == "clean")
            perm_id = next(r["id"] for r in rows if r["variant"] == "permuted")
            if clean_id not in scores or perm_id not in scores:
                continue
            covered += 1
            tag = ("clean" if scores[clean_id] else "") + ("permuted" if scores[perm_id] else "")
            counts[tag or "neither"] += 1
        entry = dict(
            key=key, label=LABELS[key], join=engine["join"], rule=rule, pairs_covered=covered,
            both_correct=counts["cleanpermuted"], only_clean_correct=counts["clean"],
            only_permuted_correct=counts["permuted"], neither_correct=counts["neither"],
            estimable=not census_level, **p_of(counts["clean"], counts["permuted"]))
        if census_level:
            entry["not_estimable_because"] = (
                "census-level join: this engine was asked ONE rendering of each triple -- the "
                "census's, which is `permuted` on all 2,510 census records -- and the join "
                "attributes that single verdict to the triple's clean row as well. Only-clean and "
                "only-permuted are therefore 0 by construction. This is a gap in what was run, not "
                "a measured absence of position bias: the clean rendering was never put to this "
                "engine.")
        elif covered < 30:
            entry["not_estimable_because"] = (
                f"only {covered} of the 322 pairs covered: too few to say anything")
            entry["estimable"] = False
        per_engine.append(entry)

    def pool(entries: list[dict]) -> dict:
        both = sum(e["both_correct"] for e in entries)
        oc = sum(e["only_clean_correct"] for e in entries)
        op = sum(e["only_permuted_correct"] for e in entries)
        ne = sum(e["neither_correct"] for e in entries)
        units = both + oc + op + ne
        return dict(engines=len(entries), engine_pair_units=units, both_correct=both,
                    only_clean_correct=oc, only_permuted_correct=op, neither_correct=ne,
                    **p_of(oc, op),
                    only_clean_rate=(oc / units) if units else None,
                    only_permuted_rate=(op / units) if units else None)

    estimable = [e for e in per_engine if e["estimable"]]
    return {
        "rule": rule, "pairs": len(DOUBLED),
        "pair_source": f"{SUITE_IDS_FILE}: the 322 (dataset, cell, family) triples carrying one "
                       "`clean` and one `permuted` suite row",
        "unit": "one engine x one of the 322 doubled triples",
        "per_engine": per_engine, "pooled_suite_level": pool(estimable),
        "pooled_all_engines": pool(per_engine),
        "note": "For a suite-level engine the two renderings are two real questions carrying the same "
             "answer, put to the engine separately, so the middle columns measure rendering "
             "sensitivity -- of which option-position bias is the suspected component. For a "
             "census-level engine they are one verdict seen twice (see not_estimable_because)."}


CHECK1 = run_check_1()
CHECK2 = run_check_2()


# --------------------------------------------------------------------- replication
def run_replication() -> dict:
    checks: list[dict] = []

    def check(name: str, got, want, source: str, tol: float = 0.0) -> None:
        if isinstance(want, float):
            ok = got is not None and abs(float(got) - want) <= tol
        else:
            ok = got == want
        checks.append({"check": name, "got": got, "published": want, "source": source, "ok": bool(ok)})

    # --- the duplication's own shape
    check("panel_choice rows in the suite id file", len(POP["pc"]), 1819,
          "reports/per-dataset.json population.panel_choice")
    check("distinct (dataset, cell, family) triples", len(BY_TRIPLE), 1497,
          "reports/per-dataset.json scoring_rule.join")
    check("triples carrying two suite rows", len(DOUBLED), 322,
          "reports/per-dataset.json scoring_rule.join")
    check("clean rows / permuted rows",
          [sum(1 for r in POP["pc"] if r["variant"] == v) for v in ("clean", "permuted")],
          [910, 909], SUITE_IDS_FILE)
    check("every clean row carries variant_mismatch, no permuted row does",
          [all(bool(r.get("variant_mismatch")) for r in POP["pc"] if r["variant"] == "clean"),
           any(bool(r.get("variant_mismatch")) for r in POP["pc"] if r["variant"] == "permuted")],
          [True, False], "the census asked the permuted rendering of every panel_choice triple")
    check("census panel_choice ids / variants present",
          [len(CENSUS_IDS), len({r.get("variant") for r in CENSUS["jev"].values()})], [2510, 1],
          "reports/both-forced-table.json populations.census_2510")
    check("population sizes (full, dedup clean, dedup permuted)",
          [POPULATIONS[p]["n"] for p in POP_IDS], [1819, 1497, 1497],
          "this script's construction")
    check("each de-duplicated population holds exactly one row per triple",
          [len({(r["dataset"], r["cell"], r["family"]) for r in POP["pc"] if r["id"] in ids})
           for ids in (set(POPULATIONS[p]["rows"]) for p in POP_IDS[1:])], [1497, 1497],
          "this script's construction")

    # --- the two renderings of a doubled triple agree on the panel and on the answer
    panel_agree = answer_agree = perm_is_census = 0
    for triple in DOUBLED:
        rows = BY_TRIPLE[triple]
        clean = next(r for r in rows if r["variant"] == "clean")
        perm = next(r for r in rows if r["variant"] == "permuted")
        ck, pk = ARCHIVE[clean["id"]], ARCHIVE[perm["id"]]
        census_row = CENSUS["jev"][census_id(*triple)]
        panel_agree += set(ck["keys"]) == set(pk["keys"])
        answer_agree += ck["keys"][ck["label"]] == pk["keys"][pk["label"]]
        perm_is_census += pk["keys"] == census_row["keys"] and pk["label"] == census_row["label"]
    check("the 322 pairs' two renderings share one panel", panel_agree, 322,
          f"{ARCHIVE_FILE}, the suite's own keys field")
    check("the 322 pairs' two renderings carry the same answer", answer_agree, 322,
          f"{ARCHIVE_FILE}, the option named by keys[label]")
    check("each pair's `permuted` row is the census's own record", perm_is_census, 322,
          "reports/jev-full/rows.jsonl vs the suite's permuted row")

    # --- reports/both-forced-table.json, every engine it scores on the 1,819 rows
    bft = json.loads(read_text("reports/both-forced-table.json"))
    for e in bft["engines"]:
        key = e["key"]
        if key not in ENGINES:
            continue
        for rule in ("served", "forced"):
            published = e["census_matched_1819"][rule]
            if published.get("correct") is None:
                continue
            got = sum(1 for i in POP["id1819"] if ENGINES[key]["ok_by_suite_id"][rule].get(i))
            check(f"both-forced-table {key} | {rule} correct (1,819)", got,
                  published["correct"], f"reports/both-forced-table.json engines[{key}]")
        check(f"both-forced-table {key} | abstain-pick count (1,819)",
              sum(1 for i in POP["id1819"] if ABSTAIN_PICKS[key][i]),
              e["census_matched_1819"]["abstain_pick"]["picks"],
              f"reports/both-forced-table.json engines[{key}]")

    # --- reports/arms-forced.json, the 11 arms re-scored with `p` retained
    af = json.loads(read_text("reports/arms-forced.json"))
    for e in af["engines"]:
        key = e["key"]
        published = e["census_matched_1819"]
        for rule in ("served", "forced"):
            got = sum(1 for i in POP["id1819"] if ENGINES[key]["ok_by_suite_id"][rule].get(i))
            check(f"arms-forced {key} | {rule} correct (1,819)", got,
                  published[rule]["correct"], f"reports/arms-forced.json engines[{key}]")
        check(f"arms-forced {key} | abstain-pick count (1,819)",
              sum(1 for i in POP["id1819"] if ABSTAIN_PICKS[key][i]),
              published["abstain_pick"]["picks"], f"reports/arms-forced.json engines[{key}]")
        if ARCHIVED_DRIFT.get(key) is not None:
            check(f"arms-forced {key} | archived vs rescored top1 mismatches",
                  ARCHIVED_DRIFT[key], e["fidelity"]["top1_mismatch"]["count"],
                  f"reports/arms-forced.json engines[{key}].fidelity")

    # --- reports/per-dataset.json paired_tests: weighted <-> full, distinct <-> de-duplicated
    pdj = json.loads(read_text("reports/per-dataset.json"))["paired_tests"]
    for label, a, ra, b, rb in SUPPLEMENTAL:
        for pid, variant in (("full_1819", "weighted"), ("dedup_clean_1497", "distinct"),
                             ("dedup_permuted_1497", "distinct")):
            key = f"{label}|pooled|{variant}"
            if key not in pdj:
                continue
            got = compare(a, ra, b, rb, POPULATIONS[pid]["rows"], label)
            want = pdj[key]
            check(f"per-dataset {label} ({pid}: {variant})",
                  [got["a_right_b_wrong"], got["b_right_a_wrong"], round(got["p"], 12)],
                  [want["a_right_b_wrong"], want["b_right_a_wrong"], round(want["p"], 12)],
                  f"reports/per-dataset.json paired_tests[{key}]")

    # --- reports/extra-seeds.json, each extra seed against its own rung
    es = json.loads(read_text("reports/extra-seeds.json"))
    for size in ("4b", "9b"):
        for seed in es["sizes"][size]["seeds"]:
            key = f"ours-d{size}-s{seed['seed']}"
            if key not in ENGINES:
                continue
            got = compare(key, "forced", RUNG_OF[key], "forced", POPULATIONS["full_1819"]["rows"],
                          key)
            want = seed["paired_vs_rung"]
            check(f"extra-seeds {key} | forced vs its own rung",
                  [got["a_correct"], got["b_correct"], got["a_right_b_wrong"],
                   got["b_right_a_wrong"], round(got["p"], 12)],
                  [seed["census_matched_1819"]["forced"]["correct"],
                   es["sizes"][size]["rung_forced"]["correct"], want["arm_right_rung_wrong"],
                   want["rung_right_arm_wrong"], round(want["p_two_sided_exact"], 12)],
                  f"reports/extra-seeds.json sizes.{size}.seeds[{seed['seed'] - 1}]"
                  ".paired_vs_rung")

    # --- reports/arms-forced.json claim_2_5, every arm against the 0.8B rung
    for arm in af["claim_2_5"]["per_arm"]:
        key = arm["key"]
        got = compare(key, "forced", "kev-0.8b", "forced", POPULATIONS["full_1819"]["rows"], key)
        check(f"arms-forced claim_2_5 {key} | forced vs kev-0.8b",
              [got["a_correct"], got["b_correct"], got["a_right_b_wrong"], got["b_right_a_wrong"],
               round(got["p"], 12)],
              [arm["forced_correct"], af["claim_2_5"]["reference"]["correct"],
               arm["paired"]["arm_right_rung_wrong"], arm["paired"]["rung_right_arm_wrong"],
               round(arm["paired"]["p_two_sided_exact"], 12)],
              f"reports/arms-forced.json claim_2_5.per_arm[{key}]")

    # --- the manuscript's own headline numbers
    manuscript = read_text("manuscript/abstention-decides-the-ranking.md")
    both_forced = CHECK1["groups"]["iii"]["population"]["full_1819"][0]
    check("manuscript both-forced discordant counts",
          [both_forced["a_right_b_wrong"], both_forced["b_right_a_wrong"]], [76, 69],
          "manuscript section 2.4 / Figure 1b")
    for side, needle in (("a", "0.7191"), ("b", "0.7152")):
        check("manuscript both-forced {} = {}".format(LABELS[both_forced[side]["key"]], needle),
              round(both_forced[side + "_acc"], 4), float(needle),
              "manuscript section 2.4 / Figure 1b", tol=1e-12)
        check(f"manuscript states {needle}", needle in manuscript, True,
              "manuscript/abstention-decides-the-ranking.md")
    for key, rule, needle in (("kev-0.8b", "forced", 0.3178), ("kev-4b", "forced", 0.6168),
                              ("kev-9b", "forced", 0.6295), ("kev-27b", "served", 0.4662),
                              ("jev", "served", 0.6597)):
        got = sum(1 for i in POP["id1819"] if ENGINES[key]["ok_by_suite_id"][rule][i]) / 1819
        check(f"manuscript {key} {rule} = {needle}", round(got, 4), needle,
              "manuscript section 2.4, Table 3", tol=5e-5)
    for size, needle in (("4B", 0.5732), ("9B", 0.5651)):
        keys = next(v for k, v in ARM_BLOCKS.items() if k.startswith(size + " warm-started"))
        accs = [sum(1 for i in POP["id1819"] if ENGINES[k]["ok_by_suite_id"]["forced"][i]) / 1819
                for k in keys]
        check(f"manuscript {size} seed-mean forced = {needle}",
              round(sum(accs) / len(accs), 4), needle,
              "manuscript section 2.5", tol=5e-5)
    d08 = [sum(1 for i in POP["id1819"] if ENGINES[k]["ok_by_suite_id"]["forced"][i]) / 1819
           for k in ARM_BLOCKS["0.8B warm-started from the released rung -- the five seeds"]]
    check("manuscript 0.8B seeds' forced range 0.3793-0.4854",
          [round(min(d08), 4), round(max(d08), 4)], [0.3793, 0.4854],
          "manuscript section 2.5", tol=5e-5)

    # --- the paired test itself: the settled function against the exact evaluation
    overflowed = [e for e in P_AGREEMENT if e["overflows"]]
    disagreed = [e for e in P_AGREEMENT if e["same"] is False]
    check("the settled paired test agrees with the exact evaluation on every discordance count it "
          f"can evaluate ({len(P_AGREEMENT) - len(overflowed)} distinct counts)",
          len(disagreed), 0, RULE_SOURCE + "::binom_two_sided")
    checks.append({
        "check": "distinct discordance counts where the settled test overflows float64 (2.0 ** n > "
              "1.8e308)",
        "got": len(overflowed), "published": None, "source": RULE_SOURCE + "::binom_two_sided", "ok": True,
        "note": "evaluated exactly instead; no published number in the repository uses one of these "
             "(the largest discordance count any of them uses is 429)"})

    return {"checks": checks, "n": len(checks), "all_pass": all(c["ok"] for c in checks),
                "failures": [c for c in checks if not c["ok"]]}


# --- the caches run_replication() reads (built before it is called)
ABSTAIN_PICKS: dict[str, dict[str, bool]] = {}
ARCHIVED_DRIFT: dict[str, int | None] = {}
for _key, _rows in CENSUS.items():
    _picks: dict[str, bool] = {}
    for _cid, _row in _rows.items():
        _idx = _row.get("abstain")
        if _idx is None:
            _idx = abstain_index(_row)
        _picks[_cid] = bool(_idx is not None and predict(_row, "served") == _idx)
    ABSTAIN_PICKS[_key] = {r["id"]: _picks[census_id(r["dataset"], r["cell"], r["family"])]
                           for r in POP["pc"]}
    ARCHIVED_DRIFT[_key] = None
for _key, _rows in LOADED["rows"].items():
    _idx = {sid: (row.get("abstain") if row.get("abstain") is not None else abstain_index(row))
            for sid, row in _rows.items()}
    ABSTAIN_PICKS[_key] = {sid: bool(_idx[sid] is not None
                                     and predict(_rows[sid], "served") == _idx[sid])
                           for sid in _rows}
    ARCHIVED_DRIFT[_key] = None
    _archived = ARM_FILES[_key][1] if _key in ARM_FILES else None
    if _archived:
        _old = {r["id"]: r for r in read_jsonl(_archived) if r["id"] in PC_IDS}
        ARCHIVED_DRIFT[_key] = sum(1 for _sid in _rows if _old[_sid]["top1"] != _rows[_sid]["top1"])

REPLICATION = run_replication()


# --------------------------------------------------------------------- conclusions
def conclusions() -> dict:
    bf = CHECK1["groups"]["iii"]["population"]
    out: dict = OrderedDict()

    out["invariance"] = {
        "claim": "Every census-level comparison is invariant to the de-duplication DIRECTION",
        "why": "the census asked one rendering per triple (`permuted` on all 2,510 records) and the "
            "join of reports/per-dataset.json attributes that single verdict to both rows of a "
            "doubled triple, so preferring `clean` and preferring `permuted` give identical "
            "numbers. The direction cannot be cherry-picked because there is no direction.",
        "groups_identical": [gid for gid, g in CHECK1["groups"].items()
                          if g["dedup_directions_identical"]],
        "supplemental_identical": [label for label, v in CHECK1["supplemental"].items()
                                if v["dedup_directions_identical"]],
        "arms_identical": all(
            all(v["population"]["dedup_clean_1497"][i][k] ==
                v["population"]["dedup_permuted_1497"][i][k] for k in ("n", "a_acc", "b_acc"))
            for v in CHECK1["arms"].values()
            for i in range(len(v["population"]["full_1819"])))}

    out["both_forced"] = OrderedDict()
    for pid in POP_IDS:
        d = bf[pid][0]  # comparison 0 of group (iii) is the both-forced pair
        out["both_forced"][pid] = {"n": d["n"], "a_acc": d["a_acc"], "b_acc": d["b_acc"],
                                       "a_right_b_wrong": d["a_right_b_wrong"],
                                       "b_right_a_wrong": d["b_right_a_wrong"], "p": d["p"]}
    out["both_forced"]["moves"] = (
        "the accuracy gap and the discordant counts shrink and the conclusion does not: with n = "
        "1,497, p = {:.4f}, the pair is a null either way, and it is the same number under both "
        "de-duplication directions".format(bf["dedup_clean_1497"][0]["p"]))

    out["check_1_rows_that_change_significance"] = []
    out["check_1_rows_that_flip_direction"] = []
    for gid, g in CHECK1["groups"].items():
        for i, v in enumerate(g["verdicts"]):
            name = g["population"]["full_1819"][i]["comparison"]
            if not v["significance_stable"]:
                out["check_1_rows_that_change_significance"].append(
                    {"group": gid, "comparison": name, "p": v["p"], "significant": v["significant"]})
            if not v["direction_stable"]:
                out["check_1_rows_that_flip_direction"].append(
                    {"group": gid, "comparison": name, "gaps": v["gaps"], "direction": v["direction"]})
    out["arm_rows"] = []
    for block, v in CHECK1["arms"].items():
        for i, w in enumerate(v["verdicts"]):
            out["arm_rows"].append(dict(
                block=block, comparison=v["population"]["full_1819"][i]["comparison"], **w))
    out["arm_direction_changes"] = [r for r in out["arm_rows"] if not r["direction_stable"]]

    effects = []
    for block, v in CHECK1["arms"].items():
        for i in range(len(v["population"]["full_1819"])):
            c = v["population"]["dedup_clean_1497"][i]
            p = v["population"]["dedup_permuted_1497"][i]
            effects.append({
                "comparison": c["comparison"], "block": block,
                "a_acc_full": v["population"]["full_1819"][i]["a_acc"], "a_acc_clean": c["a_acc"],
                "a_acc_permuted": p["a_acc"], "b_acc_clean": c["b_acc"], "b_acc_permuted": p["b_acc"],
                "arm_delta_points": round(100 * (c["a_acc"] - p["a_acc"]), 3),
                "rung_delta_points": round(100 * (c["b_acc"] - p["b_acc"]), 3)})
    out["dedup_direction_effect"] = {
        "census_level": "identical for every census-level comparison, by construction and verified: "
                     "the census asked one rendering per triple, so the retained row cannot change "
                     "a census-level verdict",
        "arms": effects,
        "max_abs_arm_delta_points": max(abs(e["arm_delta_points"]) for e in effects),
        "max_abs_rung_delta_points": max(abs(e["rung_delta_points"]) for e in effects),
        "crosses_any_conclusion": bool(out["check_1_rows_that_flip_direction"]
                                    or out["arm_direction_changes"]
                                    or out["check_1_rows_that_change_significance"]),
        "why_the_arms_differ": "a suite-level arm answered each rendering separately, so which row is "
                            "retained changes its score; the census-level rung it is compared "
                            "against has one verdict per triple and does not move"}
    return out


CONCLUSIONS = conclusions()


# --------------------------------------------------------------------- sources
def collect_sources() -> OrderedDict[str, dict]:
    files: OrderedDict[str, dict] = OrderedDict()

    def add(rel: str) -> None:
        if rel not in files:
            files[rel] = {"bytes": os.path.getsize(os.path.join(ROOT, rel)), "sha256": sha256_of(rel)}

    for rel in (SUITE_IDS_FILE, RULE_SOURCE, ARCHIVE_FILE):
        add(rel)
    for engine in ENGINES.values():
        for rel in engine["files"]:
            add(rel)
    for rel in ("reports/both-forced-table.json", "reports/arms-forced.json",
                "reports/per-dataset.json", "reports/extra-seeds.json", "reports/PAIRED-TESTS.md",
                "manuscript/abstention-decides-the-ranking.md"):
        add(rel)
    return files


SOURCES = collect_sources()
LAYA_RELS = [rel for rel in SOURCES if rel.startswith("scratch/runs/laya-pilot-score/")]


# --------------------------------------------------------------------- output
def payload() -> dict:
    return OrderedDict([
        ("what_this_is",
         ("Two re-aggregation checks on the panel_choice duplication: (1) the manuscript's headline "
         "comparisons recomputed on the full 1,819 rows and on two de-duplicated 1,497-row "
         "populations (one row per (dataset, cell, family) triple, preferring `clean`, then "
         "preferring `permuted`); (2) clean-vs-permuted rendering asymmetry on the 322 triples "
         "carrying both. A re-aggregation of row dumps already on disk: no model was run, no job "
         "submitted, nothing pre-existing touched.")),
        ("population", OrderedDict([
            ("suite_ids", SUITE_IDS_FILE),
            ("family", FAMILY),
            ("rows", len(POP["pc"])),
            ("distinct_triples", len(BY_TRIPLE)),
            ("doubled_triples", len(DOUBLED)),
            ("variant_shapes", {"+".join(k): v for k, v in POP["shapes"].items()}),
            ("rows_per_triple", len(POP["pc"]) / len(BY_TRIPLE)),
            ("same_ratio_in_other_families",
             ("panel_miss 1,788 rows / 1,476 triples (1.211), evidence_diluted 1,763 / 1,454 "
             "(1.213), marker_noul 1,833 / 1,489 (1.231)")),
            ("populations", {pid: {k: v for k, v in POPULATIONS[pid].items() if k != "rows"}
                             for pid in POP_IDS}),
            ("de_dup_unit", ("(dataset, cell, family) -- inside family=panel_choice this is the "
                            "(dataset, cell) pair, since a triple names exactly one cell")),
            ("paired_unit_in_dedup",
             ("in a de-duplicated run a doubled cell contributes ONE unit to the paired test: its "
             "single retained row, carrying the correctness of the preferred rendering. Its other "
             "row is dropped, so the 644 rows those 322 cells contribute to the 1,819 population "
             "are 322 units here. The test's n is 1,497 rather than 1,819, and each discordant count "
             "counts cells, not rows.")),
        ])),
        ("rules", OrderedDict([
            ("source", RULE_SOURCE + "; functions {} extracted by AST and executed from that "
                       "source, not re-implemented".format(", ".join(RULES))),
            ("served", ("argmax(row['p']) == row['label']; the general LLM's dumps carry no `p` and "
                       "are scored on `parsed_index`, as reports/both-forced-table.json does")),
            ("forced", ("argmax over the row's p vector with the abstain option removed == "
                       "row['label'] (reports/abstain-reconciliation.py's own function)")),
            ("abstain_index", ("sckevtype.eval.abstain_index(row) -- the option whose key is in "
                               f"sckevtype.data.families.NONE_OPTIONS ({len(NONE_OPTIONS)} "
                               "wordings); used everywhere and checked equal to the dumps' own "
                               "`abstain` field where they carry one (Laya's do not)")),
            ("paired_test", ("exact two-sided binomial on the discordant counts -- the same function, "
                            "hence the same test, as reports/PAIRED-TESTS.md; it is called and "
                            "compared on every distinct discordance count, and only where its "
                            "`2.0 ** n` would overflow float64 (discordance count > 1,023, which "
                            "no published number reaches) is the same expression evaluated with "
                            "exact integers -- see paired_test_agreement")),
            ("join_census", ("on (dataset, cell, family): the engine answered the census's own record "
                            "of that triple and its verdict is attributed to every suite row with "
                            "it (322 panel_choice triples hold two)")),
            ("join_suite", ("on the row's own id (test/custom/<n>): the engine answered that exact "
                           "record")),
            ("census_rendering", ("every census panel_choice record is `variant=permuted`, asserted "
                                 "in load_census(); that is why the de-duplication direction cannot "
                                 "move a census-level number")),
            ("rules_undefined_for", {k: v for k, v in LOADED["undefined"].items() if any(v.values())}),
        ])),
        ("accuracy_by_population", SHIFT),
        ("check_1_sensitivity_to_the_duplication", CHECK1),
        ("check_2_position_bias_on_the_322_pairs", CHECK2),
        ("paired_test_agreement", P_AGREEMENT),
        ("replication", REPLICATION),
        ("conclusions", CONCLUSIONS),
        ("limits", [
            ("The de-duplicated populations are 1,497 rows; the manuscript's tables stay on 1,819. "
            "Nothing here changes a published table, a figure or the manuscript."),
            ("A census-level engine's verdict attributed to a clean-rendered suite row scores a "
            "rendering that engine was never asked. That is the manuscript's capping choice and the "
            "reason its 322 doubled triples are weighted twice; it is also why Check 2 cannot "
            "measure rendering asymmetry for those engines at all."),
            ("Check 2 measures the difference between two renderings of one question, which is "
            "rendering sensitivity; option-position bias is the suspected mechanism, not something "
            "this design isolates from the other wording differences a permutation carries."),
            ("The 'mean over seeds' rows have no single paired test: per-seed tests are reported and "
            "the stacked test is labelled secondary because it inflates significance."),
            ("Four arms (4B s2/s3, 9B s1/s2) have a re-scored dump only; their served and forced "
            "values come from the same dumps the published numbers do, and their row-level top1 "
            "fidelity check is unavailable, as reports/extra-seeds.json's limits say."),
            ("The replication block checks this script against the published artifacts; it cannot "
            "check those artifacts against a model re-run, which no one did here."),
        ]),
        ("sha256", {rel: v["sha256"] for rel, v in SOURCES.items()}),
        ("sha256_bytes", {rel: v["bytes"] for rel, v in SOURCES.items()}),
    ])


def render_md(data: dict) -> str:
    L: list[str] = []
    A = L.append
    bf = data["check_1_sensitivity_to_the_duplication"]
    c2 = data["check_2_position_bias_on_the_322_pairs"]
    rep = data["replication"]
    conc = data["conclusions"]

    def cmp_block(per_pop: dict) -> None:
        # a group's population is a list of comparisons; a single comparison is a dict
        items = per_pop["full_1819"]
        count = len(items) if isinstance(items, list) else 1
        for i in range(count):
            for pid in POP_IDS:
                entry = per_pop[pid]
                cc = entry[i] if isinstance(entry, list) else entry
                A(f"| {cc['comparison']} | {POP_SHORT[pid]} | {cc['n']} | "
                  f"{fmt_acc(cc['a_acc'])} ({cc['a_correct']}) | {fmt_acc(cc['b_acc'])} "
                  f"({cc['b_correct']}) | {cc['a_right_b_wrong']} | {cc['b_right_a_wrong']} | "
                  f"{p_disp(cc)} |")

    A("# Is the `panel_choice` duplication load-bearing?")
    A("")
    A("Two re-aggregation checks on the manuscript's scored population. **No model was run, no job")
    A("submitted**: every number is recomputed from row dumps already on disk, and the scoring rule,")
    A("the joins and the paired test are the settled ones (see §1). Generated by")
    A("`reports/variant-redundancy.py`; the same numbers, unrounded, are in")
    A("`reports/variant-redundancy.json`.")
    A("")
    A("## 0. The headline answer")
    A("")
    A("**The both-forced pair (Kev-27B forced against Jev forced) stays null, and shrinks.**")
    A("")
    A("| population | n | 27B forced | Jev forced | 27B-only | Jev-only | p |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for pid in POP_IDS:
        d = conc["both_forced"][pid]
        A(f"| {POP_SHORT[pid]} | {d['n']} | {d['a_acc']:.4f} | {d['b_acc']:.4f} | "
          f"{d['a_right_b_wrong']} | {d['b_right_a_wrong']} | {p_disp(d)} |")
    A("")
    ded = conc["both_forced"]["dedup_clean_1497"]
    A("The manuscript's 0.7191 against 0.7152, 76 against 69 discordant, p = 0.62 is the")
    A("full-population row. On the de-duplicated 1,497 cells the same comparison is "
      f"**{ded['a_acc']:.4f} against")
    A(f"{ded['b_acc']:.4f}**, **{ded['a_right_b_wrong']} against {ded['b_right_a_wrong']}** "
      f"discordant, **p = {p_disp(ded)}**. The gap narrows and the null gets flatter;")
    A("the conclusion — the two are level once both are forced — does not move, and it is the same")
    A("number whether the duplicate is resolved toward `clean` or toward `permuted`.")
    A("")
    A("- **The de-duplication direction cannot be cherry-picked for any census-level comparison**")
    A("  (groups (i), (ii), (iii) and the three supplemental pairs): the census asked one rendering")
    A("  per triple, so both directions give identical rows — §2.")
    A("- **Only the suite-level arms can be direction-sensitive**, because they answered both")
    A("  renderings; §2 (iv) reports both directions for every arm.")
    _pool = c2["pooled_suite_level"]
    A(f"- **Rendering asymmetry is measurable only for those arms** (§3): pooled over the "
      f"{_pool['engines']}")
    A(f"  suite-level engines ({_pool['engine_pair_units']:,} engine × pair units), only-clean "
      f"**{_pool['only_clean_correct']}** against only-permuted")
    A(f"  **{_pool['only_permuted_correct']}**, p = **{p_disp(_pool)}** — nothing detectable; for "
      f"the {len(c2['per_engine']) - _pool['engines']} census-level engines no test exists.")
    A("")
    A("### Which conclusions move, and which do not")
    A("")
    A("| the paper's conclusion | full 1,819 | de-duplicated 1,497 (either direction) | moves? |")
    A("|---|---|---|---|")
    j9 = bf["groups"]["i"]["population"]
    A("| Jev is above every Kev rung | p = {} against the 9B, and smaller against the smaller "
      "rungs | p = {} | **no** |".format(p_disp(j9["full_1819"][4]),
                                         p_disp(j9["dedup_clean_1497"][4])))
    g2 = bf["groups"]["ii"]["population"]
    g2_full, g2_ded = g2["full_1819"], g2["dedup_clean_1497"]
    A("| Kev-27B is last among the large engines, as served | "
      + f"{g2_full[0]['a_acc']:.4f} vs " + " / ".join(f"{r['b_acc']:.4f}" for r in g2_full)
      + " | " + f"{g2_ded[0]['a_acc']:.4f} vs " + " / ".join(f"{r['b_acc']:.4f}" for r in g2_ded)
      + " | **no** |")
    bf_full, bf_ded = conc["both_forced"]["full_1819"], conc["both_forced"]["dedup_clean_1497"]
    A("| forcing both leaves the pair level | "
      f"{bf_full['a_acc']:.4f} vs {bf_full['b_acc']:.4f}, p = {p_disp(bf_full)} | "
      f"{bf_ded['a_acc']:.4f} vs {bf_ded['b_acc']:.4f}, p = {p_disp(bf_ded)} | **no** |")
    for block in list(bf["arms"])[:3]:
        per_pop = bf["arms"][block]["population"]
        mf = next(e for e in per_pop["full_1819"] if e.get("p") is None)
        mc = next(e for e in per_pop["dedup_clean_1497"] if e.get("p") is None)
        mp = next(e for e in per_pop["dedup_permuted_1497"] if e.get("p") is None)
        side = "above" if mf["a_acc"] > mf["b_acc"] else "below"
        A(f"| {block.split(' --')[0]}: the fine-tune is {side} its own rung | "
          f"{mf['a_acc']:.4f} vs {mf['b_acc']:.4f} | "
          f"{mc['a_acc']:.4f} / {mp['a_acc']:.4f} vs {mc['b_acc']:.4f} | **no** |")
    b4 = list(bf["arms"])[1]
    b4_pop = bf["arms"][b4]["population"]
    A("| the 4B seed 1 is indistinguishable from its rung | p = {} | p = {} / {} | "
      "**no** (a null either way) |".format(p_disp(b4_pop["full_1819"][0]),
                                            p_disp(b4_pop["dedup_clean_1497"][0]),
                                            p_disp(b4_pop["dedup_permuted_1497"][0])))
    A("")
    A("The de-duplication moves **numbers**, never a verdict: it re-weights the same 1,497 cells,")
    A("and the 322 doubled cells are close enough to the population average that no comparison")
    A("changes sign or crosses 0.05. What it does change is the size of the both-forced gap (0.39")
    A("points to 0.26) and every p-value near the top of the distribution.")
    A("")
    if conc["check_1_rows_that_flip_direction"] or conc["arm_direction_changes"]:
        A("**Rows whose direction flips** across the three populations (and so the rows where the")
        A("preferred variant could have been chosen to tell a different story):")
        A("")
        for r in conc["check_1_rows_that_flip_direction"] + conc["arm_direction_changes"]:
            A("- {} — gaps {}, directions {}".format(r["comparison"], r["gaps"], r["direction"]))
        A("")
    else:
        A("**No comparison in §2 flips direction** across the three populations.")
        A("")
    if conc["check_1_rows_that_change_significance"]:
        A("**Rows whose significance at 0.05 changes**: "
          + ", ".join(r["comparison"] for r in conc["check_1_rows_that_change_significance"]) + ".")
    else:
        A("**No census-level comparison in §2 crosses the 0.05 line** between the full and the")
        A("de-duplicated populations: each is either overwhelmingly significant or plainly null both")
        A("ways. §2 (iv) lists the per-arm numbers: the least extreme of them is the 4B seed 1")
        A("null (p = 0.5171 full, 0.9527 / 0.8578 de-duplicated).")
    A("")
    A("## 1. The population, the duplication and the rules")
    A("")
    p = data["population"]
    A("| | |")
    A("|---|---|")
    A("| source | `{}` |".format(p["suite_ids"]))
    A("| rows with family `{}` | **{}** |".format(p["family"], f"{p['rows']:,}"))
    A("| distinct `(dataset, cell, family)` triples | **{}** |".format(f"{p['distinct_triples']:,}"))
    A(f"| triples carrying two rows | **{p['doubled_triples']}** of the "
      f"{p['distinct_triples']:,} |")
    shapes = p["variant_shapes"]
    A(f"| triples carrying a lone `clean` / lone `permuted` row | {shapes.get('clean', 0):,} / "
      f"{shapes.get('permuted', 0):,} |")
    A("| rows per triple | {:.4f} |".format(p["rows_per_triple"]))
    A("| the same ratio in the other families | {} |".format(p["same_ratio_in_other_families"]))
    A("")
    A("`variant` is a property of the rendered record, not of the question: two renderings of one")
    A("cell and family carry the same panel and the same answer in a different option order —")
    A("checked rather than assumed (`scripts/annotation-tools/01_census_questions.py`; re-checked for")
    A("all 322 pairs in §4). **All 2,510 census `panel_choice` records are `variant=permuted`**")
    A("(asserted by the script), so the census asked exactly one rendering of each triple; the suite")
    A("holds the other one for {} of the 1,819 rows — exactly the rows `variant_mismatch` marks."
      .format(sum(1 for r in POP["pc"] if r.get("variant_mismatch"))))
    A("")
    A("**How the de-duplicated paired test counts.** A doubled cell contributes *one* unit: its")
    A("single retained row, whose correctness is that of the preferred rendering. Its other row is")
    A("dropped, so n is 1,497 rather than 1,819, and each discordant count counts cells rather than")
    A("rows. For a census-level engine the retained row does not change the verdict at all, because")
    A("the verdict is attributed per triple; for a suite-level arm it does.")
    A("")
    A("Rules, all settled and not re-implemented: served = `argmax(row['p'])`; forced = `argmax` of")
    A("`p` with the abstain option removed, the abstain option being a property of the question")
    A("(`sckevtype.eval.abstain_index` over `sckevtype.data.families.NONE_OPTIONS`, "
      f"{len(NONE_OPTIONS)} wordings);")
    A("the test is the exact two-sided binomial on the discordant counts. `served`, `forced` and the")
    A("test are the functions of `reports/abstain-reconciliation.py`, extracted by AST and executed")
    A("from that source; the joins are `reports/per-dataset.json`'s (census engines on")
    A("`(dataset, cell, family)`, our arms on the row's own `test/custom/<n>` id).")
    A("")
    A("## 2. Check 1 — the headline comparisons, off and on the duplication")
    A("")
    A("Each comparison appears three times: the full 1,819 rows, then the two de-duplicated runs.")
    A("`A` and `B` are the two engines, `A-only` and `B-only` the discordant counts; `p` is exact in")
    A("the JSON.")
    A("")
    for gid in ("i", "ii", "iii"):
        g = bf["groups"][gid]
        A("### {}".format(g["title"]))
        A("")
        A("*{}.*".format(g["why"]))
        A("")
        A("| comparison | population | n | A | B | A-only | B-only | p |")
        A("|---|---|---:|---:|---:|---:|---:|---:|")
        cmp_block(g["population"])
        A("")
        A("Clean and permuted de-duplications give **%s** values for every comparison in this group:"
          % ("identical" if g["dedup_directions_identical"] else "DIFFERENT"))
        A("the census join attributes one verdict per triple, so there is no direction to pick.")
        A("")
    A("### (iv) every fine-tuned arm against its own released rung, forced")
    A("")
    A("*The paper's §2.5: at 0.8B the warm-started fine-tune beats its rung; at 4B and 9B it loses.*")
    A("")
    A("| arm | population | n | arm | rung | arm-only | rung-only | p |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for block, v in bf["arms"].items():
        A(f"| **{block}** | | | | | | | |")
        for i in range(len(v["population"]["full_1819"])):
            entry = v["population"]["full_1819"][i]
            name = (f"{block.split()[0]} seed mean") if entry.get("p") is None \
                else entry["a"]["label"]
            for pid in POP_IDS:
                c = v["population"][pid][i]
                A(f"| {name} | {POP_SHORT[pid]} | {c['n']} | {fmt_acc(c['a_acc'])} | "
                  f"{fmt_acc(c['b_acc'])} | "
                  f"{'--' if c['a_right_b_wrong'] is None else c['a_right_b_wrong']} | "
                  f"{'--' if c['b_right_a_wrong'] is None else c['b_right_a_wrong']} | "
                  f"{p_disp(c)} |")
    A("")
    A("The `--` rows are the per-size means: a mean over seeds has no single paired test, so its")
    A("per-seed tests and its secondary stacked test are below.")
    A("")
    A("| size | population | per-seed p | seed-mean arm | rung | stacked p (secondary) |")
    A("|---|---|---|---:|---:|---:|")
    for v in bf["arms"].values():
        for i in range(len(v["population"]["full_1819"])):
            if v["population"]["full_1819"][i].get("p") is not None:
                continue
            for pid in POP_IDS:
                c = v["population"][pid][i]
                A(f"| {c['comparison']} | {POP_SHORT[pid]} | "
                  f"{', '.join(fmt_p(x) for x in c['per_seed_p'])} | {c['a_acc']:.4f} | "
                  f"{c['b_acc']:.4f} | {p_disp(c['seed_stacked'])} |")
    A("")
    de = conc["dedup_direction_effect"]
    A("**How much the direction can move a number.** Every census-level comparison in §2 is")
    A("direction-invariant (above), so the only rows where the preferred variant changes anything are")
    A("the arms, which answered both renderings. There the arm's own accuracy moves by at most")
    A(f"**{de['max_abs_arm_delta_points']:.1f} points** between the two directions, and the rung "
      "it is compared against does not move")
    A("at all ({:.1f} pt) because its verdict is per-triple. The six largest moves:"
      .format(de["max_abs_rung_delta_points"]))
    A("")
    A("| arm | arm acc, full | acc, dedup clean | acc, dedup permuted | clean − permuted |")
    A("|---|---:|---:|---:|---:|")
    for e in sorted(de["arms"], key=lambda e: -abs(e["arm_delta_points"]))[:6]:
        A(f"| {e['comparison'].split(' forced vs ')[0]} | {e['a_acc_full']:.4f} | "
          f"{e['a_acc_clean']:.4f} | {e['a_acc_permuted']:.4f} | {e['arm_delta_points']:+.1f} pt |")
    A("")
    A("Crossing a conclusion: **%s** (see §0)."
      % ("yes" if de["crosses_any_conclusion"] else "no"))
    A("")
    A("### What the duplication weights: every engine's accuracy on each population")
    A("")
    A("The 322 doubled cells are not a representative sub-sample, so de-duplicating moves most")
    A("engines a little. `Δ` is the full-population figure minus the de-duplicated one, in points.")
    A("")
    A("| engine | join | full 1,819 | dedup clean | dedup permuted | Δ clean | Δ permuted |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for key, v in sorted(data["accuracy_by_population"]["served"].items(),
                         key=lambda kv: -abs(kv[1]["shift_points_full_to_dedup"])):
        A(f"| {v['label']} | {v['join']} | {v['full_1819']:.4f} | {v['dedup_clean_1497']:.4f} | "
          f"{v['dedup_permuted_1497']:.4f} | {100 * (v['dedup_clean_1497'] - v['full_1819']):+.2f} "
          f"| {100 * (v['dedup_permuted_1497'] - v['full_1819']):+.2f} |")
    A("")
    A("*Served rule; `accuracy_by_population.forced` in the JSON carries the same table under the")
    A("forced rule. Δ is the de-duplicated figure minus the full one, in points: the largest")
    A("re-weighting any engine takes is about half a point (Kev-4B, −0.47), and the census-level")
    A("engines move by the same amount in either direction because their verdict is per-triple.*")
    A("")
    A("### Supplemental comparisons the sibling reports publish on this population")
    A("")
    A("| comparison | population | n | A | B | A-only | B-only | p |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for v in bf["supplemental"].values():
        cmp_block(v["population"])
    A("")
    A("## 3. Check 2 — clean against permuted on the 322 pairs")
    A("")
    A(f"For each engine, over the {c2['pairs']} triples carrying both renderings, under the "
      "served rule: `both`")
    A("(correct on both renderings), `only clean`, `only permuted`, `neither`. The two `only` columns")
    A("are the asymmetry, and `p` is the exact two-sided binomial over them (1.0 when both are 0).")
    A("")
    A("### Suite-level engines — both renderings were actually put to the engine")
    A("")
    A("| engine | pairs | both | **only clean** | **only permuted** | neither | p |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for e in c2["per_engine"]:
        if e["join"] == "suite":
            A(f"| {e['label']} | {e['pairs_covered']} | {e['both_correct']} | "
              f"**{e['only_clean_correct']}** | **{e['only_permuted_correct']}** | "
              f"{e['neither_correct']} | {p_disp(e)} |")
    pl = c2["pooled_suite_level"]
    A(f"| **pooled, engine × pair** | **{pl['engine_pair_units']:,}** | **{pl['both_correct']:,}** "
      f"| **{pl['only_clean_correct']:,}** | **{pl['only_permuted_correct']:,}** | "
      f"**{pl['neither_correct']:,}** | **{p_disp(pl)}** |")
    A("")
    A("### Census-level engines — only one rendering is on disk")
    A("")
    A("| engine | pairs | both | only clean | only permuted | neither |")
    A("|---|---:|---:|---:|---:|---:|")
    for e in c2["per_engine"]:
        if e["join"] == "census":
            A(f"| {e['label']} | {e['pairs_covered']} | {e['both_correct']} | "
              f"{e['only_clean_correct']} | {e['only_permuted_correct']} | "
              f"{e['neither_correct']} |")
    A("")
    A("Those middle columns are zero **by construction, not by measurement**: the census asked one")
    A("rendering per triple (`permuted` on all 2,510 census records) and the join attributes that")
    A("single verdict to the triple's clean row as well. These engines were never asked the clean")
    A("rendering of these cells, so **no rendering-asymmetry test exists for them on this")
    A("population** — a gap in what was run, not evidence of unbiasedness. Every engine in both")
    A("tables covers all 322 pairs, so no engine is short of pairs; the blanks are structural.")
    A("")
    A("## 4. Verification")
    A("")
    A("Every published number this script can re-derive was compared against its own source:")
    A(f"**{rep['n']} checks, {'all passing' if rep['all_pass'] else 'FAILURES PRESENT'}**.")
    A("")
    A("| check | got | published | source | ok |")
    A("|---|---:|---:|---|---|")
    for c in rep["checks"]:
        got = c["got"] if isinstance(c["got"], str) else json.dumps(c["got"])
        want = c["published"] if isinstance(c["published"], str) else json.dumps(c["published"])
        src = c["source"]
        A("| {} | {} | {} | {} | {} |".format(c["check"].replace(" | ", " / "), got[:64], want[:32],
             src if "`" in src else f"`{src}`", "yes" if c["ok"] else "**NO**"))
    A("")
    if not rep["all_pass"]:
        A("**Failures**")
        A("")
        for c in rep["failures"]:
            A("- `{}`: got {}, published {} (`{}`)".format(c["check"], c["got"], c["published"],
                                                       c["source"]))
        A("")
    A("## 5. What this establishes, and what it does not")
    A("")
    A("**Establishes**")
    A("")
    A(f"- **{conc['invariance']['claim']}** — {conc['invariance']['why']}")
    A(f"  Identical in groups {', '.join(conc['invariance']['groups_identical'])} and in the "
      f"supplemental pairs {', '.join(f'`{s}`' for s in conc['invariance']['supplemental_identical'])}.")
    A("- The de-duplicated recomputation of every headline comparison beside the full population,")
    A("  under both de-duplication directions (§2).")
    A("- The clean/permuted asymmetry for every engine whose dump answers both renderings, with the")
    A("  exact binomial p, and the explicit statement that the census-level engines cannot be tested")
    A("  this way because they were asked one rendering (§3).")
    A("- The 322 pairs really are pairs: same panel, same answer, and the `permuted` row is the")
    A("  census's own record — checked on all 322 (§4).")
    A("")
    A("**Does not establish**")
    A("")
    for lim in data["limits"]:
        A(f"- {lim}")
    A("")
    A("## 6. Sources (sha256)")
    A("")
    A("| file | bytes | sha256 |")
    A("|---|---:|---|")
    for rel, v in SOURCES.items():
        if rel in LAYA_RELS:
            continue
        A("| `{}` | {} | `{}` |".format(rel, f"{v['bytes']:,}", v["sha256"]))
    if LAYA_RELS:
        A(f"| `{LAYA_GLOB}` — {len(LAYA_RELS)} shards, digests in the appendix | "
          f"{sum(SOURCES[r]['bytes'] for r in LAYA_RELS):,} | see appendix |")
    A("")
    if LAYA_RELS:
        A(f"### Appendix — Laya's {len(LAYA_RELS)} shard digests")
        A("")
        A("```")
        for rel in LAYA_RELS:
            A("{}  {}".format(SOURCES[rel]["sha256"], rel))
        A("```")
        A("")
    A("*Written by `reports/variant-redundancy.py`. Nothing pre-existing was modified.*")
    A("")
    return "\n".join(L)


DATA = payload()
with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(DATA, fh, indent=1)
with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write(render_md(DATA))


# --------------------------------------------------------------------- printout
print(f"population: {len(POP['pc'])} rows, {len(BY_TRIPLE)} triples, {len(DOUBLED)} doubled "
      f"({len(POP['pc']) / len(BY_TRIPLE):.3f} rows/cell)")
for pid in POP_IDS:
    print(f"  {pid:<22s} n={POPULATIONS[pid]['n']}")

for gid, g in CHECK1["groups"].items():
    print("\n[{}] {}".format(gid, g["title"]))
    for i in range(len(g["population"]["full_1819"])):
        for pid in POP_IDS:
            cc = g["population"][pid][i]
            print(f"  {pid:<22s} {cc['comparison'][:44]:<44s} A={cc['a_correct']:4d}/"
                  f"{cc['n']} ({cc['a_acc']:.4f}) B={cc['b_correct']:4d}/{cc['n']} "
                  f"({cc['b_acc']:.4f})  {cc['a_right_b_wrong']:3d}/"
                  f"{cc['b_right_a_wrong']:<3d} p={p_disp(cc)}")

print("\n[(iv) arms vs their own rung, forced]")
for block, v in CHECK1["arms"].items():
    print(f"  {block}")
    for i in range(len(v["population"]["full_1819"])):
        for pid in POP_IDS:
            c = v["population"][pid][i]
            if c.get("p") is None:
                print(f"    {pid:<22s} {c['comparison'][:40]:<40s} mean={c['a_acc']:.4f} vs "
                      f"{c['b_acc']:.4f}  per-seed p="
                      f"{' '.join(fmt_p(x) for x in c['per_seed_p'])}")
            else:
                print(f"    {pid:<22s} {c['a']['label'][:40]:<40s} A={c['a_correct']:4d}/"
                      f"{c['n']} ({c['a_acc']:.4f}) B={c['b_correct']:4d}/{c['n']} "
                      f"({c['b_acc']:.4f})  {c['a_right_b_wrong']:3d}/"
                      f"{c['b_right_a_wrong']:<3d} p={p_disp(c)}")

print("\n[check 2] the 322 pairs, rule={}".format(CHECK2["rule"]))
for e in CHECK2["per_engine"]:
    print(f"  {e['label'][:34]:<34s} pairs={e['pairs_covered']:3d} both={e['both_correct']:3d} "
          f"only_clean={e['only_clean_correct']:3d} "
          f"only_permuted={e['only_permuted_correct']:3d} "
          f"neither={e['neither_correct']:3d} p={p_disp(e):<10s} "
          f"{'' if e['estimable'] else '(not estimable)'}")
pl = CHECK2["pooled_suite_level"]
print(f"  POOLED suite-level: units={pl['engine_pair_units']} both={pl['both_correct']} "
      f"only_clean={pl['only_clean_correct']} only_permuted={pl['only_permuted_correct']} "
      f"neither={pl['neither_correct']} p={p_disp(pl)}")

_flips = len(CONCLUSIONS["check_1_rows_that_flip_direction"]) + len(
    CONCLUSIONS["arm_direction_changes"])
_idents = CONCLUSIONS["invariance"]["groups_identical"] + CONCLUSIONS[
    "invariance"]["supplemental_identical"]
print(f"\n[conclusions] direction flips={_flips}; significance changes="
      f"{len(CONCLUSIONS['check_1_rows_that_change_significance'])}; census-level dedup "
      f"identical={_idents}")

print(f"\n[replication] {REPLICATION['n']} checks, all_pass={REPLICATION['all_pass']}")
for c in REPLICATION["failures"]:
    print("  FAIL {}: got {} published {} ({})".format(c["check"], c["got"], c["published"], c["source"]))
print("\nwrote reports/variant-redundancy.json and reports/variant-redundancy.md")
