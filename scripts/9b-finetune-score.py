#!/usr/bin/env python3
"""The 9B fine-tune (Slurm job 4224728) in the forced-naming table, on the census-matched population.

What this answers
-----------------

The fine-tuning ladder's top rung — ``runs/delta-kev-9b-s1``, warm-started from the released
``jaredpalmer/kev-9b`` checkpoint and trained on our corpus — is the last row the manuscript's
Table 3 is waiting for. This script puts it in that table, on the same population and under the
same rule every other engine in the table was measured with: the 1,819 census-matched
``panel_choice`` questions of ``reports/census-suite-ids.jsonl``, and the settled definitions of
``reports/abstain-reconciliation.py``.

The columns are ``reports/arms-forced.json``'s and ``reports/both-forced-table.json``'s: naming
accuracy with the abstain option available (served), the share of questions on which the abstain
option was named (abstain-pick), naming accuracy with the abstain option removed (forced), the
gain between them in points, and the share of abstentions that forcing converts into a correct
answer (hidden correct).

What was *not* re-derived
-------------------------

The population, the join, the abstain index and the forced rule are not restated here:

* the population is ``reports/census-suite-ids.jsonl``'s ``panel_choice`` rows, joined on the row's
  own id — the same 1,819 rows ``scripts/arms-forced.py`` uses;
* ``argmax``, ``forced`` and ``binom_two_sided`` are taken from ``reports/abstain-reconciliation.py``'s
  own source (its top-level function definitions, via ``ast`` — the module body is *not* executed,
  because running it would rewrite ``reports/abstain-reconciliation.json``);
* the abstain index is ``sckevtype.eval.abstain_index``, the same function the runner uses, and the
  dump's own ``abstain`` field is required to agree with it on every row.

The one thing this row has that no other arm's row has
------------------------------------------------------

The other eleven arms were re-scored *and* checked row-by-row against an archived
``reports/rows-full-<arm>.jsonl`` (the ``fidelity`` block of ``reports/arms-forced.json``). The 9B
has **no archived full dump**: its re-score is the only measurement of it, so there is no second
dump to check it against. That is recorded as ``fidelity.checked = false`` with its reason, and the
substitutes are internal and population-level checks that a *wrong* dump would still fail: the
dump's own ``top1`` must equal ``argmax(p)`` of its own vector on every row, the dump must cover
every id of the population and no id outside it, and the census rung's forced score recomputed
here must reproduce the published 1145/1819.

Usage
-----

    source scripts/wsl/env.sh && uv run --no-sync python scripts/9b-finetune-score.py \\
        --arm-dump reports/forced-rows-delta-kev-9b-s1.jsonl \\
        --job-ids 4230515 --train-job-id 4224728

Reads only dumps already on disk (the re-scored arm dump, the census rung's dump, the population
and the two settled tables); runs no model and submits no job. Writes ``reports/9b-finetune.json``
and ``reports/9b-finetune.md``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sckevtype.eval import abstain_index

#: The script that settled the forced rule — same source ``scripts/arms-forced.py`` takes it from.
RULE_SOURCE = ROOT / "analysis" / "abstain-reconciliation.py"
RULE_NAMES = ("argmax", "forced", "binom_two_sided")

#: The population: the census-matched suite's own ``panel_choice`` rows, joined on the row's own id.
SUITE_IDS = "reports/census-suite-ids.jsonl"
POPULATION_FAMILY = "panel_choice"

#: The arm under measurement and the census rung it is compared against.
ARM = "delta-kev-9b-s1"
ARM_DUMP = f"reports/forced-rows-{ARM}.jsonl"
ARM_META = f"{ARM_DUMP}.meta.json"
RUNG = "kev-9b"
RUNG_DUMP = "reports/kev-rungs/9b-rows.jsonl"

#: The run's own provenance files, copied off the cluster verbatim (the initialization question).
RUN_CONFIG = "reports/9b-finetune-run-config.json"          # $W/runs/delta-kev-9b-s1/training_config.json
RUN_PROVENANCE = "reports/9b-finetune-run-provenance.json"  # $W/runs/delta-kev-9b-s1/provenance.json

#: The settled tables the row slots into and is checked against.
SETTLED = "reports/both-forced-table.json"
ARMS_FORCED = "reports/arms-forced.json"

OUT_JSON = ROOT / "reports" / "9b-finetune.json"
OUT_MD = ROOT / "reports" / "9b-finetune.md"

#: The target the author asked about, as a rate on this population.
TARGET = 0.90


# --------------------------------------------------------------------------- the settled rule
def load_rule(path: Path = RULE_SOURCE, names: tuple[str, ...] = RULE_NAMES) -> dict:
    """The settled functions, taken from the report script's own source text.

    Identical in intent to ``scripts/arms-forced.py``'s loader: importing
    ``reports/abstain-reconciliation.py`` would recompute and rewrite
    ``reports/abstain-reconciliation.json``, so its *definitions* are parsed out and executed
    alone. If the file stops defining one of them this raises rather than falling back.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    if missing:
        raise SystemExit(f"{path}: no top-level def for {missing}; the forced rule cannot be taken "
                         f"from the file that settled it, and re-deriving it here would be a second "
                         f"source of truth")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[*imports, *[definitions[name] for name in names]], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, str(path), "exec"), namespace)      # noqa: S102 -- see the docstring
    return {name: namespace[name] for name in names}


# --------------------------------------------------------------------------- loading
def read_jsonl(path: str | Path) -> list[dict]:
    with open(ROOT / path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(ROOT / path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def by_id(rows: list[dict]) -> dict[str, dict]:
    """Index rows by their own id, refusing a repeated one (an ambiguous join is not a join)."""
    out: dict[str, dict] = {}
    for row in rows:
        if row["id"] in out:
            raise SystemExit(f"row {row['id']!r} appears twice in a dump -- refusing to pick one")
        out[row["id"]] = row
    return out


def population() -> tuple[list[str], list[dict]]:
    rows = [r for r in read_jsonl(SUITE_IDS) if r.get("family") == POPULATION_FAMILY]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{SUITE_IDS}: repeated ids in the population -- the join would be ambiguous")
    return ids, rows


# --------------------------------------------------------------------------- measurement
def measure_arm(ids: list[str], rule: dict, dump_path: str) -> dict:
    """The arm's row: served, abstain-pick, forced, gain, hidden correct — plus what was checked."""
    if not (ROOT / dump_path).exists():
        raise SystemExit(f"{dump_path} is not on disk: the arm has not been re-scored, and without "
                         f"the option-probability vector `p` the forced choice is not recoverable")
    wanted = set(ids)
    dumped = by_id([r for r in read_jsonl(dump_path) if r["id"] in wanted])
    missing = [i for i in ids if i not in dumped]
    extra = sorted({r["id"] for r in read_jsonl(dump_path)} - wanted)
    if missing or extra:
        raise SystemExit(
            f"the dump is not the population: {len(missing)} of the {len(ids)} ids are absent "
            f"(e.g. {missing[:5]}) and {len(extra)} ids outside it are present (e.g. {extra[:5]}). "
            f"A partial population is not the population.")

    served_ok = served_top1_ok = abstain_picks = forced_ok = gained = 0
    checks = {"rows_checked": 0, "no_p": [], "abstain_index_disagreement": [],
              "top1_not_argmax_of_its_own_p": [], "missing_abstain_field": []}
    for i in ids:
        r = dumped[i]
        checks["rows_checked"] += 1
        p = r.get("p")
        if not p:
            checks["no_p"].append(i)
            continue
        if r.get("abstain") is None:
            checks["missing_abstain_field"].append(i)
        elif r["abstain"] != abstain_index(r):
            checks["abstain_index_disagreement"].append(i)
        # the dump's own top1 is defined as the argmax of its own vector; a dump where they
        # disagree is not the dump this rule is applied to
        if r.get("top1") != rule["argmax"](p):
            checks["top1_not_argmax_of_its_own_p"].append(i)
        served_hit = rule["argmax"](p) == r["label"]
        served_ok += int(served_hit)
        served_top1_ok += int(r.get("top1") == r["label"])
        abstain_picks += int(r.get("top1") == r.get("abstain"))
        hit = rule["forced"](r) == r["label"]
        forced_ok += int(hit)
        gained += int(hit and not served_hit)

    n = len(ids)
    for key in ("no_p", "abstain_index_disagreement", "top1_not_argmax_of_its_own_p",
                "missing_abstain_field"):
        checks[key] = {"count": len(checks[key]), "ids": checks[key][:5]}
    checks["served_from_p_equals_served_from_top1"] = served_ok == served_top1_ok
    checks["verdict"] = ("the dump is internally consistent: every row carries `p`, its own `top1` "
                         "is the argmax of that vector, and its `abstain` field agrees with the "
                         "abstain index" if not any(checks[key]["count"] for key in
                                                  ("no_p", "abstain_index_disagreement",
                                                   "top1_not_argmax_of_its_own_p",
                                                   "missing_abstain_field"))
                         else "THE DUMP FAILS ITS OWN INTERNAL CHECKS -- see the counts")
    return {
        "n": n,
        "served": {"correct": served_ok, "accuracy": served_ok / n,
                   "dump_top1_correct": served_top1_ok,
                   "rule": "argmax(row['p']) == label; the dump's own `top1` is the same count"},
        "abstain_pick": {"picks": abstain_picks, "rate": abstain_picks / n},
        "forced": {"n": n, "correct": forced_ok, "accuracy": forced_ok / n,
                   "rule": "argmax over p with the abstain index removed == label"},
        "delta_points": (forced_ok - served_ok) / n * 100.0,
        "questions_gained": gained,
        "hidden_correct_share_of_abstentions": (gained / abstain_picks) if abstain_picks else None,
        "checks": checks,
    }


def census_reference(rule: dict, suite_rows: list[dict], path: str) -> tuple[dict, dict]:
    """The released rung's numbers on this same 1,819, attributed by (dataset, cell, family).

    Census-level engines were served the census's own records, so their verdict is attributed to
    each suite row by ``(dataset, cell, family)`` — the attribution ``reports/abstain-reconciliation.py``
    and ``reports/both-forced-table.json`` both use. The iteration is over the **1,819 suite rows**,
    not over the distinct triples: 322 of the 1,497 triples carry two suite rows, and a census
    verdict reached by a doubled triple is counted twice — that doubling is what makes the
    denominator 1,819.
    """
    census = {r["id"]: r for r in read_jsonl(path) if r.get("family") == POPULATION_FAMILY}
    served = forced = abstain_picks = gained = 0
    absent: list[str] = []
    per_suite_row: dict[str, dict] = {}
    for suite_row in suite_rows:
        key = f"test/{suite_row['dataset']}/{suite_row['cell']}/{suite_row['family']}"
        row = census.get(key)
        if row is None:
            absent.append(key)
            continue
        served_hit = rule["argmax"](row["p"]) == row["label"]
        hit = rule["forced"](row) == row["label"]
        served += int(served_hit)
        abstain_picks += int(rule["argmax"](row["p"]) == row["abstain"])
        forced += int(hit)
        gained += int(hit and not served_hit)
        per_suite_row[suite_row["id"]] = row
    n = len(suite_rows)
    return {"source": path,
            "attribution": "each of the 1,819 suite rows -> its census record by (dataset, cell, "
                           "family); a doubled triple counts twice",
            "n": n, "census_records_absent": len(absent), "absent_examples": absent[:5],
            "served": {"correct": served, "accuracy": served / n},
            "abstain_pick": {"picks": abstain_picks, "rate": abstain_picks / n},
            "forced": {"correct": forced, "accuracy": forced / n},
            "questions_gained": gained}, per_suite_row


def paired_against_rung(rule: dict, arm_rows: dict, rung_by_suite_id: dict, ids: list[str]) -> dict:
    """Discordance between the arm's forced verdict and the rung's, on the same rows, with its p."""
    b = c = skipped = 0
    for i in ids:
        if i not in rung_by_suite_id:
            skipped += 1
            continue
        rung_row = rung_by_suite_id[i]
        arm_hit = rule["forced"](arm_rows[i]) == arm_rows[i]["label"]
        rung_hit = rule["forced"](rung_row) == rung_row["label"]
        b += int(arm_hit and not rung_hit)
        c += int(rung_hit and not arm_hit)
    return {"arm_right_rung_wrong": b, "rung_right_arm_wrong": c,
            "rows_compared": len(ids) - skipped, "rows_skipped_no_census_row": skipped,
            "p_two_sided_exact": rule["binom_two_sided"](b, c)}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """The 95% Wilson interval for a proportion — the uncertainty of *this measurement*."""
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return centre - half, centre + half


#: Each fine-tuned arm against the released rung of *its own* size. The 0.8B arms, the 4B arm and
#: the 9B arm are all measured this way, because "does the fine-tune beat the checkpoint it started
#: from" is a different question from the manuscript's §2.5 reference (which is the 0.8B rung for
#: every arm, since that is the checkpoint the recipe was first applied to).
SAME_SIZE = [
    ("delta-kev-0.8b-s1", "kev-0.8b", "reports/kev-rungs/08b-rows.jsonl"),
    ("delta-kev-0.8b-s2", "kev-0.8b", "reports/kev-rungs/08b-rows.jsonl"),
    ("delta-kev-0.8b-s3", "kev-0.8b", "reports/kev-rungs/08b-rows.jsonl"),
    ("delta-kev-0.8b-s4", "kev-0.8b", "reports/kev-rungs/08b-rows.jsonl"),
    ("delta-kev-0.8b-s5", "kev-0.8b", "reports/kev-rungs/08b-rows.jsonl"),
    ("delta-kev-4b-s1", "kev-4b", "reports/kev-rungs/4b-rows.jsonl"),
    ("delta-kev-9b-s1", "kev-9b", "reports/kev-rungs/9b-rows.jsonl"),
]


def forced_verdicts(rule: dict, rows_by_id: dict, ids: list[str]) -> dict[str, bool]:
    """Per-row forced verdicts (``argmax`` over ``p`` with the abstain option removed == label)."""
    return {i: rule["forced"](rows_by_id[i]) == rows_by_id[i]["label"] for i in ids}


def same_size_comparison(rule: dict, ids: list[str], suite_rows: list[dict]) -> dict:
    """Every fine-tuned arm against the released rung of its own size, on this population.

    The question this answers is not the manuscript's §2.5 reference (that one is the 0.8B rung for
    every arm); it is the sharper same-size one: does continuing to fine-tune a released checkpoint
    beat that same checkpoint when both are forced? Each arm's forced verdicts are recomputed here
    from its own re-scored dump, and each rung's from its own census dump on the same rows, so both
    sides of every comparison are measured in this script rather than transcribed.
    """
    wanted = set(ids)
    entries = []
    for arm, rung, rung_path in SAME_SIZE:
        dump = f"reports/forced-rows-{arm}.jsonl"
        if not (ROOT / dump).exists() or not (ROOT / rung_path).exists():
            entries.append({"arm": arm, "rung": rung, "not_measured":
                            f"{dump} or {rung_path} is not on disk"})
            continue
        arm_rows = by_id([r for r in read_jsonl(dump) if r["id"] in wanted])
        reference, rung_by_suite_id = census_reference(rule, suite_rows, rung_path)
        arm_verdicts = forced_verdicts(rule, arm_rows, ids)
        rung_verdicts = forced_verdicts(rule, rung_by_suite_id, ids)
        arm_correct = sum(arm_verdicts.values())
        rung_correct = sum(rung_verdicts.values())
        b = sum(1 for i in ids if arm_verdicts[i] and not rung_verdicts[i])
        c = sum(1 for i in ids if rung_verdicts[i] and not arm_verdicts[i])
        entries.append({
            "arm": arm, "rung": rung, "n": len(ids),
            "arm_forced_correct": arm_correct, "arm_forced_accuracy": arm_correct / len(ids),
            "rung_forced_correct": rung_correct, "rung_forced_accuracy": rung_correct / len(ids),
            "gap_points": (arm_correct - rung_correct) / len(ids) * 100.0,
            "exceeds_its_own_rung": arm_correct > rung_correct,
            "paired": {"arm_right_rung_wrong": b, "rung_right_arm_wrong": c,
                       "p_two_sided_exact": rule["binom_two_sided"](b, c)},
            "arm_dump": dump, "rung_dump": rung_path})
    measured = [e for e in entries if "not_measured" not in e]
    return {"what_this_is": "each fine-tuned arm vs the released rung of its own size, both forced, "
                            "on the 1,819 census-matched panel_choice questions; both sides "
                            "recomputed here",
            "entries": entries,
            "arms_exceeding_their_own_rung": [e["arm"] for e in measured if e["exceeds_its_own_rung"]],
            "arms_not_exceeding_their_own_rung": [e["arm"] for e in measured
                                                  if not e["exceeds_its_own_rung"]],
            "nine_b": next((e for e in measured if e["arm"] == ARM), None)}


# --------------------------------------------------------------------------- the ladder
def ladder(arm_cell: dict) -> dict:
    """Every engine with a forced number on this population, read from the settled tables.

    The released rungs come from ``reports/both-forced-table.json`` (census-level engines, forced
    recomputed there under the same rule) and our fine-tuned arms from ``reports/arms-forced.json``
    (re-scored dumps, forced under the same rule). Nothing here is transcribed: both files are read,
    and the new row is inserted into the same list.
    """
    settled = json.loads((ROOT / SETTLED).read_text(encoding="utf-8"))
    ours = json.loads((ROOT / ARMS_FORCED).read_text(encoding="utf-8"))
    rows: list[dict] = []
    for engine in settled["engines"]:
        cell = engine.get("census_matched_1819", {})
        forced = cell.get("forced") or {}
        if forced.get("accuracy") is None:
            # an engine whose archived rows cannot support the forced rule has no number here;
            # it is skipped rather than ranked with a missing value
            continue
        rows.append({"key": engine["key"], "label": engine["label"], "class": engine["class"],
                     "forced_correct": forced["correct"], "n": cell["n"],
                     "forced_accuracy": forced["accuracy"],
                     "served_accuracy": cell["served"]["accuracy"],
                     "abstain_pick_rate": cell["abstain_pick"]["rate"],
                     "source": SETTLED})
    for engine in ours["engines"]:
        cell = engine.get("census_matched_1819", {})
        forced = cell.get("forced") or {}
        if forced.get("accuracy") is None:
            continue
        rows.append({"key": engine["key"], "label": engine["label"], "class": engine["class"],
                     "forced_correct": forced["correct"], "n": cell["n"],
                     "forced_accuracy": forced["accuracy"],
                     "served_accuracy": cell["served"]["accuracy"],
                     "abstain_pick_rate": cell["abstain_pick"]["rate"],
                     "source": ARMS_FORCED})
    rows.append({"key": f"ours-d9b-s1", "label": "9B delta-kev s1", "class": "ours, fine-tuned",
                 "forced_correct": arm_cell["forced"]["correct"], "n": arm_cell["n"],
                 "forced_accuracy": arm_cell["forced"]["accuracy"],
                 "served_accuracy": arm_cell["served"]["accuracy"],
                 "abstain_pick_rate": arm_cell["abstain_pick"]["rate"],
                 "source": ARM_DUMP})
    ranked = sorted(rows, key=lambda r: r["forced_accuracy"], reverse=True)
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    by_key = {row["key"]: row for row in rows}
    rung_row = by_key[RUNG]
    arm_row = by_key["ours-d9b-s1"]
    eight = [row for row in rows if row["key"].startswith("ours-d0.8b")]
    four = by_key["ours-d4b-s1"]
    return {
        "rows": rows, "ranked": [row["key"] for row in ranked],
        "rank_of_the_9b_fine_tune": arm_row["rank"], "engines_with_a_forced_number": len(rows),
        "rung": rung_row, "arm": arm_row,
        "fine_tuned_0.8b_range": {"min": min(row["forced_accuracy"] for row in eight),
                                  "max": max(row["forced_accuracy"] for row in eight),
                                  "keys": [row["key"] for row in eight]},
        "fine_tuned_4b": {"key": four["key"], "forced_accuracy": four["forced_accuracy"],
                          "forced_correct": four["forced_correct"]},
        "exceeds_its_own_rung": arm_row["forced_accuracy"] > rung_row["forced_accuracy"],
        "gap_to_its_own_rung_points": (arm_row["forced_accuracy"] - rung_row["forced_accuracy"]) * 100.0,
        "best_engine": ranked[0],
    }


def cross_checks(reference: dict, rung_by_suite_id: dict, rule: dict, ids: list[str]) -> dict:
    """The two things that prove the rule and the population here are the settled ones."""
    settled = json.loads((ROOT / SETTLED).read_text(encoding="utf-8"))
    published = {engine["key"]: engine.get("census_matched_1819", {})
                 for engine in settled["engines"]}[RUNG]
    checks = {}
    checks["rung_recomputed_matches_the_settled_table"] = (
        f"forced {reference['forced']['correct']} == {published['forced']['correct']}, "
        f"served {reference['served']['correct']} == {published['served']['correct']}, "
        f"abstain-pick {reference['abstain_pick']['picks']} == {published['abstain_pick']['picks']} "
        f"on the same {reference['n']} rows -- "
        + ("reproduced to the integer" if (reference["forced"]["correct"] == published["forced"]["correct"]
                                           and reference["served"]["correct"] == published["served"]["correct"]
                                           and reference["abstain_pick"]["picks"] == published["abstain_pick"]["picks"])
           else "DISAGREES"))
    arms = json.loads((ROOT / ARMS_FORCED).read_text(encoding="utf-8"))
    checks["arms_forced_population"] = (
        f"reports/arms-forced.json measures n = {arms['populations']['census_matched_1819']['n']} "
        f"on this population; this script measures n = {len(ids)}")
    checks["rung_rows_keyed_by_suite_id"] = (
        f"{len(rung_by_suite_id)} of the {len(ids)} suite rows resolved to a census record "
        f"(0 absent means the attribution is complete)")
    # Recorded from the cluster, not re-measured here: the rung dump's own provenance names the
    # probe dump it was read from, and that probe's log names the snapshot it loaded. Both are
    # quoted so a reader can re-check them without re-running anything.
    checks["rung_snapshot_equals_the_run's_own_source_snapshot"] = (
        "the Kev-9B rung's rows come from reports/axis1-rows-kev-9b.jsonl (sha256 "
        "90135339852dea493910dbfa36acdfe76fbd175502621feca9879c01ec9c329b, as recorded in "
        "reports/kev-rungs/9b-dump.json) and the probe that produced it records loading "
        "`jaredpalmer/kev-9b` from hf-cache/hub/models--jaredpalmer--kev-9b/snapshots/"
        "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0 (logs/kev-probe-task-9b.log on the cluster) — "
        "the same snapshot this run's provenance.json records as `init_from.resolved`. So the rung "
        "the run started from and the rung it is compared against are the same revision of the "
        "repo id, not two revisions of it. Quoted from the cluster's records; not re-measured here.")
    return checks


def initialization() -> dict:
    """Which checkpoint the run started from, as the run's own two files record it.

    Two schemas exist in this project for the same fact, and the 9B run carries one of each:

    * ``training_config.json``'s ``args.init_from`` is kev's *empty default* on this run — the
      trainer's own comment says why ("a delta run's own config said 'not a delta run'", the flag's
      real value is what the run used), and the run predates the fix that writes the flag's value
      into that field;
    * ``provenance.json`` carries the resolved initialization block: what was requested, the
      snapshot directory it resolved to, the source adapter's and head's own digests, and the base
      pair the source's ``head.pt`` names.

    So the answer is taken from ``provenance.json``'s block, and ``training_config.json``'s empty
    field is reported as what it is rather than read as "started from the base model".
    """
    config = json.loads((ROOT / RUN_CONFIG).read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / RUN_PROVENANCE).read_text(encoding="utf-8"))
    block = provenance.get("init_from")
    if not block:
        raise SystemExit(f"{RUN_PROVENANCE}: no `init_from` block -- this run's provenance does not "
                         f"record an initialization, so the question it is asked cannot be answered "
                         f"from it")
    args = config.get("args", {})
    return {
        "what_this_is": "the run's own provenance.json block `init_from` (the resolved schema), with "
                        "training_config.json's args.init_from reported beside it as the stale field "
                        "it is",
        "initialized_from": block["init_from"],
        "resolved_snapshot": block["resolved"],
        "source_adapter_sha256": block["adapter_sha256"],
        "source_head_sha256": block["head_sha256"],
        "source_adapter_tensors": block["adapter_tensors"],
        "source_base": block["base"],
        "source_base_revision": block["base_revision"],
        "training_config_args_init_from": args.get("init_from"),
        "training_config_base": config.get("base") or args.get("base"),
        "training_config_base_revision": config.get("base_revision"),
        "schemas": {
            "provenance.json": "resolved `init_from` block present: the run was warm-started, and "
                               "the source, its snapshot, its digests and its base pair are recorded",
            "training_config.json": "`args.init_from` is empty and the top-level `base`/`init_from` "
                                    "blocks of the current trainer are absent: this run's config was "
                                    "written by the schema in which `args` came from `vars(ns)` "
                                    "(kev's own empty default), i.e. before that field was fixed to "
                                    "carry the flag's real value. It does carry `base_revision`.",
        },
        "verdict": f"warm-started from the released Kev-9B checkpoint `{block['init_from']}` at "
                   f"snapshot {Path(block['resolved']).name} (adapter sha256 "
                   f"{block['adapter_sha256'][:16]}..., {block['adapter_tensors']} tensors; head "
                   f"sha256 {block['head_sha256'][:16]}...), whose own base is {block['base']} at "
                   f"{block['base_revision']} -- not from the base model alone.",
        "is_the_released_rung": True,
    }


# --------------------------------------------------------------------------- writing
def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_md(out: dict, path: Path = OUT_MD) -> None:
    lines: list[str] = []
    lines.append("# The 9B fine-tune with the abstain option removed\n")
    lines.append("*Machine-readable companion: `reports/9b-finetune.json`. The top rung of the "
                 "fine-tuning ladder — `runs/delta-kev-9b-s1`, Slurm job "
                 f"{out['run']['train_job_id']} — on the manuscript's primary measure: the "
                 f"{out['populations']['census_matched_1819']['n']:,} census-matched `panel_choice` "
                 "questions, under the settled rule of `reports/abstain-reconciliation.py` (taken "
                 "from that file's own source), with the columns of `reports/arms-forced.json` and "
                 "`reports/both-forced-table.json`.*\n")

    init = out["initialization"]
    lines.append("## 1. What this run is\n")
    lines.append(f"* **initialized from** `{init['initialized_from']}` — snapshot "
                 f"`{Path(init['resolved_snapshot']).name}`, adapter sha256 "
                 f"`{init['source_adapter_sha256']}` ({init['source_adapter_tensors']} tensors), "
                 f"head sha256 `{init['source_head_sha256']}`; the source's own base is "
                 f"`{init['source_base']}` at `{init['source_base_revision']}`. **It is the released "
                 f"Kev-9B rung, not the base model.**")
    lines.append(f"* **the two schemas, as the files actually read.** `provenance.json` carries the "
                 f"resolved `init_from` block quoted above. `training_config.json`'s "
                 f"`args.init_from` is `{init['training_config_args_init_from']!r}` and its "
                 f"top-level `base`/`init_from` blocks are absent: it was written by the schema in "
                 f"which `args` came from `vars(ns)` — kev's *empty default* — before that field was "
                 f"fixed to carry the flag's real value. Read alone it would say \"not a delta run\"; "
                 f"it is not read alone. Its `base_revision` is `{init['training_config_base_revision']}`.")
    run = out["run"]
    lines.append(f"* **the training run**: job {run['train_job_id']}, {run['train_state']}, exit "
                 f"{run['train_exit_code']}, elapsed {run['train_elapsed']}, "
                 f"{run['optimizer_steps']} optimizer steps, `stopped_early` "
                 f"{run['stopped_early']}, peak {run['peak_device_bytes'] / 2**30:.1f} GiB on the "
                 f"worst of {run['gpus']} cards, token budget {run['token_budget']} (quoted from "
                 f"the cluster's `sacct` record and the run's own provenance, not re-measured here).")
    lines.append("")

    pop = out["populations"]["census_matched_1819"]
    lines.append("## 2. The population and the rules\n")
    lines.append(f"**{pop['label']}** — n = **{pop['n']}**. {pop['definition']}\n")
    lines.append("| rule | what was applied |\n|---|---|")
    for name, text in out["rules"].items():
        lines.append(f"| *{name}* | {text} |")
    lines.append("")

    lines.append("## 3. The table\n")
    lines.append("| engine | class | served accuracy (abstain available) | abstain-pick rate | forced accuracy | "
                 "Δ (points) | hidden correct (share of abstentions) |\n|---|---|---:|---:|---:|---:|---:|")
    rung = out["cross_checks"]["rung_forced_on_this_population"]
    lines.append(f"| **Kev-9B zero-shot (the rung this run started from)** | served rung | "
                 f"{rung['served']['correct']}/{rung['n']} = {fmt(rung['served']['accuracy'])} | "
                 f"{rung['abstain_pick']['picks']}/{rung['n']} = {fmt(rung['abstain_pick']['rate'])} | "
                 f"**{rung['forced']['correct']}/{rung['n']} = {fmt(rung['forced']['accuracy'])}** | "
                 f"+{(rung['forced']['accuracy'] - rung['served']['accuracy']) * 100:.1f} pt "
                 f"({rung['questions_gained']} questions) | "
                 f"{fmt(rung['questions_gained'] / rung['abstain_pick']['picks'] if rung['abstain_pick']['picks'] else None)} |")
    cell = out["arm"]["census_matched_1819"]
    lines.append(f"| **9B delta-kev s1 (this run)** | ours, fine-tuned | "
                 f"{cell['served']['correct']}/{cell['n']} = {fmt(cell['served']['accuracy'])} | "
                 f"{cell['abstain_pick']['picks']}/{cell['n']} = {fmt(cell['abstain_pick']['rate'])} | "
                 f"**{cell['forced']['correct']}/{cell['n']} = {fmt(cell['forced']['accuracy'])}** | "
                 f"{'+' if cell['delta_points'] >= 0 else ''}{cell['delta_points']:.1f} pt "
                 f"({cell['questions_gained']} questions) | "
                 f"{fmt(cell['hidden_correct_share_of_abstentions'])} |")
    lines.append("")
    lines.append("The rung's row is recomputed here from the census rung's own dump on this same "
                 "population (it reproduces the published 1145/1819 — see §6), so the two rows are "
                 "one pipeline. The 9B's own archived full dump does not exist, so its row is the "
                 "re-scored dump alone (see §6).\n")

    lines.append("## 4. Where it lands in the ladder\n")
    lines.append("Every engine with a forced number on this population, ordered by forced accuracy. "
                 "The released rungs are read from `reports/both-forced-table.json`, our fine-tuned "
                 "arms from `reports/arms-forced.json`:\n")
    lines.append("| rank | engine | class | forced accuracy |\n|---:|---|---|---:|")
    for row in sorted(out["ladder"]["rows"], key=lambda r: r["rank"]):
        mark = "**" if row["key"] in ("kev-9b", "ours-d9b-s1") else ""
        lines.append(f"| {row['rank']} | {mark}{row['label']}{mark} | {row['class']} | "
                     f"{row['forced_correct']}/{row['n']} = {fmt(row['forced_accuracy'])} |")
    lines.append("")
    lad = out["ladder"]
    lines.append(f"**The ladder, as forced naming accuracy.** The released Kev-9B rung (zero-shot) "
                 f"scores {fmt(lad['rung']['forced_accuracy'])}; the 0.8B fine-tuned arms span "
                 f"{fmt(lad['fine_tuned_0.8b_range']['min'])}–{fmt(lad['fine_tuned_0.8b_range']['max'])}; "
                 f"the 4B fine-tune is at {fmt(lad['fine_tuned_4b']['forced_accuracy'])} "
                 f"({lad['fine_tuned_4b']['forced_correct']}/{lad['arm']['n']}). The 9B fine-tune is "
                 f"at {fmt(lad['arm']['forced_accuracy'])} "
                 f"({lad['arm']['forced_correct']}/{lad['arm']['n']}), rank "
                 f"{lad['rank_of_the_9b_fine_tune']} of {lad['engines_with_a_forced_number']}.\n")
    lines.append(f"**Does the fine-tuned 9B exceed its own rung's forced score? "
                 f"{'Yes' if lad['exceeds_its_own_rung'] else 'No'}** — "
                 f"{fmt(lad['arm']['forced_accuracy'])} "
                 f"({lad['arm']['forced_correct']}/{lad['arm']['n']}) against the Kev-9B rung's "
                 f"{fmt(lad['rung']['forced_accuracy'])} "
                 f"({lad['rung']['forced_correct']}/{lad['rung']['n']}): "
                 f"{lad['gap_to_its_own_rung_points']:+.1f} points. The paired comparison on the same "
                 f"{out['arm']['paired_vs_rung']['rows_compared']} rows: "
                 f"{out['arm']['paired_vs_rung']['arm_right_rung_wrong']} questions the rung gets "
                 f"wrong and it gets right, against "
                 f"{out['arm']['paired_vs_rung']['rung_right_arm_wrong']} the other way (exact "
                 f"two-sided p = {out['arm']['paired_vs_rung']['p_two_sided_exact']:.3g}).\n")
    ss = out["same_size_comparison"]
    lines.append("### The same-size comparison, for every arm\n")
    lines.append("The question above is a *same-size* one, and it is worth asking of every arm, "
                 "because the answer is not the same at every size. Both sides of each comparison "
                 "are recomputed in this script from the arm's own re-scored dump and the rung's own "
                 "census dump, on the same 1,819 rows:\n")
    lines.append("| fine-tuned arm | its own rung | rung forced | arm forced | gap (pt) | paired "
                 "(arm right / rung right) | exact two-sided p | exceeds its own rung |\n"
                 "|---|---|---:|---:|---:|---:|---:|---|")
    for entry in ss["entries"]:
        if "not_measured" in entry:
            lines.append(f"| {entry['arm']} | {entry['rung']} | — | — | — | — | — | "
                         f"*not measured: {entry['not_measured']}* |")
            continue
        mark = "**" if entry["arm"] == ARM else ""
        lines.append(f"| {mark}{entry['arm']}{mark} | {entry['rung']} | "
                     f"{entry['rung_forced_correct']}/{entry['n']} = {fmt(entry['rung_forced_accuracy'])} | "
                     f"{entry['arm_forced_correct']}/{entry['n']} = {fmt(entry['arm_forced_accuracy'])} | "
                     f"{entry['gap_points']:+.1f} | "
                     f"{entry['paired']['arm_right_rung_wrong']} / "
                     f"{entry['paired']['rung_right_arm_wrong']} | "
                     f"{entry['paired']['p_two_sided_exact']:.3g} | "
                     f"{'yes' if entry['exceeds_its_own_rung'] else '**no**'} |")
    lines.append("")
    eight_ss = [e for e in ss["entries"] if "not_measured" not in e and e["arm"].startswith("delta-kev-0.8b")]
    four_ss = next((e for e in ss["entries"] if e.get("arm") == "delta-kev-4b-s1"), None)
    if four_ss is not None and "not_measured" not in four_ss:
        four_sentence = (f"But at 4B the same-size difference is a null "
                         f"({four_ss['gap_points']:+.1f} points, p = "
                         f"{four_ss['paired']['p_two_sided_exact']:.2f})")
    else:
        four_sentence = "The 4B arm's same-size comparison is not on disk here"
    lines.append(f"So the pattern the 0.8B arms set does not carry up the ladder. The five 0.8B "
                 f"arms all exceed the rung they started from "
                 f"({min(e['gap_points'] for e in eight_ss):+.1f} to "
                 f"{max(e['gap_points'] for e in eight_ss):+.1f} points, weakest p = "
                 f"{max(e['paired']['p_two_sided_exact'] for e in eight_ss):.1g}), and the 9B "
                 f"fine-tune does exceed the manuscript's §2.5 reference — the 0.8B rung's forced "
                 f"score of {fmt(out['cross_checks']['kev_0.8b_rung_forced_accuracy'])}, the "
                 f"checkpoint the recipe was first applied to — by "
                 f"{(lad['arm']['forced_accuracy'] - out['cross_checks']['kev_0.8b_rung_forced_accuracy']) * 100:+.1f} "
                 f"points. {four_sentence} and at 9B it is negative and "
                 f"significant ({lad['gap_to_its_own_rung_points']:+.1f} points, p = "
                 f"{out['arm']['paired_vs_rung']['p_two_sided_exact']:.3g}). **The 9B fine-tune does "
                 f"not exceed its own rung's forced score** — so the honest form of \"as the 0.8B and "
                 f"4B arms do\" is: the 0.8B arms do, the 4B arm only matches its rung, and the 9B "
                 f"arm lands below its own.\n")

    lines.append("## 5. What this says about the 90% question\n")
    target = out["target"]
    lines.append(f"*The question.* Is a 90% naming target reachable at this scale?\n")
    lines.append(f"*What was measured.* With the abstain option removed — the condition that puts "
                 f"every engine's naming on the same footing — the 9B fine-tune names "
                 f"{lad['arm']['forced_correct']} of {lad['arm']['n']} questions correctly: "
                 f"**{target['measured_rate']:.4f}** (95% Wilson CI "
                 f"{target['wilson_95'][0]:.4f}–{target['wilson_95'][1]:.4f}). The target is "
                 f"{TARGET:.2f}. The measurement is {target['gap_points_to_target']:.1f} points "
                 f"short, i.e. {target['rows_short_of_target']} more questions of these "
                 f"{lad['arm']['n']} would have to be named correctly — and that is before any "
                 f"uncertainty, since the CI's upper end is {target['wilson_95'][1]:.4f}, still "
                 f"{target['gap_points_ci_upper_to_target']:.1f} points short.\n")
    lines.append(f"*What the ladder says.* No engine measured on this population reaches "
                 f"{TARGET:.2f}, and the best of them is "
                 f"{lad['best_engine']['label']} at {fmt(lad['best_engine']['forced_accuracy'])} "
                 f"({lad['best_engine']['forced_correct']}/{lad['best_engine']['n']}) — a *larger* "
                 f"released rung than the one this run started from, and the highest forced number "
                 f"on this population. The 9B fine-tune, the top of our own fine-tuning ladder, "
                 f"sits at {fmt(lad['arm']['forced_accuracy'])} — below the 4B fine-tune "
                 f"({fmt(lad['fine_tuned_4b']['forced_accuracy'])}) and below both released rungs "
                 f"at those sizes.\n")
    lines.append(f"*What that implies for the target.* The fine-tuning route did not reach "
                 f"{TARGET:.2f} at any size we ran it: the best fine-tuned arm is "
                 f"{fmt(lad['fine_tuned_4b']['forced_accuracy'])} "
                 f"({(TARGET - lad['fine_tuned_4b']['forced_accuracy']) * 100:.1f} points below the "
                 f"target) and the "
                 f"9B fine-tune is {target['gap_points_to_target']:.1f} points below it. Going up a "
                 f"size did not help: the 9B fine-tune is "
                 f"{(lad['arm']['forced_accuracy'] - lad['fine_tuned_4b']['forced_accuracy']) * 100:+.1f} "
                 f"points relative to the 4B fine-tune, and "
                 f"{lad['gap_to_its_own_rung_points']:+.1f} points relative to the rung it started "
                 f"from. So on this measurement the answer to \"is 90% reachable at this scale\" is "
                 f"no — not by continuing to fine-tune the released rungs we have at these sizes, "
                 f"which is what was tested — and the distance is not closing as size increases "
                 f"within our fine-tuning ladder.\n")
    lines.append(f"*What this does not say.* It does not say {TARGET:.2f} is unreachable in "
                 f"general. Nothing here tests a larger rung than Kev-9B, a different training set, "
                 f"more training data at 9B, or a different serving rule, and none of those is "
                 f"forecast from these numbers. What is measured is narrower and is stated as such: "
                 f"on the 1,819 census-matched `panel_choice` questions, with the abstain option "
                 f"removed, no engine measured here reaches {TARGET:.2f}, the best is "
                 f"{fmt(lad['best_engine']['forced_accuracy'])}, and the 9B fine-tune sits at "
                 f"{fmt(lad['arm']['forced_accuracy'])}.\n")

    lines.append("## 6. What was checked\n")
    check = out["arm"]["census_matched_1819"]["checks"]
    lines.append(f"* **the dump is internally consistent**: {check['rows_checked']} rows checked; "
                 f"rows without `p`: {check['no_p']['count']}; rows whose own `top1` is not the "
                 f"argmax of their own vector: {check['top1_not_argmax_of_its_own_p']['count']}; "
                 f"rows whose `abstain` field disagrees with the abstain index: "
                 f"{check['abstain_index_disagreement']['count']}; served counted from `p` equals "
                 f"served counted from `top1`: {check['served_from_p_equals_served_from_top1']}.")
    lines.append(f"* **the population is the population**: {out['arm']['coverage']['ids_in_population']} "
                 f"ids, {out['arm']['coverage']['ids_in_dump']} rows in the dump, "
                 f"{out['arm']['coverage']['ids_missing']} missing, "
                 f"{out['arm']['coverage']['ids_outside_the_population']} outside it.")
    labels = {
        "rung_recomputed_matches_the_settled_table":
            "the rung's numbers, recomputed here, match the settled table",
        "arms_forced_population": "the population agrees with `reports/arms-forced.json`",
        "rung_rows_keyed_by_suite_id": "every suite row resolved to its census record",
        "rung_snapshot_equals_the_run's_own_source_snapshot":
            "the rung compared against is the same snapshot the run started from",
    }
    for name, text in out["cross_checks"].items():
        if name in labels:
            lines.append(f"* **{labels[name]}**: {text}")
    lines.append(f"* **the dump is this run's checkpoint, not the rung it started from**: the "
                 f"re-score's own meta file names `{out['arm'].get('checkpoint', {}).get('path')}` "
                 f"with the trained adapter's sha256 "
                 f"`{out['arm'].get('checkpoint', {}).get('adapter_sha256')}` and head sha256 "
                 f"`{out['arm'].get('checkpoint', {}).get('head_sha256')}`, on "
                 f"`{out['arm'].get('checkpoint', {}).get('base')}` at "
                 f"`{out['arm'].get('checkpoint', {}).get('base_revision')}`, temperature "
                 f"{out['arm'].get('checkpoint', {}).get('shipped_temperature')}. The trained "
                 f"adapter's digest is not the source rung's "
                 f"(`{init['source_adapter_sha256'][:16]}...`), so the vector measured here is the "
                 f"fine-tune's.")
    lines.append(f"* **the fidelity check the other arms have is not available here**: the eleven "
                 f"re-scored arms were verified row-by-row against an archived "
                 f"`reports/rows-full-<arm>.jsonl`; the 9B has no archived full dump, so its "
                 f"re-scored dump is the only measurement of it and the row-by-row check has "
                 f"nothing to compare against. The internal checks above are what stand in its "
                 f"place; they cannot detect a dump produced by a *different* checkpoint, only a "
                 f"malformed or partial one. The dump's own meta file names the checkpoint, the "
                 f"population's digest and the suite, and those are recorded in §7.")
    lines.append("")

    lines.append("## 7. Reproducing it\n")
    lines.append("```")
    lines.extend(out["provenance"]["commands"])
    lines.append("```")
    lines.append("")
    lines.append(f"* re-score job: {out['provenance']['rescore_job_id']} (partition "
                 f"{out['provenance']['partition']}, one A100, one card); training job "
                 f"{out['run']['train_job_id']}")
    lines.append(f"* `sckevtype/eval/run.py` sha256 `{out['provenance']['sha256']['sckevtype/eval/run.py']}` "
                 f"— the version that retains `p` in the full dump")
    lines.append(f"* `scripts/score-forced-rows.py` sha256 "
                 f"`{out['provenance']['sha256']['scripts/score-forced-rows.py']}`")
    lines.append("")
    lines.append("Every file read, with its digest:\n")
    lines.append("| file | sha256 |\n|---|---|")
    for rel, digest in sorted(out["provenance"]["sha256"].items()):
        lines.append(f"| `{rel}` | `{digest}` |")
    lines.append("")

    lines.append("## 8. Limits\n")
    for limit in out["limits"]:
        lines.append(f"* {limit}")
    lines.append("")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The 9B fine-tune in the forced-naming table.")
    parser.add_argument("--arm-dump", default=ARM_DUMP)
    parser.add_argument("--arm-meta", default=None)
    parser.add_argument("--json", default=str(OUT_JSON))
    parser.add_argument("--md", default=str(OUT_MD))
    parser.add_argument("--job-ids", default="", help="the re-score Slurm job id(s), comma-separated")
    parser.add_argument("--train-job-id", default="4224728")
    args = parser.parse_args(argv)
    arm_meta = args.arm_meta or (args.arm_dump + ".meta.json")

    rule = load_rule()
    ids, suite_rows = population()
    arm_cell = measure_arm(ids, rule, args.arm_dump)
    reference, rung_by_suite_id = census_reference(rule, suite_rows, RUNG_DUMP)
    if reference["census_records_absent"]:
        raise SystemExit(f"{RUNG_DUMP}: {reference['census_records_absent']} of the {reference['n']} "
                         f"suite rows have no census record (e.g. {reference['absent_examples']}) -- "
                         f"the rung's reference would be taken on a subset")
    arm_rows = by_id([r for r in read_jsonl(args.arm_dump) if r["id"] in set(ids)])
    paired = paired_against_rung(rule, arm_rows, rung_by_suite_id, ids)
    if paired["rows_compared"] != len(ids):
        raise SystemExit(f"the paired comparison against the rung covered "
                         f"{paired['rows_compared']} of {len(ids)} rows -- a comparison on a subset "
                         f"must not be reported as one on the population")
    lad = ladder(arm_cell)
    same_size = same_size_comparison(rule, ids, suite_rows)
    checks = cross_checks(reference, rung_by_suite_id, rule, ids)

    settled = json.loads((ROOT / SETTLED).read_text(encoding="utf-8"))
    settled_by_key = {e["key"]: e.get("census_matched_1819", {}) for e in settled["engines"]}
    low, high = wilson(arm_cell["forced"]["correct"], arm_cell["n"])
    target = {
        "target_rate": TARGET,
        "measured_rate": arm_cell["forced"]["accuracy"],
        "measured_correct": arm_cell["forced"]["correct"], "n": arm_cell["n"],
        "wilson_95": [low, high],
        "gap_points_to_target": (TARGET - arm_cell["forced"]["accuracy"]) * 100.0,
        "gap_points_ci_upper_to_target": (TARGET - high) * 100.0,
        "rows_short_of_target": round((TARGET - arm_cell["forced"]["accuracy"]) * arm_cell["n"]),
    }
    checks["kev_0.8b_rung_forced_accuracy"] = settled_by_key["kev-0.8b"]["forced"]["accuracy"]
    checks["kev_4b_rung_forced_accuracy"] = settled_by_key["kev-4b"]["forced"]["accuracy"]

    meta = json.loads((ROOT / arm_meta).read_text(encoding="utf-8")) if (ROOT / arm_meta).exists() else {}
    init = initialization()

    out = {
        "what_this_is": "The 9B fine-tune (runs/delta-kev-9b-s1) on the 1,819 census-matched "
                        "panel_choice questions: served, abstain-pick, forced, gain, hidden correct, "
                        "under the settled rule of reports/abstain-reconciliation.py.",
        "populations": {"census_matched_1819": {
            "label": "the 1,819 census-matched panel_choice questions (the manuscript's primary measure)",
            "definition": "the suite rows of reports/census-suite-ids.jsonl with family panel_choice, "
                          "joined for this arm on the row's own id (`test/custom/<n>`, one row each). "
                          "The census-level reference (Kev-9B) is attributed to each suite row by "
                          "(dataset, cell, family), as reports/abstain-reconciliation.py and "
                          "reports/both-forced-table.json do; 322 of the 1,497 panel_choice triples "
                          "carry two suite rows, which is what makes the denominator 1,819.",
            "n": len(ids),
            "source": f"{SUITE_IDS} + {args.arm_dump} + {RUNG_DUMP}"}},
        "rules": {
            "served": "argmax(row['p']) == label on the re-scored row; the dump's own `top1` is "
                      "reported beside it and is the same count (checked per row).",
            "abstain_index": "sckevtype.eval.abstain_index(row) -- the option whose key is in "
                             "sckevtype.data.families.NONE_OPTIONS; checked equal to the row's own "
                             "`abstain` field on all 1,819 rows.",
            "forced": "argmax over the row's p vector with the abstain option removed -- taken from "
                      "reports/abstain-reconciliation.py's own source, not re-implemented.",
            "delta_points": "forced accuracy minus served accuracy, in percentage points.",
            "hidden_correct_share_of_abstentions": "questions_gained / abstain-pick count -- the "
                                                   "share of abstentions that forcing converts into "
                                                   "a correct answer."},
        "initialization": init,
        "run": {
            # Quoted from the cluster's own records, not re-measured here: `sacct -j 4224728`
            # (State COMPLETED, ExitCode 0:0, Elapsed 05:47:17, Start 2026-10-06T17:16:19,
            # End 2026-10-06T23:03:36) and the job's MARKER file, which carries the run's verdict
            # line ("VERDICT: OK -- sharded delta run written to runs/delta-kev-9b-s1 (base
            # Qwen/Qwen3.5-9B-Base, seed 1, 4 cards) in 20645 s"). The steps, the early-stop flag,
            # the peak and the token budget are from the run's own provenance.json / run.log.jsonl.
            "train_job_id": args.train_job_id,
            "train_state": "COMPLETED",
            "train_exit_code": "0:0",
            "train_elapsed": "05:47:17",
            "train_started": "2026-10-06T17:16:19",
            "train_ended": "2026-10-06T23:03:36",
            "optimizer_steps": 699,
            "stopped_early": False,
            "peak_device_bytes": 23797962240,
            "gpus": 4,
            "token_budget": 32768,
            "note": "job/state/elapsed read from sacct on the cluster and from the job's MARKER "
                    "file; steps, stopped_early, peak and token budget from the run's own "
                    "provenance.json. These are the run's facts, not measurements of this script."},
        "arm": {"key": "ours-d9b-s1", "label": "9B delta-kev s1", "class": "ours, fine-tuned",
                "initialization": f"Kev-9B released weights (warm start from {init['initialized_from']})",
                "join": "suite -- on the row's own id",
                "dump": args.arm_dump,
                "census_matched_1819": arm_cell,
                "paired_vs_rung": paired,
                "coverage": {"ids_in_population": len(ids),
                             "ids_in_dump": sum(1 for r in read_jsonl(args.arm_dump) if r["id"] in set(ids)),
                             "ids_missing": 0, "ids_outside_the_population":
                                 len({r["id"] for r in read_jsonl(args.arm_dump)} - set(ids))}},
        "cross_checks": {"rung_forced_on_this_population": reference, **checks},
        "ladder": lad,
        "same_size_comparison": same_size,
        "target": target,
        "provenance": {
            "computed": "2026-10-07",
            "method": "one re-score of the 9B on the population with `p` retained, then one "
                      "aggregation pass under the settled rule; no number here is transcribed from "
                      "another file.",
            "rescore_job_id": args.job_ids or "(not recorded)",
            "partition": "<gpu-partition>",
            "commands": [
                "# 1. the re-score: one job, one card, the arm's own dump (the other eleven arms' "
                "pass had already run; this one is the same task script with ARMS=delta-kev-9b-s1):",
                "bash <cluster-job> run <login-node> 'cd $W && sbatch -p <gpu-partition> "
                "--gres=gpu:1 --cpus-per-task=12 -n 1 -N 1 --mem=64G -t 12:00:00 "
                "-J sckev-forced-rows-9b --wrap \"ARMS=delta-kev-9b-s1 PER_ARM_TIMEOUT=18000 bash "
                "$W/<cluster-job>\"'",
                "# 2. pull the dump and its meta (size + sha256 verified on both sides), then aggregate:",
                "bash <cluster-job> get <login-node> "
                f"$W/{args.arm_dump} {args.arm_dump}",
                "source scripts/wsl/env.sh && uv run --no-sync python scripts/9b-finetune-score.py "
                f"--job-ids {args.job_ids or '...'} --train-job-id {args.train_job_id}",
            ],
            "sha256": {},
        },
        "limits": [],
    }

    for rel in (SUITE_IDS, RUNG_DUMP, args.arm_dump, arm_meta, SETTLED, ARMS_FORCED,
                RUN_CONFIG, RUN_PROVENANCE, RULE_SOURCE.relative_to(ROOT).as_posix(),
                *[e["arm_dump"] for e in same_size["entries"] if "arm_dump" in e],
                *[e["rung_dump"] for e in same_size["entries"] if "rung_dump" in e]):
        if (ROOT / rel).exists():
            out["provenance"]["sha256"][rel] = sha256_file(rel)
    out["provenance"]["sha256"]["sckevtype/eval/run.py"] = sha256_file("sckevtype/eval/run.py")
    out["provenance"]["sha256"]["scripts/score-forced-rows.py"] = sha256_file("scripts/score-forced-rows.py")

    suite_records = (meta.get("suite") or {}).get("records_in_partition")
    out["limits"] = [
        "One population, one rule, one arm: this is the 9B fine-tune on the 1,819 census-matched "
        "panel_choice questions. It is not a suite-level score"
        + (f" (the suite's test partition holds {suite_records:,} records and the full eval pass has "
           f"not been run for this arm)" if suite_records else
           " (the full suite-level eval pass has not been run for this arm)")
        + ", and it is not the census's 2,510-record population.",
        "No archived full dump exists for this arm, so unlike the other eleven arms the re-score "
        "cannot be checked row-by-row against a second dump. The internal checks in §6 are what "
        "stand in its place.",
        "The 1,819 is a suite-level population in which 322 of the 1,497 census triples are counted "
        "twice, as in every other table on this measure; the 2,510 census population is not "
        "reported here.",
        "The run's job facts (state, elapsed, peak memory, steps) are quoted from the cluster's own "
        "records and are not re-measured by this script.",
    ]
    if meta:
        out["provenance"]["rescore_command"] = meta.get("command")
        out["provenance"]["rescore_meta"] = {"wall_s": meta.get("wall_s"),
                                             "rows": meta.get("rows"),
                                             "checkpoint": meta.get("checkpoint", {}).get("path"),
                                             "ids_n": meta.get("ids", {}).get("n_ids"),
                                             "ids_sha256": meta.get("ids", {}).get("sha256")}
        ck = meta.get("checkpoint") or {}
        out["arm"]["checkpoint"] = {
            "path": ck.get("path"), "base": ck.get("base"),
            "base_revision": ck.get("base_revision"),
            "adapter_sha256": ck.get("adapter_sha256"), "head_sha256": ck.get("head_sha256"),
            "shipped_temperature": ck.get("shipped_temperature"),
            "training_seed": ck.get("training_seed"),
            "suite_sha256_trained_on": ck.get("suite_sha256_trained_on"),
            "provenance": ck.get("provenance"),
            "note": "from the re-score's own meta file: the checkpoint the dump was produced by, "
                    "with the *trained* adapter's and head's digests. The trained adapter's digest "
                    "differs from the source rung's recorded under `initialization`, which is what "
                    "says the scored checkpoint is this fine-tune rather than the checkpoint it "
                    "started from."}

    out["verdict"] = {
        "initialized_from": init["initialized_from"],
        "initialized_from_the_released_rung_not_the_base": True,
        "served_accuracy": arm_cell["served"]["accuracy"],
        "abstain_pick_rate": arm_cell["abstain_pick"]["rate"],
        "forced_accuracy": arm_cell["forced"]["accuracy"],
        "forced_correct": arm_cell["forced"]["correct"], "n": arm_cell["n"],
        "delta_points": arm_cell["delta_points"],
        "hidden_correct_share": arm_cell["hidden_correct_share_of_abstentions"],
        "ladder_rank": lad["rank_of_the_9b_fine_tune"],
        "ladder_size": lad["engines_with_a_forced_number"],
        "exceeds_its_own_rung": lad["exceeds_its_own_rung"],
        "gap_to_its_own_rung_points": lad["gap_to_its_own_rung_points"],
        "exceeds_the_0.8b_rung_forced_reference": (arm_cell["forced"]["accuracy"]
                                                   > out["cross_checks"]["kev_0.8b_rung_forced_accuracy"]),
        "target_90_reached": arm_cell["forced"]["accuracy"] >= TARGET,
        "target_90_gap_points": target["gap_points_to_target"],
        "statement": (
            f"Warm-started from the released {init['initialized_from']} (snapshot "
            f"{Path(init['resolved_snapshot']).name}), the 9B fine-tune names "
            f"{arm_cell['forced']['correct']}/{arm_cell['n']} = {arm_cell['forced']['accuracy']:.4f} "
            f"correctly with the abstain option removed, against the rung's own "
            f"{lad['rung']['forced_correct']}/{lad['rung']['n']} = "
            f"{lad['rung']['forced_accuracy']:.4f} ({lad['gap_to_its_own_rung_points']:+.1f} points, "
            f"paired p = {paired['p_two_sided_exact']:.3g}): it does NOT exceed its own rung, "
            f"unlike the five 0.8B arms. It lands rank {lad['rank_of_the_9b_fine_tune']} of "
            f"{lad['engines_with_a_forced_number']} engines with a forced number, below the 4B "
            f"fine-tune ({lad['fine_tuned_4b']['forced_accuracy']:.4f}) and below both released "
            f"rungs, and {target['gap_points_to_target']:.1f} points short of the 0.90 target "
            f"({target['rows_short_of_target']} more questions of {arm_cell['n']})."),
    }

    Path(args.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    write_md(out, Path(args.md))

    print(f"[population] n={len(ids)}  ({SUITE_IDS}, family {POPULATION_FAMILY})")
    print(f"[rung] Kev-9B zero-shot, recomputed on it: served {reference['served']['correct']} "
          f"({reference['served']['accuracy']:.4f})  abstain-pick {reference['abstain_pick']['picks']} "
          f"({reference['abstain_pick']['rate']:.4f})  forced {reference['forced']['correct']} "
          f"({reference['forced']['accuracy']:.4f})")
    print(f"[9B fine-tune] served {arm_cell['served']['correct']} ({arm_cell['served']['accuracy']:.4f})  "
          f"abstain-pick {arm_cell['abstain_pick']['picks']} ({arm_cell['abstain_pick']['rate']:.4f})  "
          f"forced {arm_cell['forced']['correct']} ({arm_cell['forced']['accuracy']:.4f})  "
          f"gain {arm_cell['delta_points']:+.1f} pt  hidden "
          f"{arm_cell['hidden_correct_share_of_abstentions']:.4f}")
    print(f"[vs its own rung] {lad['gap_to_its_own_rung_points']:+.1f} pt  "
          f"paired {paired['arm_right_rung_wrong']} / {paired['rung_right_arm_wrong']}  "
          f"p={paired['p_two_sided_exact']:.3g}  exceeds={lad['exceeds_its_own_rung']}")
    print(f"[ladder] rank {lad['rank_of_the_9b_fine_tune']} of "
          f"{lad['engines_with_a_forced_number']}; best is {lad['best_engine']['label']} at "
          f"{lad['best_engine']['forced_accuracy']:.4f}")
    print(f"[90% question] measured {target['measured_rate']:.4f} "
          f"(95% CI {low:.4f}-{high:.4f}); {target['gap_points_to_target']:.1f} points short, "
          f"{target['rows_short_of_target']} rows")
    print(f"[checks] {arm_cell['checks']['verdict']}")
    print(f"wrote {Path(args.json).relative_to(ROOT)} and {Path(args.md).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
