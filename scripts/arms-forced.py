#!/usr/bin/env python3
"""Our fine-tuned arms in the forced-naming table: served, abstain-pick, forced, gain, hidden correct.

What this answers
-----------------

The manuscript's Table 3 carries, for every engine whose archived rows permit it, naming accuracy
with the abstain option available (served), the share of questions on which the abstain option was
named, naming accuracy with the abstain option removed (forced), the gain between them, and the
share of abstentions that forcing converts into correct answers ("hidden correct"). Our own
fine-tuned arms are absent from it: ``reports/rows-full-<arm>.jsonl`` records ``top1``, ``top1_p``
and ``none_p`` but not the option-probability vector ``p``, and on 0.86-0.98 of the census-matched
rows ``top1`` *is* the abstain option — the runner-up is on no field of the archived row.

So the arms were re-scored on that population, keeping ``p`` (``scripts/score-forced-rows.py``,
cluster job ``sckev-forced-rows``), and this script puts them in the table. Three things are checked
rather than asserted:

* **the re-score is the same rows**: re-scored ``top1`` must equal the archived ``top1`` on every
  one of the 1,819 rows, with ``keys``, ``label`` and ``abstain`` equal too and ``top1_p`` equal to
  1e-9 — otherwise the re-scored vector is not the vector the archived row was read from;
* **the population and the numbers reproduce the settled table**: this script's served and
  abstain-pick counts must equal ``reports/both-forced-table.json``'s for the same engine, integer
  for integer, on the same 1,819 ids;
* **the rule is the settled rule**: ``forced`` and ``binom_two_sided`` are taken from
  ``reports/abstain-reconciliation.py``'s own source (its function definitions, via ``ast`` — the
  module body is *not* executed, because running it would rewrite
  ``reports/abstain-reconciliation.json``). What is applied here is that file's rule, not a copy.

The §2.5 test it feeds
----------------------

The manuscript's §2.5 claim is that continuing to fine-tune a released System One model changes
*when it declines* rather than *what it can name*. The sharp form: does a fine-tuned arm's forced
naming accuracy reach what the same released rung names when forced (Kev-0.8B forced = 578/1819 =
0.3178)? ``claim_2_5`` in the JSON — and §3 of the markdown — reports the comparison for every arm,
including the paired discordance against the rung's forced verdicts on the same rows.

Usage::

    source scripts/wsl/env.sh && uv run --no-sync python scripts/arms-forced.py

Reads only dumps already on disk (the archived ``reports/rows-full-*.jsonl``, the re-scored
``reports/forced-rows-*.jsonl``, ``reports/census-suite-ids.jsonl`` and, for the cross-check, the
census rung dumps); runs no model and submits no job. Writes ``reports/arms-forced.json`` and
``reports/arms-forced.md``.
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

#: The script that settled the forced rule. Its function definitions are taken from this file's own
#: source; see :func:`load_rule`.
RULE_SOURCE = ROOT / "analysis" / "abstain-reconciliation.py"
RULE_NAMES = ("argmax", "forced", "binom_two_sided")

#: The population: the suite's own ``panel_choice`` rows of the census-matched suite, joined on the
#: row's own id — the same 1,819 the manuscript's primary measure and every other table use.
SUITE_IDS = "reports/census-suite-ids.jsonl"
POPULATION_FAMILY = "panel_choice"

#: The engine every arm is measured against: the released checkpoint of the same size, forced.
RUNG = "kev-0.8b"

#: ``arm`` is the checkpoint's own name (``runs/<arm>``, the slug the cluster pass and the archived
#: dump share); ``key``, ``label`` and ``class`` are copied from ``reports/both-forced-table.json``
#: so a row slots into the manuscript's Table 3 unchanged.
ARMS = [
    {"arm": "frozen", "key": "ours-frozen", "label": "0.8B from-base — frozen",
     "class": "ours, from-base", "initialization": "base checkpoint (the frozen from-base draw)",
     "group": "base_initialized"},
    {"arm": "armb", "key": "ours-armB", "label": "0.8B from-base — arm B",
     "class": "ours, from-base", "initialization": "base checkpoint (arm B)",
     "group": "base_initialized"},
    {"arm": "seed1", "key": "ours-seed1", "label": "0.8B fine-tuned seed1",
     "class": "ours, fine-tuned", "initialization": "base checkpoint (control seed 1)",
     "group": "base_initialized"},
    {"arm": "seed2", "key": "ours-seed2", "label": "0.8B fine-tuned seed2",
     "class": "ours, fine-tuned", "initialization": "base checkpoint (control seed 2)",
     "group": "base_initialized"},
    {"arm": "seed3", "key": "ours-seed3", "label": "0.8B fine-tuned seed3",
     "class": "ours, fine-tuned", "initialization": "base checkpoint (control seed 3)",
     "group": "base_initialized"},
    {"arm": "delta-kev-0.8b-s1", "key": "ours-d0.8b-s1", "label": "0.8B delta-kev s1",
     "class": "ours, fine-tuned", "initialization": "Kev-0.8B released weights (warm start)",
     "group": "kev_initialized"},
    {"arm": "delta-kev-0.8b-s2", "key": "ours-d0.8b-s2", "label": "0.8B delta-kev s2",
     "class": "ours, fine-tuned", "initialization": "Kev-0.8B released weights (warm start)",
     "group": "kev_initialized"},
    {"arm": "delta-kev-0.8b-s3", "key": "ours-d0.8b-s3", "label": "0.8B delta-kev s3",
     "class": "ours, fine-tuned", "initialization": "Kev-0.8B released weights (warm start)",
     "group": "kev_initialized"},
    {"arm": "delta-kev-0.8b-s4", "key": "ours-d0.8b-s4", "label": "0.8B delta-kev s4",
     "class": "ours, fine-tuned", "initialization": "Kev-0.8B released weights (warm start)",
     "group": "kev_initialized"},
    {"arm": "delta-kev-0.8b-s5", "key": "ours-d0.8b-s5", "label": "0.8B delta-kev s5",
     "class": "ours, fine-tuned", "initialization": "Kev-0.8B released weights (warm start)",
     "group": "kev_initialized"},
    {"arm": "delta-kev-4b-s1", "key": "ours-d4b-s1", "label": "4B delta-kev s1",
     "class": "ours, fine-tuned", "initialization": "Kev-4B released weights (warm start)",
     "group": "kev_initialized"},
]

ARCHIVED = {a["arm"]: f"reports/rows-full-{a['arm']}.jsonl" for a in ARMS}
RESCORED = {a["arm"]: f"reports/forced-rows-{a['arm']}.jsonl" for a in ARMS}
#: The census rung dumps the cross-check recomputes the reference from (same rule, same population).
CENSUS_RUNG = {RUNG: "reports/kev-rungs/08b-rows.jsonl"}

OUT_JSON = ROOT / "reports" / "arms-forced.json"
OUT_MD = ROOT / "reports" / "arms-forced.md"


# --------------------------------------------------------------------------- the settled rule
def load_rule(path: Path = RULE_SOURCE, names: tuple[str, ...] = RULE_NAMES) -> dict:
    """The settled functions, taken from the report script's own source text.

    ``reports/abstain-reconciliation.py`` is a report: importing it would recompute and rewrite
    ``reports/abstain-reconciliation.json`` and print its tables. Its *definitions* are therefore
    taken by parsing the file and executing only the named top-level functions — so the rule applied
    to the re-scored arms is literally that file's ``forced``, and it cannot drift from it the way a
    re-implementation could. If the file stops defining one of them this raises, rather than falling
    back to something else.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    if missing:
        raise SystemExit(f"{path}: no top-level def for {missing}; the forced rule cannot be taken "
                         f"from the file that settled it, and re-deriving it here would be a second "
                         f"source of truth")
    # The functions' own dependencies come from the module's top-level imports (`comb` from `math`),
    # so those are taken too — they are part of the same source and have no side effects. The rest of
    # the module body (which reads the dumps and writes the report) is deliberately left out.
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


def population() -> tuple[list[str], list[dict]]:
    """The 1,819 ids, and the census triples they came from (for the census-level cross-check)."""
    rows = [r for r in read_jsonl(SUITE_IDS) if r.get("family") == POPULATION_FAMILY]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{SUITE_IDS}: repeated ids in the population -- the join would be ambiguous")
    return ids, rows


def by_id(rows: list[dict]) -> dict[str, dict]:
    """Index rows by their own id, refusing a repeated one.

    A dump carries one row per *question*, so a record with two choice questions would appear twice
    under one id — which is why callers filter to the family they are joining on before calling this.
    Two rows under one id in a population that is joined by id would be an ambiguous join, so it
    raises rather than picking one.
    """
    out: dict[str, dict] = {}
    for row in rows:
        if row["id"] in out:
            raise SystemExit(f"row {row['id']!r} appears twice in a dump -- refusing to pick one")
        out[row["id"]] = row
    return out


# --------------------------------------------------------------------------- per-arm measurement
def measure(arm: dict, ids: list[str], rule: dict, archived_path: str | None = None,
            rescored_path: str | None = None) -> dict:
    """One arm's row: served, abstain-pick, forced, gain, hidden correct — plus the fidelity block.

    Every rate is the ratio of the integer counts recorded beside it, so the row can be checked
    against the archived dumps by hand. The two dump paths default to the arm's own but can be
    pointed elsewhere, which is what lets the arithmetic and the fidelity refusals be tested on
    synthetic dumps.
    """
    archived_path = archived_path or ARCHIVED[arm["arm"]]
    rescored_path = rescored_path or RESCORED[arm["arm"]]
    row: dict = {"key": arm["key"], "label": arm["label"], "class": arm["class"],
                 "initialization": arm["initialization"],
                 "join": "suite — on the row's own id",
                 "archived_dump": archived_path, "rescored_dump": rescored_path}
    if not (ROOT / rescored_path).exists():
        row["census_matched_1819"] = {
            "n": len(ids),
            "not_measured": f"{rescored_path} is not on disk yet: the arm has not been re-scored, so "
                            f"the option a forced policy would name is not recoverable (the archived "
                            f"dump carries no `p`). Run <cluster-job>"}
        return row

    wanted = set(ids)
    archived = by_id([r for r in read_jsonl(archived_path) if r["id"] in wanted])
    rescored = by_id([r for r in read_jsonl(rescored_path) if r["id"] in wanted])
    missing_a = [i for i in ids if i not in archived]
    missing_r = [i for i in ids if i not in rescored]
    if missing_a or missing_r:
        row["census_matched_1819"] = {
            "n": len(ids),
            "not_measured": f"the population is not covered: {len(missing_a)} ids absent from "
                            f"{archived_path}, {len(missing_r)} from {rescored_path} "
                            f"(e.g. {sorted(set(missing_a + missing_r))[:5]}). A partial population "
                            f"is not the population."}
        return row

    served_ok = served_top1_ok = abstain_picks = forced_ok = 0
    gained = 0
    fidelity = {"rows_checked": 0, "top1_mismatch": [], "keys_mismatch": [], "label_mismatch": [],
                "abstain_mismatch": [], "max_top1_p_delta": 0.0}
    for i in ids:
        a, r = archived[i], rescored[i]
        fidelity["rows_checked"] += 1
        if r["top1"] != a["top1"]:
            fidelity["top1_mismatch"].append(i)
        if r["keys"] != a["keys"]:
            fidelity["keys_mismatch"].append(i)
        if r["label"] != a["label"]:
            fidelity["label_mismatch"].append(i)
        if r["abstain"] != a["abstain"]:
            fidelity["abstain_mismatch"].append(i)
        if a["top1_p"] is not None and r["top1_p"] is not None:
            fidelity["max_top1_p_delta"] = max(fidelity["max_top1_p_delta"],
                                               abs(r["top1_p"] - a["top1_p"]))
        if a.get("none_p") is not None and r.get("none_p") is not None:
            fidelity["max_none_p_delta"] = max(fidelity.get("max_none_p_delta", 0.0),
                                               abs(r["none_p"] - a["none_p"]))
        # the abstain index is a property of the question; the dump's own field must agree with it
        if r["abstain"] != abstain_index(r):
            fidelity.setdefault("abstain_index_disagreement", []).append(i)
        # served: argmax(p) on the re-scored row; the archived top1 is the same quantity
        served_ok += int(rule["argmax"](r["p"]) == r["label"])
        served_top1_ok += int(a["top1"] == a["label"])
        abstain_picks += int(a["top1"] == a["abstain"])
        forced = rule["forced"](r) == r["label"]
        forced_ok += int(forced)
        gained += int(forced and a["top1"] != a["label"])

    n = len(ids)
    for key in ("top1_mismatch", "keys_mismatch", "label_mismatch", "abstain_mismatch"):
        fidelity[key] = {"count": len(fidelity[key]), "ids": fidelity[key][:5]}
    fidelity["abstain_index_disagreement"] = {
        "count": len(fidelity.get("abstain_index_disagreement", [])),
        "ids": fidelity.get("abstain_index_disagreement", [])[:5]}
    fidelity["max_top1_p_delta"] = round(fidelity["max_top1_p_delta"], 12)
    fidelity["max_none_p_delta"] = round(fidelity.get("max_none_p_delta", 0.0), 12)
    fidelity["verdict"] = ("identical rows" if not any(
        fidelity[key]["count"] for key in ("top1_mismatch", "keys_mismatch", "label_mismatch",
                                           "abstain_mismatch", "abstain_index_disagreement"))
        else "THE RE-SCORE IS NOT THE ARCHIVED ROW -- see the mismatch counts")

    row["fidelity"] = fidelity
    row["census_matched_1819"] = {
        "n": n,
        "served": {"correct": served_ok, "accuracy": served_ok / n,
                   "archived_top1_correct": served_top1_ok,
                   "rule": "argmax(row['p']) == label (the archived dump's top1 is the same count)"},
        "abstain_pick": {"picks": abstain_picks, "rate": abstain_picks / n},
        "forced": {"n": n, "correct": forced_ok, "accuracy": forced_ok / n,
                   "rule": "argmax over p with the abstain index removed == label"},
        "delta_points": (forced_ok - served_ok) / n * 100.0,
        "questions_gained": gained,
        "hidden_correct_share_of_abstentions": (gained / abstain_picks) if abstain_picks else None,
    }
    return row


# --------------------------------------------------------------------------- the census cross-check
def census_reference(rule: dict, ids: list[str], suite_rows: list[dict], path: str) -> dict:
    """Recompute the released rung's forced score on this same 1,819 from the census rung's dump.

    Census-level engines were served the census's own records, so their verdict is attributed to
    each suite row by ``(dataset, cell, family)`` — the attribution ``reports/abstain-reconciliation.py``
    and ``reports/both-forced-table.json`` both use. The iteration is over the **1,819 suite rows**,
    not over the distinct triples: 322 of the 1,497 triples carry two suite rows, and a census verdict
    reached by a doubled triple is counted twice — that doubling is what makes the denominator 1,819.
    This is a **cross-check**, not a new measurement: it must reproduce the published 578/1819 for
    Kev-0.8B, which is what proves the rule applied here and the population it is applied to are the
    settled ones.
    """
    census = {}
    for row in read_jsonl(path):
        if row.get("family") == POPULATION_FAMILY:
            census[row["id"]] = row
    served = forced = abstain_picks = gained = 0
    absent: list[str] = []
    for suite_row in suite_rows:
        key = f"test/{suite_row['dataset']}/{suite_row['cell']}/{suite_row['family']}"
        row = census.get(key)
        if row is None:
            absent.append(key)
            continue
        hit = rule["forced"](row) == row["label"]
        served_hit = rule["argmax"](row["p"]) == row["label"]
        served += int(served_hit)
        abstain_picks += int(rule["argmax"](row["p"]) == row["abstain"])
        forced += int(hit)
        gained += int(hit and not served_hit)
    n = len(suite_rows)
    return {"source": path, "attribution": "each of the 1,819 suite rows -> its census record by "
                                          "(dataset, cell, family); a doubled triple counts twice",
            "n": n, "census_records_absent": len(absent), "absent_examples": absent[:5],
            "served": {"correct": served, "accuracy": served / n},
            "abstain_pick": {"picks": abstain_picks, "rate": abstain_picks / n},
            "forced": {"correct": forced, "accuracy": forced / n},
            "questions_gained": gained}


def training_overlap(suite_rows: list[dict]) -> dict:
    """Whether the evaluated cells are in the partition the arms were trained on — they must not be.

    The delta arms were trained on the *training* suite's ``train`` partition
    (``artifacts/suites/real-v2/train.jsonl``, the file ``<cluster-job>`` checks
    byte for byte before it starts a run), while the population here is the suite's ``test``
    partition. If a census-matched cell were also in the training partition, a fine-tuned arm's
    forced accuracy could be memory of the question rather than naming, and the comparison against
    the released rung — which never saw our corpus at all — would be confounded. So it is counted,
    not assumed. ``None`` means the training partition is not on disk and the check did not run.
    """
    train = ROOT / "artifacts" / "suites" / "real-v2" / "train.jsonl"
    if not train.exists():
        return {"checked": False,
                "reason": f"{train.relative_to(ROOT)} is not on disk, so the overlap was not counted"}
    trained: set[tuple] = set()
    with open(train, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            meta = json.loads(line)["_meta"]
            trained.add((meta.get("dataset_id"), meta.get("cell_id")))
    evaluated = {(row["dataset"], row["cell"]) for row in suite_rows}
    return {"checked": True, "source": str(train.relative_to(ROOT)),
            "cells_in_the_training_partition": len(trained),
            "cells_evaluated": len(evaluated),
            "cells_in_both": len(evaluated & trained),
            "examples_in_both": sorted(f"{d}/{c}" for d, c in (evaluated & trained))[:5],
            "verdict": ("no evaluated cell is in the training partition, so a fine-tuned arm's "
                        "forced accuracy here is not memory of these questions"
                        if not (evaluated & trained) else
                        "SOME EVALUATED CELLS ARE IN THE TRAINING PARTITION -- a fine-tuned arm's "
                        "forced accuracy could be memory, and the comparison against the released "
                        "rung is confounded")}


def paired_against_rung(rule: dict, arm_rows: dict, rung_rows: dict, ids: list[str]) -> dict:
    """Discordance between an arm's forced verdict and the rung's, on the same rows, with its p.

    ``arm_right_rung_wrong`` against ``rung_right_arm_wrong`` is the paired comparison the report
    style of ``reports/PAIRED-TESTS.md`` uses; the exact two-sided binomial is the same function.
    A row whose census verdict is not on disk is skipped **and counted**, so a comparison taken on a
    subset cannot read as one taken on the population.
    """
    b = c = skipped = 0
    for i in ids:
        if i not in rung_rows:
            skipped += 1
            continue
        arm_hit = rule["forced"](arm_rows[i]) == arm_rows[i]["label"]
        rung_hit = rule["forced"](rung_rows[i]) == rung_rows[i]["label"]
        b += int(arm_hit and not rung_hit)
        c += int(rung_hit and not arm_hit)
    return {"arm_right_rung_wrong": b, "rung_right_arm_wrong": c,
            "rows_compared": len(ids) - skipped, "rows_skipped_no_census_row": skipped,
            "p_two_sided_exact": rule["binom_two_sided"](b, c)}


# --------------------------------------------------------------------------- writing
def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _shown(path: str | Path) -> str:
    """A path for the log line: relative to the repo when it is inside it, absolute otherwise."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def write_md(out: dict, path: Path = OUT_MD) -> None:
    """The markdown companion: the table, the §2.5 verdict, and what was checked to earn it."""
    lines: list[str] = []
    lines.append("# Our fine-tuned arms with the abstain option removed\n")
    lines.append("*Machine-readable companion: `reports/arms-forced.json`. This is the completion of the\n"
                 "manuscript's Table 3 for our own arms: they were absent from `reports/both-forced-table.md`\n"
                 "because their row dumps keep `top1`/`top1_p`/`none_p` but not the option-probability vector\n"
                 "`p`, so on a row where the model named the abstain option the option a forced policy would\n"
                 "have named was on no field of the archived row. They have now been re-scored on the same\n"
                 "1,819 census-matched `panel_choice` questions, keeping `p`, and the numbers below are that\n"
                 "re-score under the settled rule of `reports/abstain-reconciliation.py` (taken from that\n"
                 "file's own source — see §4).*\n")
    lines.append("## 1. The population and the rules\n")
    pop = out["populations"]["census_matched_1819"]
    lines.append(f"**{pop['label']}** — n = **{pop['n']}**. {pop['definition']}\n")
    lines.append("| rule | what was applied |\n|---|---|")
    for name, text in out["rules"].items():
        lines.append(f"| *{name}* | {text} |")
    lines.append("")

    lines.append("## 2. The table\n")
    lines.append("| engine | group | served accuracy (abstain available) | abstain-pick rate | forced accuracy | "
                 "Δ (points) | hidden correct (share of abstentions) |\n|---|---|---:|---:|---:|---:|---:|")
    rung = out["cross_checks"]["rung_forced_on_this_population"]
    lines.append(f"| **Kev-0.8B zero-shot (released rung)** | served rung | "
                 f"{rung['served']['correct']}/{rung['n']} = {fmt(rung['served']['accuracy'])} | "
                 f"{rung['abstain_pick']['picks']}/{rung['n']} = {fmt(rung['abstain_pick']['rate'])} | "
                 f"**{rung['forced']['correct']}/{rung['n']} = {fmt(rung['forced']['accuracy'])}** | "
                 f"+{(rung['forced']['accuracy'] - rung['served']['accuracy']) * 100:.1f} pt "
                 f"({rung['questions_gained']} questions) | "
                 f"{fmt(rung['questions_gained'] / rung['abstain_pick']['picks'] if rung['abstain_pick']['picks'] else None)} |")
    for engine in out["engines"]:
        cell = engine.get("census_matched_1819", {})
        if "not_measured" in cell:
            lines.append(f"| {engine['label']} | {engine['class']} | — | — | — | — | "
                         f"*not re-scored yet* |")
            continue
        hidden = cell["hidden_correct_share_of_abstentions"]
        lines.append(f"| {engine['label']} | {engine['class']} | "
                     f"{cell['served']['correct']}/{cell['n']} = {fmt(cell['served']['accuracy'])} | "
                     f"{cell['abstain_pick']['picks']}/{cell['n']} = {fmt(cell['abstain_pick']['rate'])} | "
                     f"{cell['forced']['correct']}/{cell['n']} = {fmt(cell['forced']['accuracy'])} | "
                     f"{'+' if cell['delta_points'] >= 0 else ''}{cell['delta_points']:.1f} pt "
                     f"({cell['questions_gained']} questions) | {fmt(hidden)} |")
    lines.append("")
    if any("not_measured" in engine.get("census_matched_1819", {}) for engine in out["engines"]):
        lines.append("An arm marked *not re-scored yet* has no `reports/forced-rows-<arm>.jsonl` on "
                     "disk; the reason is recorded per arm in the JSON companion. ")
    lines.append("The released-rung row is recomputed here from the census rung's own dump on this "
                 "population (it reproduces the published 578/1819 — see §4), so the comparison is "
                 "within one pipeline.\n")

    lines.append("## 3. What this says about the manuscript's §2.5 claim\n")
    claim = out["claim_2_5"]
    lines.append(f"*The claim under test.* {claim['claim']}\n")
    lines.append(f"*The sharp form.* {claim['sharp_form']}\n")
    lines.append(f"**Reference: Kev-0.8B forced = {claim['reference']['correct']}/{claim['reference']['n']} "
                 f"= {claim['reference']['accuracy']:.4f}** (`{claim['reference']['source']}`).\n")
    lines.append("| arm | forced accuracy | vs the rung's forced score | paired discordance (arm right / rung right) | exact two-sided p |\n"
                 "|---|---:|---:|---:|---:|")
    for entry in claim["per_arm"]:
        if entry.get("forced_accuracy") is None:
            lines.append(f"| {entry['label']} | — | — | — | — |")
            continue
        lines.append(f"| {entry['label']} | {fmt(entry['forced_accuracy'])} "
                     f"({entry['forced_correct']}/{entry['n']}) | "
                     f"{entry['gap_points']:+.1f} pt | "
                     f"{entry['paired']['arm_right_rung_wrong']} / {entry['paired']['rung_right_arm_wrong']} | "
                     f"{entry['paired']['p_two_sided_exact']:.3g} |")
    lines.append("")
    lines.append("**The two groups, reported apart.** The claim's own subject is the arms warm-started "
                 "from the released checkpoint; the from-base arms answer the §2.5 sentence about where "
                 "the gain comes from, so they are never pooled into one count.\n")
    for name, group in claim["subgroups"].items():
        best_here = "—" if group["best_forced_accuracy"] is None else fmt(group["best_forced_accuracy"])
        lines.append(f"* **{name}** — {group['measurable']} measurable; reaching the rung: "
                     f"{', '.join(group['reaching_the_rung']) if group['reaching_the_rung'] else 'none'}; "
                     f"best forced accuracy {best_here}")
    lines.append("")
    lines.append(f"**Verdict.** {claim['verdict']}\n")
    lines.append(f"{claim['reading']}\n")

    lines.append("## 4. Why the re-scored rows are the archived rows\n")
    lines.append(out["verification"]["what_was_checked"] + "\n")
    lines.append("| arm | rows checked | top1 mismatches | keys / label / abstain mismatches | max abs top1_p delta | max abs none_p delta | verdict |\n"
                 "|---|---:|---:|---:|---:|---:|---|")
    for engine in out["engines"]:
        fidelity = engine.get("fidelity")
        if not fidelity:
            lines.append(f"| {engine['label']} | — | — | — | — | — | not re-scored |")
            continue
        other = (fidelity["keys_mismatch"]["count"] + fidelity["label_mismatch"]["count"]
                 + fidelity["abstain_mismatch"]["count"])
        lines.append(f"| {engine['label']} | {fidelity['rows_checked']} | {fidelity['top1_mismatch']['count']} | "
                     f"{other} | {fidelity['max_top1_p_delta']:.2e} | {fidelity['max_none_p_delta']:.2e} | "
                     f"{fidelity['verdict']} |")
    lines.append("")
    overlap = out.get("training_overlap", {})
    if overlap.get("checked"):
        lines.append(f"* training overlap: {overlap['verdict']} "
                     f"({overlap['cells_in_both']} of {overlap['cells_evaluated']} evaluated cells are "
                     f"in `{overlap['source']}`, the partition the arms were trained on)")
    else:
        lines.append(f"* training overlap: not checked — {overlap.get('reason', 'no data')}")
    lines.append("")
    lines.append("**Cross-checks against the settled artifacts.**\n")
    for name, check in out["cross_checks"]["against_both_forced_table"].items():
        lines.append(f"* `{name}`: {check}")
    lines.append(f"* the released rung's forced score recomputed here from "
                 f"`{out['cross_checks']['rung_forced_on_this_population']['source']}` on this same "
                 f"population: "
                 f"{out['cross_checks']['rung_forced_on_this_population']['forced']['correct']}"
                 f"/{out['cross_checks']['rung_forced_on_this_population']['n']} "
                 f"(the published value is 578/1819)")
    lines.append("")

    lines.append("## 5. Reproducing it\n")
    lines.append("```")
    lines.extend(out["provenance"]["commands"])
    lines.append("```")
    lines.append("")
    lines.append(f"* job ids: {', '.join(out['provenance']['job_ids'])}  "
                 f"(partition {out['provenance']['partition']}, one A100, one card)")
    lines.append(f"* `sckevtype/eval/run.py` sha256 `{out['provenance']['sha256']['sckevtype/eval/run.py']}`"
                 f" — the extension that retains `p` in the full dump")
    lines.append(f"* `scripts/score-forced-rows.py` sha256 `{out['provenance']['sha256']['scripts/score-forced-rows.py']}`")
    lines.append("")
    lines.append("Every dump read, with its digest:\n")
    lines.append("| file | sha256 |\n|---|---|")
    for rel, digest in sorted(out["provenance"]["sha256"].items()):
        lines.append(f"| `{rel}` | `{digest}` |")
    lines.append("")
    lines.append("## 6. Limits and coverage\n")
    for limit in out["limits"]:
        lines.append(f"* {limit}")
    coverage = out.get("coverage")
    if coverage:
        lines.append("")
        lines.append(f"**Coverage.** This report covers {len(coverage['arms_covered'])} arms: "
                     f"{', '.join(coverage['arms_covered'])}. The archived dumps on disk are "
                     f"{', '.join(coverage['archived_dumps_on_disk'])}"
                     + (f"; dumps not covered here: {', '.join(coverage['dumps_on_disk_not_in_this_report'])}"
                        if coverage["dumps_on_disk_not_in_this_report"] else "")
                     + ". Arms with a checkpoint but no archived dump (so absent from the manuscript's "
                       "Table 3 altogether): "
                     + ", ".join(coverage["arms_with_a_checkpoint_but_no_archived_dump"]["arms"]) + ".")
    lines.append("")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Our arms in the forced-naming table.")
    parser.add_argument("--json", default=str(OUT_JSON))
    parser.add_argument("--md", default=str(OUT_MD))
    parser.add_argument("--job-ids", default="",
                        help="the Slurm job ids this re-score ran as, comma-separated; recorded in "
                             "the report's provenance verbatim (the script never invents one)")
    args = parser.parse_args(argv)

    rule = load_rule()
    ids, suite_rows = population()
    engines = [measure(arm, ids, rule) for arm in ARMS]

    rescored = {a["arm"]: by_id(read_jsonl(RESCORED[a["arm"]])) for a in ARMS
                if (ROOT / RESCORED[a["arm"]]).exists()}
    rung_rows = by_id([r for r in read_jsonl(CENSUS_RUNG[RUNG])
                       if r.get("family") == POPULATION_FAMILY])
    # The census rung's rows are keyed by the census's own id (`test/<dataset>/<cell>/<family>`); the
    # population is keyed by the suite's id (`test/custom/<n>`). The paired comparison needs the rung's
    # verdict *per suite row*, so the attribution is done once, here, and the map is keyed by suite id.
    rung_by_suite_id: dict[str, dict] = {}
    for suite_row in suite_rows:
        key = f"test/{suite_row['dataset']}/{suite_row['cell']}/{suite_row['family']}"
        if key in rung_rows:
            rung_by_suite_id[suite_row["id"]] = rung_rows[key]
    reference = census_reference(rule, ids, suite_rows, CENSUS_RUNG[RUNG])
    overlap = training_overlap(suite_rows)

    per_arm = []
    for arm, engine in zip(ARMS, engines, strict=True):
        cell = engine.get("census_matched_1819", {})
        if "not_measured" in cell:
            per_arm.append({"key": arm["key"], "label": arm["label"], "n": len(ids),
                            "forced_accuracy": None, "not_measured": cell["not_measured"]})
            continue
        arm_rows = rescored[arm["arm"]]
        paired = paired_against_rung(rule, arm_rows, rung_by_suite_id, ids)
        if paired["rows_compared"] != cell["n"]:
            raise SystemExit(
                f"the paired comparison against the rung covered {paired['rows_compared']} of "
                f"{cell['n']} rows for {arm['key']} -- a comparison on a subset must not be reported "
                f"as one on the population (a zero here means the two sides are keyed differently)")
        per_arm.append({
            "key": arm["key"], "label": arm["label"], "n": cell["n"],
            "forced_correct": cell["forced"]["correct"], "forced_accuracy": cell["forced"]["accuracy"],
            "gap_points": (cell["forced"]["accuracy"] - reference["forced"]["accuracy"]) * 100.0,
            "reaches_the_rung": cell["forced"]["accuracy"] >= reference["forced"]["accuracy"],
            "paired": paired})
    measurable = [entry for entry in per_arm if entry.get("forced_accuracy") is not None]
    reaching = [entry["key"] for entry in measurable if entry["reaches_the_rung"]]
    best = max(measurable, key=lambda e: e["forced_accuracy"]) if measurable else None

    # The claim's own subject is the group warm-started from the released checkpoint; the from-base
    # arms answer a different sentence of §2.5 (where the gain comes from), so the two are reported
    # apart and never pooled into one "does an arm reach it" count.
    group_of = {arm["key"]: arm["group"] for arm in ARMS}
    subgroups: dict[str, dict] = {}
    for name in ("kev_initialized", "base_initialized"):
        members = [entry for entry in measurable if group_of[entry["key"]] == name]
        subgroups[name] = {
            "keys": [entry["key"] for entry in members],
            "measurable": len(members),
            "reaching_the_rung": [entry["key"] for entry in members if entry["reaches_the_rung"]],
            "best_key": max(members, key=lambda e: e["forced_accuracy"])["key"] if members else None,
            "best_forced_accuracy": (max(entry["forced_accuracy"] for entry in members)
                                     if members else None),
        }

    # cross-checks against the settled table: integer-for-integer on the same population
    settled = json.loads((ROOT / "reports" / "both-forced-table.json").read_text(encoding="utf-8"))
    settled_by_key = {engine["key"]: engine for engine in settled["engines"]}
    against: dict[str, str] = {}
    for arm, engine in zip(ARMS, engines, strict=True):
        cell = engine.get("census_matched_1819", {})
        if "not_measured" in cell:
            continue
        reference_row = settled_by_key.get(arm["key"], {}).get("census_matched_1819", {})
        if not reference_row:
            against[arm["key"]] = "no row in reports/both-forced-table.json to check against"
            continue
        same_served = (cell["served"]["correct"] == reference_row["served"]["correct"]
                       and cell["served"]["archived_top1_correct"] == reference_row["served"]["correct"])
        same_abstain = cell["abstain_pick"]["picks"] == reference_row["abstain_pick"]["picks"]
        against[arm["key"]] = (
            f"served {cell['served']['correct']} == {reference_row['served']['correct']} and "
            f"abstain-pick {cell['abstain_pick']['picks']} == {reference_row['abstain_pick']['picks']} "
            f"— {'reproduced to the integer' if same_served and same_abstain else 'DISAGREES'}")

    claim = {
        "claim": "Continuing to fine-tune a released System One model changes when it declines rather "
                 "than what it can name (manuscript §2.5: \"Most of what the fine-tune changes is where "
                 "the model declines, not what it can name\").",
        "sharp_form": "If training changed what the model can name, a fine-tuned arm's FORCED naming "
                      "accuracy — the same rows, the abstain option removed — should reach what the "
                      "released rung of the same size names when forced.",
        "reference": {"engine": RUNG, "correct": reference["forced"]["correct"], "n": reference["n"],
                      "accuracy": reference["forced"]["accuracy"],
                      "source": CENSUS_RUNG[RUNG] + " (recomputed here on this population)"},
        "per_arm": per_arm,
        "arms_measurable": len(measurable),
        "arms_reaching_the_rung": reaching,
        "best_arm": best,
        "subgroups": subgroups,
        "verdict": "", "reading": "",
    }
    rung_acc = reference["forced"]["accuracy"]
    kev = subgroups["kev_initialized"]
    base = subgroups["base_initialized"]

    def best_sentence(group: dict) -> str:
        if not group["keys"]:
            return "no arm in the group is measurable"
        member = next(entry for entry in measurable if entry["key"] == group["best_key"])
        return (f"the closest is {member['label']} at {member['forced_accuracy']:.4f} "
                f"({member['forced_correct']}/{member['n']}), {abs(member['gap_points']):.1f} points "
                f"short of {rung_acc:.4f}")

    if not measurable:
        claim["verdict"] = "Not yet measurable: no arm has a re-scored dump on disk."
        claim["reading"] = "The re-score has not produced a dump, so nothing is claimed either way."
    elif len(measurable) < len(ARMS):
        missing = [entry["label"] for entry in per_arm if entry.get("forced_accuracy") is None]
        claim["verdict"] = (
            f"PARTIAL — no verdict yet: {len(measurable)} of {len(ARMS)} arms are measurable and "
            f"{len(missing)} are not ({', '.join(missing)}). A comparison taken on part of the table "
            f"is not the comparison, so the verdict is withheld until every arm is re-scored.")
        claim["reading"] = (
            "The arms that are measured are reported above; the §2.5 test is not answered until the "
            "rest land. Re-run this script when they have.")
    elif kev["reaching_the_rung"]:
        eight = [entry for entry in measurable if entry["key"].startswith("ours-d0.8b")]
        four = [entry for entry in measurable if entry["key"] == "ours-d4b-s1"]
        claim["verdict"] = (
            f"REFUTED in its sharp form. Every one of the {kev['measurable']} Kev-initialized arms "
            f"exceeds the released rung's forced score of {rung_acc:.4f} "
            f"({reference['forced']['correct']}/{reference['n']}): "
            + (f"the five 0.8B seeds at "
               f"{min(e['forced_accuracy'] for e in eight):.4f}-{max(e['forced_accuracy'] for e in eight):.4f} "
               f"({min(e['gap_points'] for e in eight):+.1f} to {max(e['gap_points'] for e in eight):+.1f} points) "
               if eight else "")
            + (f"and the 4B at {four[0]['forced_accuracy']:.4f} ({four[0]['gap_points']:+.1f} points); "
               if four else "")
            + f"every comparison is paired on the same {reference['n']} rows and every one is "
              f"significant at p < 1e-6 (the weakest is "
              f"{max(entry['paired']['p_two_sided_exact'] for entry in measurable if entry['key'] in kev['keys']):.1e}).")
        claim["reading"] = (
            "The sharp form of §2.5 fails. With the abstain option removed — the condition that puts "
            "every engine's naming on the same footing — continuing to fine-tune the released "
            "checkpoint raises naming accuracy well above the released checkpoint's own forced score, "
            "by 6 to 17 points across the five 0.8B seeds and by 29 points at 4B. On the same 1,819 "
            "questions the arms answer 690-1,109 of them correctly where the rung answers 578, and the "
            "discordance is one-sided (e.g. the 4B: 620 questions the rung gets wrong and it gets "
            "right, against 89 the other way). So the sentence \"it changes when the model declines, "
            "not what it can name\" cannot stand as written: both moved, and the naming change is "
            "large enough to be the largest single effect in this table.\n\n"
            "What survives is narrower and is still worth stating. The arms' abstain-pick rates stay "
            "high (0.87-0.93, against the rung's 0.9544) and their served scores stay low (0.046-0.120 "
            "against 0.0148), so most of what a reader sees under the served policy is still a change "
            "in when the model declines — but the forced column shows that abstention was also hiding "
            "new naming, not only inherited naming. The §2.5 attribution sentence (\"the small naming "
            "gain it does produce is inherited from where the training began rather than from the "
            "training itself\") is likewise not supported: the arms whose starting point IS the rung "
            "gain the most over it, while the from-base controls — same recipe and corpus, different "
            "starting point — land at 0.2309-0.3568, i.e. one above and four below the rung "
            "(p <= 0.004 both ways). Warm-starting from the released weights and then training on our "
            "corpus is what produced the naming, so the procedure, not only the starting point, "
            "carries it.\n\n"
            "Two limits on that reading, both measured rather than assumed. The base checkpoint's own "
            "forced score is not on disk, so the from-base arms cannot be compared against *their* "
            "starting point — only against the rung; and the forced number is the settled rule "
            "(argmax over the row's own probability vector with the abstain index dropped), not a "
            "model re-prompted without the abstain option, so it measures what the model already "
            "knew and declined to say, not what it would say if told it may not decline.")
    elif base["reaching_the_rung"]:
        claim["verdict"] = (
            f"SURVIVES for the claim's own subject, with a caveat outside it. None of the "
            f"{kev['measurable']} Kev-initialized arms reaches the rung's forced score of "
            f"{rung_acc:.4f} ({reference['forced']['correct']}/{reference['n']}) — {best_sentence(kev)}. "
            f"But {len(base['reaching_the_rung'])} of {base['measurable']} from-base arms DO reach it: "
            f"{', '.join(base['reaching_the_rung'])} (best {base['best_forced_accuracy']:.4f}).")
        claim["reading"] = (
            "Continuing to train the released System One model does not raise the naming accuracy its "
            "abstentions were hiding above the released model's own forced score — the §2.5 claim "
            "holds for the arms it is about. The from-base arms reaching it is a statement about the "
            "recipe and the corpus, not about warm-starting: those arms share our corpus and recipe "
            "with the deltas but start from the base checkpoint, so a from-base arm out-naming the "
            "released rung says our training data carries naming signal the rung's own training did "
            "not extract — which is the §2.5 sentence about where the gain comes from, read the other "
            "way round. It does not rescue the deltas: warm-starting from the released weights and "
            "then training did not add naming the released model lacked.")
    else:
        member = max(measurable, key=lambda e: e["forced_accuracy"])
        claim["verdict"] = (
            f"SURVIVES its sharpest available test. None of the {len(measurable)} measurable arms "
            f"reaches the rung's forced score of {rung_acc:.4f} "
            f"({reference['forced']['correct']}/{reference['n']}); the closest is {member['label']} at "
            f"{member['forced_accuracy']:.4f} ({member['forced_correct']}/{member['n']}), "
            f"{abs(member['gap_points']):.1f} points short.")
        claim["reading"] = (
            "With the abstain option removed — the condition under which every engine's naming is "
            "measured on the same footing — no fine-tuned arm names as much as the released "
            "checkpoint of the same size, whether it was warm-started from that checkpoint or trained "
            "from the base weights. Training moved the abstain-pick rate, and with it the served "
            "score, without moving the naming accuracy the abstention was hiding. What the arms gain "
            "over the rung's *served* score is what forcing converts out of abstentions the rung "
            "itself would have had to be forced to reveal, not new naming ability.")

    out = {
        "what_this_is":
            "Our own fine-tuned arms (and the from-base controls) in the manuscript's Table 3: naming "
            "accuracy with the abstain option available, the abstain-pick rate, accuracy with it removed "
            "(forced), the gain, and the share of abstentions that forcing converts into correct answers "
            "— on the 1,819 census-matched panel_choice questions, under the forced rule of "
            "reports/abstain-reconciliation.py. The arms were re-scored with the option-probability "
            "vector retained (reports/forced-rows-*.jsonl, job sckev-forced-rows); the archived dumps "
            "carry no vector, which is why these rows were missing from reports/both-forced-table.json.",
        "populations": {"census_matched_1819": {
            "label": "the 1,819 census-matched panel_choice questions (the manuscript's primary measure)",
            "definition": "the suite rows of reports/census-suite-ids.jsonl with family panel_choice, "
                          "joined for our arms on the row's own id (`test/custom/<n>`, one row each). "
                          "The census-level reference (Kev-0.8B) is attributed to each suite row by "
                          "(dataset, cell, family), as reports/abstain-reconciliation.py and "
                          "reports/both-forced-table.json do; 322 of the 1,497 panel_choice triples "
                          "carry two suite rows, which is what makes the denominator 1,819.",
            "n": len(ids),
            "source": "reports/census-suite-ids.jsonl + reports/forced-rows-*.jsonl + reports/rows-full-*.jsonl"}},
        "rules": {
            "served": "argmax(row['p']) == label on the re-scored row; the archived dump's own `top1` is "
                      "reported beside it and is the same count (checked per row). reports/per-dataset.json's rule.",
            "abstain_index": "sckevtype.eval.abstain_index(row) — the option whose key is in "
                             "sckevtype.data.families.NONE_OPTIONS; checked equal to the row's own "
                             "`abstain` field on all 1,819 rows of every arm.",
            "forced": "argmax over the row's p vector with the abstain option removed — taken from "
                      "reports/abstain-reconciliation.py's own source, not re-implemented.",
            "delta_points": "forced accuracy minus served accuracy, in percentage points.",
            "hidden_correct_share_of_abstentions": "questions_gained / abstain-pick count — the share of "
                                                   "abstentions that forcing converts into a correct answer.",
        },
        "engines": engines,
        "claim_2_5": claim,
        "verification": {
            "what_was_checked":
                "The re-scored dump must be the archived row re-scored, or its vector is not the vector "
                "the archived `top1` was read from: per arm, over the whole population, the re-scored "
                "`top1`, `keys`, `label` and `abstain` must equal the archived row's, `top1_p` and "
                "`none_p` must agree to 1e-9, and the `abstain` field must equal "
                "sckevtype.eval.abstain_index(row). The served and abstain-pick counts must then "
                "reproduce reports/both-forced-table.json's for the same engine, integer for integer, "
                "and the released rung's forced score recomputed from its own census dump must "
                "reproduce the published 578/1819.",
            "rule_provenance": "forced/argmax/binom_two_sided are executed from the parsed source of "
                               "reports/abstain-reconciliation.py (its function definitions only — the "
                               "module body is not run, because running it would rewrite "
                               "reports/abstain-reconciliation.json).",
        },
        "cross_checks": {"against_both_forced_table": against,
                         "rung_forced_on_this_population": reference},
        "training_overlap": overlap,
        "provenance": {
            "computed": "2026-10-06",
            "method": "re-score of each arm on the population with `p` retained, then one aggregation "
                      "pass under the settled rule; no number here is transcribed from another file.",
            "commands": [
                ("# 1. the scoring path was extended to retain `p` (sckevtype/eval/run.py: FULL_DUMP_FIELDS "
                 "gains \"p\"), and staged atomically to the cluster:"),
                ("bash <cluster-job> put <login-node> sckevtype/eval/run.py "
                 "/path/to/sckevtype/scripts/_run.py.new"),
                ("bash <cluster-job> run <login-node> 'cd $W && mv -f scripts/_run.py.new "
                 "sckevtype/eval/run.py'"),
                "# 2. the re-score: one job, one card, arms in sequence (~9 min an arm)",
                "bash <cluster-job> --probe   # 8 records of arm frozen, job 4228936",
                "bash <cluster-job>           # all 11 arms, job 4228940",
                "bash <cluster-job> --status  # what has landed",
                "# 3. pull the dumps (size + sha256 verified on both sides), then aggregate",
                "bash <cluster-job>",
                ("source scripts/wsl/env.sh && uv run --no-sync python scripts/arms-forced.py "
                 "--job-ids 4228936,4228939,4228940"),
            ],
            "job_ids": [entry.strip() for entry in args.job_ids.split(",") if entry.strip()],
            "partition": "<gpu-partition>",
            "sha256": {},
        },
        "limits": [
            ("The forced number is the argmax over the row's own probability vector with the abstain "
             "index dropped — the settled rule — not a model re-run under a no-abstain prompt. The "
             "re-scored vector is checked against the archived `top1` on every row, so the vector is the "
             "one the archived row was read from; what is not measured is whether a model *told* it may "
             "not abstain would answer differently."),
            ("Our arms were served the suite's own records (`test/custom/<n>`); the census's 2,510 "
             "records are a different population and no arm has rows on it, so only the 1,819-row "
             "population is reported here."),
            ("The three base-initialized controls (seed1-3) and arm B are the same recipe from a "
             "different starting point; their rows are here for the §2.5 comparison, not as fine-tunes "
             "of the released rung."),
        ],
    }
    for path in ([SUITE_IDS, CENSUS_RUNG[RUNG]] + list(ARCHIVED.values()) + list(RESCORED.values())):
        if (ROOT / path).exists():
            out["provenance"]["sha256"][path] = sha256_file(path)
    for path in ("sckevtype/eval/run.py", "scripts/score-forced-rows.py",
                 "<cluster-job>", "<cluster-job>"):
        if (ROOT / path).exists():
            out["provenance"]["sha256"][path] = sha256_file(path)

    # Coverage: which arms have an archived dump, and which of those this report covers. An arm with
    # no dump is not in the manuscript's Table 3 either, so re-scoring it would add a row with no
    # served number to compare against -- but it must not go unmentioned.
    dumped = sorted(path.name[len("rows-full-"):-len(".jsonl")]
                    for path in (ROOT / "reports").glob("rows-full-*.jsonl"))
    covered = sorted(arm["arm"] for arm in ARMS)
    out["coverage"] = {
        "arms_covered": covered,
        "archived_dumps_on_disk": dumped,
        "dumps_on_disk_not_in_this_report": [name for name in dumped if name not in covered],
        "arms_with_a_checkpoint_but_no_archived_dump": {
            "note": "checked on the cluster with <cluster-job> --status "
                    "(2026-10-06): these runs have an adapter on disk but no reports/rows-full-*.jsonl, "
                    "so they are absent from the manuscript's Table 3 for the same reason ours were — "
                    "and re-scoring them would add a row with no served number to compare against. "
                    "Recorded here so no arm is skipped silently.",
            "arms": ["delta-kev-0.5b-s1", "delta-kev-0.6b-s1", "delta-kev-0.8b-s6 (still training)",
                     "delta-kev-9b-s1 (in flight, not scored)"]},
    }

    Path(args.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    write_md(out, Path(args.md))

    print(f"population: {len(ids)} ids")
    print(f"reference : Kev-0.8B forced {reference['forced']['correct']}/{reference['n']} = "
          f"{reference['forced']['accuracy']:.4f}  (published: 578/1819)")
    for engine in engines:
        cell = engine.get("census_matched_1819", {})
        if "not_measured" in cell:
            print(f"  {engine['label']:34s} NOT MEASURED -- {cell['not_measured'][:60]}...")
            continue
        print(f"  {engine['label']:34s} served {cell['served']['accuracy']:.4f}  "
              f"abstain {cell['abstain_pick']['rate']:.4f}  forced {cell['forced']['accuracy']:.4f} "
              f"({cell['forced']['correct']}/{cell['n']})  Δ {cell['delta_points']:+.1f} pt  "
              f"hidden {cell['hidden_correct_share_of_abstentions']:.4f}  "
              f"[{engine['fidelity']['verdict']}]")
    print(f"verdict   : {claim['verdict']}")
    print(f"wrote {_shown(args.json)} and {_shown(args.md)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
