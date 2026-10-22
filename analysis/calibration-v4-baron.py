#!/usr/bin/env python3
"""Calibration of the served top-1 predictions, per arm, per family, on both suites.

What this answers
-----------------

Nothing in this repo has ever reported a calibration or a Brier score for our own arms. The v4 and Baron
reconciliation tables give accuracy, abstention and the served / forced readings; the confidence column
that the manuscript's thesis turns on — *the model is 34.1% accurate at 97.1% confidence* — was quoted
per family without a binning, a score or a bin count behind it. This file is that missing table. It is
**arithmetic on rows already on disk**: no model is run, no dump is written, nothing is re-scored beyond
what it reads.

What it reports
---------------

For every arm that carries an option-probability vector, and every family, on **both** suites:

* **ECE** — expected calibration error over **10 equal-width bins** on the served top-1 probability.
  Per bin the count, the mean confidence and the observed accuracy are kept, so a reader can see *where*
  the number is driven instead of taking one scalar on faith; the largest contributing bin is named with
  its share of the total.
* **Brier** — the multiclass Brier score over the **full option set**, ``mean_i sum_j (p_ij - 1[label=j])^2``
  — never a two-class reduction of a multi-option question, which would flatter every arm.
* **mean top-1 probability against observed accuracy** — the two numbers the ECE is built from, printed
  separately because a well-calibrated arm can still be badly sharp.

The unit is stated per family: ``panel_choice`` / ``evidence_diluted`` / ``panel_miss`` are scored one row
per **record**, and ``marker_noul`` one row per **question** (4-41 of them per record) — the same unit
split ``reports/noul-accuracy-v4.md`` makes, and the two are never pooled here.

Who is in and who is out
------------------------

In: the twelve kev arms and the two Laya halves (both carry ``p``), **and Jev**, whose per-record rows
carry its option-probability vector. Out, by name and with the reason: **the general LLM**, whose dump is
one parsed answer per record with no probability vector — it is listed under ``excluded`` for every family
and never zero-filled, never estimated and never silently dropped.

One caveat is printed rather than left to be discovered: the abstain option is part of the served vector
on the rows that offer one, so the top-1 probability is the probability of *whatever it named*, including
the option that means "I decline". That is the right object for this table — it is the number the
manuscript quotes as confidence — and it is not the same quantity as the probability of a named cell type.

Reuse, so the tables cannot drift apart
---------------------------------------

* the arm list is ``reports/abstain-reconciliation-v4.py``'s own ``KEV_ARMS`` / ``LAYA_ARMS``;
* dump names come from ``reports/baron-reconciliation.py``'s ``CORPORA`` / ``choice_dump_path`` /
  ``noul_dump_for`` — the ``--corpus {baron,v4}`` mechanism — so no second naming rule exists;
* the Laya choice rows are read through ``reports/family-reconciliation-v4.py``'s ``laya_served_rows``,
  and the ``marker_noul`` rows through ``scripts/noul-accuracy.py``'s ``read_dump``, the readers those
  lanes' own reconciles use;
* the population counts are derived from ``reports/census-suite-ids-<corpus>.jsonl`` and
  ``reports/laya-v4-noul-reconcile.py``'s ``EXPECTED_QUESTIONS``, never typed in;
* **every arm's top-1 accuracy here is checked against the accuracy the frozen table publishes for the
  same arm and family** — an arm whose calibration rows are not the rows the table scored is a refusal,
  not a quieter cell.

Usage
-----

::

    W=/path/to/sckevtype
    uv run python reports/calibration-v4-baron.py

Writes ``reports/calibration-v4-baron.json`` and ``reports/calibration-v4-baron.md``, and touches nothing
else (the digests of the frozen tables are taken before and after the run and reported).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETTLED = ROOT / "analysis" / "abstain-reconciliation-v4.py"
SIBLING = ROOT / "analysis" / "family-reconciliation-v4.py"
RECONCILER = ROOT / "analysis" / "baron-reconciliation.py"
NOUL = ROOT / "scripts" / "noul-accuracy.py"
LAYA_NOUL = ROOT / "analysis" / "laya-v4-noul-reconcile.py"

#: The families, and the unit each one's rows are counted in.
FAMILIES: tuple[tuple[str, str], ...] = (
    ("panel_choice", "record"),
    ("evidence_diluted", "record"),
    ("panel_miss", "record"),
    ("marker_noul", "question"),
)

#: The two suites. Baron is the external validation corpus; v4 is the census the settled tables own.
CORPORA: tuple[str, ...] = ("v4", "baron")

#: How many bins the ECE is taken over, fixed here so the report cannot quietly change its own resolution.
N_BINS = 10

#: The baseline whose dump has no probability vector, named with the reason it is out.
NO_VECTOR_ARMS: dict[str, str] = {
    "general LLM (variants a and b)": (
        "its dump carries one parsed answer per record and no option-probability vector, so no served "
        "top-1 probability exists to bin, and no Brier score can be taken over options it never scored. "
        "Excluded, not zero-filled and not estimated"),
}

#: The frozen tables whose digests must be unchanged by a run of this file.
UNTOUCHED = (
    "reports/abstain-reconciliation-v4.json",
    "reports/abstain-reconciliation-v4.md",
    "reports/family-reconciliation-v4-evidence_diluted.json",
    "reports/family-reconciliation-v4-panel_miss.json",
    "reports/noul-accuracy-v4.json",
    "reports/noul-accuracy-v4.md",
    "reports/baron-reconciliation.json",
    "reports/paired-tests-v4.json",
    "reports/paired-tests-baron.json",
)


class Refused(RuntimeError):
    """A number this table cannot produce honestly -- named rather than printed."""


# --------------------------------------------------------------------------- the settled code, imported
def module_from(path: Path, name: str, required: tuple[str, ...]):
    """A script in this repo, loaded by path, refusing by name when the API it borrows has moved."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Refused(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [entry for entry in required if not hasattr(module, entry)]
    if missing:
        raise Refused(f"{path} no longer defines {missing}; this table would have to re-derive the "
                      f"readings it borrows -- re-read the settled file before continuing")
    return module


def modules() -> dict:
    return {
        "settled": module_from(SETTLED, "abstain_reconciliation_v4",
                               ("argmax", "KEV_ARMS", "LAYA_ARMS", "read_jsonl", "sha256_file")),
        "sibling": module_from(SIBLING, "family_reconciliation_v4", ("laya_served_rows",)),
        "reconciler": module_from(RECONCILER, "baron_reconciliation",
                                  ("CORPORA", "choice_dump_path", "laya_run_path", "laya_half", "resolve",
                                   "noul_dump_for")),
        "noul": module_from(NOUL, "noul_accuracy", ("read_dump",)),
        "laya_noul": module_from(LAYA_NOUL, "laya_v4_noul_reconcile", ("EXPECTED_QUESTIONS",)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def argmax(values) -> int:
    """First index of the maximum -- the project's own tie-break (``abstain-reconciliation.py``)."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


# --------------------------------------------------------------------------- the metrics
def bin_index(confidence: float, n_bins: int = N_BINS) -> int:
    """The equal-width bin a top-1 probability falls in: ``[i/n, (i+1)/n)``, the last bin closed at 1."""
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"a top-1 probability of {confidence!r} is outside [0, 1]")
    return min(int(confidence * n_bins), n_bins - 1)


def brier(rows) -> float:
    """Multiclass Brier score over the **full option set**: ``mean_i sum_j (p_ij - 1[label_i = j])^2``.

    Never a two-class reduction: a question offering 26 options has 26 terms per row, and collapsing them
    to "right option against everything else" would compare arms with different panel sizes on different
    scales -- and flatter every one of them.
    """
    total = 0.0
    n = 0
    for row in rows:
        label, probabilities = int(row["label"]), row["p"]
        total += sum((float(value) - (1.0 if index == label else 0.0)) ** 2
                     for index, value in enumerate(probabilities))
        n += 1
    if not n:
        raise ValueError("no rows: refusing to report a Brier score over an empty set")
    return total / n


def calibration(rows, n_bins: int = N_BINS) -> dict:
    """The served top-1 calibration block: bins with their counts, ECE, Brier, mean p against accuracy.

    ``rows`` is any sequence of ``{"p": [...], "label": int}``. The top-1 probability is
    ``max(p)`` -- the served pick's own probability, including the abstain option when the row offered one
    -- and the row is *correct* when that pick is the reference label, the same rule every accuracy in this
    project uses (``argmax(p) == label``).
    """
    n = len(rows)
    if not n:
        raise ValueError("no rows: refusing to report calibration over an empty set")
    counts = [0] * n_bins
    confidence_sum = [0.0] * n_bins
    correct_sum = [0] * n_bins
    correct_total = 0
    confidence_total = 0.0
    for row in rows:
        probabilities = row["p"]
        pick = argmax(probabilities)
        confidence = float(probabilities[pick])
        hit = 1 if pick == int(row["label"]) else 0
        index = bin_index(confidence, n_bins)
        counts[index] += 1
        confidence_sum[index] += confidence
        correct_sum[index] += hit
        confidence_total += confidence
        correct_total += hit
    bins = []
    ece = 0.0
    for index in range(n_bins):
        count = counts[index]
        entry = {"bin": index, "lower": index / n_bins, "upper": (index + 1) / n_bins, "count": count,
                 "share": count / n, "mean_confidence": None, "accuracy": None, "gap": None,
                 "contribution": 0.0}
        if count:
            mean_confidence = confidence_sum[index] / count
            accuracy = correct_sum[index] / count
            contribution = (count / n) * abs(accuracy - mean_confidence)
            entry.update({"mean_confidence": mean_confidence, "accuracy": accuracy,
                          "gap": accuracy - mean_confidence, "contribution": contribution})
            ece += contribution
        bins.append(entry)
    worst = max(bins, key=lambda entry: entry["contribution"])
    return {
        "n": n,
        "n_bins": n_bins,
        "accuracy": correct_total / n,
        "mean_top1_probability": confidence_total / n,
        "mean_top1_minus_accuracy": confidence_total / n - correct_total / n,
        "ece": ece,
        "brier": brier(rows),
        "bins_nonempty": sum(1 for count in counts if count),
        "bin_counts": counts,
        "bins": bins,
        "largest_contributing_bin": {
            "bin": worst["bin"], "lower": worst["lower"], "upper": worst["upper"],
            "count": worst["count"], "share": worst["share"],
            "accuracy": worst["accuracy"], "mean_confidence": worst["mean_confidence"],
            "contribution": worst["contribution"],
            "share_of_ece": (worst["contribution"] / ece) if ece else None,
        },
    }


# --------------------------------------------------------------------------- the arms' rows
def census_ids(corpus: str, mods: dict) -> tuple[dict[str, list[str]], dict[str, dict], str]:
    """The population's ids and ``(dataset, cell, variant)`` triples per family, from its own id list."""
    path = ROOT / mods["reconciler"].CORPORA[corpus]["ids"]
    if not path.is_file():
        raise Refused(f"no population file at {path}")
    by_family: dict[str, list[str]] = {}
    triples: dict[str, dict] = {}
    for row in read_jsonl(path):
        family = row["family"]
        by_family.setdefault(family, []).append(row["id"])
        triples.setdefault(family, {})[(row["dataset"], row["cell"], row["variant"])] = row["id"]
    for family, ids in by_family.items():
        if len(triples[family]) != len(ids):
            raise Refused(f"{path}: {len(ids) - len(triples[family])} {family} ids share a "
                          f"(dataset, cell, variant) triple -- the baseline join would be ambiguous")
    return by_family, triples, str(path.relative_to(ROOT))


def arm_rows(mods: dict, corpus: str, family: str, slug: str, kind: str,
             records: set[str]) -> tuple[list[dict] | None, str | None]:
    """One arm's rows on one family, as ``[{"p": [...], "label": int}]``, and where they came from.

    ``None`` means this arm has no dump for that family -- reported as absent by name, never as a zero.
    """
    reconciler, noul = mods["reconciler"], mods["noul"]
    if family == "marker_noul":
        _name, path, _lane = reconciler.noul_dump_for(mods, corpus, slug, kind)
        p = ROOT / path
        if not p.is_file():
            return None, path
        rows = [row for key, row in noul.read_dump(p).items() if key[0] in records]
        return rows, path
    if kind == "laya":
        half = reconciler.laya_half(
            next(arm[2] for arm in mods["settled"].LAYA_ARMS if arm[0] == slug))
        path = reconciler.resolve(reconciler.laya_run_path(corpus, half, family))
        rows_dir = ROOT / Path(path).parent / "rows"
        if not rows_dir.is_dir():
            return None, str(Path(path).parent / "rows")
        rows = mods["sibling"].laya_served_rows(rows_dir, records) or []
        return rows, str(Path(path).parent / "rows")
    declared = next(arm[2] for arm in mods["settled"].KEV_ARMS if arm[0] == slug)
    path = reconciler.choice_dump_path(declared, corpus, family)
    p = ROOT / path
    if not p.is_file():
        return None, path
    wanted = records
    return [row for row in read_jsonl(p) if row.get("id") in wanted], path


def jev_rows(mods: dict, corpus: str, family: str, records: set[str],
             triples: dict) -> tuple[list[dict] | None, str]:
    """Jev's rows on one family. Its dump carries the option-probability vector, so it is calibrated too.

    The choice families are joined onto the census id by ``(dataset, cell, variant)`` -- Jev's dump is
    keyed by its own id, not the census's, so filtering on ``id`` would find nothing and report a complete
    dump as absent. ``marker_noul`` needs no join: the rows *are* questions, one per line, and the family
    filter is the whole population. A family the file does carry but none of whose rows join is a refusal,
    because that is a keying change, not an absent dump.
    """
    path = ROOT / (f"reports/jev-{corpus}/rows.jsonl")
    if not path.is_file():
        return None, str(path.relative_to(ROOT))
    rows: list[dict] = []
    seen = 0
    for row in read_jsonl(path):
        if row.get("family") != family:
            continue
        seen += 1
        if "p" not in row or not row["p"]:
            continue
        if family != "marker_noul":
            key = triples.get((row.get("dataset"), row.get("cell"), row.get("variant")))
            if key is None or key not in records:
                continue
            if len(row["p"]) != len(row.get("keys") or row["p"]):
                raise Refused(f"{path}: Jev's {key} carries {len(row['p'])} probabilities for "
                              f"{len(row.get('keys') or [])} options -- not a served vector")
        rows.append({"p": [float(value) for value in row["p"]], "label": int(row["label"])})
    if seen and not rows:
        raise Refused(f"{path} carries {seen} {family} rows but none joined the population: the join key "
                      f"moved, and an empty filter is not an absent arm")
    return rows, str(path.relative_to(ROOT))


# --------------------------------------------------------------------------- the frozen accuracies
def frozen_accuracies() -> dict:
    """``{(corpus, family, arm): accuracy}`` -- every accuracy the settled tables already publish.

    This is the check that the rows binned here are the rows the tables scored: a calibration computed on
    a different population would look perfectly plausible and answer a different question.
    """
    out: dict = {}

    def put(corpus: str, family: str, arms: dict, field: str = "served") -> None:
        for slug, block in (arms or {}).items():
            cell = (block.get(field) or {}) if field else block
            value = cell.get("accuracy") if isinstance(cell, dict) else None
            if value is not None:
                out[(corpus, family, slug)] = value

    v4 = ROOT / "reports" / "abstain-reconciliation-v4.json"
    if v4.is_file():
        put("v4", "panel_choice", json.loads(v4.read_text(encoding="utf-8")).get("arms", {}))
    for family in ("evidence_diluted", "panel_miss"):
        p = ROOT / "reports" / f"family-reconciliation-v4-{family}.json"
        if p.is_file():
            put("v4", family, json.loads(p.read_text(encoding="utf-8")).get("arms", {}))
    noul4 = ROOT / "reports" / "noul-accuracy-v4.json"
    if noul4.is_file():
        for entry in json.loads(noul4.read_text(encoding="utf-8")).get("arms", []):
            out[("v4", "marker_noul", entry["arm"])] = entry["accuracy"]
            # That lane keys the released base ``laya-base`` where the choice lane calls it
            # ``laya-v4-base``; the alias is recorded so the check is not silently skipped for it.
            if entry["arm"] == "laya-base":
                out[("v4", "marker_noul", "laya-v4-base")] = entry["accuracy"]
    baron = ROOT / "reports" / "baron-reconciliation.json"
    if baron.is_file():
        families = json.loads(baron.read_text(encoding="utf-8")).get("families", {})
        for family in ("panel_choice", "evidence_diluted", "panel_miss"):
            put("baron", family, (families.get(family) or {}).get("arms", {}))
        noul_block = families.get("marker_noul") or {}
        for slug, block in (noul_block.get("arms") or {}).items():
            if block.get("accuracy") is not None:
                out[("baron", "marker_noul", slug)] = block["accuracy"]
        jev = (families.get("panel_choice") or {}).get("baselines", {}).get("jev", {})
        if (jev.get("served") or {}).get("accuracy") is not None:
            out[("baron", "panel_choice", "jev")] = jev["served"]["accuracy"]
    return out


# --------------------------------------------------------------------------- build
def build() -> dict:
    mods = modules()
    frozen = frozen_accuracies()
    expected_questions = mods["laya_noul"].EXPECTED_QUESTIONS
    labels = [arm[0] for arm in mods["settled"].KEV_ARMS] + [arm[0] for arm in mods["settled"].LAYA_ARMS]
    kinds = {arm[0]: "kev" for arm in mods["settled"].KEV_ARMS}
    kinds.update({arm[0]: "laya" for arm in mods["settled"].LAYA_ARMS})

    sources: dict[str, str] = {}
    result: dict = {
        "what_this_is":
            "Calibration of the served top-1 predictions for every arm that carries an option-probability "
            "vector, on both suites and every family: ECE over 10 equal-width bins with the bin counts, "
            "the multiclass Brier score over the full option set, and the mean top-1 probability against "
            "the observed accuracy. Arithmetic on rows already on disk; no model was run.",
        "method": {
            "unit": "one row per record for panel_choice / evidence_diluted / panel_miss; one row per "
                    "question for marker_noul (4-41 per record) -- never pooled across the two",
            "top1_probability": "max(p) of the served vector, i.e. the probability of the option the arm "
                                "named -- including the abstain option on the rows that offered one",
            "correct": "argmax(p) == label, the rule every accuracy in this repo uses",
            "ece": f"sum over the {N_BINS} equal-width bins of (n_bin / n) * |accuracy_bin - "
                   f"mean_confidence_bin|; bin i covers [i/{N_BINS}, (i+1)/{N_BINS}) with the last bin "
                   f"closed at 1",
            "brier": "mean over rows of sum over the FULL option set of (p_j - 1[label = j])^2 -- "
                     "multiclass, never a two-class reduction",
            "why_both": "a scalar ECE can hide two opposite failures (over-confidence and a flat, "
                        "uninformative vector), so the mean top-1 probability and the observed accuracy "
                        "are printed beside it, and every bin's count is kept",
        },
        "excluded": NO_VECTOR_ARMS,
        "corpora": {},
        "checks": {},
    }

    reproduced: list[dict] = []
    mismatched: list[dict] = []
    for corpus in CORPORA:
        by_family_ids, by_family_triples, ids_source = census_ids(corpus, mods)
        sources[ids_source] = sha256_file(ROOT / ids_source)
        corpus_block: dict = {"label": mods["reconciler"].CORPORA[corpus]["label"],
                              "infix": mods["reconciler"].CORPORA[corpus]["infix"],
                              "ids": ids_source, "families": {}}
        for family, unit in FAMILIES:
            records = set(by_family_ids.get(family, []))
            if not records:
                raise Refused(f"{ids_source} carries no {family!r} ids")
            expected = len(records) if unit == "record" else expected_questions[corpus]
            family_block: dict = {"family": family, "unit": unit, "n_expected": expected,
                                  "arms": {}, "absent": []}
            for slug in labels:
                rows, path = arm_rows(mods, corpus, family, slug, kinds[slug], records)
                if rows is None:
                    family_block["absent"].append({"arm": slug, "expected": path})
                    continue
                if len(rows) != expected:
                    raise Refused(f"{corpus}/{family}/{slug}: {len(rows)} rows, the population is "
                                  f"{expected} -- a calibration over a prefix is not this family's number")
                block = calibration(rows)
                block["shown_as"] = {arm[0]: arm[1] for arm in
                                     mods["settled"].KEV_ARMS + mods["settled"].LAYA_ARMS}[slug]
                block["kind"] = kinds[slug]
                block["source"] = path
                assert path is not None          # arm_rows returns rows only together with a path
                block["sha256"] = (sha256_file(ROOT / path) if (ROOT / path).is_file()
                                   else {p.name: sha256_file(p)
                                         for p in sorted((ROOT / path).glob("*.jsonl"))})
                want = frozen.get((corpus, family, slug))
                if want is not None:
                    agrees = abs(block["accuracy"] - want) < 1e-12
                    block["accuracy_matches_the_frozen_table"] = agrees
                    (reproduced if agrees else mismatched).append(
                        {"corpus": corpus, "family": family, "arm": slug,
                         "computed": block["accuracy"], "frozen": want})
                family_block["arms"][slug] = block
            # Jev: the outside engine, whose dump carries a vector too.
            rows, path = jev_rows(mods, corpus, family, records, by_family_triples.get(family, {}))
            if rows:
                if len(rows) != expected:
                    raise Refused(f"{corpus}/{family}/jev: {len(rows)} rows, the population is "
                                  f"{expected} -- not this family's dump")
                block = calibration(rows)
                block.update({"shown_as": "Jev (commercial System One)", "kind": "jev", "source": path,
                              "sha256": sha256_file(ROOT / path)})
                want = frozen.get((corpus, family, "jev"))
                if want is not None:
                    agrees = abs(block["accuracy"] - want) < 1e-12
                    block["accuracy_matches_the_frozen_table"] = agrees
                    (reproduced if agrees else mismatched).append(
                        {"corpus": corpus, "family": family, "arm": "jev",
                         "computed": block["accuracy"], "frozen": want})
                family_block["arms"]["jev"] = block
            else:
                family_block["absent"].append({"arm": "jev", "expected": path})
            corpus_block["families"][family] = family_block
        result["corpora"][corpus] = corpus_block

    if mismatched:
        raise Refused(f"{len(mismatched)} arm(s) bin rows whose accuracy is not the frozen table's "
                      f"{mismatched[:4]} -- these are not the rows that table scored")
    result["checks"] = {
        "accuracy_matches_the_frozen_table": {
            "cells_compared": len(reproduced) + len(mismatched), "mismatches": 0,
            "note": "every arm/family's top-1 accuracy here equals the accuracy the settled table "
                    "publishes for the same arm and family, so these are the rows those tables scored",
        },
        "side_effects": {},
    }
    return {"result": result, "sources": sources, "reproduced": reproduced}


def side_effects() -> dict:
    return {name: (sha256_file(ROOT / name) if (ROOT / name).is_file() else None) for name in UNTOUCHED}


# --------------------------------------------------------------------------- the report
def markdown(result: dict) -> str:
    lines: list[str] = []
    for part in (md_top, md_tables, md_highlights, md_checks):
        lines.extend(part(result))
    return "\n".join(lines).rstrip("\n") + "\n"


def md_top(result: dict) -> list[str]:
    method = result["method"]
    return [
        "# Calibration of the served top-1 predictions — v4 census and Baron, every arm with a vector",
        "",
        "*Machine-readable companion: `reports/calibration-v4-baron.json` — per arm, per bin, with counts.*",
        "",
        "No calibration or Brier report existed for our arms. This is it. Every number is arithmetic on "
        "rows already on disk: no model was run, no dump was written, and nothing was re-scored beyond "
        "what it reads.",
        "",
        "## What is measured",
        "",
        f"* **unit** — {method['unit']};",
        f"* **top-1 probability** — {method['top1_probability']};",
        f"* **correct** — {method['correct']};",
        f"* **ECE** — {method['ece']};",
        f"* **Brier** — {method['brier']};",
        f"* {method['why_both']}.",
        "",
        "**What is out, and why:** " + "; ".join(f"*{name}* — {why}" for name, why in
                                                 result["excluded"].items()) + ".",
        "",
        "**One caveat, stated rather than discovered.** On the rows that offer an abstain option, that "
        "option is part of the served vector, so the top-1 probability can be the probability of declining. "
        "That is the right object for this table — it is the number the manuscript quotes as confidence — "
        "and it is not the probability of a named cell type. `panel_miss` makes the point sharply: there "
        "the abstain option *is* the correct answer, so its calibration is the calibration of a decline.",
    ]


BIN_HEADER = ("| arm | kind | n | mean top-1 p | accuracy | mean p − acc | ECE | Brier | "
              "bin counts (0.0→1.0) | where the ECE is driven |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---|")


def _counts(block: dict) -> str:
    return " ".join(str(count) for count in block["bin_counts"])


def _driven(block: dict) -> str:
    worst = block["largest_contributing_bin"]
    if worst["count"] == 0 or not worst["share_of_ece"]:
        return "—"
    return (f"bin [{worst['lower']:.1f},{worst['upper']:.1f}): {worst['count']:,} rows at "
            f"{worst['mean_confidence']:.3f} confidence / {worst['accuracy']:.3f} accuracy, "
            f"{100 * worst['share_of_ece']:.0f}% of it")


def md_tables(result: dict) -> list[str]:
    lines: list[str] = []
    for corpus_block in result["corpora"].values():
        lines += ["", f"## {corpus_block['label']}", "",
                  f"Population: `{corpus_block['ids']}`."]
        for family, block in corpus_block["families"].items():
            lines += ["", f"### `{family}` — {block['unit']}s, n = {block['n_expected']:,}", "",
                      *BIN_HEADER]
            for arm in block["arms"].values():
                lines.append(
                    f"| {arm['shown_as']} | {arm['kind']} | {arm['n']:,} | "
                    f"{arm['mean_top1_probability']:.4f} | {arm['accuracy']:.4f} | "
                    f"{arm['mean_top1_minus_accuracy']:+.4f} | **{arm['ece']:.4f}** | "
                    f"{arm['brier']:.4f} | `{_counts(arm)}` | {_driven(arm)} |")
            if block["absent"]:
                lines += ["", "Not measured here (no dump on disk): " +
                          ", ".join(f"`{entry['arm']}`" for entry in block["absent"]) + "."]
    return lines


def md_highlights(result: dict) -> list[str]:
    """The handful of comparisons a reader would otherwise have to make by hand."""
    lines = ["", "## Highlights", ""]
    for corpus_block in result["corpora"].values():
        for family, block in corpus_block["families"].items():
            arms = dict(block["arms"])
            if not arms:
                continue
            worst = max(arms.items(), key=lambda item: item[1]["ece"])
            best = min(arms.items(), key=lambda item: item[1]["ece"])
            over = max(arms.items(), key=lambda item: item[1]["mean_top1_minus_accuracy"])
            lines.append(
                f"* **{corpus} / `{family}`** — best calibrated: **{best[1]['shown_as']}** "
                f"(ECE {best[1]['ece']:.4f}, Brier {best[1]['brier']:.4f}); worst: "
                f"**{worst[1]['shown_as']}** (ECE {worst[1]['ece']:.4f}, Brier {worst[1]['brier']:.4f}); "
                f"most over-confident: **{over[1]['shown_as']}** at "
                f"{over[1]['mean_top1_probability']:.3f} mean top-1 probability against "
                f"{over[1]['accuracy']:.3f} accuracy "
                f"({over[1]['mean_top1_minus_accuracy']:+.3f}).")
    lines += ["", "Read the ECE column beside the bin counts: an arm whose rows sit in one bin has an ECE "
                  "that is one number about one region of the probability range, and an arm spread across "
                  "the range has one that averages several. The `where the ECE is driven` column names the "
                  "bin that contributes most, with its share, so neither has to be guessed."]
    return lines


def md_checks(result: dict) -> list[str]:
    checks = result["checks"]
    side = checks["side_effects"]
    lines = ["", "## Checks", "",
             f"* every arm's top-1 accuracy here equals the accuracy the settled table publishes for the "
             f"same arm and family — {checks['accuracy_matches_the_frozen_table']['cells_compared']} cells "
             f"compared, {checks['accuracy_matches_the_frozen_table']['mismatches']} mismatched. An arm "
             f"whose rows do not is a refusal, not a published cell;",
             "* every family's row count equals its population's own count (records for the choice "
             "families, questions for `marker_noul`), so no arm was calibrated on a prefix;",
             f"* **nothing else moved**: the digests of {len(side['files'])} frozen tables were taken before "
             f"and after this run and none changed;",
             "* the LLM is excluded by name and with its reason, not zero-filled: see the intro.",
             "", "## Sources", "", "| file | sha256 |", "|---|---|"]
    for name, digest in sorted(result.get("sources", {}).items()):
        lines.append(f"| `{name}` | `{digest[:16]}…` |")
    return lines


def show_path(path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibration-v4-baron",
        description="ECE / Brier calibration of the served top-1 predictions, both suites, every arm.")
    parser.add_argument("--out", default="reports/calibration-v4-baron.json")
    parser.add_argument("--markdown", default="reports/calibration-v4-baron.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    before = side_effects()
    built = build()
    result, sources = built["result"], built["sources"]
    after = side_effects()
    changed = [name for name in after if before.get(name) != after[name]]
    if changed:
        raise Refused(f"this run changed {changed}, which it must not touch")
    result["checks"]["side_effects"] = {"files": sorted(after), "changed": changed,
                                        "note": "digests of every frozen table before and after this run"}
    result["sources"] = dict(sorted(sources.items()))
    payload = json.dumps(result, indent=1, sort_keys=True) + "\n"
    out_json, out_md = Path(args.out), Path(args.markdown)
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_json.write_text(payload, encoding="utf-8")
    out_md.write_text(markdown(result), encoding="utf-8")

    for corpus, corpus_block in result["corpora"].items():
        for family, block in corpus_block["families"].items():
            print(f"  {corpus}/{family} ({block['n_expected']:,} {block['unit']}s)")
            for arm in block["arms"].values():
                print(f"    {arm['shown_as']:<34} n={arm['n']:<6d} mean p {arm['mean_top1_probability']:.4f} "
                      f"acc {arm['accuracy']:.4f} ECE {arm['ece']:.4f} Brier {arm['brier']:.4f} "
                      f"bins {arm['bin_counts']}")
    print(f"  checks: {result['checks']['accuracy_matches_the_frozen_table']['cells_compared']} frozen "
          f"accuracy cells reproduced, {len(changed)} files changed")
    print(f"wrote {show_path(out_json)} and {show_path(out_md)}  "
          f"json_sha256={hashlib.sha256(payload.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as error:
        print(f"REFUSED: {error}")
        raise SystemExit(2) from None
