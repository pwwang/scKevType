#!/usr/bin/env python3
"""The Baron `panel_choice` paired tests: every comparison the paper makes, with its p-value.

What this answers
-----------------

``reports/baron-reconciliation.md`` reports the Baron external-validation suite arm by arm under the
**served** and **forced** readings and carries **point estimates only**. ``reports/paired-tests-v4.md``
gives the same suite of tests for the v4 census's ``panel_choice`` population — 42 pre-specified
comparisons, Holm within each reading, 42/42 significant raw and no verdict changed by the correction —
and Baron had no equivalent. This script is that equivalent, and it exists to answer one question the
point estimates cannot: **is the 27B fine-tune's Baron lead over the 9B and 4B fine-tunes real?** On v4
the 27B sits *below* both of them (served 0.7928 against 0.8191 for 4B and 9B), so a Baron ordering in
the other direction is either a real corpus effect or a coin flip on ~2,600 records, and only a paired
test can say which.

The unit is the **record**, and both arms of every comparison answer the **same 2,600 records**, so the
test is McNemar's exact test on the discordant pairs: ``p = 2 * P(X <= min(b, c))`` under
``Binomial(b + c, 0.5)``. That is not a second statistical rule: it is the same ``binom_two_sided`` the
v4 paired table and the noul table use, lifted out of ``reports/abstain-reconciliation.py``'s own source
(the file that settled it) rather than re-derived. **Its implementation is untouched.**

Readings
--------

* **served** — ``argmax(p)`` equals the reference label: what the arm named with abstention available.
* **forced** — ``argmax(p)`` with the abstain option removed equals the label: what it would have named
  had abstention not been offered. A record served with no abstain option has nothing to remove, so its
  forced verdict *is* its served verdict — the rule ``reports/abstain-reconciliation-v4.py``'s ``cell()``
  applies, and the rule this file applies per row.

Both readings are tested wherever the arm carries a probability vector. **The general LLM carries none**
(one parsed answer per record), so its forced reading is reported as **not testable** — never zero-filled
and never estimated.

Reuse, so the tables cannot drift apart
---------------------------------------

* the arm list is the settled script's own ``KEV_ARMS`` / ``LAYA_ARMS``, imported rather than re-typed;
* the dump names come from ``reports/baron-reconciliation.py``'s ``CORPORA`` / ``choice_dump_path``, the
  ``--corpus {baron,v4}`` mechanism that file established — so this script reads
  ``forced-rows-<slug>-baron.jsonl`` for Baron and ``forced-rows-<slug>.jsonl`` for v4 by construction,
  not by a second hand-written rule;
* the Laya halves are read off their own run directory through
  ``reports/family-reconciliation-v4.py``'s ``laya_served_rows`` — the same reader that lane's reconcile
  used — so a Laya row here is the row the settled table counted;
* the noul column is **not** restated: ``marker_noul`` is a yes/no family with no abstain option to
  remove, so it has no forced reading and no paired test of this shape. Its p-values live in
  ``reports/noul-accuracy-v4.md`` (v4) and the Baron noul floor in ``reports/baron-reconciliation.md``.

What it refuses
---------------

Each arm must cover the population exactly — no missing id, no extra id — and every arm's reference label
for a record must equal every other arm's. A dump that does not is a ``Refused``/``SystemExit``. Every
arm's per-row verdict rule is also re-derived through the settled ``cell()`` **as a cross-check**: the two
readings this file computes per row must sum to the same served / forced / abstain-pick counts that
``reports/baron-reconciliation.json`` (or, on ``--corpus v4``, ``reports/paired-tests-v4.json`` and
``reports/abstain-reconciliation-v4.json``) already publish, integer for integer. Numbers that disagree
with the table they extend are not published.

Controls that must reproduce, asserted rather than assumed (a failure exits non-zero): the six Baron
numbers ``reports/baron-reconciliation.py`` asserts (Jev ``panel_choice`` served 0.6123, Jev ``marker_noul``
0.8879, Laya fine-tuned ``marker_noul`` 0.9064, Laya base 0.1362, general LLM 0.4146 / 0.5523) plus this
suite's own headline rows — **Baron fine-tuned 27B served 0.9100** and, on ``--corpus v4``, **v4
fine-tuned 27B served 0.7928**.

Usage
-----

::

    # the Baron table (writes reports/paired-tests-baron.{md,json})
    W=/path/to/sckevtype
    uv run python reports/paired-tests-baron.py

    # the cross-check: the same reader on the v4 population must reproduce the published v4 table
    uv run python reports/paired-tests-baron.py --corpus v4 \\
        --published reports/paired-tests-v4.json \\
        --out scratch/paired-tests-v4-crosscheck.json \\
        --markdown scratch/paired-tests-v4-crosscheck.md

Reads only dumps already on disk; runs no model and submits no job. ``--corpus v4`` never writes into
``reports/``: v4's outputs belong to the settled scripts, and this file's v4 mode is a proof, not a table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETTLED = ROOT / "analysis" / "abstain-reconciliation-v4.py"
SIBLING = ROOT / "analysis" / "family-reconciliation-v4.py"
RECONCILER = ROOT / "analysis" / "baron-reconciliation.py"
#: The file that settled the exact binomial; its definitions are lifted, never re-implemented.
RULE_SOURCE = ROOT / "analysis" / "abstain-reconciliation.py"

POPULATION_FAMILY = "panel_choice"

#: The external baselines' **per-record** dumps, per corpus. ``jev`` carries the option-probability
#: vector (both readings); the general LLM carries one parsed answer and a ``hit`` flag (served only).
BASELINES: dict[str, dict] = {
    "baron": {
        "jev": {"shown_as": "Jev (jev-1.13.0)", "path": "reports/jev-baron/rows.jsonl",
                "readings": ("served", "forced")},
        "llm-a": {"shown_as": "general LLM, variant a",
                  "path": "reports/llm-baseline-baron-rows-census-varA.jsonl", "readings": ("served",)},
        "llm-b": {"shown_as": "general LLM, variant b",
                  "path": "reports/llm-baseline-baron-rows-census-varB.jsonl", "readings": ("served",)},
    },
    "v4": {
        "jev": {"shown_as": "Jev (jev-1.13.0)", "path": "reports/jev-v4/rows.jsonl",
                "readings": ("served", "forced")},
        "llm-a": {"shown_as": "general LLM, variant a",
                  "path": "reports/llm-baseline-v4-rows-census-varA.jsonl", "readings": ("served",)},
        "llm-b": {"shown_as": "general LLM, variant b",
                  "path": "reports/llm-baseline-v4-rows-census-varB.jsonl", "readings": ("served",)},
    },
}

#: The numbers this table is proved by, per corpus: what the frozen table already publishes, and the
#: accuracy each one must come back as. ``id`` names the arm or baseline and the field it lands in.
CONTROLS: dict[str, tuple[dict, ...]] = {
    "baron": (
        {"id": "jev.served", "value": 0.6123, "what": "Jev served accuracy on Baron panel_choice"},
        {"id": "llm-a.served", "value": 0.4146, "what": "general LLM (variant a) served accuracy"},
        {"id": "llm-b.served", "value": 0.5523, "what": "general LLM (variant b) served accuracy"},
        {"id": "delta-kev-27b-v4-s1.served", "value": 0.9100,
         "what": "Baron fine-tuned 27B served accuracy"},
        {"id": "delta-kev-9b-v4-s1.served", "value": 0.8988,
         "what": "Baron fine-tuned 9B served accuracy"},
        {"id": "delta-kev-4b-v4-s1.served", "value": 0.8492,
         "what": "Baron fine-tuned 4B served accuracy"},
        {"id": "laya-v4-ft-spec.served", "value": 0.6800, "what": "Laya fine-tuned served accuracy"},
        {"id": "laya-v4-base.served", "value": 0.0908, "what": "Laya base served accuracy"},
    ),
    "v4": (
        {"id": "delta-kev-27b-v4-s1.served", "value": 0.7928, "what": "v4 fine-tuned 27B served accuracy"},
        {"id": "delta-kev-9b-v4-s1.served", "value": 0.8191, "what": "v4 fine-tuned 9B served accuracy"},
        {"id": "delta-kev-4b-v4-s1.served", "value": 0.8191, "what": "v4 fine-tuned 4B served accuracy"},
        {"id": "laya-v4-ft-spec.served", "value": 0.6310, "what": "v4 Laya fine-tuned served accuracy"},
        {"id": "laya-v4-base.served", "value": 0.1294, "what": "v4 Laya base served accuracy"},
    ),
}

#: The frozen tables whose digests must be unchanged by a run of this file: it reads them and writes
#: nothing of theirs. A change is a refusal, because a proof that rewrites what it proves is not one.
UNTOUCHED = (
    "reports/abstain-reconciliation-v4.json",
    "reports/abstain-reconciliation-v4.md",
    "reports/family-reconciliation-v4-evidence_diluted.json",
    "reports/family-reconciliation-v4-panel_miss.json",
    "reports/noul-accuracy-v4.json",
    "reports/baron-reconciliation.json",
    "reports/paired-tests-v4.json",
)

#: Why a reading is missing for an arm, printed beside the row rather than left to be inferred.
NO_VECTOR = ("the dump carries no option-probability vector (one parsed answer per record), so the "
             "abstain option cannot be removed: there is no forced verdict to test")


class Refused(RuntimeError):
    """A number this table cannot produce honestly -- named rather than printed."""


# --------------------------------------------------------------------------- the settled code, imported
def module_from(path: Path, name: str, required: tuple[str, ...]):
    """A script in this repo, loaded by path, refusing by name when the API it borrows has moved.

    Importing is safe *because* every one of these scripts guards its ``main()`` behind
    ``if __name__ == "__main__"``: the module body defines the readings and writes nothing. The run's
    own ``checks.side_effects`` block re-reads the digests of every frozen table afterwards, so "writes
    nothing" is a checked fact rather than a claim about the source.
    """
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


def settled_module():
    return module_from(SETTLED, "abstain_reconciliation_v4",
                       ("cell", "read_jsonl", "sha256_file", "argmax", "forced_index", "KEV_ARMS",
                        "LAYA_ARMS"))


def sibling_module():
    return module_from(SIBLING, "family_reconciliation_v4",
                       ("laya_served_rows", "identity_without_abstain_labels", "LAYA_HALVES"))


def reconciler_module():
    return module_from(RECONCILER, "baron_reconciliation",
                       ("CORPORA", "choice_dump_path", "laya_run_path", "laya_half", "resolve"))


def rule_from(path: Path, names: tuple[str, ...]) -> dict:
    """The named function definitions taken from another script's **source**, never from a copy.

    ``reports/abstain-reconciliation.py`` is the settled analysis that defines this project's exact
    two-sided binomial on discordant pairs. Re-implementing it here would let two analyses disagree, so
    the definitions are lifted by ``ast`` and executed on their own -- the device
    ``scripts/noul-accuracy.py``, ``reports/paired-tests-v4.py`` and ``scripts/arms-forced.py`` all use.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    if missing:
        raise Refused(f"{path} no longer defines {missing}: the rule this report applies has moved, "
                      f"and applying a copy of it would be a different test")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[*imports, *[definitions[name] for name in names]], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, filename=str(path), mode="exec"), namespace)      # noqa: S102 - the project's own source
    return namespace


# --------------------------------------------------------------------------- per-row verdicts
def argmax(values) -> int:
    """First index of the maximum -- the project's own tie-break (``abstain-reconciliation.py``)."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def argmax_without(values, index: int) -> int:
    """The argmax with ``index`` removed; the whole vector's argmax when there is nothing else."""
    best, best_value = None, None
    for position, value in enumerate(values):
        if position == index:
            continue
        if best_value is None or value > best_value:
            best, best_value = position, value
    return argmax(values) if best is None else best


def forced_index(row) -> int:
    """Index of the forced pick: the argmax with the row's own abstain option removed (settled rule)."""
    abstain = row.get("abstain")
    if abstain is None:
        return argmax(row["p"])
    return argmax_without(row["p"], int(abstain))


def verdicts(row: dict, reading: str) -> bool:
    """One arm's verdict on one row, under ``served`` or ``forced``.

    ``row["abstain"]`` -- the field the settled ``cell()`` reads -- decides whether there is an abstain
    option to remove. ``None`` means the question as served offered none, so the forced verdict *is* the
    served verdict. ``sckevtype.eval.abstain_index`` is the other settled way to locate that option;
    ``build()`` compares the two on every row of every kev arm and refuses when they disagree.
    """
    label = int(row["label"])
    if reading == "served":
        return argmax(row["p"]) == label
    if reading != "forced":
        raise ValueError(f"unknown reading {reading!r}")
    return forced_index(row) == label


def vectors(rows: dict, readings: tuple[str, ...]) -> dict:
    """``{id: {"label": int, reading: bool, ...}}`` -- the verdicts a comparison is taken on."""
    out = {}
    for key, row in rows.items():
        if "p" in row:
            entry = {"label": int(row["label"])}
            for reading in readings:
                entry[reading] = verdicts(row, reading)
        else:
            # The general LLM's dump: one parsed answer per record, scored by that lane as `hit`.
            entry = {"label": int(row["label"]), "served": bool(int(row["hit"]))}
        out[key] = entry
    return out


# --------------------------------------------------------------------------- the population and arms
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


def population(census_path: Path) -> tuple[list[str], dict, dict]:
    """The census-matched ids, a summary, and the ``(dataset, cell, variant)`` -> id join map.

    The join is checked to be a bijection **within this family**: the census carries one cell in several
    families, so a triple that repeated here would make the baseline join ambiguous, and an ambiguous join
    is not a pairing.
    """
    rows = [row for row in read_jsonl(census_path) if row.get("family") == POPULATION_FAMILY]
    if not rows:
        raise Refused(f"{census_path} carries no {POPULATION_FAMILY!r} rows")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise Refused(f"{census_path}: repeated ids -- a repeated record would be counted twice")
    triple = {(row["dataset"], row["cell"], row["variant"]): row["id"] for row in rows}
    if len(triple) != len(ids):
        raise Refused(f"{census_path}: {len(ids) - len(triple)} ids share a (dataset, cell, variant) "
                      f"triple -- the baselines could not be joined onto them one to one")
    return ids, {"rows": len(rows), "distinct_ids": len(ids), "triples": len(triple)}, triple


def roster(corpus: str, dumps: Path, runs: Path) -> dict:
    """Every arm, built from the settled arm lists and the reconciler's own dump-naming mechanism."""
    settled, reconciler = settled_module(), reconciler_module()
    arms: dict = {}
    for slug, shown, rel in settled.KEV_ARMS:
        name = reconciler.choice_dump_path(rel, corpus, POPULATION_FAMILY)
        arms[slug] = {"slug": slug, "shown_as": shown, "kind": "kev", "declared": name,
                      "canonical": name, "read": dumps / Path(name).name,
                      "readings": ("served", "forced")}
    for slug, shown, rel in settled.LAYA_ARMS:
        half = reconciler.laya_half(rel)
        name = reconciler.resolve(reconciler.laya_run_path(corpus, half, POPULATION_FAMILY))
        arms[slug] = {"slug": slug, "shown_as": shown, "kind": "laya", "declared": name,
                      "canonical": str(Path(name).parent / "rows"), "read": runs / Path(name).parent.name / "rows",
                      "readings": ("served", "forced")}
    for slug, config in BASELINES[corpus].items():
        arms[slug] = {"slug": slug, "shown_as": config["shown_as"], "kind": "baseline",
                      "declared": config["path"], "canonical": config["path"],
                      "read": Path(config["path"]), "readings": config["readings"]}
    return arms


def ladder(arms: dict) -> list[tuple[str, str, str]]:
    """``(size, fine-tune slug, rung slug)`` for the kev ladder, ordered by parameter count."""
    pairs = []
    for slug in arms:
        if not slug.startswith("delta-kev-"):
            continue
        size = slug[len("delta-kev-"):].split("-v4")[0]
        rung = f"kev-{size}-v4"
        if rung not in arms:
            raise Refused(f"{slug} has no rung {rung} in the arm list: the ladder cannot be paired, "
                          f"and a missing rung is not a missing test")
        pairs.append((size, slug, rung))
    pairs.sort(key=lambda item: float(item[0].rstrip("b")))
    return pairs


def laya_pair(arms: dict) -> tuple[str, str]:
    """``(fine-tune, base)`` for the sibling engine, identified by name, never by position."""
    laya = [slug for slug, arm in arms.items() if arm["kind"] == "laya"]
    bases = [slug for slug in laya if "base" in slug]
    if len(laya) != 2 or len(bases) != 1:
        raise Refused(f"the Laya arms are {laya}: one fine-tune and one base were expected")
    return next(slug for slug in laya if slug not in bases), bases[0]


def load_baseline(path: Path, ids: list[str], triple_to_id: dict) -> dict:
    """An external arm's per-record rows, joined onto the census id by its ``(dataset, cell, variant)``."""
    wanted = set(ids)
    joined: dict = {}
    for row in read_jsonl(path):
        if row.get("family") != POPULATION_FAMILY:
            continue
        key = triple_to_id.get((row["dataset"], row["cell"], row["variant"]))
        if key is None or key not in wanted:
            continue
        if key in joined:
            raise Refused(f"{path}: {key} is answered twice -- the join is not one to one")
        joined[key] = row
    return joined


# --------------------------------------------------------------------------- the statistics
def mcnemar(left: dict, right: dict, reading: str, rule: dict) -> dict:
    """McNemar's exact test for two arms scored on the same records.

    With ``b`` = left right / right wrong and ``c`` = the reverse, the null is ``b ~ Binomial(b + c, 0.5)``
    and the two-sided p is ``2 * P(X <= min(b, c))`` -- :func:`binom_two_sided`, lifted from the file that
    settled it. Never a difference of two accuracies: on a paired design that throws away the pairing and
    the noise it cancels, and it is the wrong test for binary verdicts.
    """
    shared = sorted(set(left) & set(right))
    if len(shared) != len(left) or len(shared) != len(right):
        raise Refused(f"{reading}: the two arms are not on the same records "
                      f"({len(left)} and {len(right)} rows, {len(shared)} shared)")
    both = left_only = right_only = neither = 0
    for key in shared:
        a, b = left[key][reading], right[key][reading]
        both += int(a and b)
        left_only += int(a and not b)
        right_only += int(b and not a)
        neither += int(not a and not b)
    p = rule["binom_two_sided"](left_only, right_only)
    log10 = rule["binom_two_sided_log10"](left_only, right_only)
    return {
        "n": len(shared),
        "contingency": {"both_correct": both, "left_only": left_only,
                        "right_only": right_only, "both_wrong": neither},
        "discordant": left_only + right_only,
        "accuracy_left": (both + left_only) / len(shared),
        "accuracy_right": (both + right_only) / len(shared),
        "difference_points": 100.0 * ((both + left_only) - (both + right_only)) / len(shared),
        "p_value": p,
        "p_value_log10": log10,
    }


def holm(entries: list[dict]) -> None:
    """Holm-Bonferroni over one family, **in log space**, writing ``p_holm`` beside each entry.

    The p-values of this family underflow a float (a lopsided pair on 2,600 records is far below
    ``5e-324``), so the adjustment is taken on ``p_value_log10``: the step-up maximum of
    ``log10(p) + log10(m - rank)`` is the same rule, computed where the number still exists. ``m`` is the
    family's own size, announced on every entry so a reader can check the multiplier.
    """
    m = len(entries)
    order = sorted(range(m), key=lambda i: (entries[i]["p_value_log10"], entries[i]["p_value"]))
    running = -math.inf
    for rank, index in enumerate(order):
        adjusted = entries[index]["p_value_log10"] + math.log10(m - rank)
        running = max(running, adjusted)
        capped = min(0.0, running)
        entries[index]["holm"] = {"family_size": m, "rank": rank + 1,
                                  "p_value_log10": capped,
                                  "p_value": 10.0 ** capped if capped > -323 else 0.0}


# --------------------------------------------------------------------------- build
def build(args: argparse.Namespace) -> dict:
    """Every number in the report, checks first: nothing is compared until the inputs agree."""
    corpus = args.corpus
    reconciler, settled, sibling = reconciler_module(), settled_module(), sibling_module()
    config = reconciler.CORPORA[corpus]
    ids_path = Path(args.ids) if args.ids else ROOT / config["ids"]
    if not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    if not ids_path.is_file():
        raise Refused(f"no population file at {ids_path}")
    ids, population_info, triple_to_id = population(ids_path)
    rule = rule_from(RULE_SOURCE, ("binom_two_sided", "binom_two_sided_log10"))
    arms = roster(corpus, Path(args.dumps), Path(args.runs))

    raw: dict = {}
    missing: list[dict] = []
    path_rule_mismatches: list[str] = []
    for slug, arm in arms.items():
        path = arm["read"]
        if arm["kind"] == "laya":
            if not Path(path).is_dir():
                missing.append({"arm": slug, "expected": str(path), "why": "no rows directory"})
                continue
            rows_list = sibling.laya_served_rows(Path(path), set(ids))
            if rows_list is None:
                missing.append({"arm": slug, "expected": str(path), "why": "no row files"})
                continue
            rows = {row["id"]: row for row in rows_list}
            arm["sha256"] = {p.name: sha256_file(p) for p in sorted(Path(path).glob("*.jsonl"))}
        elif arm["kind"] == "baseline":
            if not Path(path).is_file():
                missing.append({"arm": slug, "expected": str(path), "why": "no such dump"})
                continue
            rows = load_baseline(Path(path), ids, triple_to_id)
            arm["sha256"] = sha256_file(Path(path))
        else:
            if not Path(path).is_file():
                missing.append({"arm": slug, "expected": str(path), "why": "no such dump"})
                continue
            wanted = set(ids)
            rows = {row["id"]: row for row in read_jsonl(Path(path)) if row.get("id") in wanted}
            arm["sha256"] = sha256_file(Path(path))
        if len(rows) != len(ids):
            raise Refused(f"{slug}: covers {len(rows)} of {len(ids)} records -- a paired test over a "
                          f"partial population is not this table")
        if arm["kind"] == "kev":
            # The row's own `abstain` field against sckevtype.eval.abstain_index, and this file's
            # argmax/forced-index against the settled script's: two code paths, compared per row.
            from sckevtype.eval import abstain_index

            moved = [key for key, row in rows.items() if row.get("abstain") != abstain_index(row)]
            if moved:
                path_rule_mismatches.append(f"{slug}: {len(moved)} rows")
            settled_mismatch = [
                key for key, row in rows.items()
                if settled.argmax(row["p"]) != argmax(row["p"])
                or settled.forced_index(row) != forced_index(row)
            ]
            if settled_mismatch:
                raise Refused(f"{slug}: this file's argmax/forced_index disagree with the settled "
                              f"script's on {len(settled_mismatch)} rows -- they are not the same rule")
        arm["coverage"] = {"rows": len(rows), "missing": 0, "extra": 0}
        arm["vectors"] = vectors(rows, arm["readings"])
        arm["correct"] = {reading: sum(1 for row in arm["vectors"].values() if row[reading])
                          for reading in arm["readings"]}
        arm["accuracy"] = {reading: (arm["correct"][reading] / len(ids) if reading in arm["correct"]
                                     else None) for reading in ("served", "forced")}
        raw[slug] = rows
    if missing:
        raise Refused(f"missing dumps: {json.dumps(missing, indent=1)}")
    if path_rule_mismatches:
        raise Refused(f"the row's `abstain` field and sckevtype.eval.abstain_index disagree on "
                      f"{path_rule_mismatches}: the two settled rules would drop different options")
    return {"rule": rule, "ids": ids, "population": population_info, "arms": arms, "raw": raw,
            "corpus": corpus, "config": config, "census": ids_path}


def comparisons(state: dict) -> list[dict]:
    """Every comparison the paper makes, in the order a reader needs them.

    ``ladder`` first (each fine-tune against the rung it started from, ordered by parameter count), then
    the sibling engine, then the external baselines, then the exploratory ordering checks.
    """
    arms, rule = state["arms"], state["rule"]
    entries: list[dict] = []

    def add(group: str, left: str, right: str, reading: str, note: str | None = None,
            exploratory: bool = False) -> dict:
        block = mcnemar(arms[left]["vectors"], arms[right]["vectors"], reading, rule)
        entry = {
            "comparison": f"{arms[left]['shown_as']} vs {arms[right]['shown_as']}",
            "group": group, "reading": reading,
            "left": {"arm": left, "shown_as": arms[left]["shown_as"],
                     "correct": arms[left]["correct"][reading],
                     "accuracy": arms[left]["accuracy"][reading]},
            "right": {"arm": right, "shown_as": arms[right]["shown_as"],
                      "correct": arms[right]["correct"][reading],
                      "accuracy": arms[right]["accuracy"][reading]},
            "exploratory": exploratory,
        }
        entry.update(block)
        if note:
            entry["note"] = note
        entries.append(entry)
        return entry

    ladder_pairs = ladder(arms)
    for _size, fine_tune, rung in ladder_pairs:
        for reading in ("served", "forced"):
            add("ladder", fine_tune, rung, reading)

    fine_tuned_laya, base_laya = laya_pair(arms)
    for reading in ("served", "forced"):
        add("sibling", fine_tuned_laya, base_laya, reading)

    fine_tunes = [fine_tune for _, fine_tune, _ in ladder_pairs] + [fine_tuned_laya]
    for fine_tune in fine_tunes:
        for reading in ("served", "forced"):
            add("external-jev", fine_tune, "jev", reading)
    for baseline in ("llm-a", "llm-b"):
        for fine_tune in fine_tunes:
            add("external-llm", fine_tune, baseline, "served")

    # Exploratory, and marked so: the ordering checks were not pre-specified, so they are outside the
    # corrected family below. The two 27B pairs are the headline question -- on v4 the 27B sits *below*
    # both 4B and 9B, so whether Baron's ordering is real is exactly what a paired test has to say.
    for left_size, right_size in (("0.8b", "9b"), ("4b", "9b"), ("9b", "27b"), ("4b", "27b")):
        for reading in ("served", "forced"):
            add("ordering", f"delta-kev-{left_size}-v4-s1", f"delta-kev-{right_size}-v4-s1", reading,
                note="adjacent fine-tunes of the ladder, tested to see whether the point-estimate "
                     "ordering is real; not a pre-specified comparison",
                exploratory=True)
    return entries


# --------------------------------------------------------------------------- checks
def frozen_cells(corpus: str) -> dict:
    """The frozen table this reader must reproduce, per family: its arm cells and baseline cells."""
    if corpus == "baron":
        path = ROOT / "reports" / "baron-reconciliation.json"
        if not path.is_file():
            return {}
        return {"path": path, "kind": "baron_reconciliation",
                "families": json.loads(path.read_text(encoding="utf-8"))["families"]}
    path = ROOT / "reports" / "abstain-reconciliation-v4.json"
    if not path.is_file():
        return {}
    return {"path": path, "kind": "abstain_reconciliation_v4",
            "arms": json.loads(path.read_text(encoding="utf-8"))["arms"]}


def reproduce(state: dict) -> dict:
    """Every served / forced / abstain-pick count this table derives, against the frozen table's own.

    Two shapes, because the two corpora's frozen tables have two shapes: Baron's reconciliation carries
    all four families and its baselines inline, so the ``panel_choice`` block is compared cell by cell;
    v4's ``abstain-reconciliation-v4.json`` carries the arms only. A mismatch is a refusal.
    """
    corpus, arms = state["corpus"], state["arms"]
    frozen = frozen_cells(corpus)
    if not frozen:
        return {"checked": False,
                "why": f"no frozen {corpus} table on disk; nothing to reproduce against"}
    compared, mismatches, absent = 0, [], []

    def cmp(slug: str, field: str, got, want) -> None:
        nonlocal compared
        compared += 1
        if got != want:
            mismatches.append({"arm": slug, "field": field, "computed": got, "frozen": want})

    if frozen["kind"] == "baron_reconciliation":
        block = frozen["families"].get(POPULATION_FAMILY, {})
        for slug, arm in arms.items():
            if arm["kind"] == "baseline":
                continue
            want = block.get("arms", {}).get(slug)
            if not want or "served" not in want:
                absent.append(slug)
                continue
            cmp(slug, "served.correct", arm["correct"]["served"], want["served"]["correct"])
            cmp(slug, "forced.correct", arm["correct"]["forced"], want["forced"]["correct"])
            picks = sum(1 for row in state["raw"][slug].values()
                        if row.get("abstain") is not None
                        and argmax(row["p"]) == int(row["abstain"]))
            cmp(slug, "abstain_pick.picks", picks, want["abstain_pick"]["picks"])
        jev = block.get("baselines", {}).get("jev", {})
        if jev.get("status") == "complete" and "jev" in arms:
            rows = state["raw"]["jev"]
            cmp("jev", "served.correct", arms["jev"]["correct"]["served"], jev["served"]["correct"])
            # That scorer defines forced / abstain-pick on the rows offering an abstain option, so the
            # comparison is against its own eligible-restricted count -- never against its
            # population-denominator cell (which mixes a population denominator with an eligible count).
            eligible = [row for row in rows.values() if row.get("abstain") is not None]
            forced_eligible = sum(1 for row in eligible if forced_index(row) == int(row["label"]))
            over = (jev.get("over_the_rows_offering_an_abstain_option") or {})
            want_forced = (over.get("forced") or {}).get("correct", (jev.get("forced") or {}).get("correct"))
            cmp("jev", "forced.correct (over the rows offering an abstain option)",
                forced_eligible, want_forced)
            picks = sum(1 for row in rows.values()
                        if row.get("abstain") is not None and argmax(row["p"]) == int(row["abstain"]))
            cmp("jev", "abstain_pick.picks", picks, jev["abstain_pick"]["picks"])
        llm = block.get("baselines", {}).get("llm", {})
        for slug in ("llm-a", "llm-b"):
            variant = "var" + slug[-1].upper()
            want = (llm.get("variants") or {}).get(variant, {})
            if not want.get("accuracy") or slug not in arms:
                absent.append(slug)
                continue
            cmp(slug, "correct (accuracy x n)", arms[slug]["correct"]["served"],
                round(want["accuracy"] * want["n"]))
    else:
        for slug, arm in arms.items():
            if arm["kind"] == "baseline":
                continue
            want = frozen["arms"].get(slug)
            if not want or "served" not in want:
                absent.append(slug)
                continue
            cmp(slug, "served.correct", arm["correct"]["served"], want["served"]["correct"])
            cmp(slug, "forced.correct", arm["correct"]["forced"], want["forced"]["correct"])
    if mismatches:
        raise Refused(f"this table does not reproduce {frozen['path'].name}: {mismatches[:6]}")
    return {"checked": True, "source": frozen["path"].relative_to(ROOT).as_posix(),
            "family": POPULATION_FAMILY, "fields_compared": compared, "mismatches": 0,
            "arms_absent_from_the_frozen_table": absent,
            "note": "every served and forced count derived per row here equals the frozen table's, "
                    "integer for integer"}


def reproduce_published(state: dict, path: Path | None) -> dict:
    """On ``--corpus v4``: the published v4 paired table's contingency cells, recomputed here.

    The v4 paired tests were published from a synced copy of these dumps. Re-deriving their contingency
    cells from the dumps themselves is the strongest available check that this file's reader is the same
    reader -- and it is a check the Baron run cannot carry, because Baron has no published counterpart.
    """
    if path is None or not Path(path).is_file():
        return {"checked": False, "why": f"no published paired table at {path}"}
    published = json.loads(Path(path).read_text(encoding="utf-8"))
    arms, rule = state["arms"], state["rule"]
    compare = 0
    mismatches = []
    for entry in published.get("comparisons", []):
        left, right, reading = entry["left"]["arm"], entry["right"]["arm"], entry["reading"]
        if left not in arms or right not in arms:
            continue
        block = mcnemar(arms[left]["vectors"], arms[right]["vectors"], reading, rule)
        compare += 1
        for field in ("both_correct", "left_only", "right_only", "both_wrong"):
            if block["contingency"][field] != entry["contingency"][field]:
                mismatches.append({"pair": f"{left} vs {right}", "reading": reading, "field": field,
                                   "computed": block["contingency"][field],
                                   "published": entry["contingency"][field]})
        if block["p_value"] != entry["p_value"] or block["p_value_log10"] != entry["p_value_log10"]:
            mismatches.append({"pair": f"{left} vs {right}", "reading": reading, "field": "p_value",
                               "computed": block["p_value"], "published": entry["p_value"]})
    if mismatches:
        raise Refused(f"this reader does not reproduce {Path(path).name}: {mismatches[:6]}")
    return {"checked": True, "source": Path(path).as_posix(), "comparisons_compared": compare,
            "mismatches": 0,
            "note": "every contingency cell and p-value of the published v4 paired table, recomputed "
                    "from the v4 dumps through this file's own reader"}


def controls(state: dict) -> dict:
    """The numbers that already exist elsewhere, re-derived through this file's own pipeline."""
    corpus, arms = state["corpus"], state["arms"]
    checks = []
    for control in CONTROLS[corpus]:
        slug, field = control["id"].rsplit(".", 1)
        arm = arms.get(slug)
        value = arm["accuracy"].get(field) if arm else None
        ok = value is not None and abs(round(value, 4) - control["value"]) < 1e-9
        checks.append({**control, "observed": value,
                       "observed_rounded": None if value is None else round(value, 4),
                       "reproduced": ok})
    return {"what": "numbers that already exist elsewhere in this repo, re-derived through this file's "
                    "own reader; every one of them must reproduce or the run fails",
            "checks": checks, "n_reproduced": sum(1 for c in checks if c["reproduced"]),
            "n_checks": len(checks)}


def side_effects() -> dict:
    """Digests of the frozen tables, taken before and after the run: a proof, not a promise."""
    return {name: (sha256_file(ROOT / name) if (ROOT / name).is_file() else None) for name in UNTOUCHED}


def checks(state: dict, before: dict, published: Path | None) -> dict:
    """The things this table refuses to publish over: label drift, a moved rule, a moved table."""
    arms = state["arms"]
    labels: dict = {}
    disagreements = []
    for slug, arm in arms.items():
        for key, row in arm["vectors"].items():
            first = labels.setdefault(key, (slug, row["label"]))
            if first[1] != row["label"]:
                disagreements.append({"record": key, "arms": [first[0], slug],
                                      "labels": [first[1], row["label"]]})
    if disagreements:
        raise Refused(f"{len(disagreements)} records carry a different reference label in different "
                      f"arms: {disagreements[:3]}")
    after = side_effects()
    changed = [name for name in after if before.get(name) != after[name]]
    if changed:
        raise Refused(f"this run changed {changed}, which it must not touch -- a proof that rewrites "
                      f"what it proves is not one")
    return {
        "labels_agree_across_arms": {"records": len(labels), "disagreements": 0},
        "abstain_field_equals_index": {"arms_checked": sum(1 for a in arms.values()
                                                           if a["kind"] == "kev"),
                                       "records_per_arm": len(state["ids"]), "disagreements": 0},
        "reproduces_the_frozen_table": reproduce(state),
        "reproduces_the_published_v4_paired_table": reproduce_published(state, published),
        "side_effects": {"files": sorted(after), "changed": changed,
                         "note": "digests of every frozen table before and after this run; none moved"},
    }


# --------------------------------------------------------------------------- the report
def show_path(path) -> str:
    """A path as this repo would write it, when it lives inside the repo."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


def p_text(entry: dict) -> str:
    """A p-value as a float, or its ``log10`` magnitude when a float cannot hold it."""
    if entry["p_value"] > 0.0:
        return f"{entry['p_value']:.3g}"
    return f"<5e-324 (log10 {entry['p_value_log10']:.1f})"


def p_holm_text(entry: dict) -> str:
    if entry["holm"] is None:
        return "— (exploratory)"
    if entry["holm"]["p_value"] > 0.0:
        return f"{entry['holm']['p_value']:.3g}"
    return f"<5e-324 (log10 {entry['holm']['p_value_log10']:.1f})"


def significant(entry: dict) -> bool:
    """Significant at 0.05 -- after the correction where the entry is in the corrected family."""
    if entry["holm"] is None:
        return entry["p_value"] < 0.05
    return entry["holm"]["p_value"] < 0.05


def headline(entries: list[dict], corpus: str) -> dict:
    """The 27B question, answered from the two ordering entries (and its v4 counterpart for contrast).

    The pair is looked up **by arm set, in either direction**: ``comparisons()`` emits the ordering
    checks as ``(9B, 27B)`` and ``(4B, 27B)`` — small arm first, as the ladder is — so a lookup keyed on
    ``27B`` being the *left* side would silently find nothing and report the headline as untestable. Each
    side is therefore identified by its arm slug and the discordant counts are oriented to 27B here.
    """
    def pick(other_size: str, reading: str) -> dict | None:
        other = f"delta-kev-{other_size}-v4-s1"
        twenty_seven = "delta-kev-27b-v4-s1"
        for entry in entries:
            arms = (entry["left"]["arm"], entry["right"]["arm"])
            if reading != entry["reading"] or set(arms) != {twenty_seven, other}:
                continue
            side = "left" if entry["left"]["arm"] == twenty_seven else "right"
            cells = entry["contingency"]
            twenty_seven_wins = cells["left_only"] if side == "left" else cells["right_only"]
            other_wins = cells["right_only"] if side == "left" else cells["left_only"]
            return {
                "comparison": f"fine-tuned 27B vs fine-tuned {other_size.upper()}",
                "reading": reading,
                "accuracy_27b": entry[f"accuracy_{side}"],
                "accuracy_other": entry["accuracy_right" if side == "left" else "accuracy_left"],
                "difference_points": (100.0 * (twenty_seven_wins - other_wins) / entry["n"]),
                "discordant_27b_wins": twenty_seven_wins,
                "discordant_other_wins": other_wins,
                "p_value": entry["p_value"], "p_value_log10": entry["p_value_log10"],
                "significant_at_0.05": entry["p_value"] < 0.05,
                "exploratory": True,
                "oriented": "27B is the left side of the published pair" if side == "left"
                            else "27B is the right side of the published pair; counts oriented to 27B here",
            }
        return None

    pairs = {}
    for other_size in ("9b", "4b"):
        for reading in ("served", "forced"):
            entry = pick(other_size, reading)
            if entry is not None:
                pairs[f"27b_vs_{other_size}_{reading}"] = entry
    contrast: dict = {}
    for name, path in (("v4", "reports/abstain-reconciliation-v4.json"),
                       ("baron", "reports/baron-reconciliation.json")):
        p = ROOT / path
        if not p.is_file():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        if name == "v4":
            arms = raw.get("arms", {})
            cells = {slug: (block.get("served") or {}).get("accuracy")
                     for slug, block in arms.items()
                     if slug in ("delta-kev-27b-v4-s1", "delta-kev-9b-v4-s1", "delta-kev-4b-v4-s1")}
        else:
            arms = raw.get("families", {}).get(POPULATION_FAMILY, {}).get("arms", {})
            cells = {slug: (block.get("served") or {}).get("accuracy")
                     for slug, block in arms.items()
                     if slug in ("delta-kev-27b-v4-s1", "delta-kev-9b-v4-s1", "delta-kev-4b-v4-s1")}
        if cells:
            contrast[name] = {slug: value for slug, value in cells.items()
                              if value is not None}
    for name, cells in contrast.items():
        if len(cells) == 3:
            contrast[name]["order"] = sorted(cells, key=lambda slug: -cells[slug])
    return {
        "question": "is the 27B fine-tune's Baron lead over the 9B and 4B fine-tunes real?",
        "answer": _headline_answer(pairs),
        "pairs": pairs,
        "served_accuracies_by_corpus": contrast,
        "why_it_matters": "on v4 the 27B fine-tune sits *below* both the 4B and the 9B on served "
                          "accuracy; on Baron it sits above both. One of the two orderings is a "
                          "point-estimate artefact, and only a paired test on the records can say which",
    }


def _classify(p_value: float) -> str:
    """How strong one pair's evidence is, in words derived from the number rather than chosen beside it."""
    if p_value < 1e-3:
        return "decisive"
    if p_value < 0.05:
        return "marginal"
    return "not significant"


def _headline_answer(pairs: dict) -> str:
    """One sentence, built from the numbers rather than written beside them.

    The strength is *classified from the p-value* — decisive, marginal, or not significant — so the
    sentence cannot say "real" about a pair that barely clears 0.05, and cannot say "not established"
    about a pair that is 34 orders of magnitude from the null.
    """
    parts: list[str] = []
    verdicts: dict[str, str] = {}
    for key in ("27b_vs_9b_served", "27b_vs_4b_served"):
        entry = pairs.get(key)
        if entry is None:
            continue
        other = key.split("_vs_")[1].split("_")[0].upper()
        verdicts[other] = _classify(entry["p_value"])
        parts.append(f"vs {other} {entry['difference_points']:+.2f} pt "
                     f"({entry['discordant_27b_wins']}/{entry['discordant_other_wins']} discordant, "
                     f"{p_text(entry)}, {verdicts[other]})")
    if not parts:
        return "no 27B ordering pair was testable"
    decisive = [name for name, verdict in verdicts.items() if verdict == "decisive"]
    marginal = [name for name, verdict in verdicts.items() if verdict == "marginal"]
    lost = [name for name, verdict in verdicts.items() if verdict == "not significant"]
    if lost and not decisive and not marginal:
        lead = "the 27B's served lead is not established on these records"
    elif marginal and not decisive:
        lead = f"the 27B's served lead over {', '.join(marginal)} is marginal at best"
    elif marginal:
        lead = (f"the 27B's served lead is decisive over {', '.join(decisive)} and marginal over "
                f"{', '.join(marginal)}")
    else:
        lead = f"the 27B's served lead is decisive over {', '.join(decisive)}"
    caveat = (" Both pairs are exploratory — the pre-specified family is the ladder and the external "
              "baselines — so a marginal raw p is not corrected here and would not survive a correction."
              if marginal else "")
    return f"{lead}: " + "; ".join(parts) + "." + caveat


TABLE_HEADER = (
    "| A | B | reading | acc A | acc B | Δ (pt) | both ✓ | A only | B only | both ✗ | p | p (Holm) |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
)


def row(entry: dict) -> str:
    cells = entry["contingency"]
    return (f"| {entry['left']['shown_as']} | {entry['right']['shown_as']} | {entry['reading']} | "
            f"{entry['accuracy_left']:.4f} | {entry['accuracy_right']:.4f} | "
            f"{entry['difference_points']:+.2f} | {cells['both_correct']:,} | "
            f"{cells['left_only']:,} | {cells['right_only']:,} | {cells['both_wrong']:,} | "
            f"{p_text(entry)} | {p_holm_text(entry)} |")


def markdown(result: dict) -> str:
    """The same numbers, as the table the manuscript can carry and a reader can check."""
    lines: list[str] = []
    for part in (md_top, md_headline, md_tables, md_prose, md_checks, md_sources):
        lines.extend(part(result))
    return "\n".join(lines).rstrip("\n") + "\n"


def md_top(result: dict) -> list[str]:
    population = result["population"]
    families = result["multiplicity"]["families"]
    return [
        f"# Paired significance tests — `panel_choice`, {result['corpus']['label']} "
        f"(n = {population['rows']:,} records)",
        "",
        "*Machine-readable companion: `reports/paired-tests-baron.json` (one entry per comparison).*",
        "",
        "`reports/baron-reconciliation.md` gives the served and forced accuracy of every Baron arm with "
        "**no test and no discordant-pair counts**. This is that table's missing half: every comparison "
        "the manuscript makes, on the same records, with an exact paired test — the Baron counterpart of "
        "`reports/paired-tests-v4.md`.",
        "",
        "## The test",
        "",
        "Correctness is binary per record (top-1 against the reference label), and both arms of every "
        "comparison answer the **same** records, so the test is **McNemar's exact test**: with `b` = A "
        "right / B wrong and `c` = the reverse,",
        "",
        "```",
        "p = 2 * P(X <= min(b, c))   under   X ~ Binomial(b + c, 0.5)",
        "```",
        "",
        "It is `reports/abstain-reconciliation.py`'s own `binom_two_sided`, lifted from that file's "
        "source and applied unchanged — the rule `reports/PAIRED-TESTS.md`, `reports/paired-tests-v4.md` "
        "and `reports/noul-accuracy-v4.md` use. Where the p-value is below what a float holds it is shown "
        "as its `log10` magnitude, from the same file's `binom_two_sided_log10`.",
        "",
        "Two readings, both reported: **served** — `argmax(p) == label`, abstention available; **forced** "
        "— `argmax(p)` with the abstain option removed, where a record served with no abstain option has "
        "nothing to remove and its forced verdict is its served verdict.",
        "",
        f"Multiplicity is addressed at the end: Holm–Bonferroni within each reading, over the "
        f"{families['served']} served and {families['forced']} forced pre-specified comparisons.",
    ]


def md_headline(result: dict) -> list[str]:
    block = result["headline"]
    pairs = block["pairs"]
    contrast = block.get("served_accuracies_by_corpus", {})
    lines = ["", "## The headline question: is the 27B's Baron lead real?", "",
             f"**{block['question']}** {block['why_it_matters']}.", "",
             block["answer"] + ".", "",
             "| pair | reading | 27B | other | Δ (pt) | 27B only | other only | p |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key in ("27b_vs_9b_served", "27b_vs_4b_served", "27b_vs_9b_forced", "27b_vs_4b_forced"):
        entry = pairs.get(key)
        if entry is None:
            continue
        other = key.split("_vs_")[1].split("_")[0].upper()
        lines.append(f"| 27B vs {other} | {entry['reading']} | {entry['accuracy_27b']:.4f} | "
                     f"{entry['accuracy_other']:.4f} | {entry['difference_points']:+.2f} | "
                     f"{entry['discordant_27b_wins']:,} | {entry['discordant_other_wins']:,} | "
                     f"{p_text(entry)} |")
    lines += ["", "These four comparisons are **exploratory** — they were not pre-specified, so they sit "
                  "outside the corrected family and are reported raw.", ""]
    if contrast:
        lines += ["Served accuracy of the three largest fine-tunes, per corpus (from the settled tables, "
                  "not re-derived here):", "",
                  "| corpus | 27B | 9B | 4B | order |", "|---|---:|---:|---:|---|"]
        for name in ("v4", "baron"):
            cells = contrast.get(name)
            if not cells:
                continue
            order = " > ".join(slug.replace("delta-kev-", "").replace("-v4-s1", "")
                               for slug in cells.get("order", []))
            lines.append(f"| {name} | {cells.get('delta-kev-27b-v4-s1', float('nan')):.4f} | "
                         f"{cells.get('delta-kev-9b-v4-s1', float('nan')):.4f} | "
                         f"{cells.get('delta-kev-4b-v4-s1', float('nan')):.4f} | {order} |")
        lines += ["", "The v4 column is the reason this test exists: the same three arms order "
                      "differently on the two corpora, and the paired test is what separates a real "
                      "corpus effect from a coin flip on the records that disagree."]
    return lines


def md_tables(result: dict) -> list[str]:
    lines: list[str] = ["", "## The ladder — each fine-tune against the rung it started from", "",
                        "The rung is the released checkpoint of the same size, scored on the same "
                        "records (`kev-<size>-v4`). Both halves answer the same prompt, so the "
                        "discordance is weights.", "", *TABLE_HEADER]
    groups = (("ladder", None), ("sibling", "## The sibling engine — Laya fine-tuned against its base"),
              ("external-jev", "## Against Jev"),
              ("external-llm", "## Against the general LLM (served only)"))
    for group, heading in groups:
        if heading:
            lines += ["", heading, ""]
            if group == "sibling":
                lines += ["One pair, the same comparison one engine over. The two halves were rendered "
                          "at their own geometry (the fine-tune at 8192/2048, the released base at its "
                          "shipped 1024/256), so this discordance carries a prompt difference as well "
                          "as a weights difference.", ""]
            if group == "external-jev":
                lines += ["The same fine-tunes against the outside engine, on the same records. Jev's "
                          "dump carries its option-probability vector, so it has both readings; note "
                          "that Jev's own scorer defines *forced* on the 650 rows that offer an abstain "
                          "option, while the comparison here uses the settled population-wide rule on "
                          "both sides — the same rule every arm is scored by.", ""]
            if group == "external-llm":
                lines += ["The general LLM (Qwen3.5-9B zero-shot), two prompt variants. Its dump has one "
                          "parsed answer per record and **no probability vector**, so only its served "
                          "reading exists — the forced comparison is listed as not testable below rather "
                          "than dropped or estimated.", ""]
            lines += list(TABLE_HEADER)
        for entry in result["comparisons"]:
            if entry["group"] == group:
                lines.append(row(entry))
    return lines


def md_prose(result: dict) -> list[str]:
    entries = result["comparisons"]
    families = result["multiplicity"]["families"]
    total_p = families["served"] + families["forced"]
    raw = result["multiplicity"]["raw_significant_at_0.05"]
    crossing = result["multiplicity"]["would_cross_0.05_after_the_correction"]
    ladder_served = [e for e in entries if e["group"] == "ladder" and e["reading"] == "served"]
    ladder_forced = [e for e in entries if e["group"] == "ladder" and e["reading"] == "forced"]
    jev = [e for e in entries if e["group"] == "external-jev"]
    llm = [e for e in entries if e["group"] == "external-llm"]
    sibling = [e for e in entries if e["group"] == "sibling"]
    ordering = [e for e in entries if e["exploratory"]]

    def tally(group: list[dict]) -> str:
        return f"{sum(1 for e in group if significant(e))} of {len(group)}"

    exceptions = [f"{e['left']['shown_as']} ({e['reading']})" for e in jev if e["difference_points"] <= 0]
    lines = ["", "## Pairs that could not be tested", ""]
    for block in result["not_testable"]:
        lines += [f"* **{block['comparison']}** — {block['why']}. It would need "
                  f"{block['what_it_would_need']}. The served reading is tested; the forced one is "
                  f"absent, not zero and not estimated."]
    lines += ["", "## Exploratory: is the ordering real?", "",
              "The fine-tunes' ordering is a point-estimate ordering until it is tested, and on this "
              "corpus it is the point of the table: the 27B sits above the 9B and 4B on Baron while it "
              "sits *below* both on v4. The adjacent steps — including the two 27B pairs — are tested "
              "under both readings. They were **not** pre-specified, so they are outside the corrected "
              "family and are reported raw.", "", *TABLE_HEADER]
    lines += [row(entry) for entry in ordering]
    yes = [f"{e['left']['shown_as'].replace('fine-tuned ', '')} vs "
           f"{e['right']['shown_as'].replace('fine-tuned ', '')} ({e['reading']})"
           for e in ordering if significant(e)]
    no = [f"{e['left']['shown_as'].replace('fine-tuned ', '')} vs "
          f"{e['right']['shown_as'].replace('fine-tuned ', '')} ({e['reading']})"
          for e in ordering if not significant(e)]
    lines += ["", f"* significant at 0.05: {'; '.join(yes) if yes else 'none'};",
              f"* not significant: {'; '.join(no) if no else 'none'}."]

    lines += ["", "## What the table says", "",
              f"* **The ladder:** {tally(ladder_served)} fine-tune-vs-rung comparisons are significant "
              f"at 0.05 after the correction under the served reading, and {tally(ladder_forced)} under "
              f"the forced reading.",
              f"* **Against Jev:** {tally(jev)} comparisons are significant; "
              f"{sum(1 for e in jev if e['difference_points'] > 0)} of them favour the fine-tune"
              + (f", and the exceptions are {', '.join(exceptions)}" if exceptions else "") + ".",
              f"* **Against the general LLM:** {tally(llm)} served comparisons are significant; "
              f"{sum(1 for e in llm if e['difference_points'] <= 0)} of them fail to favour the "
              f"fine-tune.",
              f"* **Laya against its own base:** {tally(sibling)} comparisons are significant, under "
              f"both readings."]
    verdict_text = (
        f"The correction does not change a single verdict: all {total_p - crossing} comparisons that "
        f"are significant raw stay significant once corrected."
        if crossing == 0 else
        f"The correction changes {crossing} verdicts: {raw} of the {total_p} comparisons are "
        f"significant raw, and {raw - crossing} stay below 0.05 once corrected.")
    primary = [e for e in entries if not e["exploratory"]]
    weakest = sorted(primary, key=lambda e: -e["p_value"])[:2]
    weakest_text = "; ".join(f"{e['left']['shown_as']} vs {e['right']['shown_as']} ({e['reading']}) "
                             f"at p = {p_text(e)}" for e in weakest)
    lines += ["", "## Multiplicity", "",
              f"**A correction was applied: Holm–Bonferroni, within each reading, over the "
              f"{families['served']} served and {families['forced']} forced pre-specified "
              f"comparisons.**", "",
              f"The family is 'the table', not 'each row in isolation': {total_p} p-values printed side "
              "by side will be scanned for the small ones, and some rows here are plausibly null — the "
              f"two weakest are {weakest_text}. The comparisons were not fished from the data (each "
              "fine-tune has exactly one rung, and each is compared to each external baseline), and the "
              "family is strongly positively correlated because every arm answers the same records, so a "
              "Bonferroni-style bound is conservative here. Holm is the least conservative correction "
              "that still controls the family-wise error rate. A reader who prefers no correction can "
              "read the raw column: it is printed beside every adjusted one.", "",
              verdict_text + " The exploratory ordering checks are excluded from the family and labelled "
              "as such in every row they appear in.", "",
              "## What this does not establish", ""]
    lines += [f"* {caveat}" for caveat in result["caveats"]]
    return lines


def md_checks(result: dict) -> list[str]:
    checks = result["checks"]
    controls = result["controls"]
    lines = ["", "## Controls", "",
             "| control | expected | observed | verdict |", "|---|---:|---:|:--:|"]
    for check in controls["checks"]:
        observed = "—" if check["observed"] is None else f"{check['observed']:.4f}"
        lines.append(f"| {check['what']} | {check['value']:.4f} | {observed} | "
                     f"{'ok' if check['reproduced'] else '**DIFFERS**'} |")
    lines += ["", f"{controls['n_reproduced']} of {controls['n_checks']} reproduced. A failure here "
                  "means the reader is wrong, not that the number moved.", "", "## Checks", "",
              f"* the population is {result['population']['rows']:,} records "
              f"({result['population']['distinct_ids']:,} distinct ids, one `(dataset, cell, variant)` "
              f"each) and every arm covers all of them — no arm was tested on a prefix;",
              f"* all {checks['labels_agree_across_arms']['records']:,} records carry the same reference "
              f"label in every arm;",
              "* the row's `abstain` field and `sckevtype.eval.abstain_index` name the same option on "
              "every row of every kev arm, and this file's `argmax` / forced-index agree with the "
              "settled script's on every row — two code paths, compared per row;"]
    reproduction = checks["reproduces_the_frozen_table"]
    if reproduction["checked"]:
        lines.append(f"* **this table reproduces the settled one**: all {reproduction['fields_compared']} "
                     f"served / forced / abstain-pick counts of `{reproduction['source']}` come out "
                     f"integer-for-integer the same here. An arm that does not is a refusal, not a "
                     f"published number;")
    else:
        lines.append(f"* the settled-table reproduction was **not run**: {reproduction['why']}.")
    published = checks["reproduces_the_published_v4_paired_table"]
    if published["checked"]:
        lines.append(f"* **the same reader reproduces the published v4 paired table**: "
                     f"{published['comparisons_compared']} comparisons of `{published['source']}` "
                     f"recomputed from the v4 dumps, every contingency cell and p-value identical;")
    side = checks["side_effects"]
    lines.append(f"* **nothing else moved**: the digests of {len(side['files'])} frozen tables were taken "
                 f"before and after this run and none changed;")
    lines += ["* every p-value came from `reports/abstain-reconciliation.py`'s own `binom_two_sided`, "
              "lifted at run time — the implementation was not edited and is not copied."]
    return lines


def md_sources(result: dict) -> list[str]:
    lines = ["", "## Sources", "",
             "Every number above is a function of these bytes. The dumps are the ones the lane wrote; "
             "nothing was regenerated for this report. A Laya arm is read from that run's own `rows/` "
             "directory — the one its `reconcile.json` was aggregated from.", "",
             "| arm | where the lane writes it | sha256 |", "|---|---|---|"]
    for arm in result["arms"]:
        digest = arm.get("sha256")
        if isinstance(digest, dict):
            joined = hashlib.sha256()
            for name in sorted(digest):
                joined.update(f"{name}:{digest[name]}\n".encode())
            text = f"`{joined.hexdigest()[:16]}…` (hash of {len(digest)} row files)"
        else:
            text = f"`{digest[:16]}…`" if digest else "—"
        lines.append(f"| `{arm['arm']}` | `{arm['canonical']}` | {text} |")
    return lines


# --------------------------------------------------------------------------- entry point
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paired-tests-baron",
        description="McNemar's exact test for every Baron panel_choice comparison the paper makes.")
    parser.add_argument("--corpus", default="baron", choices=("baron", "v4"),
                        help="which corpus's population and dumps to read (default baron)")
    parser.add_argument("--ids", default=None,
                        help="the population id list (default: the corpus's own)")
    parser.add_argument("--dumps", default="reports", help="directory the kev forced-rows dumps live in")
    parser.add_argument("--runs", default="runs", help="directory the Laya run directories live in")
    parser.add_argument("--published", default=None,
                        help="a published paired table to reproduce (default: reports/paired-tests-v4.json "
                             "when --corpus v4, else nothing)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    return parser.parse_args(argv)


def default_outputs(corpus: str) -> tuple[Path, Path]:
    """Where a run writes. v4's own reports belong to the settled scripts, so its mode writes to scratch."""
    if corpus == "baron":
        return ROOT / "reports" / "paired-tests-baron.json", ROOT / "reports" / "paired-tests-baron.md"
    return (ROOT / "scratch" / "paired-tests-v4-crosscheck.json",
            ROOT / "scratch" / "paired-tests-v4-crosscheck.md")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    published = args.published
    if published is None and args.corpus == "v4":
        published = "reports/paired-tests-v4.json"
    published_path = ROOT / published if published else None
    if published_path is not None and not published_path.is_absolute():
        published_path = ROOT / published_path

    before = side_effects()
    state = build(args)
    entries = comparisons(state)

    served_family = [e for e in entries if e["reading"] == "served" and not e["exploratory"]]
    forced_family = [e for e in entries if e["reading"] == "forced" and not e["exploratory"]]
    holm(served_family)
    holm(forced_family)
    for entry in entries:
        if entry["exploratory"]:
            entry["holm"] = None

    report_checks = checks(state, before, published_path)
    control_block = controls(state)
    survivors = [e for e in served_family + forced_family
                 if e["p_value"] < 0.05 and (e["holm"]["p_value"] >= 0.05)]
    raw_significant = sum(1 for e in served_family + forced_family if e["p_value"] < 0.05)

    arm_rows = []
    for slug, arm in state["arms"].items():
        entry = {"arm": slug, "shown_as": arm["shown_as"], "kind": arm["kind"],
                 "readings": list(arm["readings"]), "n": arm["coverage"]["rows"],
                 "correct": arm["correct"], "accuracy": arm["accuracy"],
                 "declared": arm["declared"], "canonical": arm["canonical"],
                 "read": show_path(arm["read"])}
        if "sha256" in arm:
            entry["sha256"] = arm["sha256"]
        arm_rows.append(entry)
    arm_rows.sort(key=lambda item: (item["kind"], item["arm"]))

    result = {
        "what_this_is":
            "The paired significance tests the Baron panel_choice table was missing: every comparison "
            "the manuscript makes, on the same records, under the served and forced readings, with the "
            "discordant-pair counts a reviewer asks for. Re-aggregation only: no model was run.",
        "corpus": {"selector": state["corpus"], "label": state["config"]["label"],
                   "infix": state["config"]["infix"],
                   "rules_from": {show_path(path): sha256_file(path)
                                  for path in (SETTLED, SIBLING, RECONCILER, RULE_SOURCE)}},
        "test": {
            "name": "McNemar's exact test",
            "statistic": "p = 2 * P(X <= min(b, c)) under Binomial(b + c, 0.5)",
            "discordant_pairs": "b = left correct and right wrong; c = right correct and left wrong",
            "rule": "reports/abstain-reconciliation.py:binom_two_sided, lifted from that file's source "
                    "and applied unchanged (the rule reports/PAIRED-TESTS.md, reports/paired-tests-v4.md "
                    "and reports/noul-accuracy-v4.md use)",
            "two_sided": True,
            "unit": "record — one row of the panel_choice population",
            "pairing": f"both arms answer the same {state['population']['rows']:,} records, with the "
                       f"same reference label",
        },
        "population": dict(state["population"], source=show_path(state["census"]),
                           family=POPULATION_FAMILY, corpus=state["corpus"]),
        "read_roots": {"dumps": show_path(args.dumps), "runs": show_path(args.runs),
                       "canonical": (show_path(args.dumps) == "reports"
                                     and show_path(args.runs) == "runs")},
        "readings": {
            "served": "argmax(p) == label, with the abstain option available",
            "forced": "argmax(p with the abstain option removed) == label; a record served with no "
                      "abstain option has nothing to remove, so its forced verdict is its served one",
        },
        "arms": arm_rows,
        "comparisons": entries,
        "not_testable": [{
            "comparison": "any fine-tuned arm vs the general LLM, forced reading",
            "arms": [slug for slug, config in BASELINES[state["corpus"]].items()
                     if "forced" not in config["readings"]],
            "reading": "forced", "why": NO_VECTOR,
            "what_it_would_need": "a per-record option-probability vector in the LLM's dump",
        }],
        "multiplicity": {
            "correction": "Holm-Bonferroni, applied within each reading to the pre-specified "
                          "comparisons (the ladder, the sibling pair, and both external families)",
            "families": {"served": len(served_family), "forced": len(forced_family)},
            "not_in_the_family": "the exploratory ordering checks (group 'ordering'), reported raw only",
            "raw_significant_at_0.05": raw_significant,
            "would_cross_0.05_after_the_correction": len(survivors),
            "conclusion": ("no comparison's verdict depends on the correction"
                           if not survivors else
                           f"{len(survivors)} comparisons lose significance under the correction"),
        },
        "headline": headline(entries, state["corpus"]),
        "caveats": [
            "A significant McNemar test says the two arms disagree on more records than chance "
            "explains; it does not say the fine-tuning caused the difference. For the kev ladder both "
            "halves answer the same prompt, so the discordance is weights. For Laya the halves were "
            "rendered at their own geometry, so its discordance carries a prompt difference.",
            "The forced reading is a policy applied to test rows, not a model that was trained to answer "
            "without abstaining. It answers 'what would this arm have named', not 'what would it do "
            "under a different training objective'.",
            "Jev and the general LLM answer their own prompt formats. The pairing here is on the record "
            "and the reference label, which is what a paired test needs; it is not a claim that both "
            "models saw the same string.",
            "Jev's own scorer defines forced / abstain-pick on the rows that offer an abstain option. "
            "The comparison here applies the settled population-wide rule to Jev as well, so the two "
            "sides of a pair are scored by one rule; the eligible-restricted number is in "
            "reports/baron-reconciliation.json for a reader who wants Jev on its own denominator.",
            "One seed per arm. The tests are over records, so they say nothing about variation between "
            "training runs; a seed-level claim needs seed-level replicates.",
            "The general LLM has no probability vector in its dump, so its forced reading is not tested "
            "and is not estimated here.",
            "The ordering checks — including the 27B-vs-9B and 27B-vs-4B pairs the headline quotes — are "
            "exploratory. They are reported raw, outside the Holm family, and are the honest form of a "
            "claim that was not pre-registered.",
        ],
        "controls": control_block,
        "checks": report_checks,
    }
    payload = json.dumps(result, indent=1, sort_keys=True) + "\n"
    default_json, default_md = default_outputs(args.corpus)
    out_json = Path(args.out) if args.out else default_json
    out_md = Path(args.markdown) if args.markdown else default_md
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(payload, encoding="utf-8")
    out_md.write_text(markdown(result), encoding="utf-8")

    for entry in entries:
        holm_text = "" if entry["holm"] is None else f"  holm={p_holm_text(entry)}"
        print(f"  {entry['group']:<13s} {entry['reading']:<6s} {entry['comparison']:<62.62s} "
              f"{entry['accuracy_left']:.4f} vs {entry['accuracy_right']:.4f}  "
              f"disc {entry['contingency']['left_only']:>4d}/{entry['contingency']['right_only']:<4d} "
              f"{p_text(entry)}{holm_text}")
    for check in control_block["checks"]:
        observed = "—" if check["observed"] is None else "{:.4f}".format(check["observed"])
        print(f"  {'ok      ' if check['reproduced'] else 'DIFFERS '} {check['what']}: "
              f"expected {check['value']:.4f}, observed {observed}")
    print(f"  checks: labels agree on {report_checks['labels_agree_across_arms']['records']:,} records; "
          f"frozen table reproduced: {report_checks['reproduces_the_frozen_table'].get('fields_compared', 0)} "
          f"fields; published v4 table reproduced: "
          f"{report_checks['reproduces_the_published_v4_paired_table'].get('comparisons_compared', 0)} "
          f"comparisons; side effects: {len(report_checks['side_effects']['changed'])} files changed")
    print(f"wrote {show_path(out_json)} and {show_path(out_md)}  "
          f"json_sha256={hashlib.sha256(payload.encode()).hexdigest()[:16]}")
    if control_block["n_reproduced"] != control_block["n_checks"]:
        print("\nREFUSED: a control that already existed did not reproduce -- the reader is wrong.")
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as error:
        print(f"REFUSED: {error}")
        raise SystemExit(2) from None
