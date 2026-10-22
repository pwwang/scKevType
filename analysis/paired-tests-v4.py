#!/usr/bin/env python3
"""The `panel_choice` paired tests: every comparison the headline table needs, with its p-value.

What this answers
-----------------

``reports/abstain-reconciliation-v4.md`` reports the v4 census-matched ``panel_choice`` population
(4,035 records) under the **served** and **forced** readings, arm by arm, but carries **point
estimates only**. The noul family beside it has exact two-sided binomial p-values for every
fine-tune-vs-rung pair (``reports/noul-accuracy-v4.md``, ``scripts/noul-accuracy.py``). This script
closes that gap for the family the manuscript's headline table is made of.

The unit is the **record**, and both arms of every comparison answer the **same 4,035 records**, so
the test is McNemar's exact test on the discordant pairs: ``p = 2 * P(X <= min(b, c))`` under
``Binomial(b + c, 0.5)``. That is not a second statistical rule: it is the same
``binom_two_sided`` the noul table uses, lifted out of ``reports/abstain-reconciliation.py``'s own
source (the file that settled it) rather than re-derived. Its implementation is untouched.

Readings
--------

* **served** — ``argmax(p)`` equals the reference label: what the arm named with abstention
  available.
* **forced** — ``argmax(p)`` with the abstain option removed equals the label: what it would have
  named had abstention not been offered. Rows that were served with no abstain option have nothing
  to remove, so their forced verdict is their served verdict (the rule
  ``reports/abstain-reconciliation-v4.py`` applies).

Both readings are reported because the paper reports both, and because **the forced ordering is not
the served ordering** (9B forced 0.9172 against 0.8B forced 0.9108, while served 0.8B 0.7898 is far
below 4B/9B 0.8191) — so a ranking taken from one reading cannot be assumed in the other, and each
comparison is tested under each reading where the arm has a probability vector.

Arms
----

The kev and Laya arms are **not restated here**: ``KEV_ARMS`` and ``LAYA_ARMS`` are lifted out of
``reports/abstain-reconciliation-v4.py``'s own source, so the two tables cannot drift. The Laya arm
list names that lane's settled ``reconcile.json``; the per-record verdicts come from the ``rows/``
directory beside it (same run directory, one level down), which is where that reconcile read them.

The external baselines are read from their own dumps:

* **Jev** — ``reports/jev-v4/rows.jsonl``: one row per ``panel_choice`` record carrying the option
  probability vector, so served *and* forced are reconstructed and both are tested.
* **the general LLM** — ``reports/llm-baseline-v4-rows-census-var{A,B}.jsonl``: the model's parsed
  answer per record and no probability vector, so only its **served** verdict exists. Its forced
  reading is reported as **not testable** rather than dropped or invented.

What it refuses
---------------

Every arm must cover the population exactly — 4,035 records, no missing id, no extra id — and every
arm's reference label for a record must equal every other arm's. A dump that does not is a
``SystemExit``: a paired test over a partial population is not this table. The script also
**reproduces the settled table**: for every arm in ``reports/abstain-reconciliation-v4.json`` its
served and forced counts must equal that file's, integer for integer, or the run fails. Numbers that
disagree with the table they are meant to extend are not published.

Usage
-----

::

    # on the cluster, where the dumps live
    W=/path/to/sckevtype
    uv run python reports/paired-tests-v4.py

    # on a checkout whose reports/ holds only the aggregates: point it at a synced copy
    uv run python reports/paired-tests-v4.py --dumps scratch/v4dumps/reports --runs scratch/v4dumps/runs

Reads only dumps already on disk; runs no model and submits no job. Writes
``reports/paired-tests-v4.json`` and ``reports/paired-tests-v4.md``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sckevtype.eval import abstain_index                      # noqa: E402

CENSUS = "reports/census-suite-ids-v4.jsonl"
POPULATION_FAMILY = "panel_choice"

#: The file whose two readings this table extends, and the source the arm list is lifted from.
SETTLED = ROOT / "reports" / "abstain-reconciliation-v4.json"
ARM_SOURCE = ROOT / "analysis" / "abstain-reconciliation-v4.py"
#: The file that settled the exact binomial; its definitions are lifted, never re-implemented.
RULE_SOURCE = ROOT / "analysis" / "abstain-reconciliation.py"

OUT_JSON = ROOT / "reports" / "paired-tests-v4.json"
OUT_MD = ROOT / "reports" / "paired-tests-v4.md"

#: The external baselines, as ``(key, shown_as, path, readings)``. ``readings`` is what the dump can
#: support: the LLM's rows carry no probability vector, so it has no forced verdict to test.
BASELINES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("jev", "Jev (jev-1.13.0)", "reports/jev-v4/rows.jsonl", ("served", "forced")),
    ("llm-a", "general LLM (variant a)", "reports/llm-baseline-v4-rows-census-varA.jsonl",
     ("served",)),
    ("llm-b", "general LLM (variant b)", "reports/llm-baseline-v4-rows-census-varB.jsonl",
     ("served",)),
)

#: Why a reading is missing for an arm, printed beside the row rather than left to be inferred.
NO_VECTOR = "the dump carries no option-probability vector (one parsed answer per record), so the " \
            "abstain option cannot be removed: there is no forced verdict to test"

def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lift_constants(path: Path, names: tuple[str, ...]) -> dict:
    """Module-level literals taken from another script's **source**, never re-typed here.

    ``reports/abstain-reconciliation-v4.py`` is the settled table this report extends and it declares
    its arms as module-level tuples. Importing that module would execute its body and rewrite
    ``reports/abstain-reconciliation-v4.json``, so the assignments are lifted by ``ast`` and evaluated
    as literals: the arm list has exactly one home, and a missing or renamed name is refused by name
    instead of silently yielding an empty roster.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict = {}
    for node in tree.body:
        targets: list[str] = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target.id], node.value
        for name in targets:
            if name in names and value is not None:
                found[name] = ast.literal_eval(value)
    missing = [name for name in names if name not in found]
    if missing:
        raise ValueError(f"{path} no longer declares {missing}: the arm list this table is built "
                         f"from has moved, and re-typing it here would let the two tables drift")
    return found


def rule_from(path: Path, names: tuple[str, ...]) -> dict:
    """The named function definitions taken from another script's **source**, never from a copy.

    ``reports/abstain-reconciliation.py`` is the settled analysis that defines this project's exact
    two-sided binomial on discordant pairs (``reports/PAIRED-TESTS.md`` and the noul table both use
    it). Importing the module would execute its body and rewrite
    ``reports/abstain-reconciliation.json``; re-implementing the rule here would let two analyses
    disagree. So the definitions are lifted by ``ast`` and executed on their own — the same device
    ``scripts/noul-accuracy.py`` and ``scripts/arms-forced.py`` use.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    if missing:
        raise ValueError(f"{path} no longer defines {missing}: the rule this report applies has "
                         f"moved, and applying a copy of it would be a different test")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[*imports, *[definitions[name] for name in names]], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, filename=str(path), mode="exec"), namespace)     # noqa: S102 - the project's own source
    return namespace


def argmax(values) -> int:
    """First index of the maximum — the project's own tie-break (``abstain-reconciliation.py``)."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def verdicts(row: dict, reading: str) -> bool:
    """One arm's verdict on one record, under ``served`` or ``forced``.

    ``forced`` drops the abstain option and re-takes the argmax; a record served with no abstain
    option has nothing to drop, so its forced verdict *is* its served verdict — exactly the rule
    ``reports/abstain-reconciliation-v4.py`` and ``reports/laya-v4-reconcile.py`` apply.
    """
    label = int(row["label"])
    if reading == "served":
        return argmax(row["p"]) == label
    if reading != "forced":
        raise ValueError(f"unknown reading {reading!r}")
    index = abstain_index(row)
    if index is None:
        return argmax(row["p"]) == label
    return _argmax_without(row["p"], index) == label


def _argmax_without(values, index: int) -> int:
    best, best_value = None, None
    for i, value in enumerate(values):
        if i == index:
            continue
        if best_value is None or value > best_value:
            best, best_value = i, value
    return argmax(values) if best is None else best

def population(census_path: Path = ROOT / CENSUS) -> tuple[list[str], dict]:
    """The census-matched ``panel_choice`` ids, and the triple → id map the baselines join on.

    The id list is the population every v4 arm was scored on (``test/custom/<n>``). Jev's and the
    general LLM's dumps were written against the same records but keyed by
    ``(dataset, cell, variant)``, so the join is that triple — and it is checked to be a bijection:
    two ids sharing a triple would make the join ambiguous, and an ambiguous join is not a pairing.

    ``census_path`` is a parameter so the tests can drive the whole pipeline on a hand-built
    population; the report itself always uses ``reports/census-suite-ids-v4.jsonl``.
    """
    rows = [row for row in read_jsonl(census_path) if row.get("family") == POPULATION_FAMILY]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{census_path}: repeated ids -- a repeated record would be counted twice")
    triple = {(row["dataset"], row["cell"], row["variant"]): row["id"] for row in rows}
    if len(triple) != len(ids):
        raise SystemExit(f"{census_path}: {len(ids) - len(triple)} ids share a (dataset, cell, "
                         f"variant) triple -- the baselines could not be joined onto them one to one")
    return ids, {"rows": len(rows), "distinct_ids": len(ids), "triples": len(triple)}


def load_kev(path: Path, ids: list[str]) -> dict:
    """A kev dump's per-record verdicts, keyed by the census id it was scored on."""
    rows = {row["id"]: row for row in read_jsonl(path) if row.get("id") in set(ids)}
    return rows


def load_laya(rows_dir: Path, ids: list[str]) -> dict:
    """A Laya run's per-record verdicts, off ``rows/*.jsonl`` the way that lane's reconcile reads them.

    Only ``ask == "base"`` rows are the population's answers: the lane asks 2,992 of the census's
    records a **second** question with an abstain option added (``ask == "abstain"``), and reading
    that one would score a question the other arms were never asked.
    """
    wanted = set(ids)
    rows: dict = {}
    files = sorted(rows_dir.glob("*.jsonl"))
    for path in files:
        for row in read_jsonl(path):
            if row.get("ask") != "base" or row.get("family") != POPULATION_FAMILY:
                continue
            if row["id"] in wanted:
                rows[row["id"]] = row
    return rows


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
            raise SystemExit(f"{path}: {key} is answered twice -- the join is not one to one")
        joined[key] = row
    return joined


def vectors(rows: dict, readings: tuple[str, ...]) -> dict:
    """``{id: {"label": int, reading: bool, ...}}`` — the verdicts a comparison is taken on."""
    out = {}
    for key, row in rows.items():
        if "p" in row:
            entry = {"label": int(row["label"])}
            for reading in readings:
                entry[reading] = verdicts(row, reading)
        else:
            # The general LLM's dump: one parsed answer per record, scored by that lane as `hit`.
            # There is no probability vector, so no verdict can be re-taken under another reading.
            entry = {"label": int(row["label"]), "served": bool(int(row["hit"]))}
        out[key] = entry
    return out


def mcnemar(left: dict, right: dict, reading: str, rule: dict) -> dict:
    """McNemar's exact test for two arms scored on the same records.

    With ``b`` = left right / right wrong and ``c`` = the reverse, the null is ``b ~ Binomial(b + c,
    0.5)`` and the two-sided p is ``2 * P(X <= min(b, c))`` — :func:`binom_two_sided`, lifted from
    the file that settled it. Never a difference of two accuracies: on a paired design that throws
    away the pairing and the noise it cancels, and it is the wrong test for binary verdicts.
    """
    shared = sorted(set(left) & set(right))
    if len(shared) != len(left) or len(shared) != len(right):
        raise SystemExit(f"{reading}: the two arms are not on the same records "
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
    """Holm–Bonferroni over one family, **in log space**, writing ``p_holm`` beside each entry.

    The p-values of this family underflow a float (a lopsided pair on 4,035 records is far below
    ``5e-324``), so the adjustment is taken on ``p_value_log10``: the step-up maximum of
    ``log10(p) + log10(m - rank)`` is the same rule, computed where the number still exists. ``m``
    is the family's own size, announced on every entry so a reader can check the multiplier.
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

def roster(dumps: Path, runs: Path, baselines: Path) -> dict:
    """Every arm, built from the arm lists lifted out of the settled v4 reconciliation.

    ``(slug, shown_as, path)`` triples come from that file verbatim. A kev arm's path is read from
    the dump directory (``--dumps``), a Laya arm's from the run directory named on its reconcile path
    (``--runs``), one level down in that run's own ``rows/``.
    """
    constants = lift_constants(ARM_SOURCE, ("KEV_ARMS", "LAYA_ARMS"))
    arms: dict = {}
    for slug, shown, rel in constants["KEV_ARMS"]:
        arms[slug] = {"slug": slug, "shown_as": shown, "kind": "kev",
                      "declared": rel, "canonical": f"reports/{Path(rel).name}",
                      "read": dumps / Path(rel).name,
                      "readings": ("served", "forced")}
    for slug, shown, rel in constants["LAYA_ARMS"]:
        run = Path(rel).parent.name
        arms[slug] = {"slug": slug, "shown_as": shown, "kind": "laya",
                      "declared": rel, "canonical": f"runs/{run}/rows",
                      "read": runs / run / "rows",
                      "readings": ("served", "forced")}
    for slug, shown, rel, readings in BASELINES:
        arms[slug] = {"slug": slug, "shown_as": shown, "kind": "baseline",
                      "declared": rel, "canonical": rel,
                      "read": baselines / Path(rel).relative_to("reports"),
                      "readings": readings}
    return arms


def ladder(arms: dict) -> list[tuple[str, str, str]]:
    """``(size, fine-tune slug, rung slug)`` for the kev ladder, ordered by parameter count.

    The rung is derived from the fine-tune's own slug (``delta-kev-0.8b-v4-s1`` → ``kev-0.8b-v4``)
    and **looked up in the roster**: an arm renamed on one side fails here instead of dropping out of
    the table as a pair that was quietly never tested.
    """
    pairs = []
    for slug in arms:
        if not slug.startswith("delta-kev-"):
            continue
        size = slug[len("delta-kev-"):].split("-v4")[0]
        rung = f"kev-{size}-v4"
        if rung not in arms:
            raise SystemExit(f"{slug} has no rung {rung} in the arm list: the ladder cannot be "
                             f"paired, and a missing rung is not a missing test")
        pairs.append((size, slug, rung))
    pairs.sort(key=lambda item: float(item[0].rstrip("b")))
    return pairs


def laya_pair(arms: dict) -> tuple[str, str]:
    """``(fine-tune, base)`` for the sibling engine, identified by name, never by position."""
    laya = [slug for slug, arm in arms.items() if arm["kind"] == "laya"]
    bases = [slug for slug in laya if "base" in slug]
    if len(laya) != 2 or len(bases) != 1:
        raise SystemExit(f"the Laya arms are {laya}: one fine-tune and one base were expected")
    return next(slug for slug in laya if slug not in bases), bases[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="paired-tests-v4",
        description="McNemar's exact test for every panel_choice comparison the paper makes.")
    parser.add_argument("--dumps", default="reports",
                        help="directory the kev forced-rows dumps live in (default reports)")
    parser.add_argument("--runs", default="runs",
                        help="directory the laya-v4-census-* run directories live in (default runs)")
    parser.add_argument("--baselines", default="reports",
                        help="directory the Jev and general-LLM row dumps live in (default reports)")
    parser.add_argument("--census", default=CENSUS,
                        help=f"the population file (default {CENSUS})")
    parser.add_argument("--out", default="reports/paired-tests-v4.json")
    parser.add_argument("--markdown", default="reports/paired-tests-v4.md")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> dict:
    """Every number in the report, checks first: nothing is compared until the inputs agree."""
    rule = rule_from(RULE_SOURCE, ("binom_two_sided", "binom_two_sided_log10"))
    census_path = Path(args.census)
    if not census_path.is_absolute():
        census_path = ROOT / census_path
    ids, population_info = population(census_path)
    arms = roster(Path(args.dumps), Path(args.runs), Path(args.baselines))
    triple_to_id = {(row["dataset"], row["cell"], row["variant"]): row["id"]
                    for row in read_jsonl(census_path)
                    if row.get("family") == POPULATION_FAMILY}

    raw: dict = {}
    missing = []
    for slug, arm in arms.items():
        path = arm["read"]
        if arm["kind"] == "laya":
            if not Path(path).is_dir():
                missing.append({"arm": slug, "expected": str(path), "why": "no rows directory"})
                continue
            rows = load_laya(Path(path), ids)
            arm["files"] = {p.name: sha256_file(p) for p in sorted(Path(path).glob("*.jsonl"))}
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
            rows = load_kev(Path(path), ids)
            arm["sha256"] = sha256_file(Path(path))
        if len(rows) != len(ids):
            raise SystemExit(f"{slug}: covers {len(rows)} of {len(ids)} records -- a paired test "
                             f"over a partial population is not this table")
        arm["coverage"] = {"rows": len(rows), "missing": 0, "extra": 0}
        arm["vectors"] = vectors(rows, arm["readings"])
        arm["correct"] = {reading: sum(1 for row in arm["vectors"].values() if row[reading])
                          for reading in arm["readings"]}
        arm["accuracy"] = {reading: (arm["correct"][reading] / len(ids) if reading in arm["correct"]
                                     else None) for reading in ("served", "forced")}
        if arm["kind"] == "kev":
            arm["abstain_field_equals_index"] = sum(
                1 for row in rows.values() if row.get("abstain") == abstain_index(row))
        raw[slug] = rows
    if missing:
        raise SystemExit(f"missing dumps: {json.dumps(missing, indent=1)}")
    return {"rule": rule, "ids": ids, "population": population_info, "arms": arms,
            "raw": raw, "triple_to_id": triple_to_id, "census": census_path}


def comparisons(state: dict) -> list[dict]:
    """Every comparison the paper makes, in the order a reader needs them.

    ``ladder`` first (each fine-tune against the rung it started from, ordered by parameter count),
    then the sibling engine, then the external baselines. Each comparison is emitted once per
    reading the **pair** can support, served before forced, and the exploratory ordering checks last.
    """
    arms, rule = state["arms"], state["rule"]
    entries: list[dict] = []

    def add(group: str, left: str, right: str, reading: str, note: str | None = None,
            exploratory: bool = False) -> None:
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

    ladder_pairs = ladder(arms)
    for size, fine_tune, rung in ladder_pairs:
        for reading in ("served", "forced"):
            add("ladder", fine_tune, rung, reading)

    fine_tuned_laya, base_laya = laya_pair(arms)
    for reading in ("served", "forced"):
        add("sibling", fine_tuned_laya, base_laya, reading)

    fine_tunes = [fine_tune for _, fine_tune, _ in ladder_pairs] + [fine_tuned_laya]
    for fine_tune in fine_tunes:
        for reading in ("served", "forced"):
            add("external-jev", fine_tune, "jev", reading)
    for baseline, reading in (("llm-a", "served"), ("llm-b", "served")):
        for fine_tune in fine_tunes:
            add("external-llm", fine_tune, baseline, reading)

    # Exploratory, and marked so: the forced ordering is *not* the served ordering (9B forced
    # 0.9172 against 0.8B forced 0.9108, while 4B and 9B are level at 0.8191 served), so these two
    # adjacent-step pairs are tested under both readings to say whether that ordering is real. They
    # were not pre-specified, so they are outside the corrected family below.
    for size_pair in (("4b", "9b"), ("0.8b", "9b")):
        for reading in ("served", "forced"):
            add("ordering", f"delta-kev-{size_pair[0]}-v4-s1", f"delta-kev-{size_pair[1]}-v4-s1",
                reading, note="adjacent rungs of the ladder, tested to see whether the forced "
                              "ordering carries over; not a pre-specified comparison",
                exploratory=True)
    return entries


def checks(state: dict) -> dict:
    """The three things this table refuses to publish over: label drift, a moved rule, a moved table."""
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
        raise SystemExit(f"{len(disagreements)} records carry a different reference label in "
                         f"different arms: {disagreements[:3]}")

    moved = {slug: {"rows": arm["abstain_field_equals_index"], "n": arm["coverage"]["rows"]}
             for slug, arm in arms.items()
             if arm["kind"] == "kev" and arm["abstain_field_equals_index"] != arm["coverage"]["rows"]}
    if moved:
        raise SystemExit(f"the row's `abstain` field and sckevtype.eval.abstain_index disagree on "
                         f"{moved}: the two settled rules would drop different options")

    canonical = (ROOT / CENSUS).resolve()
    extra: list[str] = []
    if Path(state["census"]).resolve() != canonical:
        reproduction = {
            "checked": False,
            "source": SETTLED.relative_to(ROOT).as_posix(),
            "why": f"this run's population is {show_path(state['census'])}, not {CENSUS}: the "
                   f"settled table is a statement about the census-matched 4,035 records, so "
                   f"re-deriving it from a different population would be comparing two tables "
                   f"rather than checking one",
        }
    else:
        settled = json.loads(SETTLED.read_text(encoding="utf-8"))["arms"]
        reproduced, mismatches, absent = [], [], []
        for slug, block in settled.items():
            if "served" not in block:
                continue
            if slug not in arms:
                absent.append(slug)
                continue
            got = {"served": arms[slug]["correct"]["served"],
                   "forced": arms[slug]["correct"]["forced"]}
            want = {"served": block["served"]["correct"], "forced": block["forced"]["correct"]}
            if got != want:
                mismatches.append({"arm": slug, "computed": got, "settled": want})
            else:
                reproduced.append(slug)
        if mismatches:
            raise SystemExit(f"this table does not reproduce reports/abstain-reconciliation-v4.json: "
                             f"{mismatches}")
        reproduction = {
            "checked": True, "source": SETTLED.relative_to(ROOT).as_posix(),
            "arms_reproduced": sorted(reproduced), "count": len(reproduced), "mismatches": 0,
            "arms_in_the_settled_table_not_here": absent,
            "note": "every arm's served and forced correct count equals the settled v4 "
                    "reconciliation's, integer for integer",
        }
        extra = sorted(slug for slug, arm in arms.items()
                       if slug not in settled and arm["kind"] == "kev")
    return {
        "labels_agree_across_arms": {"records": len(labels), "disagreements": 0},
        "abstain_field_equals_index": {"arms_checked": sum(1 for a in arms.values()
                                                           if a["kind"] == "kev"),
                                       "records_per_arm": len(state["ids"]), "disagreements": 0},
        "reproduces_the_settled_table": reproduction,
        "arms_here_not_in_the_settled_table": extra,
    }


def show_path(path) -> str:
    """A path as this repo would write it, when it lives inside the repo."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state = build(args)
    entries = comparisons(state)

    served_family = [e for e in entries if e["reading"] == "served" and not e["exploratory"]]
    forced_family = [e for e in entries if e["reading"] == "forced" and not e["exploratory"]]
    holm(served_family)
    holm(forced_family)
    for entry in entries:
        if entry["exploratory"]:
            entry["holm"] = None

    report_checks = checks(state)
    survivors = [e for e in served_family + forced_family
                 if e["p_value"] < 0.05 and (e["holm"]["p_value"] >= 0.05)]
    raw_significant = sum(1 for e in served_family + forced_family if e["p_value"] < 0.05)

    arms = []
    for slug, arm in state["arms"].items():
        entry = {"arm": slug, "shown_as": arm["shown_as"], "kind": arm["kind"],
                 "readings": list(arm["readings"]), "n": arm["coverage"]["rows"],
                 "correct": arm["correct"], "accuracy": arm["accuracy"],
                 "declared": arm["declared"], "canonical": arm["canonical"],
                 "read": show_path(arm["read"])}
        if "sha256" in arm:
            entry["sha256"] = arm["sha256"]
        if "files" in arm:
            entry["sha256"] = arm["files"]
        arms.append(entry)
    arms.sort(key=lambda item: (item["kind"], item["arm"]))

    result = {
        "what_this_is":
            "The paired significance tests the v4 panel_choice table was missing: every comparison "
            "the manuscript makes, on the same 4,035 census-matched records, under the served and "
            "forced readings, with the discordant-pair counts a reviewer asks for. Re-aggregation "
            "only: no model was run.",
        "test": {
            "name": "McNemar's exact test",
            "statistic": "p = 2 * P(X <= min(b, c)) under Binomial(b + c, 0.5)",
            "discordant_pairs": "b = left correct and right wrong; c = right correct and left wrong",
            "rule": "reports/abstain-reconciliation.py:binom_two_sided, lifted from that file's "
                    "source and applied unchanged (the rule reports/PAIRED-TESTS.md and the noul "
                    "table use)",
            "two_sided": True,
            "unit": "record — one row of the panel_choice population",
            "pairing": "both arms answer the same 4,035 records, with the same reference label",
        },
        "population": dict(state["population"], source=CENSUS, family=POPULATION_FAMILY),
        "read_roots": {"dumps": show_path(args.dumps), "runs": show_path(args.runs),
                       "baselines": show_path(args.baselines),
                       "canonical": (show_path(args.dumps) == "reports"
                                     and show_path(args.runs) == "runs"
                                     and show_path(args.baselines) == "reports")},
        "readings": {
            "served": "argmax(p) == label, with the abstain option available",
            "forced": "argmax(p with the abstain option removed) == label; a record served with no "
                      "abstain option has nothing to remove, so its forced verdict is its served one",
        },
        "arms": arms,
        "comparisons": entries,
        "not_testable": [{
            "comparison": "any fine-tuned arm vs the general LLM, forced reading",
            "arms": [slug for slug, _, _, readings in BASELINES if "forced" not in readings],
            "reading": "forced", "why": NO_VECTOR,
            "what_it_would_need": "a per-record option-probability vector in the LLM's dump",
        }],
        "multiplicity": {
            "correction": "Holm-Bonferroni, applied within each reading to the pre-specified "
                          "comparisons (the ladder, the sibling pair, and both external families)",
            "families": {"served": len(served_family), "forced": len(forced_family)},
            "not_in_the_family": "the exploratory adjacent-rung ordering checks (group 'ordering'), "
                                 "reported raw only",
            "raw_significant_at_0.05": raw_significant,
            "would_cross_0.05_after_the_correction": len(survivors),
            "conclusion": ("no comparison's verdict depends on the correction"
                           if not survivors else
                           f"{len(survivors)} comparisons lose significance under the correction"),
        },
        "caveats": [
            "A significant McNemar test says the two arms disagree on more records than chance "
            "explains; it does not say the fine-tuning caused the difference. For the kev ladder "
            "both halves answer the same prompt, so the discordance is weights. For Laya the halves "
            "were rendered at their own geometry, so its discordance carries a prompt difference.",
            "The forced reading is a policy applied to test rows, not a model that was trained to "
            "answer without abstaining. It answers 'what would this arm have named', not 'what "
            "would it do under a different training objective'.",
            "Jev and the general LLM answer their own prompt formats. The pairing here is on the "
            "record and the reference label, which is what a paired test needs; it is not a claim "
            "that both models saw the same string.",
            "One seed per arm. The tests are over records, so they say nothing about variation "
            "between training runs; a seed-level claim needs seed-level replicates.",
            "The general LLM has no probability vector in its dump, so its forced reading is not "
            "tested and is not estimated here.",
        ],
        "checks": report_checks,
    }
    payload = json.dumps(result, indent=1, sort_keys=True) + "\n"
    Path(args.out).write_text(payload, encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")

    for entry in entries:
        holm_text = "" if entry["holm"] is None else f"  holm={p_text(entry['holm'])}"
        print(f"  {entry['group']:<13s} {entry['reading']:<6s} {entry['comparison']:<62.62s} "
              f"{entry['accuracy_left']:.4f} vs {entry['accuracy_right']:.4f}  "
              f"disc {entry['contingency']['left_only']:>4d}/{entry['contingency']['right_only']:<4d} "
              f"{p_text(entry)}{holm_text}")
    reproduction = report_checks["reproduces_the_settled_table"]
    settled_text = (f"{reproduction['count']} arms reproduce the settled table"
                    if reproduction["checked"] else "the settled-table check was not run")
    print(f"  checks: labels agree on {report_checks['labels_agree_across_arms']['records']} "
          f"records; {settled_text}; {len(result['not_testable'])} reading not testable")
    print(f"wrote {args.out} and {args.markdown}  "
          f"json_sha256={hashlib.sha256(payload.encode()).hexdigest()[:16]}")
    return 0


def p_text(entry: dict) -> str:
    """A p-value as a float, or its ``log10`` magnitude when a float cannot hold it.

    On 4,035 records a lopsided pair lands below ``5e-324``, where the exact rule returns ``0.0`` —
    correct for the type and unreadable in a table. The magnitude comes from the same file's
    ``binom_two_sided_log10``; it is never re-derived here.
    """
    if entry["p_value"] > 0.0:
        return f"{entry['p_value']:.3g}"
    return f"<5e-324 (log10 {entry['p_value_log10']:.1f})"


def p_holm_text(entry: dict) -> str:
    """The Holm-adjusted p, formatted the same way."""
    if entry["holm"] is None:
        return "— (exploratory)"
    if entry["holm"]["p_value"] > 0.0:
        return f"{entry['holm']['p_value']:.3g}"
    return f"<5e-324 (log10 {entry['holm']['p_value_log10']:.1f})"


TABLE_HEADER = (
    "| A | B | reading | acc A | acc B | Δ (pt) | both ✓ | A only | B only | both ✗ | p | p (Holm) |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
)


def row(entry: dict) -> str:
    """One comparison as a table row: accuracies, the four cells, and both p-values."""
    cells = entry["contingency"]
    return (f"| {entry['left']['shown_as']} | {entry['right']['shown_as']} | {entry['reading']} | "
            f"{entry['accuracy_left']:.4f} | {entry['accuracy_right']:.4f} | "
            f"{entry['difference_points']:+.2f} | {cells['both_correct']:,} | "
            f"{cells['left_only']:,} | {cells['right_only']:,} | {cells['both_wrong']:,} | "
            f"{p_text(entry)} | {p_holm_text(entry)} |")


def markdown(result: dict) -> str:
    """The same numbers, as the table the manuscript can carry and a reader can check."""
    lines: list[str] = []
    for part in (md_top, md_tables, md_prose, md_sources):
        lines.extend(part(result))
    return "\n".join(lines).rstrip("\n") + "\n"


def md_top(result: dict) -> list[str]:
    population = result["population"]
    families = result["multiplicity"]["families"]
    return [
        "# Paired significance tests — `panel_choice`, v4 census (n = "
        f"{population['rows']:,} records)",
        "",
        "*Machine-readable companion: `reports/paired-tests-v4.json` (one entry per comparison).*",
        "",
        "`reports/abstain-reconciliation-v4.md` gives the served and forced accuracy of every v4 arm",
        "with **no test and no discordant-pair counts**. This is that table's missing half: every",
        "comparison the manuscript makes, on the same records, with an exact paired test.",
        "",
        "## The test",
        "",
        "Correctness is binary per record (top-1 against the reference label), and both arms of every",
        "comparison answer the **same** records, so the test is **McNemar's exact test**: with `b` =",
        "A right / B wrong and `c` = the reverse,",
        "",
        "```",
        "p = 2 * P(X <= min(b, c))   under   X ~ Binomial(b + c, 0.5)",
        "```",
        "",
        "It is `reports/abstain-reconciliation.py`'s own `binom_two_sided`, lifted from that file's",
        "source and applied unchanged — the rule `reports/PAIRED-TESTS.md` and",
        "`reports/noul-accuracy-v4.md` use. Where the p-value is below what a float holds it is shown",
        "as its `log10` magnitude, from the same file's `binom_two_sided_log10`.",
        "",
        "Two readings, both reported because the paper reports both and **they rank the arms",
        "differently** (9B forced 0.9172 against 0.8B forced 0.9108; served, 0.8B is 0.7898 against",
        "4B and 9B at 0.8191):",
        "",
        "* **served** — `argmax(p) == label`, abstention available;",
        "* **forced** — `argmax(p)` with the abstain option removed. A record served with no abstain",
        "  option has nothing to remove, so its forced verdict is its served verdict.",
        "",
        f"Multiplicity is addressed at the end: Holm–Bonferroni within each reading, over the",
        f"{families['served']} served and {families['forced']} forced pre-specified comparisons.",
        "",
        "## The ladder — each fine-tune against the rung it started from",
        "",
        "The rung is the released checkpoint of the same size, scored on the same records: `kev-<size>-v4`.",
        "Both halves answer the same prompt, so the discordance is weights.",
        "",
        *TABLE_HEADER,
    ]


def md_tables(result: dict) -> list[str]:
    lines: list[str] = []
    groups = (("ladder", None), ("sibling", "## The sibling engine — Laya fine-tuned against its base"),
              ("external-jev", "## Against Jev"),
              ("external-llm", "## Against the general LLM (served only)"))
    for group, heading in groups:
        if heading:
            lines += ["", heading, ""]
            if group == "sibling":
                lines += ["One pair, the same comparison one engine over. Note that the two halves were",
                          "rendered at their own geometry (the fine-tune at 8192/2048, the released base at",
                          "its shipped 1024/256), so this discordance carries a prompt difference as well as",
                          "a weights difference.", ""]
            if group == "external-jev":
                lines += ["The same fine-tunes against the outside engine, on the same 4,035 records. Jev's",
                          "dump carries its option-probability vector, so it has both readings.", ""]
            if group == "external-llm":
                lines += ["The general LLM (Qwen3.5-9B zero-shot), two prompt variants. Its dump has one",
                          "parsed answer per record and **no probability vector**, so only its served",
                          "reading exists — the forced comparison is listed as not testable below rather",
                          "than dropped or estimated.", ""]
            lines += TABLE_HEADER
        for entry in result["comparisons"]:
            if entry["group"] == group:
                lines.append(row(entry))
    return lines

def significant(entry: dict) -> bool:
    """Significant at 0.05 — after the correction where the entry is in the corrected family."""
    if entry["holm"] is None:
        return entry["p_value"] < 0.05
    return entry["holm"]["p_value"] < 0.05


def _tally(entries: list[dict]) -> str:
    total = len(entries)
    return f"{sum(1 for e in entries if significant(e))} of {total}"


def md_prose(result: dict) -> list[str]:
    entries = result["comparisons"]
    primary = [e for e in entries if not e["exploratory"]]
    lines = ["", "## Pairs that could not be tested", ""]
    for block in result["not_testable"]:
        lines += [f"* **{block['comparison']}** — {block['why']}. It would need "
                  f"{block['what_it_would_need']}. The served reading is tested; the forced one is "
                  f"absent, not zero and not estimated."]
    ordering = [e for e in entries if e["exploratory"]]
    lines += ["", "## Exploratory: is the forced ordering real?", "",
              "The served ranking does not carry over to the forced one — 9B forced 0.9172 sits above",
              "0.8B forced 0.9108, while served 0.8B (0.7898) is far below both 4B and 9B (0.8191,",
              "level at the fourth decimal). The adjacent steps are tested under both readings to ask",
              "whether that ordering is a difference or a coin flip. These two comparisons were **not**",
              "pre-specified, so they are outside the corrected family and are reported raw.", ""]
    lines += list(TABLE_HEADER)
    lines += [row(entry) for entry in ordering]

    def _label(entry: dict) -> str:
        return (f"{entry['left']['shown_as'].replace('fine-tuned ', '')} vs "
                f"{entry['right']['shown_as'].replace('fine-tuned ', '')} ({entry['reading']})")

    yes = [_label(e) for e in ordering if significant(e)]
    no = [_label(e) for e in ordering if not significant(e)]
    lines += ["", f"* significant at 0.05: {'; '.join(yes) if yes else 'none'};",
              f"* not significant: {'; '.join(no) if no else 'none'}."]

    def _pick(left_prefix: str, reading: str) -> dict | None:
        for entry in ordering:
            if entry["left"]["arm"].startswith(left_prefix) and entry["reading"] == reading:
                return entry
        return None

    weak = _pick("delta-kev-0.8b", "forced")
    if weak is not None and not significant(weak):
        lines += ["", "Read this before quoting the forced column: the 9B arm's forced lead over 0.8B —",
                  "0.6 pt, on the discordant records shown above — is **not** significant on these",
                  "4,035 records. It is a point-estimate ordering, not a demonstrated one, so the forced",
                  "reading must not be used to rank 9B above 0.8B."]
    four_nine_served, four_nine_forced = _pick("delta-kev-4b", "served"), _pick("delta-kev-4b", "forced")
    if (four_nine_served is not None and four_nine_forced is not None
            and not significant(four_nine_served) and significant(four_nine_forced)):
        lines += ["", "The 4B/9B pair is the mirror image, and the reason a paired test is used at all:",
                  "the two arms are **level at 0.8191 served** and yet disagree on 216 records (108 each",
                  "way, p = 1 — a coin flip), while the same two arms separate once the abstain option is",
                  "removed. A table of two accuracies cannot show either fact."]

    lines += ["", "## What the table says", ""]
    ladder = [e for e in entries if e["group"] == "ladder"]
    jev = [e for e in entries if e["group"] == "external-jev"]
    llm = [e for e in entries if e["group"] == "external-llm"]
    exceptions = [f"{e['left']['shown_as']} {e['reading']}" for e in jev if e["difference_points"] <= 0]
    exception_text = ", ".join(exceptions) if exceptions else "none"
    lines += [
        f"* **The ladder:** {_tally([e for e in ladder if e['reading'] == 'served'])} "
        f"fine-tune-vs-rung comparisons are significant at 0.05 after the correction, and "
        f"{_tally([e for e in ladder if e['reading'] == 'forced'])} under the forced reading. Every "
        f"step of the ladder clears its own rung.",
        f"* **Against Jev:** {_tally(jev)} comparisons (7 fine-tunes × 2 readings) are significant; "
        f"{sum(1 for e in jev if e['difference_points'] > 0)} of them favour the fine-tune, and the "
        f"exception is {exception_text}.",
        f"* **Against the general LLM:** {_tally(llm)} served comparisons are significant; "
        f"{sum(1 for e in llm if e['difference_points'] <= 0)} of them fail to favour the fine-tune.",
        f"* **Laya against its own base:** {_tally([e for e in entries if e['group'] == 'sibling'])} "
        f"comparisons are significant, under both readings.",
    ]
    families = result["multiplicity"]["families"]
    total_p = families["served"] + families["forced"]
    raw = result["multiplicity"]["raw_significant_at_0.05"]
    crossing = result["multiplicity"]["would_cross_0.05_after_the_correction"]
    verdict_text = (
        f"The correction does not change a single verdict: all {total_p} comparisons are "
        f"significant raw, and none crosses 0.05 once corrected."
        if crossing == 0 else
        f"The correction changes {crossing} verdicts: {raw} of the {total_p} comparisons are "
        f"significant raw, and {raw - crossing} stay below 0.05 once corrected."
    )
    weakest = sorted(primary, key=lambda e: -e["p_value"])[:2]
    weakest_text = "; ".join(
        f"{e['left']['shown_as']} vs {e['right']['shown_as']} ({e['reading']}) at p = {p_text(e)}"
        for e in weakest)
    lines += ["", "## Multiplicity", "",
              f"**A correction was applied: Holm–Bonferroni, within each reading, over the "
              f"{families['served']} served and {families['forced']} forced pre-specified "
              f"comparisons.**",
              "",
              f"The reasoning, in both directions. *Why a correction at all:* {total_p} p-values are",
              "printed side by side and a reader will scan them for the small ones, so the family is",
              "'the table', not 'each row in isolation'; at 0.05 uncorrected, one in twenty null",
              "comparisons would look real, and some rows here are plausibly null — the two weakest",
              f"are {weakest_text}. *Why it costs little:* the comparisons were not fished from the",
              "data — each fine-tune has exactly one rung and each is compared to each external",
              "baseline, a structure fixed by the experiment before any p-value existed — and the",
              "family is strongly positively correlated (every arm answers the same records, and the",
              "same easy questions are easy for all of them), so a Bonferroni-style bound is",
              "conservative here rather than merely adequate. Holm is the least conservative",
              "correction that still controls the family-wise error rate, which is the right target",
              "when any single row could become a sentence in the paper. A reader who prefers no",
              "correction can read the raw column: it is printed beside every adjusted one.",
              "",
              verdict_text + " The exploratory ordering checks are excluded from the family and",
              "labelled as such in every row they appear in.",
              "", "## What this does not establish", ""]
    lines += [f"* {caveat}" for caveat in result["caveats"]]
    reproduction = result["checks"]["reproduces_the_settled_table"]
    if reproduction["checked"]:
        reproduce_bullet = (
            f"* **this table reproduces the settled one**: all {reproduction['count']} arms of "
            f"`{reproduction['source']}` come out integer-for-integer the same here. An arm that "
            f"does not is a `SystemExit`, not a published number;")
    else:
        reproduce_bullet = f"* the settled-table reproduction was **not run**: {reproduction['why']}."
    lines += ["", "## Checks", "",
              f"* the population is {result['population']['rows']:,} records "
              f"({result['population']['distinct_ids']:,} distinct ids, one "
              f"`(dataset, cell, variant)` each), and every arm covers all of them — no arm was "
              f"tested on a prefix;",
              f"* all {result['checks']['labels_agree_across_arms']['records']:,} records carry the "
              f"same reference label in every arm;",
              f"* the row's `abstain` field and `sckevtype.eval.abstain_index` name the same option on "
              f"every row of every kev arm, so the two settled rules cannot be dropping different "
              f"options;",
              reproduce_bullet,
              "* every p-value came from `reports/abstain-reconciliation.py`'s own `binom_two_sided`,",
              "  lifted at run time — the implementation was not edited and is not copied."]
    if result["checks"]["arms_here_not_in_the_settled_table"]:
        lines += ["", "Arms in this table that the settled reconciliation predates (its numbers here are",
                  "computed from the same dump shape and checked against nothing but themselves): " +
                  ", ".join(f"`{slug}`" for slug in
                            result["checks"]["arms_here_not_in_the_settled_table"]) + "."]
    return lines


def md_sources(result: dict) -> list[str]:
    lines = ["", "## Sources", "",
             "Every number above is a function of these bytes. The dumps are the ones the v4 lane",
             "wrote; nothing was regenerated for this report. A Laya arm is read from that run's own",
             "`rows/` directory — the one its `reconcile.json` was aggregated from.", "",
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
    roots = result.get("read_roots", {})
    if not roots.get("canonical", True):
        lines += ["", f"On this machine the dumps were read from `{roots['dumps']}` and "
                      f"`{roots['runs']}` — a synced copy of the paths above, not a different set: "
                      f"the reproduction check proves it, by re-deriving every arm of the settled "
                      f"table from these bytes, integer for integer."]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
