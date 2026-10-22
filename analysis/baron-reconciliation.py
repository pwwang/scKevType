#!/usr/bin/env python3
"""The Baron external-validation suite under the settled readings — every arm, every family.

`reports/abstain-reconciliation-v4.py` (``panel_choice``) and `reports/family-reconciliation-v4.py`
(``evidence_diluted``, ``panel_miss``) are the settled tables for the **v4** census. The Baron suite is a
different population (Baron pancreas, never seen in training) carrying the same four families, and until
this file it had no reconciliation table: every arm's Baron number was a per-dump print rather than a
table the paper could read beside the v4 one. This is that table, and it adds nothing to what the
readings mean:

* **the readings are imported, never re-implemented.** ``cell()`` (served / abstain-pick / forced, and
  the identity that ties them together) comes from ``reports/abstain-reconciliation-v4.py`` by path,
  exactly as ``reports/family-reconciliation-v4.py`` takes it; the Laya helpers come from that same
  sibling; the ``marker_noul`` column comes from ``scripts/noul-accuracy.py``'s own ``read_dump`` /
  ``summarise``; and the suite's question count comes from ``reports/laya-v4-noul-reconcile.py``'s
  ``EXPECTED_QUESTIONS``. Nothing here restates a rule, so a Baron cell and its v4 counterpart cannot
  drift apart;
* **the arm list has one source** — the settled script's own ``KEV_ARMS`` / ``LAYA_ARMS`` — with the
  corpus infix inserted into the dump name, so an arm cannot be added to one corpus and forgotten in the
  other. ``--corpus`` selects the infix, the id list and the expected counts (the pattern
  ``reports/laya-v4-noul-reconcile.py`` established): Baron writes
  ``forced-rows-<slug>-baron[-<family>].jsonl``, v4 writes ``forced-rows-<slug>[-<family>].jsonl``;
* **every denominator is quoted.** On Baron ``panel_choice`` only 650 of 2,600 rows are served with an
  abstain option (v4's census row counts differ again), so ``forced`` and ``abstain-pick`` are reported
  over the population *and* over the rows that actually offered the option — the two are different
  numbers and neither is left implicit;
* **``panel_miss`` is a decline rate, not an accuracy.** The reference label *is* the abstain option on
  that family's rows, so served accuracy and the abstain-pick rate are the same number by construction
  and ``forced`` is 0. The count that shows it is printed per arm and checked across arms, and the
  forced column is never quoted as a capability reading;
* **``marker_noul`` carries its floor.** The accuracy is quoted beside the majority-class floor computed
  from the reference labels (0.8708 on Baron, against v4's 0.8680) and the margin over it, plus the
  says-yes rate against the reference's own yes rate. The question count is the unit and the acceptance;
* **the external baselines are rows of the same table, marked as such.** Jev is read off its own settled
  scorers (``reports/jev-baron/census-score-choice.json``, ``...-noul.json``) and the general LLM off its
  two run summaries. **The LLM has no forced reading**: a text answer has no probability vector, so the
  column is absent rather than invented;
* **the v4 behaviour is provably untouched.** This file writes nothing of v4's; ``--corpus v4`` reads the
  v4 ids and dumps, compares this reader's cells against the frozen v4 JSONs integer for integer, then
  re-runs the two settled aggregators with their outputs redirected to a scratch directory and
  sha256-compares them against the frozen files. A difference is a refusal, never a quieter row.

Controls that must reproduce, asserted rather than assumed (a failure exits non-zero): Jev
``panel_choice`` served 0.6123 and ``marker_noul`` 0.8879; Laya fine-tuned ``marker_noul`` 0.9064 and
Laya base 0.1362; the general LLM ``panel_choice`` 0.4146 / 0.5523 for variants a / b.

Usage::

    # the Baron table (writes reports/baron-reconciliation.{md,json})
    N=reports/baron-reconciliation
    uv run python reports/baron-reconciliation.py

    # the v4 proof: this reader on the v4 dumps against the frozen v4 tables, plus the byte diff
    uv run python reports/baron-reconciliation.py --corpus v4 --verify-bytes

Reads dumps already on disk. Runs no model, submits no job, and re-scores nothing but what is written.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETTLED = ROOT / "analysis" / "abstain-reconciliation-v4.py"
SIBLING = ROOT / "analysis" / "family-reconciliation-v4.py"
NOUL = ROOT / "scripts" / "noul-accuracy.py"
LAYA_NOUL = ROOT / "analysis" / "laya-v4-noul-reconcile.py"

CHOICE_FAMILIES = ("panel_choice", "evidence_diluted", "panel_miss")
FAMILIES = CHOICE_FAMILIES + ("marker_noul",)
#: The family whose reference label is the abstain option on every row: a decline rate, not an accuracy.
DECLINE_FAMILY = "panel_miss"

#: One entry per corpus. ``infix`` is inserted between the arm slug and the family tag -- Baron's ladder
#: writes ``forced-rows-<slug>-baron[-<family>].jsonl`` and v4's writes ``forced-rows-<slug>[-<family>]``
#: (``<cluster-job>`` / ``submit-v4-ladder.sh``). ``laya_runs`` is the run
#: directory that lane reconciles into; ``laya_prefix`` is what its own arm keys are built from.
CORPORA: dict[str, dict] = {
    "baron": {
        "label": "Baron external-validation suite",
        "ids": "reports/census-suite-ids-baron.jsonl",
        "infix": "-baron",
        "laya_runs": "runs/laya-baron-census-{half}{suffix}",
        "jev_choice": "reports/jev-baron/census-score-choice.json",
        "jev_noul": "reports/jev-baron/census-score-noul.json",
        "llm": {"varA": "reports/llm-baseline-baron-census-varA.summary.json",
                "varB": "reports/llm-baseline-baron-census-varB.summary.json"},
        "out_md": "reports/baron-reconciliation.md",
        "out_json": "reports/baron-reconciliation.json",
        "frozen": {},
    },
    "v4": {
        "label": "v4 census (the settled tables' own population)",
        "ids": "reports/census-suite-ids-v4.jsonl",
        "infix": "",
        "laya_runs": "runs/laya-v4-census-{half}{suffix}",
        "jev_choice": "reports/jev-v4/census-score-choice.json",
        "jev_noul": "reports/jev-v4/census-score-noul.json",
        "llm": {"varA": "reports/llm-baseline-v4-census-varA.summary.json",
                "varB": "reports/llm-baseline-v4-census-varB.summary.json"},
        "out_md": None,       # the settled scripts own v4's outputs; this file never writes them
        "out_json": None,
        "frozen": {"panel_choice": "reports/abstain-reconciliation-v4.json",
                   "evidence_diluted": "reports/family-reconciliation-v4-evidence_diluted.json",
                   "panel_miss": "reports/family-reconciliation-v4-panel_miss.json"},
    },
}

#: The controls this table is proved by: numbers that already exist elsewhere, quoted to four decimals.
#: A Baron aggregation that disagrees with one of these is a wrong reader, and the run says so and stops.
CONTROLS: tuple[dict, ...] = (
    {"id": "jev.panel_choice.served", "value": 0.6123,
     "what": "Jev served accuracy on Baron panel_choice",
     "source": "reports/jev-baron/census-score.json (primary.served_accuracy)"},
    {"id": "jev.marker_noul.accuracy", "value": 0.8879,
     "what": "Jev accuracy on the Baron marker_noul questions",
     "source": "reports/jev-baron/census-score-noul.json (primary.accuracy)"},
    {"id": "laya.ft.marker_noul.accuracy", "value": 0.9064,
     "what": "Laya fine-tuned (spec) accuracy on the Baron marker_noul questions",
     "source": "runs/laya-baron-census-spec-marker_noul/reconcile.json"},
    {"id": "laya.base.marker_noul.accuracy", "value": 0.1362,
     "what": "Laya base accuracy on the Baron marker_noul questions",
     "source": "runs/laya-baron-census-base-marker_noul/reconcile.json"},
    {"id": "llm.panel_choice.varA", "value": 0.4146,
     "what": "general LLM (variant a) accuracy on Baron panel_choice",
     "source": "reports/llm-baseline-baron-census-varA.summary.json (per_family)"},
    {"id": "llm.panel_choice.varB", "value": 0.5523,
     "what": "general LLM (variant b) accuracy on Baron panel_choice",
     "source": "reports/llm-baseline-baron-census-varB.summary.json (per_family)"},
)

#: The frozen v4 cells this reader is checked against, as dotted paths into that JSON's arm blocks.
FROZEN_FIELDS = ("n", "served.correct", "abstain_pick.picks", "forced.correct", "hidden_correct")
FROZEN_RESTRICTED_FIELDS = ("identity.forced_correct_minus_served_correct",
                            "identity.declines_times_hidden_correct_share")


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


def load_settled() -> dict:
    """The four files whose rules this table applies, each loaded by path."""
    return {
        "settled": module_from(SETTLED, "abstain_reconciliation_v4",
                               ("cell", "read_jsonl", "sha256_file", "fmt", "argmax", "forced_index",
                                "KEV_ARMS", "LAYA_ARMS")),
        "sibling": module_from(SIBLING, "family_reconciliation_v4",
                               ("laya_served_rows", "identity_without_abstain_labels",
                                "LAYA_HALVES")),
        "noul": module_from(NOUL, "noul_accuracy", ("read_dump", "summarise")),
        "laya_noul": module_from(LAYA_NOUL, "laya_v4_noul_reconcile", ("EXPECTED_QUESTIONS",)),
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


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


# --------------------------------------------------------------------------- the corpus's own paths
def laya_half(reconcile_path: str) -> str:
    """``runs/laya-v4-census-spec/reconcile.json`` -> ``spec`` (the half's own slug).

    A run directory may also carry the family as a trailing tag (``…-census-base-evidence_diluted``),
    and stripping only the lane prefix would return ``base-evidence_diluted`` -- a slug no dump name is
    built from. The tag is stripped by name, against the corpus's own family vocabulary.
    """
    name = Path(reconcile_path).parent.name
    for token in ("laya-v4-census-", "laya-baron-census-"):
        if name.startswith(token):
            half = name[len(token):]
            for family in FAMILIES:
                if half.endswith(f"-{family}"):
                    return half[: -len(family) - 1]
            return half
    raise Refused(f"{reconcile_path}: cannot read the Laya half out of {name!r}")


def choice_dump_path(panel_choice_path: str, corpus: str, family: str) -> str:
    """The family's dump name, from the settled script's own ``panel_choice`` path plus the infix."""
    if not panel_choice_path.endswith(".jsonl"):
        raise Refused(f"{panel_choice_path} is not a .jsonl dump path")
    infix = CORPORA[corpus]["infix"]
    stem = panel_choice_path[: -len(".jsonl")]
    return stem + infix + ("" if family == "panel_choice" else f"-{family}") + ".jsonl"


def laya_run_path(corpus: str, half: str, family: str) -> list[str]:
    """The family's Laya run directories, best first.

    The two corpora disagree on where the ``panel_choice`` half lives: the v4 lane reconciles it straight
    into ``runs/laya-v4-census-<half>/``, the Baron lane into ``runs/laya-baron-census-<half>-
    panel_choice/``. Both spellings are therefore candidates and the one on disk wins, so the rule is a
    search rather than a guess that is right for one corpus and silently absent for the other.
    """
    suffix = "" if family == "panel_choice" else f"-{family}"
    template = CORPORA[corpus]["laya_runs"]
    candidates = [template.format(half=half, suffix=suffix) + "/reconcile.json"]
    if family == "panel_choice":
        candidates.append(template.format(half=half, suffix=f"-{family}") + "/reconcile.json")
    return candidates


def resolve(candidates) -> str:
    """The first candidate that exists under the repo root, else the first (so the absence is reported)."""
    if isinstance(candidates, str):
        candidates = [candidates]
    for candidate in candidates:
        if (ROOT / candidate).is_file():
            return candidate
    return candidates[0]


def laya_noul_dump(corpus: str, label: str) -> str:
    """The composed noul dump name, used only when that lane's own reconcile is absent."""
    return f"reports/forced-rows-{label}{CORPORA[corpus]['infix']}-marker_noul.jsonl"


# --------------------------------------------------------------------------- the readings, per arm
def eligible_block(cell_fn, rows: list[dict]) -> dict | None:
    """The same three readings over the rows that actually offered an abstain option.

    ``cell()`` counts a row with ``abstain: None`` as served-with-no-option and forces it to its own
    top1, so ``forced`` over the population is diluted by exactly those rows. Baron ``panel_choice`` is
    75% such rows, so both denominators are reported and neither is left to be inferred.
    """
    offering = [row for row in rows if row.get("abstain") is not None]
    if not offering:
        return None
    block = cell_fn(offering)
    return {"n": block["n"], "served": block["served"], "abstain_pick": block["abstain_pick"],
            "forced": block["forced"], "hidden_correct": block["hidden_correct"],
            "identity": block["identity"], "delta_points": block["delta_points"]}


def choice_arm_block(cell_fn, rows: list[dict], identity_fn) -> dict:
    """The settled cell, plus the eligible-restricted reading and the decline-vs-accuracy facts.

    ``identity_fn`` is the sibling table's own ``identity_without_abstain_labels``: the settled identity
    re-run on the rows whose label is *not* the abstain option, which is the only place it is a
    statement about the model rather than about how the family was built.
    """
    block = cell_fn(rows)
    offering = eligible_block(cell_fn, rows)
    without_abstain_labels = identity_fn(cell_fn, rows)
    label_is_abstain = sum(1 for row in rows if row["label"] == row.get("abstain"))
    block.update({
        "rows_whose_label_is_the_abstain_option": label_is_abstain,
        "share_whose_label_is_the_abstain_option": label_is_abstain / len(rows) if rows else None,
        "over_the_rows_offering_an_abstain_option": offering,
        "population_where_the_label_is_not_the_abstain_option":
            {key: without_abstain_labels[key]
             for key in ("n", "served", "forced", "hidden_correct")},
        "identity_where_the_label_is_not_the_abstain_option": without_abstain_labels["identity"],
    })
    return block


def noul_arm_block(noul_mod, name: str, path: Path, records: set[str], expected_questions: int) -> dict:
    """One arm's ``marker_noul`` column, by **question**, with its floor and its acceptance.

    The unit is the question: a ``marker_noul`` record asks 4-41 of them, so a dump keyed by record
    would score a fraction of the family and read like a family result. A dump holding fewer rows than
    the suite's question count is a **prefix** of a pass still writing (the 4B rung's top-up is one), so
    its accuracy is kept out of the cell and reported under ``if_scored_as_a_prefix`` instead.
    """
    try:
        rows = noul_mod.read_dump(path)
    except ValueError as error:
        return {"status": "unreadable", "why": str(error), "dump": rel(path),
                "questions": count_lines(path), "questions_expected": expected_questions}
    kept = {key: row for key, row in rows.items() if key[0] in records}
    outside = len(rows) - len(kept)
    if not kept:
        return {"status": "empty", "dump": rel(path), "questions": 0,
                "questions_expected": expected_questions,
                "why": f"{len(rows)} rows in the dump, none on this population's records"}
    entry = noul_mod.summarise(name, path, kept)
    short = expected_questions - entry["questions"]
    block = {
        "dump": rel(path), "questions": entry["questions"], "questions_expected": expected_questions,
        "records": entry["records"], "rows_outside_the_population": outside,
        "reference": {"yes": round(entry["reference_yes_rate"] * entry["questions"]),
                      "yes_rate": entry["reference_yes_rate"]},
        "majority_floor": entry["majority_floor"], "majority_floor_note": entry["majority_floor_note"],
        "degenerate": entry["degenerate"],
        "sha256": entry["sha256"],
        "forced_reading": "none: a yes/no question offers no abstain option to remove, so there is no "
                          "forced reading of this family (and none is invented)",
    }
    if short > 0:
        block.update({
            "status": "short",
            "why": f"the dump holds {entry['questions']} of the {expected_questions} questions: a prefix "
                   f"of a pass still writing, so it is NOT this family's number",
            "if_scored_as_a_prefix": {
                "correct": entry["correct"], "accuracy": entry["accuracy"],
                "answered_yes_rate": entry["answered_yes_rate"],
                "note": "arithmetic over the prefix, printed so nothing is lost -- never quoted as the "
                        "arm's accuracy",
            },
        })
        return block
    block.update({
        "status": "complete",
        "accuracy": entry["accuracy"], "correct": entry["correct"],
        "margin_over_floor": entry["margin_over_floor"],
        "says_yes": {"count": entry["answered_yes"], "rate": entry["answered_yes_rate"]},
        "says_yes_minus_reference_yes_rate": entry["answered_yes_rate"] - entry["reference_yes_rate"],
    })
    return block


# --------------------------------------------------------------------------- the arms
def collect_kev_arms(mods: dict, corpus: str, wanted: set[str], population_n: int,
                     allow_missing: bool, sources: dict, out: dict) -> None:
    settled, sibling = mods["settled"], mods["sibling"]
    for label, shown, pc_path in settled.KEV_ARMS:
        path = choice_dump_path(pc_path, corpus, out["family"])
        p = ROOT / path
        if not p.is_file():
            print(f"  MISSING {path} -- {shown} is not measured here")
            out["arms"][label] = {"shown_as": shown, "kind": "kev", "dump": path, "status": "absent"}
            continue
        meta_path = Path(str(p) + ".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("ids", {}).get("family") not in (None, out["family"]):
                raise Refused(f"{rel(meta_path)} scored family {meta['ids']['family']!r}, not "
                              f"{out['family']!r} -- another family's dump under this name")
            if meta.get("ids", {}).get("n_ids") not in (None, population_n):
                raise Refused(f"{rel(meta_path)} scored {meta['ids']['n_ids']} ids, the population is "
                              f"{population_n} -- not this family's dump")
            written = meta.get("rows", {}).get("written")
            if written is not None and written != population_n:
                raise Refused(f"{rel(meta_path)} wrote {written} rows, the population is "
                              f"{population_n} -- not this family's dump")
            sources[rel(meta_path)] = sha256_file(meta_path)

        rows = [row for row in read_jsonl(p) if row.get("id") in wanted]
        families_in_dump = sorted({row.get("family") for row in rows})
        if families_in_dump != [out["family"]]:
            raise Refused(f"{path} holds rows of {families_in_dump}, not [{out['family']!r}]")
        missing = population_n - len(rows)
        if missing and not allow_missing:
            raise Refused(f"{path} covers {len(rows)} of {population_n} {out['family']} rows "
                          f"({missing} missing); pass --allow-missing to report a partial population")
        block = choice_arm_block(settled.cell, rows, sibling.identity_without_abstain_labels)
        block.update({"shown_as": shown, "kind": "kev", "dump": path, "rows_in_dump": count_lines(p),
                      "missing": missing, "meta": rel(meta_path) if meta_path.is_file() else None,
                      "status": "complete" if not missing else "partial",
                      "abstain_source_in_dump": sorted({str(row.get("abstain_source")) for row in rows})})
        out["arms"][label] = block
        sources[path] = sha256_file(p)
        print_choice_row(shown, block)


def collect_laya_arms(mods: dict, corpus: str, wanted: set[str], population_n: int,
                      allow_missing: bool, sources: dict, out: dict) -> None:
    settled, sibling = mods["settled"], mods["sibling"]
    for label, shown, reconcile in settled.LAYA_ARMS:
        half = laya_half(reconcile)
        path = resolve(laya_run_path(corpus, half, out["family"]))
        p = ROOT / path
        if not p.is_file():
            print(f"  MISSING {path} -- {shown} is not measured here")
            out["arms"][label] = {"shown_as": shown, "kind": "laya", "reconcile": path,
                                  "status": "absent"}
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("population", {}).get("family") != out["family"]:
            raise Refused(f"{path} was reconciled onto {raw.get('population', {}).get('family')!r}, not "
                          f"{out['family']!r}")
        keys = [key for key in raw if key.startswith("census_matched_")]
        if len(keys) != 1:
            raise Refused(f"{path} has {len(keys)} census_matched_* cells; expected exactly one")
        cell = raw[keys[0]]
        n, coverage = cell["n"], raw.get("coverage", {})
        if coverage.get("missing") and not allow_missing:
            raise Refused(f"{path} is missing {coverage['missing']} of the population's rows; pass "
                          f"--allow-missing to report it")
        if n != population_n - int(coverage.get("missing", 0)):
            raise Refused(f"{path}: cell n={n} does not match the population ({population_n} rows, "
                          f"{coverage.get('missing')} missing)")

        rows_dir = p.parent / "rows"
        row_files = sorted(glob.glob(str(rows_dir / "*.jsonl")))
        rows = sibling.laya_served_rows(rows_dir, wanted) if row_files else None
        if row_files:
            if rows is None or len(rows) != n:
                raise Refused(f"{rel(rows_dir)} holds {0 if rows is None else len(rows)} served rows, "
                              f"the reconcile reports n={n} -- the cell cannot be reproduced from bytes")
            reproduced = settled.cell(rows)
            got = (reproduced["served"]["correct"], reproduced["abstain_pick"]["picks"],
                   reproduced["forced"]["correct"])
            want = (cell["served"]["correct"], cell["abstain_pick"]["picks"], cell["forced"]["correct"])
            if got != want:
                raise Refused(f"{path}'s cell {want} is not what its own rows give {got} -- neither "
                              f"number goes in this table")
            for row_file in row_files:
                sources[rel(row_file)] = sha256_file(Path(row_file))
            block = choice_arm_block(settled.cell, rows,
                                     sibling.identity_without_abstain_labels)
            block["cell_reproduced_from_the_lane_rows"] = True
        else:
            block = {
                "n": n,
                "served": {"correct": cell["served"]["correct"], "accuracy": cell["served"]["accuracy"]},
                "abstain_pick": {"picks": cell["abstain_pick"]["picks"],
                                 "rate": cell["abstain_pick"]["rate"],
                                 "rows_offering_an_abstain_option":
                                     (None if coverage.get("rows_without_an_abstain_option") is None
                                      else n - coverage["rows_without_an_abstain_option"]),
                                 "rate_among_rows_offering_it": None},
                "forced": {"correct": cell["forced"]["correct"], "accuracy": cell["forced"]["accuracy"]},
                "hidden_correct": cell.get("questions_gained"),
                "delta_points": 100 * (cell["forced"]["correct"] - cell["served"]["correct"]) / n,
                "rows_whose_serving_offered_no_abstain_option":
                    coverage.get("rows_without_an_abstain_option"),
                "identity": {"forced_correct_minus_served_correct":
                                 cell["forced"]["correct"] - cell["served"]["correct"],
                             "declines_times_hidden_correct_share": cell.get("questions_gained"),
                             "holds": (cell.get("questions_gained") is None
                                       or cell["forced"]["correct"] - cell["served"]["correct"]
                                       == cell.get("questions_gained"))},
                "over_the_rows_offering_an_abstain_option": None,
                "rows_whose_label_is_the_abstain_option": None,
                "population_where_the_label_is_not_the_abstain_option": None,
                "identity_where_the_label_is_not_the_abstain_option": None,
                "cell_reproduced_from_the_lane_rows": False,
                "note": "the reconcile was read without its row dump; nothing here was reproduced "
                        "from bytes, and the column carries no eligible-restricted reading",
            }
        block.update({"shown_as": shown, "kind": "laya", "reconcile": path,
                      "status": "complete" if not coverage.get("missing") else "partial",
                      "reconcile_rows_source": raw.get("rows_source")})
        out["arms"][label] = block
        sources[path] = sha256_file(p)
        print_choice_row(shown, block)


def print_choice_row(shown: str, block: dict) -> None:
    pick = block["abstain_pick"]
    offering = pick.get("rows_offering_an_abstain_option")
    print(f"  {shown:<34} n={block['n']:<5} served {block['served']['accuracy']:.4f} "
          f"abstain {pick['rate']:.4f}/{block['n']}"
          + (f" ({pick['rate_among_rows_offering_it']:.4f} of {offering})" if offering else "")
          + f" forced {block['forced']['accuracy']:.4f}"
          f" (of the offering rows: "
          + ("—" if not block.get("over_the_rows_offering_an_abstain_option")
             else f"{block['over_the_rows_offering_an_abstain_option']['forced']['accuracy']:.4f}")
          + f") identity {'ok' if block['identity']['holds'] else 'DIFFERS'}")


def noul_dump_for(mods: dict, corpus: str, label: str, kind: str) -> tuple[str, str, dict | None]:
    """``(arm key, dump path, the lane's own reconcile or None)`` for one arm's noul dump.

    The KeV dumps are named from the settled arm list plus the corpus infix. **The Laya halves are not**:
    their noul lane keys the base arm ``laya-base`` where the choice lane calls the same model
    ``laya-v4-base``, so composing the name from the choice lane's label asks for
    ``forced-rows-laya-v4-base-baron-marker_noul.jsonl`` and reads a complete dump as absent. That lane
    writes a reconcile that names its own arm key and its own dump, so the name is **read** from it and
    the cell is reproduced from the dump it points at -- a mismatch is then a refusal rather than a row
    that quietly disappears.
    """
    infix = CORPORA[corpus]["infix"]
    if kind != "laya":
        key = f"{label}{infix}"
        return key, f"reports/forced-rows-{key}-marker_noul.jsonl", None
    reconcile = next(arm[2] for arm in mods["settled"].LAYA_ARMS if arm[0] == label)
    half = laya_half(reconcile)
    path = f"{CORPORA[corpus]['laya_runs'].format(half=half, suffix='-marker_noul')}/reconcile.json"
    p = ROOT / path
    if not p.is_file():
        key = f"{label}{infix}"
        return key, f"reports/forced-rows-{key}-marker_noul.jsonl", None
    raw = json.loads(p.read_text(encoding="utf-8"))
    cells = [name for name in raw if name.startswith("noul_questions_")]
    if len(cells) != 1:
        raise Refused(f"{path} has {len(cells)} noul_questions_* cells; expected exactly one")
    key = raw.get("label") or f"{label}{infix}"
    dump = rel((raw.get("dump") or {}).get("path") or f"reports/forced-rows-{key}-marker_noul.jsonl")
    return key, dump, {"reconcile": path, "arm_key": key, "cell": raw[cells[0]],
                       "coverage": raw.get("coverage", {}), "acceptance": raw.get("acceptance", {})}


def collect_noul_arms(mods: dict, corpus: str, records: set[str], expected_questions: int,
                      sources: dict, out: dict) -> None:
    settled = mods["settled"]
    kev_labels = {arm[0] for arm in settled.KEV_ARMS}
    for label, shown, _pc in settled.KEV_ARMS + settled.LAYA_ARMS:
        kind = "kev" if label in kev_labels else "laya"
        name, path, lane = noul_dump_for(mods, corpus, label, kind)
        p = ROOT / path
        if not p.is_file():
            print(f"  MISSING {path} -- {shown} is not measured on marker_noul")
            out["arms"][label] = {"shown_as": shown, "dump": path, "status": "absent"}
            continue
        block = noul_arm_block(mods["noul"], name, p, records, expected_questions)
        block["shown_as"] = shown
        block["kind"] = kind
        if lane is not None:
            # The lane's own reconcile as the cross-check: the dump it names must give the cell it wrote.
            theirs = lane["cell"]
            block["lane_reconcile"] = {
                "reconcile": lane["reconcile"], "arm_key": lane["arm_key"],
                "questions": theirs.get("n"),
                "correct": (theirs.get("served") or {}).get("correct"),
                "majority_floor": theirs.get("majority_floor"),
                "cell_reproduced_from_the_lane_dump":
                    block.get("status") == "complete"
                    and theirs.get("n") == block.get("questions")
                    and (theirs.get("served") or {}).get("correct") == block.get("correct")
                    and abs(float(theirs.get("majority_floor") or 0) - float(block["majority_floor"])) < 1e-12,
            }
            if block["lane_reconcile"]["questions"] is not None and \
                    block["lane_reconcile"]["questions"] != block.get("questions"):
                raise Refused(f"{shown}: the lane's reconcile reports "
                              f"{block['lane_reconcile']['questions']} questions, its dump gives "
                              f"{block.get('questions')} -- the cell cannot be reproduced from bytes")
        if block["status"] == "complete":
            print(f"  {shown:<34} {block['questions']}/{expected_questions} questions "
                  f"accuracy {block['accuracy']:.4f} floor {block['majority_floor']:.4f} "
                  f"margin {block['margin_over_floor']:+.4f} says-yes {block['says_yes']['rate']:.4f} "
                  f"vs reference {block['reference']['yes_rate']:.4f}"
                  + (" (lane reconcile reproduced)" if (block.get("lane_reconcile") or {}).get(
                      "cell_reproduced_from_the_lane_dump") else ""))
        else:
            print(f"  {shown:<34} {block.get('questions', 0)}/{expected_questions} questions  "
                  f"{block['status'].upper()} -- not scored")
        out["arms"][label] = block
        sources[path] = sha256_file(p)


# --------------------------------------------------------------------------- the external baselines
def jev_rows(corpus: str, family: str, sources: dict) -> dict:
    """Jev's cell, read off its own settled scorer (never re-scored here)."""
    config = CORPORA[corpus]
    if family == "marker_noul":
        path = config["jev_noul"]
        p = ROOT / path
        if not p.is_file():
            return {"status": "absent", "why": f"no {path}"}
        raw = json.loads(p.read_text(encoding="utf-8"))
        primary = raw["primary"]
        sources[path] = sha256_file(p)
        return {
            "status": "complete", "kind": "jev", "shown_as": "Jev (commercial System One)",
            "unit": "questions", "questions": primary["questions"], "records": primary["records"],
            "accuracy": primary["accuracy"], "correct": primary["correct"],
            "majority_floor": primary["majority_floor"],
            "margin_over_floor": primary["margin_over_floor"],
            "says_yes": {"count": primary["answered_yes"], "rate": primary["answered_yes_rate"]},
            "reference": {"yes_rate": primary["reference_yes_rate"]},
            "says_yes_minus_reference_yes_rate":
                primary["answered_yes_rate"] - primary["reference_yes_rate"],
            "forced_reading": "none: a yes/no question offers no abstain option to remove",
            "source": path,
            "note": "read off Jev's own settled noul scorer (scripts/jev/score-noul-v4.py's output); "
                    "this table did not re-score the dump",
        }
    path = config["jev_choice"]
    p = ROOT / path
    if not p.is_file():
        return {"status": "absent", "why": f"no {path}"}
    raw = json.loads(p.read_text(encoding="utf-8"))
    block = (raw.get("by_family") or {}).get(family)
    if block is None:
        return {"status": "absent", "why": f"{path} carries no {family!r} block"}
    sources[path] = sha256_file(p)
    n, eligible = block["n"], block["eligible"]
    return {
        "status": "complete", "kind": "jev", "shown_as": "Jev (commercial System One)",
        "n": n, "eligible": eligible,
        "served": {"correct": block["served_correct"], "accuracy": block["served_accuracy"]},
        "abstain_pick": {"picks": block["abstain_picks"],
                         "rate": block["abstain_picks"] / n if n else None,
                         "rows_offering_an_abstain_option": eligible,
                         "rate_among_rows_offering_it": block["abstain_pick_rate"]},
        "forced": {"correct": block["forced_correct"],
                   "accuracy": block["forced_correct"] / n if n else None,
                   "accuracy_among_rows_offering_it": block["forced_accuracy"]},
        "hidden_correct": block["questions_gained"],
        "delta_points": 100 * block["questions_gained"] / n if n else None,
        "always_abstain_floor": block["always_abstain_floor"],
        "identity": {"forced_correct_minus_served_correct": block["questions_gained"],
                     "declines_times_hidden_correct_share": block["questions_gained"],
                     "holds": (block["forced_correct"] - block["served_correct_on_eligible"])
                              == block["questions_gained"]},
        "source": path,
        "note": "read off Jev's own settled census scorer. That scorer defines forced / abstain-pick on "
                "the rows that offer an abstain option, so its own accuracy is the "
                "accuracy_among_rows_offering_it; the population-denominator figure beside it is that "
                "count over the population, which is the denominator this table's arm columns use",
    }


def llm_rows(corpus: str, family: str, sources: dict) -> dict:
    """The general LLM's cell, read off its run summary. A text answer has no probability vector."""
    out: dict = {"kind": "llm", "status": "absent", "variants": {}}
    for name, path in CORPORA[corpus]["llm"].items():
        p = ROOT / path
        if not p.is_file():
            out["variants"][name] = {"status": "absent", "why": f"no {path}"}
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        block = (raw.get("per_family") or {}).get(family)
        if block is None:
            out["variants"][name] = {"status": "absent", "why": f"{path} carries no {family!r} block"}
            continue
        sources[path] = sha256_file(p)
        out["variants"][name] = {
            "status": "complete", "shown_as": f"general LLM, variant {name[-1]}",
            "model": raw.get("model"), "n": block["n"], "accuracy": block["accuracy"],
            "accuracy_strict_rules_only": block.get("accuracy_strict_rules_only"),
            "n_strict_rules": block.get("n_strict_rules"),
            "abstain_pick_rate": block.get("abstain_pick_rate"),
            "abstain_rows": block.get("abstain_rows"),
            "parse_failures": block.get("parse_failures"),
            "source": path,
            "forced_reading": "none: a text answer carries no option-probability vector, so there is "
                              "no forced reading and none is invented",
        }
    measured = {name: v for name, v in out["variants"].items() if v.get("status") == "complete"}
    out["status"] = "complete" if measured else "absent"
    out["shown_as"] = "general LLM (Qwen3.5-9B, zero-shot)"
    return out


# --------------------------------------------------------------------------- control + identity checks
def apply_controls(result: dict) -> dict:
    """Compare the six known numbers against this table; a disagreement is a refusal."""
    def observed(control_id: str):
        family, rest = control_id.split(".", 1)
        if rest == "marker_noul.accuracy":
            block = result["families"]["marker_noul"].get("baselines", {}).get("jev", {})
            return block.get("accuracy"), block.get("status")
        if family.startswith("jev"):
            block = result["families"]["panel_choice"].get("baselines", {}).get("jev", {})
            return (block.get("served") or {}).get("accuracy"), block.get("status")
        if family == "laya":
            half, _ = rest.split(".", 1)
            label = "laya-v4-ft-spec" if half == "ft" else "laya-v4-base"
            block = result["families"]["marker_noul"]["arms"].get(label, {})
            return block.get("accuracy"), block.get("status")
        if family == "llm":
            variant = rest.rsplit(".", 1)[-1]
            block = result["families"]["panel_choice"].get("baselines", {}).get("llm", {})
            cell = (block.get("variants") or {}).get(variant, {})
            return cell.get("accuracy"), cell.get("status")
        raise Refused(f"control {control_id!r} has no reader in this table")

    checks = []
    for control in CONTROLS:
        value, status = observed(control["id"])
        ok = value is not None and abs(round(value, 4) - control["value"]) < 1e-9
        checks.append({**control, "observed": value, "observed_rounded": None if value is None else round(value, 4),
                       "on_disk": status, "reproduced": ok})
    result["controls"] = {
        "what": "numbers that already exist elsewhere in this repo, re-derived through this table's own "
                "pipeline; every one of them must reproduce or the run fails",
        "checks": checks,
        "n_reproduced": sum(1 for c in checks if c["reproduced"]),
        "n_checks": len(checks),
    }
    return result


def identity_report(result: dict) -> dict:
    """The settled identity, per arm per family, and the restriction it needs on two of them."""
    out: dict = {}
    for family in CHOICE_FAMILIES:
        block = result["families"][family]
        rows = []
        for label, arm in block["arms"].items():
            if "served" not in arm:
                continue
            restricted = arm.get("identity_where_the_label_is_not_the_abstain_option")
            sub = arm.get("population_where_the_label_is_not_the_abstain_option", {}) or {}
            rows.append({
                "arm": label, "shown_as": arm["shown_as"], "kind": arm["kind"],
                "forced_correct_minus_served_correct":
                    arm["identity"]["forced_correct_minus_served_correct"],
                "hidden_correct": arm["identity"]["declines_times_hidden_correct_share"],
                "holds": arm["identity"]["holds"],
                "restricted_n": sub.get("n"),
                "restricted_holds": None if restricted is None else restricted["holds"],
                "restricted_empty": sub.get("n") in (None, 0),
            })
        measured = [row for row in rows if not row["restricted_empty"]]
        out[family] = {
            "what": "forced_correct - served_correct == #{declined and forced-correct}, per arm",
            "restriction": "reported twice: over every row of the family, and over the rows whose label "
                           "is not the abstain option (where a decline cannot be the right answer, which "
                           "is the only place the settled identity is a statement about the model)",
            "n_arms": len(rows),
            "n_holding_all_rows": sum(1 for row in rows if row["holds"]),
            "n_arms_restricted": len(measured),
            "n_holding_restricted": sum(1 for row in measured if row["restricted_holds"]),
            "n_with_an_empty_restricted_population": len(rows) - len(measured),
            "per_arm": rows,
        }
    return out


# --------------------------------------------------------------------------- the v4 proof
def dotted(block: dict, path: str):
    value = block
    for token in path.split("."):
        if not isinstance(value, dict) or token not in value:
            return None
        value = value[token]
    return value


def verify_against_frozen(result: dict) -> dict:
    """This reader's v4 cells against the frozen v4 tables, integer for integer."""
    report: dict = {"what": "this table's own pipeline on the v4 dumps, against the frozen v4 tables",
                    "families": {}}
    for family, frozen_path in CORPORA["v4"]["frozen"].items():
        p = ROOT / frozen_path
        if not p.is_file():
            report["families"][family] = {"frozen": frozen_path, "status": "absent"}
            continue
        frozen = json.loads(p.read_text(encoding="utf-8"))
        mine = result["families"][family]["arms"]
        compared = mismatched = missing = 0
        problems: list[str] = []
        for label, arm in frozen["arms"].items():
            if "served" not in arm:
                continue
            if label not in mine or "served" not in mine[label]:
                missing += 1
                problems.append(f"{label}: frozen has a cell, this run has none")
                continue
            fields = list(FROZEN_FIELDS)
            restricted = arm.get("identity_where_the_label_is_not_the_abstain_option")
            if restricted is not None:
                fields += [f"identity_where_the_label_is_not_the_abstain_option.{name}"
                           for name in ("forced_correct_minus_served_correct",
                                        "declines_times_hidden_correct_share")]
            for field in fields:
                want, got = dotted(arm, field), dotted(mine[label], field)
                compared += 1
                if want is None:
                    continue
                if got is None or (isinstance(want, float) and abs(got - want) > 1e-12) or (
                        isinstance(want, int) and not isinstance(want, bool) and got != want):
                    mismatched += 1
                    problems.append(f"{label}.{field}: frozen {want!r}, this run {got!r}")
        report["families"][family] = {
            "frozen": frozen_path, "frozen_sha256": sha256_file(p), "status": "compared",
            "arms_compared": len([label for label, arm in frozen["arms"].items() if "served" in arm]),
            "fields_compared": compared, "mismatched": mismatched, "arms_absent_here": missing,
            "problems": problems[:20], "agrees": mismatched == 0 and missing == 0,
        }
    report["agrees"] = all(block.get("agrees") for block in report["families"].values()
                           if block.get("status") == "compared") and bool(report["families"])
    return report


def leaf_diff(frozen: dict, fresh: dict, prefix: str = "") -> dict:
    """The leaves that differ between two JSON trees, grouped so a difference can be *named*.

    A byte difference is not by itself a finding: the frozen family reports were written before the 27B
    arms' dumps existed, so a fresh run legitimately carries two more arm blocks. What has to be shown is
    that the difference is *confined* to that -- arms only on one side, and the keys that moved -- rather
    than a cell that changed under a name both files carry.
    """
    out: dict = {"only_in_frozen": [], "only_in_fresh": [], "changed": []}

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() - b.keys():
                out["only_in_frozen"].append(f"{path}.{key}" if path else key)
            for key in b.keys() - a.keys():
                out["only_in_fresh"].append(f"{path}.{key}" if path else key)
            for key in a.keys() & b.keys():
                walk(a[key], b[key], f"{path}.{key}" if path else key)
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for index, (left, right) in enumerate(zip(a, b)):
                walk(left, right, f"{path}[{index}]")
        elif a != b:
            out["changed"].append(path)

    walk(frozen, fresh, "")
    # Sorted, because these come from set operations: an unsorted list makes the published table
    # differ between two runs on the same bytes, which reads like drift and is not.
    for key in ("only_in_frozen", "only_in_fresh", "changed"):
        out[key] = sorted(out[key])
    out["arms_only_in_fresh"] = sorted({entry.split(".")[1] for entry in out["only_in_fresh"]
                                        if entry.startswith("arms.") and len(entry.split(".")) > 2})
    out["arms_only_in_frozen"] = sorted({entry.split(".")[1] for entry in out["only_in_frozen"]
                                         if entry.startswith("arms.") and len(entry.split(".")) > 2})
    out["n_leaves"] = {name: len(values) for name, values in out.items()
                       if isinstance(values, list)}
    return out


def verify_bytes(scratch: Path) -> dict:
    """Re-run the two settled v4 aggregators into scratch and sha256 them against the frozen files.

    Nothing under ``reports/`` is written: the settled scripts' output globals are redirected to a
    temporary directory before their ``main()`` is called, and the frozen files are only read.
    """
    report: dict = {"what": "the settled v4 aggregators re-run into a scratch directory, compared "
                            "byte for byte against the frozen reports",
                    "settled_scripts": {}, "checks": []}
    abstain = module_from(SETTLED, "abstain_reconciliation_v4_rerun", ("main",))
    abstain.OUT_MD = scratch / "abstain-reconciliation-v4.md"
    abstain.OUT_JSON = scratch / "abstain-reconciliation-v4.json"
    code = abstain.main([])
    report["settled_scripts"]["abstain-reconciliation-v4.py"] = {
        "sha256": sha256_file(SETTLED), "exit": code,
        "output": "reports/abstain-reconciliation-v4.json"}

    sibling = module_from(SIBLING, "family_reconciliation_v4_rerun", ("main",))
    for family in ("evidence_diluted", "panel_miss"):
        code = sibling.main(["--family", family, "--out-md", str(scratch / f"family-{family}.md"),
                             "--out-json", str(scratch / f"family-{family}.json")])
        report["settled_scripts"][f"family-reconciliation-v4.py --family {family}"] = {
            "sha256": sha256_file(SIBLING), "exit": code,
            "output": f"reports/family-reconciliation-v4-{family}.json"}

    targets = [("reports/abstain-reconciliation-v4.json", scratch / "abstain-reconciliation-v4.json")]
    targets += [(f"reports/family-reconciliation-v4-{family}.json", scratch / f"family-{family}.json")
                for family in ("evidence_diluted", "panel_miss")]
    for frozen_name, fresh in targets:
        frozen = ROOT / frozen_name
        entry = {"frozen": frozen_name,
                 # The scratch directory's name is random per run, so the record names its *shape*:
                 # embedding the real path made the report differ between two runs on the same bytes.
                 "fresh": f"scratch/baron-v4-bytes-*/{fresh.name}", "frozen_exists": frozen.is_file()}
        if frozen.is_file():
            entry["frozen_sha256"] = sha256_file(frozen)
        entry["fresh_sha256"] = sha256_file(fresh) if fresh.is_file() else None
        entry["identical"] = bool(frozen.is_file() and fresh.is_file()
                                  and entry["frozen_sha256"] == entry["fresh_sha256"])
        if not entry["identical"] and frozen.is_file() and fresh.is_file():
            entry["leaf_diff"] = leaf_diff(json.loads(frozen.read_text(encoding="utf-8")),
                                           json.loads(fresh.read_text(encoding="utf-8")))
        report["checks"].append(entry)
    report["all_identical"] = all(entry["identical"] for entry in report["checks"])
    return report


# --------------------------------------------------------------------------- authoring the table
def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def write_md(result: dict, path: Path) -> None:
    corpus = result["corpus"]["selector"]
    families = result["families"]
    controls = result["controls"]
    ident = result["identity"]
    lines = [
        f"# {result['corpus']['label']} — every arm, every family, under the settled readings",
        "",
        "*Machine-readable companion: `reports/baron-reconciliation.json`.*" if corpus == "baron" else "",
        "",
        (f"Population: `{result['population']['ids']}` — **{result['population']['rows']:,} records** "
        f"({', '.join(f'{name} {count:,}' for name, count in result['population']['families'].items())}). "
        f"The readings are `reports/abstain-reconciliation-v4.py`'s own `cell()` (imported, not "
        f"re-implemented): **served** `argmax(p)` over every option the question offered; **abstain-pick** "
        f"the served pick is the abstain option; **forced** `argmax(p)` with the abstain option removed — "
        f"a re-scoring of rows on disk, no model run. `marker_noul` is read by **question** through "
        f"`scripts/noul-accuracy.py`'s own reader."),
        "",
        (f"**Every denominator is quoted.** The abstain-pick and forced columns are reported over the "
        f"population *and* over the rows that actually offered an abstain option; on this corpus those are "
        f"different sets for `panel_choice` "
        f"({families['panel_choice']['checks'].get('rows_offering_an_abstain_option', '?')} of "
        f"{families['panel_choice']['records']:,} rows offer one) and identical for `panel_miss`."),
        "",
    ]

    # ---- the controls
    lines += ["## Controls: numbers that already existed, re-derived here", "",
              "| control | expected | observed | verdict |", "|---|---:|---:|:--:|"]
    for check in controls["checks"]:
        verdict = "ok" if check["reproduced"] else f"**DIFFERS** ({check['on_disk']})"
        lines.append(f"| {check['what']} | {check['value']:.4f} | {fmt(check['observed'])} | {verdict} |")
    lines += ["", (f"{controls['n_reproduced']} of {controls['n_checks']} reproduced. "
                  "A failure here means the reader is wrong, not that the number moved."), ""]

    # ---- panel_choice / evidence_diluted
    for family in ("panel_choice", "evidence_diluted"):
        block = families[family]
        lines += [f"## `{family}` — served, forced, abstain-pick ({block['records']:,} rows)", "",
                  ("| arm | class | n | served | abstain-pick (of the rows offering one) | forced | "
                  "forced − served (pt) | identity all / label≠abstain |"),
                  "|---|---|---:|---:|---:|---:|---:|:--:|"]
        for arm in block["arms"].values():
            if "served" not in arm:
                lines.append(f"| {arm['shown_as']} | {arm['kind']} | — | — | — | — | — | *not measured* |")
                continue
            pick = arm["abstain_pick"]
            offering = pick.get("rows_offering_an_abstain_option")
            cell_txt = fmt(pick["rate"])
            if offering:
                cell_txt += f" ({fmt(pick.get('rate_among_rows_offering_it'))} of {offering:,})"
            restricted = arm.get("identity_where_the_label_is_not_the_abstain_option")
            sub = arm.get("population_where_the_label_is_not_the_abstain_option", {}) or {}
            ident_txt = "ok" if arm["identity"]["holds"] else "**DIFFERS**"
            ident_txt += " / " + ("— (no such row)" if sub.get("n") == 0 else
                                  (("ok" if restricted["holds"] else "**DIFFERS**") if restricted
                                   else "—"))
            lines.append(
                f"| {arm['shown_as']} | {arm['kind']} | {arm['n']:,} | "
                f"**{fmt(arm['served']['accuracy'])}** | {cell_txt} | "
                f"{fmt(arm['forced']['accuracy'])} | {arm['delta_points']:+.2f} | {ident_txt} |")
        for name, base in (block.get("baselines") or {}).items():
            if name == "llm":
                for cell in (base.get("variants") or {}).values():
                    if cell.get("status") != "complete":
                        continue
                    lines.append(f"| {cell['shown_as']} — external baseline | llm | {cell['n']:,} | "
                                 f"**{fmt(cell['accuracy'])}** | "
                                 f"{fmt(cell['abstain_pick_rate'])} (of {cell['abstain_rows']:,} rows "
                                 f"offering one) | "
                                 f"*none — no probability vector* | — | — |")
            else:
                if base.get("status") != "complete":
                    continue
                offering = base["abstain_pick"].get("rows_offering_an_abstain_option")
                cell_txt = f"{fmt(base['abstain_pick']['rate'])}"
                if offering:
                    cell_txt += f" ({fmt(base['abstain_pick']['rate_among_rows_offering_it'])} of {offering:,})"
                lines.append(
                    f"| {base['shown_as']} — external baseline | jev | {base['n']:,} | "
                    f"**{fmt(base['served']['accuracy'])}** | {cell_txt} | "
                    f"{fmt(base['forced']['accuracy'])} | {base['delta_points']:+.2f} | "
                    f"{'ok' if base['identity']['holds'] else '**DIFFERS**'} |")

        eligible = block["checks"].get("rows_offering_an_abstain_option")
        if eligible is not None and eligible != block["records"]:
            lines += ["", (f"**The two forced denominators.** Only **{eligible:,} of {block['records']:,}** "
                          f"rows offer an abstain option, so `forced` over the population is diluted by the "
                          f"rows where forcing changes nothing. Both readings are in the JSON "
                          f"(`forced.accuracy` and `over_the_rows_offering_an_abstain_option.forced.accuracy`) "
                          f"and the second is the one comparable to Jev's own scorer, which defines forced on "
                          f"the offering rows.")]
        elif eligible is not None:
            lines += ["", (f"**The two forced denominators coincide here**: every one of the "
                          f"{block['records']:,} rows offers an abstain option, so `forced` over the "
                          f"population and over the offering rows are the same number. On `panel_choice` "
                          f"they are not, and both are quoted there.")]
        if family == "evidence_diluted":
            label_abstain = block["checks"].get("rows_whose_label_is_the_abstain_option")
            if label_abstain:
                lines += ["", (f"**{label_abstain:,} of {block['records']:,}** rows carry the abstain option as "
                              f"their *label*, which is where the settled identity cannot hold as stated — "
                              f"hence the two identity columns.")]
        lines.append("")

    # ---- panel_miss
    block = families["panel_miss"]
    lines += [f"## `panel_miss` — a **decline rate**, not an accuracy ({block['records']:,} rows)", "",
              ("The marker is missing from the panel, so the reference label **is** the abstain option on "
              "every row: an arm that declines is an arm that is right. Served accuracy and the "
              "abstain-pick rate are therefore the same number for every arm by construction, and the "
              "forced column is 0 because removing the abstain option removes the only correct answer. "
              "**Read the served column as a decline rate and never quote the forced column as a "
              "capability.**"), "",
              "| arm | class | n | decline rate (= served \"accuracy\") | rows whose label is the abstain option | forced |",
              "|---|---|---:|---:|---:|---:|"]
    for arm in block["arms"].values():
        if "served" not in arm:
            lines.append(f"| {arm['shown_as']} | {arm['kind']} | — | — | — | *not measured* |")
            continue
        lines.append(f"| {arm['shown_as']} | {arm['kind']} | {arm['n']:,} | "
                     f"**{fmt(arm['abstain_pick']['rate'])}** | "
                     f"{arm['rows_whose_label_is_the_abstain_option']:,} | "
                     f"{fmt(arm['forced']['accuracy'])} |")
    for name, base in (block.get("baselines") or {}).items():
        if name == "llm":
            for cell in (base.get("variants") or {}).values():
                if cell.get("status") != "complete":
                    continue
                lines.append(f"| {cell['shown_as']} — external baseline | llm | {cell['n']:,} | "
                             f"{fmt(cell['accuracy'])} | — | *none* |")
        elif base.get("status") == "complete":
            lines.append(f"| {base['shown_as']} — external baseline | jev | {base['n']:,} | "
                         f"{fmt(base['abstain_pick']['rate_among_rows_offering_it'])} | — | *none* |")
    checks = block["checks"]
    lines += ["", f"The count that shows it agrees across every measured arm: "
                  f"**{checks.get('rows_whose_label_is_the_abstain_option', '?')} of "
                  f"{block['records']:,}** rows carry the abstain option as their label"
                  + (" — the whole family." if checks.get("label_is_the_abstain_option_on_every_row")
                     else "."), ""]

    # ---- marker_noul
    block = families["marker_noul"]
    lines += [(f"## `marker_noul` — accuracy against its majority-class floor "
              f"({block['questions']:,} questions over {block['records']:,} records)"), "",
              ("The unit is the **question**, not the record: a `marker_noul` record asks 4-41 yes/no "
              "questions, so the acceptance is the suite's own question count. An accuracy here is only "
              "readable beside the floor a rule that answers the majority class every time would score "
              "reading nothing."), "",
              "| arm | class | questions | accuracy | floor | margin over floor | says yes | reference yes |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in block["arms"].values():
        if arm.get("status") != "complete":
            shown = f"{arm.get('questions', 0):,}/{block['questions']:,}"
            state = arm.get("status", "absent")
            lines.append(f"| {arm['shown_as']} | {arm.get('kind', '?')} | {shown} | "
                         + ("*SHORT — not scored (a prefix of a run in flight)*"
                            if state == "short" else "*not measured*")
                         + " | — | — | — | — |")
            continue
        lines.append(f"| {arm['shown_as']} | {arm['kind']} | {arm['questions']:,} | "
                     f"**{fmt(arm['accuracy'])}** | {fmt(arm['majority_floor'])} | "
                     f"{arm['margin_over_floor']:+.4f} | {fmt(arm['says_yes']['rate'])} | "
                     f"{fmt(arm['reference']['yes_rate'])} |")
    for name, base in (block.get("baselines") or {}).items():
        if name == "llm":
            for cell in (base.get("variants") or {}).values():
                if cell.get("status") != "complete":
                    continue
                margin = (None if block["floor"] is None else cell["accuracy"] - block["floor"])
                lines.append(f"| {cell['shown_as']} — external baseline | llm | {cell['n']:,} | "
                             f"{fmt(cell['accuracy'])} | {fmt(block['floor'])} | "
                             f"{'—' if margin is None else f'{margin:+.4f}'} | — | — |")
        elif base.get("status") == "complete":
            lines.append(f"| {base['shown_as']} — external baseline | jev | {base['questions']:,} | "
                         f"{fmt(base['accuracy'])} | {fmt(base['majority_floor'])} | "
                         f"{base['margin_over_floor']:+.4f} | {fmt(base['says_yes']['rate'])} | "
                         f"{fmt(base['reference']['yes_rate'])} |")
    lines += ["", (f"The floor is **{block['floor']:.4f}** on this corpus "
                  f"(the reference is {block['reference_yes_rate']:.4f} *yes*, so always answering *no* "
                  f"reads nothing and scores that) — not v4's 0.8680, and the two must never be "
                  f"interchanged. The LLM rows carry no says-yes column: its summary reports accuracy and "
                  f"the abstain-pick rate, not a probability readout."), ""]

    # ---- identity
    lines += ["## The identity check, and the restriction it needs", ""]
    for family in CHOICE_FAMILIES:
        block = ident[family]
        if block["n_arms_restricted"] == 0:
            lines.append(f"* `{family}`: on **every one of the {block['n_arms']} arms** the restricted "
                         f"population — the rows whose label is not the abstain option — is **empty**, so "
                         f"the identity is not checked there. Counted as empty, never as a pass; the raw "
                         f"column is reported for the record only.")
            continue
        line = (f"* `{family}`: holds over **every row on {block['n_holding_all_rows']} of "
                f"{block['n_arms']} arms**; over the rows whose label is not the abstain option — the "
                f"only place the identity is a statement about the model — the restricted population "
                f"exists on {block['n_arms_restricted']} arms and it holds on "
                f"**{block['n_holding_restricted']} of those**")
        if block["n_with_an_empty_restricted_population"]:
            line += (f"; {block['n_with_an_empty_restricted_population']} arm(s) have no such row at all "
                     f"and are counted as empty, never as a pass")
        lines.append(line + ".")
    lines += ["", ("`panel_miss` is the family where the restriction is the whole population: every row's "
                  "label *is* the abstain option, so the restricted population is empty and the raw "
                  "identity is reported for the record only, not as a check on the model."), ""]

    # ---- the v4 proof, when it was run
    frozen = result.get("frozen_comparison")
    identity_bytes = result.get("v4_byte_identity")
    if frozen or identity_bytes:
        lines += ["", "## That this reader is right: the v4 proof", ""]
        if frozen:
            lines.append("* **The frozen v4 tables, cell for cell.** The same pipeline on the **v4** ids and "
                         "dumps reproduces "
                         + ", ".join(f"`{family}` ({entry.get('arms_compared')} arms, "
                                     f"{entry.get('fields_compared')} fields)"
                                     for family, entry in frozen["families"].items())
                         + f" against `reports/abstain-reconciliation-v4.json`, "
                           f"`reports/family-reconciliation-v4-evidence_diluted.json` and "
                           f"`…-panel_miss.json`: **{'every compared field agrees' if frozen['agrees'] else 'MISMATCHES'}**"
                           f" (mismatches: "
                         + ", ".join(f"{family} {entry.get('mismatched')}"
                                     for family, entry in frozen["families"].items())
                         + ").")
        if identity_bytes:
            same = [entry["frozen"] for entry in identity_bytes["checks"] if entry["identical"]]
            differ = [entry for entry in identity_bytes["checks"] if not entry["identical"]]
            if same:
                lines.append("* **Re-running the settled v4 aggregator writes byte-identical bytes.** "
                             + ", ".join(f"`{name}`" for name in same)
                             + " reproduced from the same dumps, with the script's own output path "
                               "redirected into a scratch directory — nothing under `reports/` was "
                               "written.")
            for entry in differ:
                diff = entry.get("leaf_diff") or {}
                lines.append(
                    f"* `{entry['frozen']}` is **not** byte-identical on a re-run, and the difference is "
                    f"confined to the arms its frozen copy predates: arms only in the fresh run "
                    f"{diff.get('arms_only_in_fresh')}, arms only in the frozen file "
                    f"{diff.get('arms_only_in_frozen')}; leaves changed "
                    f"{diff['n_leaves']['changed']} ({', '.join(diff['changed'][:4])}…). Every cell both "
                    f"files carry agrees — that is the check above — so this is a stale snapshot, not "
                    f"drift, and the frozen file was left exactly as it was.")
        lines += ["", ("Both proofs run read-only: the frozen v4 files and the v4 dumps were hashed before "
                      "and after this run and are unchanged."), ""]

    # ---- status
    lines += ["## What is missing or short", ""]
    status = result["status"]
    noul = families["marker_noul"]
    lines.append(f"* Measured, complete: **{len(status['complete'])} arm-family cells** "
                 f"({len(families['panel_choice']['arms'])} arms × `panel_choice`, "
                 f"`evidence_diluted`, `panel_miss` and `marker_noul`).")
    lines.append(f"* `marker_noul`'s acceptance is a **question** count: "
                 f"{noul['questions']:,} questions over {noul['records']:,} records, and the dumps report "
                 f"{noul.get('questions_seen_in_the_dumps')}. A dump short of it would be listed below as "
                 f"SHORT and **not scored** — a prefix of a pass in flight is not the family's number.")
    if status["short"]:
        for entry in status["short"]:
            lines.append(f"* **SHORT** — `{entry['arm']}` / `{entry['family']}`: {entry['detail']}")
    else:
        lines.append("* Nothing short: every dump covers its family's whole population.")
    if status["absent"]:
        lines.append("* Not measured: " + ", ".join(f"`{entry['arm']}`/`{entry['family']}`"
                                                   for entry in status["absent"]) + ".")
    lines += ["", "## Checks", "",
              ("* Every cell is a re-aggregation of a dump already on disk; no model was run for this "
              "table, and no Slurm job was submitted."),
              ("* The readings are imported from the settled v4 aggregators by path; this file restates "
              "none of them."),
              "* A dash means **not measured** — never a zero and never an estimate.", "",
              "## Sources", "", "| file | sha256 |", "|---|---|"]
    for name, digest in result["sources"].items():
        lines.append(f"| `{name}` | `{digest[:16]}…` |")
    body = [line for line in lines if line is not None]
    # A blank line before every heading, however the sections above were assembled: a heading that
    # follows a paragraph is a heading in some renderers and part of the paragraph in others.
    spaced: list[str] = []
    for line in body:
        if line.startswith("## ") and spaced and spaced[-1].strip():
            spaced.append("")
        spaced.append(line)
    path.write_text("\n".join(spaced) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--corpus", default="baron", choices=sorted(CORPORA))
    parser.add_argument("--ids", default=None, help="the population id list (default: the corpus's)")
    parser.add_argument("--allow-missing", action="store_true",
                        help="report a choice dump that covers part of its population instead of "
                             "refusing (a short marker_noul dump is always reported as SHORT)")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--verify-bytes", action="store_true",
                        help="also re-run the settled v4 aggregators into a scratch directory and "
                             "sha256-compare against the frozen v4 reports")
    parser.add_argument("--no-controls-fatal", action="store_true",
                        help="report a control that did not reproduce instead of failing on it")
    args = parser.parse_args(argv)

    corpus = args.corpus
    config = CORPORA[corpus]
    mods = load_settled()
    settled, _sibling = mods["settled"], mods["sibling"]
    read_jsonl_ = settled.read_jsonl

    ids_path = Path(args.ids) if args.ids else ROOT / config["ids"]
    if not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    if not ids_path.is_file():
        raise Refused(f"no population file at {ids_path}")
    population = read_jsonl_(ids_path)
    by_family: dict[str, list[dict]] = {}
    for row in population:
        by_family.setdefault(row["family"], []).append(row)
    for family, rows in by_family.items():
        ids = {row["id"] for row in rows}
        if len(ids) != len(rows):
            raise Refused(f"{rel(ids_path)}: {len(rows) - len(ids)} repeated ids in "
                          f"{family} -- a repeat would be counted twice")
    expected_questions = mods["laya_noul"].EXPECTED_QUESTIONS[corpus]
    print(f"population: {len(population):,} records from {rel(ids_path)} "
          f"({', '.join(f'{name} {len(rows):,}' for name, rows in sorted(by_family.items()))})")
    print(f"corpus: {config['label']}  infix {config['infix']!r}  marker_noul expects "
          f"{expected_questions:,} questions")

    result: dict = {
        "what_this_is": (f"{config['label']} under the settled served / abstain-pick / forced readings, "
                         f"for every arm with a dump, plus the marker_noul column with its "
                         f"majority-class floor, plus the external baselines as their own rows. "
                         f"Re-aggregation only: no model was run. The readings are "
                         f"reports/abstain-reconciliation-v4.py's own cell(), imported, not "
                         f"re-implemented."),
        "corpus": {"selector": corpus, "label": config["label"], "infix": config["infix"],
                   "confidence": "reports/abstain-reconciliation-v4.py, "
                                 "reports/family-reconciliation-v4.py, scripts/noul-accuracy.py",
                   "rules_from": {rel(path): sha256_file(path)
                                  for path in (SETTLED, SIBLING, NOUL, LAYA_NOUL)}},
        "population": {"ids": rel(ids_path), "rows": len(population),
                       "families": {name: len(rows) for name, rows in sorted(by_family.items())},
                       "sha256": sha256_file(ids_path)},
        "families": {}, "sources": {}, "status": {"complete": [], "short": [], "absent": []},
    }
    sources: dict[str, str] = {rel(ids_path): sha256_file(ids_path)}

    # ------------------------------------------------------------------ the choice families
    for family in CHOICE_FAMILIES:
        rows_of_family = by_family.get(family, [])
        if not rows_of_family:
            raise Refused(f"{rel(ids_path)} carries no {family!r} rows")
        wanted = {row["id"] for row in rows_of_family}
        print(f"\n{family}: {len(rows_of_family):,} records")
        block: dict = {"family": family, "kind": "choice", "records": len(rows_of_family),
                       "arms": {}, "baselines": {}, "checks": {}}
        collect_kev_arms(mods, corpus, wanted, len(rows_of_family), args.allow_missing, sources, block)
        collect_laya_arms(mods, corpus, wanted, len(rows_of_family), args.allow_missing, sources, block)
        measured = {k: v for k, v in block["arms"].items() if "served" in v}
        if measured:
            offerings = {v["abstain_pick"].get("rows_offering_an_abstain_option") for v in measured.values()}
            block["checks"]["rows_offering_an_abstain_option"] = (
                offerings.pop() if len(offerings) == 1 else sorted(o for o in offerings if o is not None))
            labels_abstain = {v.get("rows_whose_label_is_the_abstain_option") for v in measured.values()}
            if None in labels_abstain and len(labels_abstain) > 1:
                labels_abstain.discard(None)   # a lane read without its rows cannot state the count
            if len(labels_abstain) != 1:
                raise Refused(f"the arms disagree on how many {family} rows have the abstain option as "
                              f"their label ({sorted(labels_abstain)}) -- they are not answering the same "
                              f"questions, so this table would compare different things")
            count = labels_abstain.pop()
            block["checks"]["rows_whose_label_is_the_abstain_option"] = count
            block["checks"]["label_is_the_abstain_option_on_every_row"] = count == len(rows_of_family)
            block["checks"]["forced_denominator"] = (
                "the population (forced.accuracy) and the rows offering an abstain option "
                "(over_the_rows_offering_an_abstain_option.forced.accuracy); on this family "
                + ("every row offers one, so the two coincide."
                   if block["checks"]["rows_offering_an_abstain_option"] == len(rows_of_family)
                   else f"only {block['checks']['rows_offering_an_abstain_option']} of "
                        f"{len(rows_of_family)} rows offer one, so they do not."))
            if family == DECLINE_FAMILY:
                block["checks"]["reading"] = (
                    "a DECLINE RATE, not an accuracy: the reference label is the abstain option on every "
                    "row, so the served accuracy and the abstain-pick rate are the same number by "
                    "construction and forced is 0")
        block["baselines"]["jev"] = jev_rows(corpus, family, sources)
        block["baselines"]["llm"] = llm_rows(corpus, family, sources)
        result["families"][family] = block

    # ------------------------------------------------------------------ marker_noul
    print(f"\nmarker_noul: {len(by_family.get('marker_noul', [])):,} records, "
          f"{expected_questions:,} questions expected")
    noul_records = {row["id"] for row in by_family.get("marker_noul", [])}
    if not noul_records:
        raise Refused(f"{rel(ids_path)} carries no marker_noul records")
    noul_block_out: dict = {"family": "marker_noul", "kind": "noul", "unit": "questions",
                            "records": len(noul_records), "questions": expected_questions,
                            "arms": {}, "baselines": {}, "checks": {}}
    collect_noul_arms(mods, corpus, noul_records, expected_questions, sources, noul_block_out)
    measured_noul = {k: v for k, v in noul_block_out["arms"].items() if v.get("status") == "complete"}
    floors = {round(v["majority_floor"], 12) for v in measured_noul.values()}
    reference_rates = {round(v["reference"]["yes_rate"], 12) for v in measured_noul.values()}
    if len(floors) != 1 or len(reference_rates) != 1:
        raise Refused(f"the arms disagree on the reference's own yes rate ({sorted(reference_rates)}) -- "
                      f"they are not answering the same questions, so no floor can be quoted")
    noul_block_out["floor"] = floors.pop()
    noul_block_out["reference_yes_rate"] = reference_rates.pop()
    noul_block_out["questions"] = expected_questions
    noul_block_out["questions_seen_in_the_dumps"] = sorted({v["questions"] for v in measured_noul.values()})
    noul_block_out["baselines"]["jev"] = jev_rows(corpus, "marker_noul", sources)
    noul_block_out["baselines"]["llm"] = llm_rows(corpus, "marker_noul", sources)
    noul_block_out["checks"]["floor_source"] = (
        "computed from the reference labels of the questions the dumps answer; the same value on every "
        "measured arm, which is what makes it one floor rather than one per arm")
    noul_block_out["checks"]["accepted_at"] = (
        f"{expected_questions:,} questions over {len(noul_records):,} records -- a *question* count, the "
        f"acceptance the lane itself uses (reports/laya-v4-noul-reconcile.py's EXPECTED_QUESTIONS)")
    result["families"]["marker_noul"] = noul_block_out

    # ------------------------------------------------------------------ roll-up
    for family, block in result["families"].items():
        for label, arm in block["arms"].items():
            status = arm.get("status")
            if status == "complete":
                result["status"]["complete"].append(f"{label}/{family}")
            elif status == "short":
                result["status"]["short"].append({"arm": label, "family": family, "detail":
                                                  arm.get("why", "the dump is a prefix")})
            elif status in ("absent", "empty", "unreadable", "partial"):
                result["status"]["absent"].append({"arm": label, "family": family, "status": status,
                                                   "why": arm.get("why")})

    if corpus == "baron":
        apply_controls(result)
    else:
        # The six controls are Baron's own published numbers. On another corpus the control is the
        # frozen-table comparison below, and substituting it here would silently compare two populations.
        result["controls"] = {
            "skipped": f"the six controls this table asserts are Baron's own numbers (Jev, Laya and the "
                       f"general LLM on the Baron population); on --corpus {corpus} the control is the "
                       f"frozen v4 table comparison, reported as frozen_comparison",
            "checks": [], "n_reproduced": 0, "n_checks": 0}
    result["identity"] = identity_report(result)
    result["sources"] = dict(sorted(sources.items()))
    result["marker_noul_floor"] = result["families"]["marker_noul"]["floor"]

    if corpus == "v4":
        result["frozen_comparison"] = verify_against_frozen(result)
    if args.verify_bytes:
        # Inside the repo, not the system temp dir: the settled script prints its own output paths with
        # `relative_to(ROOT)`, which raises on a path outside the tree. Nothing under reports/ is written.
        scratch_root = ROOT / "scratch"
        scratch_root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="baron-v4-bytes-", dir=scratch_root))
        try:
            result["v4_byte_identity"] = verify_bytes(scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    # ------------------------------------------------------------------ print + write
    print("\ncontrols:")
    for check in result["controls"]["checks"]:
        seen = "—" if check["observed"] is None else "{:.4f}".format(check["observed"])
        print(f"  {'ok      ' if check['reproduced'] else 'DIFFERS '} {check['what']}: "
              f"expected {check['value']:.4f}, observed {seen}")
    print("\nidentity: " + "; ".join(
        f"{family} {block['n_holding_all_rows']}/{block['n_arms']} all rows, "
        f"{block['n_holding_restricted']}/{block['n_arms_restricted']} label!=abstain"
        for family, block in result["identity"].items()))
    if result["status"]["short"]:
        print("\nSHORT: " + "; ".join(f"{e['arm']}/{e['family']} ({e['detail']})"
                                      for e in result["status"]["short"]))
    if result["status"]["absent"]:
        print("NOT MEASURED: " + ", ".join(f"{e['arm']}/{e['family']}[{e['status']}]"
                                           for e in result["status"]["absent"]))
    if "frozen_comparison" in result:
        check = result["frozen_comparison"]
        print(f"\nv4 frozen comparison: {'AGREES' if check['agrees'] else 'DIFFERS'}")
        for family, entry in check["families"].items():
            print(f"  {family}: {entry.get('arms_compared')} arms, {entry.get('fields_compared')} fields, "
                  f"mismatched {entry.get('mismatched')}, absent here {entry.get('arms_absent_here')}")
            for problem in entry.get("problems", [])[:5]:
                print(f"    {problem}")
    if "v4_byte_identity" in result:
        check = result["v4_byte_identity"]
        print(f"\nv4 byte identity: {'ALL IDENTICAL' if check['all_identical'] else 'DIFFERS'}")
        for entry in check["checks"]:
            print(f"  {'same' if entry['identical'] else 'DIFF':<5} {entry['frozen']} "
                  f"{str(entry.get('frozen_sha256'))[:16]} vs {str(entry.get('fresh_sha256'))[:16]}")
            diff = entry.get("leaf_diff")
            if diff:
                print(f"        leaves: only-frozen {diff['n_leaves']['only_in_frozen']}, "
                      f"only-fresh {diff['n_leaves']['only_in_fresh']}, "
                      f"changed {diff['n_leaves']['changed']}")
                if diff["arms_only_in_fresh"]:
                    print(f"        arms only in the fresh run: {diff['arms_only_in_fresh']}")
                if diff["arms_only_in_frozen"]:
                    print(f"        arms only in the frozen file: {diff['arms_only_in_frozen']}")
                print(f"        keys that moved: {diff['changed'][:6]}")

    # ------------------------------------------------------------------ the verdict, then the write
    ok = True
    checks_run = result["controls"].get("n_checks") or 0
    if not args.no_controls_fatal and checks_run and (
            result["controls"]["n_reproduced"] != result["controls"]["n_checks"]):
        print("\nREFUSED: a control that already existed did not reproduce -- the reader is wrong, so "
              "nothing is written")
        ok = False
    if "frozen_comparison" in result and not result["frozen_comparison"]["agrees"]:
        print("\nREFUSED: this reader does not reproduce the frozen v4 table -- the reader is wrong, so "
              "nothing is written")
        ok = False
    if "v4_byte_identity" in result and not result["v4_byte_identity"]["all_identical"]:
        print("\nNOTE: the settled v4 script's fresh output is not byte-identical to its frozen report "
              "(see the diff above) -- v4 was not modified by this run, which writes only into a "
              "scratch directory")

    out_json = Path(args.out_json) if args.out_json else (ROOT / config["out_json"]
                                                          if config["out_json"] else None)
    out_md = Path(args.out_md) if args.out_md else (ROOT / config["out_md"]
                                                    if config["out_md"] else None)
    if out_json and not out_json.is_absolute():
        out_json = ROOT / out_json
    if out_md and not out_md.is_absolute():
        out_md = ROOT / out_md
    if ok:
        if out_json:
            out_json.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            print(f"\nwrote {rel(out_json)}")
        if out_md:
            write_md(result, out_md)
            print(f"wrote {rel(out_md)}")

    print("\nBARON_RECONCILIATION_DONE" if ok else "\nBARON_RECONCILIATION_CONTROLS_FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as error:
        print(f"REFUSED: {error}")
        raise SystemExit(2) from None
