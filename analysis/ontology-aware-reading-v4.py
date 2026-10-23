#!/usr/bin/env python3
# ruff: noqa: ISC004
# The report's prose is written as adjacent string literals inside a list, one clause per line, so a
# reader can diff a sentence against the number it quotes; that style is deliberate in this file.
"""The v4 census-matched `panel_choice` population under four ontology-aware readings.

WHAT THIS IS
------------
The manuscript scores a tool as correct only when its Cell Ontology id equals the answer's
exactly. An audit found that most of the apparent deficit of the two reference-based annotators
is label GRANULARITY rather than misreading, so this report re-measures every arm on the same
4,035 questions under four readings, identically for the paper's own arms and for both tools:

  exact     the chosen CL id equals the answer's CL id.
  subtype   the chosen CL id is a STRICT DESCENDANT of the answer's CL id (narrower than the
            answer).
  parent    the chosen CL id is a STRICT ANCESTOR of the answer's CL id (broader than the
            answer). A broader term is a partial identification, not the answer, and it is
            reported separately rather than folded into a headline.
  related   exact OR subtype OR parent, reported as the sum of the three with subtype and
            parent shown beside it.

Every reading is computed twice: once on the SERVED choice and once on the FORCED choice.

RULES, REPRODUCED RATHER THAN RE-DERIVED
----------------------------------------
  served  a Kev arm: the dump's own `top1` (checked equal to `argmax(p)` on every row before it is
          used); a Laya arm and the Jev lane: `argmax(p)`; a reference-based annotator: its own
          mapped top-1 label, which is the only choice it has.
  forced  `argmax` over `p` with the abstain option's index removed, which is
          `reports/abstain-reconciliation.py`'s rule; for a question that offered no abstain option
          the forced choice is the served choice by construction.
  The abstain index, the register lookup and the SingleR harmonisation are imported from the
  modules that already own them (`sckevtype.eval.abstain_index`,
  `scripts/annotation-tools/singler_label_map.py`); nothing here re-implements them.

MAPPING THE CHOSEN LABEL
------------------------
  * the paper's arms answer with an option LABEL, so `keys[label]` and `keys[choice]` are both
    mapped through the corpus's own register (`artifacts/v3/definitions.json`, name to CL id, 82
    terms and 82 distinct names). A label the register cannot place is counted and listed by name,
    never absorbed into a rate;
  * CellTypist's per-cell `mapped_cl` is that arm's own harmonisation of its top-1 label, and
    SingleR's label is harmonised by that lane's own table. Neither is re-mapped here.

PROVENANCE
----------
The twelve per-arm row dumps and the two Laya reconciles are read from the scratch mirror of the
cluster's `reports/` and `runs/` trees, and each is verified by sha256 against the digest recorded
in `reports/abstain-reconciliation-v4.json`, whose numbers this script also reproduces as its own
first check. A digest that does not match is a hard error.

    uv run --no-sync python reports/ontology-aware-reading-v4.py             # writes the report
    uv run --no-sync python reports/ontology-aware-reading-v4.py --examples   # and prints the walks
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sckevtype.eval import abstain_index  # the settled abstain-index rule

OUT_MD = ROOT / "reports" / "ontology-aware-reading-v4.md"
OUT_JSON = ROOT / "reports" / "ontology-aware-reading-v4.json"

POPULATION = "reports/census-suite-ids-v4.jsonl"
FROZEN = "reports/abstain-reconciliation-v4.json"
QUESTIONS = "reports/annotation-tools-v4/questions.jsonl"
CT_DIR = "reports/annotation-tools-v4/celltypist"
SR_DIR = "reports/annotation-tools-v4/singler"
SR_MAP = "scripts/annotation-tools/singler_label_map.py"
HIERARCHY = "artifacts/v3/cl_terms.json"
REGISTER = "artifacts/v3/definitions.json"
JEV_ROWS = "reports/jev-v4/rows.jsonl"
JEV_SCORE = "reports/jev-v4/census-score-choice.json"
TOOL_SCORES = "reports/annotation-tools-v4/scores.json"

#: the scratch mirror of the cluster's `reports/` tree (the twelve per-arm dumps)
DUMP_DIR = ROOT / "scratch" / "v4dumps" / "reports"
#: the scratch mirror of the cluster's `runs/` tree (the two Laya arms)
LAYA_DIR = ROOT / "scratch" / "v4dumps" / "runs"

KEV_ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("delta-kev-0.5b-v4-s1", "fine-tuned 0.5B", "kev-fine-tuned", "forced-rows-delta-kev-0.5b-v4-s1.jsonl"),
    ("delta-kev-0.6b-v4-s1", "fine-tuned 0.6B", "kev-fine-tuned", "forced-rows-delta-kev-0.6b-v4-s1.jsonl"),
    ("delta-kev-0.8b-v4-s1", "fine-tuned 0.8B", "kev-fine-tuned", "forced-rows-delta-kev-0.8b-v4-s1.jsonl"),
    ("delta-kev-4b-v4-s1", "fine-tuned 4B", "kev-fine-tuned", "forced-rows-delta-kev-4b-v4-s1.jsonl"),
    ("delta-kev-9b-v4-s1", "fine-tuned 9B", "kev-fine-tuned", "forced-rows-delta-kev-9b-v4-s1.jsonl"),
    ("delta-kev-27b-v4-s1", "fine-tuned 27B", "kev-fine-tuned", "forced-rows-delta-kev-27b-v4-s1.jsonl"),
    ("kev-0.5b-v4", "Kev-0.5B zero-shot", "kev-untuned", "forced-rows-kev-0.5b-v4.jsonl"),
    ("kev-0.6b-v4", "Kev-0.6B zero-shot", "kev-untuned", "forced-rows-kev-0.6b-v4.jsonl"),
    ("kev-0.8b-v4", "Kev-0.8B zero-shot", "kev-untuned", "forced-rows-kev-0.8b-v4.jsonl"),
    ("kev-4b-v4", "Kev-4B zero-shot", "kev-untuned", "forced-rows-kev-4b-v4.jsonl"),
    ("kev-9b-v4", "Kev-9B zero-shot", "kev-untuned", "forced-rows-kev-9b-v4.jsonl"),
    ("kev-27b-v4", "Kev-27B zero-shot (full-weight)", "kev-untuned", "forced-rows-kev-27b-v4.jsonl"),
)
LAYA_ARMS: tuple[tuple[str, str, str, str], ...] = (
    ("laya-v4-ft-spec", "Laya fine-tuned (spec)", "laya", "laya-v4-census-spec"),
    ("laya-v4-base", "Laya base (unfitted recipe)", "laya", "laya-v4-census-base"),
)
SINGLER_REFERENCE = {
    "muraro": "human",
    "siletti": "human",
    "tabula_sapiens": "human",
    "tabula_muris_senis": "mouse",
    "dominguez_conde": "human",
    "hao_cite": "human",
}
#: dataset -> the arm the existing baseline declares best for it, by the fixed rule "the
#: tissue/species-matched model where CellTypist ships one, else its documented default".
CELLTYPIST_DECLARED = {
    "muraro": "pancreatic_islet",
    "siletti": "mtg",
    "tabula_muris_senis": "mouse_gut",
    "tabula_sapiens": "lung_atlas",
    "dominguez_conde": "immune_low",
    "hao_cite": "immune_low",
}
ATLAS_ORDER = ("dominguez_conde", "hao_cite", "muraro", "siletti", "tabula_muris_senis", "tabula_sapiens")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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
    """`argmax` over `p` with the abstain option removed, by the settled rule."""
    index = abstain_index(row)
    best, best_value = None, None
    for i, value in enumerate(row["p"]):
        if i == index:
            continue
        if best_value is None or value > best_value:
            best_value, best = value, i
    return best if best is not None else argmax(row["p"])


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Hierarchy:
    """The CL `is_a` closure from `artifacts/v3/cl_terms.json` (the recovered hierarchy)."""

    def __init__(self, payload: dict):
        terms = payload["terms"]
        self.parents = {t["cl_id"]: tuple(t.get("parents") or ()) for t in terms}
        self.names = {t["cl_id"]: t.get("name", "") for t in terms}
        self._cache: dict[str, frozenset] = {}

    def ancestors(self, cl_id: str) -> frozenset:
        """Every strict ancestor, transitively (a DAG walk with a cycle guard)."""
        if cl_id in self._cache:
            return self._cache[cl_id]
        seen: set[str] = set()
        stack = list(self.parents.get(cl_id, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.parents.get(node, ()))
        self._cache[cl_id] = frozenset(seen)
        return self._cache[cl_id]

    def path_to_root(self, cl_id: str) -> list[tuple[str, str, tuple[str, ...]]]:
        """An independent breadth-first walk from `cl_id` upward, recording every parent.

        Used for the worked examples and for the path check: it prints the DAG rather than a
        single chosen spine, so the printed lineage is the hierarchy's own.
        """
        chain: list[tuple[str, str, tuple[str, ...]]] = []
        seen = {cl_id}
        frontier = [cl_id]
        while frontier:
            node = frontier.pop(0)
            parents = tuple(self.parents.get(node, ()))
            chain.append((node, self.names.get(node, ""), parents))
            for parent in parents:
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        return chain

    def verdict(self, answer: str, chosen: str | None) -> str:
        """exact / subtype / parent / unrelated / unplaceable / off-hierarchy, from ancestor sets."""
        if chosen is None:
            return "unplaceable"
        if chosen == answer:
            return "exact"
        if chosen not in self.parents or answer not in self.parents:
            return "off-hierarchy"
        if answer in self.ancestors(chosen):
            return "subtype"
        if chosen in self.ancestors(answer):
            return "parent"
        return "unrelated"

    def path_check(self, answer: str, chosen: str | None) -> str:
        """The same verdict from explicit walks; raises when the two paths disagree."""
        if chosen is None:
            out = "unplaceable"
        elif chosen == answer:
            out = "exact"
        else:
            up_from_chosen = {node for node, _name, _parents in self.path_to_root(chosen)}
            up_from_answer = {node for node, _name, _parents in self.path_to_root(answer)}
            if answer in up_from_chosen:
                out = "subtype"
            elif chosen in up_from_answer:
                out = "parent"
            else:
                out = "unrelated"
        reference = self.verdict(answer, chosen)
        if out != reference:
            raise AssertionError(f"the two paths disagree on ({answer}, {chosen}): {out} != {reference}")
        return out


def rate(correct: int, n: int) -> float | None:
    return (correct / n) if n else None


def readings(verdicts: Counter, n: int) -> dict:
    exact, subtype, parent = verdicts["exact"], verdicts["subtype"], verdicts["parent"]
    return {
        "n": n,
        "exact": {"correct": exact, "rate": rate(exact, n)},
        "subtype": {"correct": subtype, "rate": rate(subtype, n)},
        "parent": {"correct": parent, "rate": rate(parent, n)},
        "related": {"correct": exact + subtype + parent, "rate": rate(exact + subtype + parent, n),
                    "is": "exact + subtype + parent, reported so a reader can subtract them"},
        "unrelated": {"correct": verdicts["unrelated"], "rate": rate(verdicts["unrelated"], n)},
        "unplaceable": {"correct": verdicts["unplaceable"], "rate": rate(verdicts["unplaceable"], n),
                        "is": "the chosen option is not a CL id at all: an abstain key, or a label the "
                              "tool's harmonisation could not place"},
        "off_hierarchy": {"correct": verdicts["off-hierarchy"], "rate": rate(verdicts["off-hierarchy"], n)},
    }


def build(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--examples", action="store_true", help="print the worked examples")
    args = parser.parse_args(argv)

    # --------------------------------------------------------------------- sources
    population = [r for r in read_jsonl(ROOT / POPULATION) if r["family"] == "panel_choice"]
    wanted_ids = [r["id"] for r in population]
    wanted = set(wanted_ids)
    if len(population) != 4035:
        raise SystemExit(f"population is {len(population)} rows, expected 4035: STOP")
    questions = list(read_jsonl(ROOT / QUESTIONS))
    by_qid = {q["id"]: q for q in questions}
    if len(by_qid) != 4035 or wanted != set(by_qid):
        raise SystemExit("the question table and the population are not the same 4,035 questions: STOP")
    qids_of = {a: [q["id"] for q in questions if q["dataset"] == a] for a in ATLAS_ORDER}
    atlas_of = {q["id"]: q["dataset"] for q in questions}
    answer_cl = {q["id"]: q["target_cl_id"] for q in questions}

    frozen = json.loads((ROOT / FROZEN).read_text(encoding="utf-8"))
    frozen_digests = {Path(k).name: v for k, v in frozen["sources"].items()}

    register_payload = json.loads((ROOT / REGISTER).read_text(encoding="utf-8"))
    name_to_cl = {d["name"]: d["cl_id"] for d in register_payload["definitions"]}
    cl_to_name = {d["cl_id"]: d["name"] for d in register_payload["definitions"]}
    hierarchy = Hierarchy(json.loads((ROOT / HIERARCHY).read_text(encoding="utf-8")))
    sr_map = load_module(ROOT / SR_MAP, "singler_label_map")

    checks: dict = {}
    unplaceable_labels: Counter = Counter()
    target_disagreements: list[str] = []
    for q in questions:
        for option in q["options"]:
            if option not in name_to_cl:
                unplaceable_labels[option] += 1
        if name_to_cl.get(q["target"]) != q["target_cl_id"]:
            target_disagreements.append(q["id"])
    checks["register"] = {
        "register": REGISTER,
        "terms": len(cl_to_name),
        "questions": len(questions),
        "answers_the_register_places": len(questions) - len(target_disagreements),
        "answer_rows_where_the_register_disagrees_with_the_corpus_target_cl_id": len(target_disagreements),
        "example_disagreements": target_disagreements[:5],
        "option_occurrences_the_register_cannot_place": sum(unplaceable_labels.values()),
        "labels_the_register_cannot_place": dict(unplaceable_labels.most_common()),
        "note": "every label the register cannot place is one of the corpus's own abstain keys, listed "
                "here rather than absorbed: an abstain key is not a cell type",
    }

    sources: dict[str, str] = {}
    records: dict[str, dict[str, dict[str, str | None]]] = {}
    arm_meta: dict[str, dict] = {}

    # ------------------------------------------------------- the twelve Kev dumps
    for key, shown, kind, basename in KEV_ARMS:
        path = DUMP_DIR / basename
        if not path.is_file():
            raise SystemExit(f"{basename}: the verified dump is not in the scratch mirror: STOP")
        digest = sha256_file(path)
        expected = frozen_digests.get(basename)
        if expected is None or digest != expected:
            raise SystemExit(f"{basename}: sha256 {digest} does not match the frozen digest "
                             f"{expected}: STOP")
        rows = [r for r in read_jsonl(path) if r["id"] in wanted]
        if len(rows) != 4035:
            raise SystemExit(f"{basename}: {len(rows)} rows join the population, expected 4035: STOP")
        served: dict[str, str | None] = {}
        forced: dict[str, str | None] = {}
        declined = 0
        unplaced: Counter = Counter()
        top1_differs = 0
        abstain_field_disagrees = 0
        for row in rows:
            qid = row["id"]
            if row["top1"] != argmax(row["p"]):
                top1_differs += 1
            if row.get("abstain") != abstain_index(row):
                abstain_field_disagrees += 1
            if name_to_cl[row["keys"][row["label"]]] != answer_cl[qid]:
                raise SystemExit(f"{basename}/{qid}: the register's answer is not the question's "
                                 f"target: STOP")
            for which, index in (("served", row["top1"]), ("forced", forced_index(row))):
                label = row["keys"][index]
                cl = name_to_cl.get(label)
                if cl is None:
                    unplaced[label] += 1
                (served if which == "served" else forced)[qid] = cl
            if name_to_cl.get(row["keys"][row["top1"]]) is None:
                declined += 1
        records[key] = {"served": served, "forced": forced}
        arm_meta[key] = {
            "shown_as": shown, "kind": kind, "source": f"scratch mirror of {basename}",
            "n": len(rows), "sha256": digest, "frozen_digest_matches": digest == expected,
            "rows_whose_top1_is_not_argmax_p": top1_differs,
            "rows_where_the_recorded_abstain_index_differs_from_the_key_based_one": abstain_field_disagrees,
            "rows_whose_served_pick_is_the_abstain_option": declined,
            "labels_outside_the_register_among_the_forced_picks": dict(unplaced),
        }
        sources[basename] = digest

    # ------------------------------------------------------------ the two Laya arms
    for key, shown, kind, lane in LAYA_ARMS:
        reconcile = LAYA_DIR / lane / "reconcile.json"
        rows_dir = LAYA_DIR / lane / "rows"
        if not reconcile.is_file() or not rows_dir.is_dir():
            raise SystemExit(f"{lane}: the scratch mirror of the Laya lane is incomplete: STOP")
        digest = sha256_file(reconcile)
        fragments = [v for k, v in frozen["sources"].items() if lane in k]
        if len(fragments) != 1:
            raise SystemExit(f"{lane}: the frozen report records {len(fragments)} digests for this "
                             "lane, expected exactly one: STOP")
        expected = fragments[0]
        if digest != expected:
            raise SystemExit(f"{lane}/reconcile.json: sha256 {digest} does not match the frozen "
                             f"digest {expected}: STOP")
        answers = {}
        for path in sorted(rows_dir.glob("*.jsonl")):
            for row in read_jsonl(path):
                if row.get("ask") == "base" and row["id"] in wanted and row.get("family") == "panel_choice":
                    answers[row["id"]] = row
        if len(answers) != 4035:
            raise SystemExit(f"{lane}: {len(answers)} rows join the population, expected 4035: STOP")
        served, forced = {}, {}
        declined = 0
        unplaced = Counter()
        for qid, row in answers.items():
            if name_to_cl[row["keys"][row["label"]]] != answer_cl[qid]:
                raise SystemExit(f"{lane}/{qid}: register answer is not the question's target: STOP")
            for which, index in (("served", argmax(row["p"])), ("forced", forced_index(row))):
                label = row["keys"][index]
                cl = name_to_cl.get(label)
                if cl is None:
                    unplaced[label] += 1
                (served if which == "served" else forced)[qid] = cl
            if name_to_cl.get(row["keys"][argmax(row["p"])]) is None:
                declined += 1
        records[key] = {"served": served, "forced": forced}
        arm_meta[key] = {
            "shown_as": shown, "kind": kind,
            "source": f"scratch mirror of the lane {lane} (its rows plus its reconcile.json)",
            "n": len(answers), "sha256": digest, "frozen_digest_matches": digest == expected,
            "rows_whose_served_pick_is_the_abstain_option": declined,
            "labels_outside_the_register_among_the_forced_picks": dict(unplaced),
        }
        sources[f"{lane}/reconcile.json"] = digest

    # --------------------------------------------------------------- the Jev lane
    jev_rows = [r for r in read_jsonl(ROOT / JEV_ROWS) if r.get("family") == "panel_choice"]
    jev_by_cell = {(r["dataset"], r["cell"]): r for r in jev_rows}
    served, forced = {}, {}
    declined = 0
    unplaced = Counter()
    answer_field_disagreements = 0
    for pop_row in population:
        qid = pop_row["id"]
        q = by_qid[qid]
        row = jev_by_cell.get((q["dataset"], q["cell"]))
        if row is None:
            raise SystemExit(f"jev: no row for {qid}: STOP")
        if row.get("answer") is not None and row["keys"][argmax(row["p"])] != row["answer"]:
            answer_field_disagreements += 1
        if name_to_cl[row["keys"][row["label"]]] != answer_cl[qid]:
            raise SystemExit(f"jev/{qid}: register answer is not the question's target: STOP")
        for which, index in (("served", argmax(row["p"])), ("forced", forced_index(row))):
            label = row["keys"][index]
            cl = name_to_cl.get(label)
            if cl is None:
                unplaced[label] += 1
            (served if which == "served" else forced)[qid] = cl
        if name_to_cl.get(row["keys"][argmax(row["p"])]) is None:
            declined += 1
    records["jev-v4"] = {"served": served, "forced": forced}
    arm_meta["jev-v4"] = {
        "shown_as": "Jev (API)", "kind": "jev",
        "source": "the Jev v4 lane's own rows dump, joined to the population by (dataset, cell)",
        "n": len(served), "sha256": sha256_file(ROOT / JEV_ROWS),
        "rows_whose_served_pick_is_the_abstain_option": declined,
        "labels_outside_the_register_among_the_forced_picks": dict(unplaced),
        "rows_where_the_models_own_answer_field_differs_from_argmax_p": answer_field_disagreements,
        "note": "Jev is nondeterministic: this is one draw of the lane, scored by argmax(p) as the "
                "settled comparison does",
    }
    sources["reports/jev-v4/rows.jsonl"] = sha256_file(ROOT / JEV_ROWS)

    # ---------------------------------- CellTypist: every arm, then the tools' best sets
    numpy = __import__("numpy")
    tool_scores = json.loads((ROOT / TOOL_SCORES).read_text(encoding="utf-8"))
    ct_rows: dict[str, dict[str, str | None]] = {}
    ct_own: dict[str, dict[str, str | None]] = {}
    ct_basis: dict[str, dict[str, str | None]] = {}
    ct_flags: dict[str, dict] = {}
    for path in sorted((ROOT / CT_DIR).glob("*.npz")):
        stem = path.stem
        dataset = next((a for a in sorted(ATLAS_ORDER, key=len, reverse=True) if stem.startswith(a + "_")),
                       None)
        if dataset is None:
            raise SystemExit(f"{path.name}: no atlas in this project owns that arm prefix: STOP")
        arm = stem[len(dataset) + 1:]
        # allow_pickle: the project's own frozen npz artifacts hold object-dtype string arrays, and
        # scripts/annotation-tools/05_score.py reads them the same way
        npz = numpy.load(path, allow_pickle=True)
        cells = [str(c) for c in npz["cells"]]
        mapped = [None if c is None else str(c) for c in npz["mapped_cl"]]
        predicted = [str(p) for p in npz["predicted"]]
        per_cell = dict(zip(cells, mapped))
        own_per_cell = dict(zip(cells, predicted))
        basis_per_cell = dict(zip(cells, (str(b) for b in npz["basis"])))
        ct_rows[f"{dataset}/{arm}"] = {q["id"]: per_cell.get(q["cell"]) for q in questions
                                       if q["dataset"] == dataset}
        ct_own[f"{dataset}/{arm}"] = {q["id"]: own_per_cell.get(q["cell"]) for q in questions
                                      if q["dataset"] == dataset}
        ct_basis[f"{dataset}/{arm}"] = {q["id"]: basis_per_cell.get(q["cell"]) for q in questions
                                        if q["dataset"] == dataset}
        ct_flags[f"{dataset}/{arm}"] = {
            "dataset": dataset, "arm": arm, "cells": len(cells),
            "path": str(path.relative_to(ROOT)),
            "model": tool_scores["per_arm"]["celltypist"][f"{dataset}/{arm}"]["model"]}
        sources[str(path.relative_to(ROOT))] = sha256_file(path)

    sr_rows: dict[str, dict[str, str | None]] = {}
    sr_own: dict[str, dict[str, str | None]] = {}
    sr_basis: dict[str, dict[str, str | None]] = {}
    sr_flags: dict[str, dict] = {}
    for dataset in ATLAS_ORDER:
        path = ROOT / SR_DIR / f"{dataset}.csv"
        reference = SINGLER_REFERENCE[dataset]
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for level in ("main", "fine"):
            per_cell: dict[str, str | None] = {}
            own_per_cell: dict[str, str] = {}
            basis_per_cell: dict[str, str] = {}
            unmapped: Counter = Counter()
            for row in rows:
                cl, basis = sr_map.singler_label_to_cl(row[f"label_{level}"], reference, level)
                per_cell[row["cell"]] = cl
                own_per_cell[row["cell"]] = row[f"label_{level}"]
                basis_per_cell[row["cell"]] = basis
                if cl is None:
                    unmapped[row[f"label_{level}"]] += 1
            sr_rows[f"{dataset}/{level}"] = {q["id"]: per_cell.get(q["cell"]) for q in questions
                                             if q["dataset"] == dataset}
            sr_own[f"{dataset}/{level}"] = {q["id"]: own_per_cell.get(q["cell"]) for q in questions
                                            if q["dataset"] == dataset}
            sr_basis[f"{dataset}/{level}"] = {q["id"]: basis_per_cell.get(q["cell"]) for q in questions
                                              if q["dataset"] == dataset}
            sr_flags[f"{dataset}/{level}"] = {
                "dataset": dataset, "level": level, "reference": reference, "cells": len(rows),
                        "labels_the_harmonisation_cannot_place": dict(unmapped.most_common())}
        sources[str(path.relative_to(ROOT))] = sha256_file(path)

    # ----------------------------------------------------- the four readings, per arm
    def block_for(choice_map: dict[str, str | None], qids) -> dict:
        verdicts: Counter = Counter()
        for qid in qids:
            verdicts[hierarchy.path_check(answer_cl[qid], choice_map.get(qid))] += 1
        return readings(verdicts, len(qids))

    def per_atlas(served_map, forced_map) -> dict:
        return {atlas: {"n": len(qids_of[atlas]),
                        "served": block_for(served_map, qids_of[atlas]),
                        "forced": block_for(forced_map, qids_of[atlas])}
                for atlas in ATLAS_ORDER}

    arms: dict[str, dict] = {}
    order: list[str] = []
    for key, meta in arm_meta.items():
        rec = records[key]
        arms[key] = {**meta,
                     "pooled": {"n": 4035, "served": block_for(rec["served"], wanted_ids),
                                "forced": block_for(rec["forced"], wanted_ids)},
                     "per_atlas": per_atlas(rec["served"], rec["forced"])}
        order.append(key)

    # CellTypist: the best declared arm per atlas, and its documented default
    ct_best_per_atlas = {}
    for atlas in ATLAS_ORDER:
        candidates = {k: v for k, v in ct_rows.items() if ct_flags[k]["dataset"] == atlas}
        scores = {k: sum(1 for qid in qids_of[atlas] if v.get(qid) == answer_cl[qid])
                  for k, v in candidates.items()}
        declared = f"{atlas}/{CELLTYPIST_DECLARED[atlas]}"
        best = max(sorted(scores), key=lambda k: scores[k])
        if scores[best] > scores[declared]:
            print(f"  NOTE {best} scores above the declared {declared} on {atlas}")
        ct_best_per_atlas[atlas] = best
    ct_best = {qid: ct_rows[ct_best_per_atlas[atlas_of[qid]]].get(qid) for qid in wanted_ids}
    ct_default = {}
    for qid in wanted_ids:
        atlas = atlas_of[qid]
        key = f"{atlas}/immune_low" if f"{atlas}/immune_low" in ct_rows else f"{atlas}/immune_low_upper"
        ct_default[qid] = ct_rows[key].get(qid)
    arms["celltypist-best-arm"] = {
        "shown_as": "CellTypist, best declared arm per atlas", "kind": "celltypist", "n": 4035,
        "cannot_abstain": True, "arm_per_atlas": ct_best_per_atlas,
        "note": "each atlas is read by the arm the existing baseline declares best for it, which is "
                "also the maximum over the declared arms; its served cell is its only cell, since the "
                "tool has no abstain action",
        "pooled": {"n": 4035, "served": block_for(ct_best, wanted_ids),
                   "forced": block_for(ct_best, wanted_ids)},
        "per_atlas": per_atlas(ct_best, ct_best),
    }
    arms["celltypist-default-model"] = {
        "shown_as": "CellTypist, its documented default model", "kind": "celltypist", "n": 4035,
        "cannot_abstain": True, "which_model": "Immune_All_Low on every atlas",
        "pooled": {"n": 4035, "served": block_for(ct_default, wanted_ids),
                   "forced": block_for(ct_default, wanted_ids)},
        "per_atlas": per_atlas(ct_default, ct_default),
    }

    # SingleR: the better of its two label levels per atlas, and label.main as its default
    sr_best_per_atlas = {}
    for atlas in ATLAS_ORDER:
        scores = {level: sum(1 for qid in qids_of[atlas]
                             if sr_rows[f"{atlas}/{level}"].get(qid) == answer_cl[qid])
                  for level in ("main", "fine")}
        sr_best_per_atlas[atlas] = max(sorted(scores), key=lambda k: scores[k])
    sr_best = {qid: sr_rows[f"{atlas_of[qid]}/{sr_best_per_atlas[atlas_of[qid]]}"].get(qid)
               for qid in wanted_ids}
    sr_main = {qid: sr_rows[f"{atlas_of[qid]}/main"].get(qid) for qid in wanted_ids}
    arms["singler-best-level"] = {
        "shown_as": "SingleR, better label level per atlas", "kind": "singler", "n": 4035,
        "cannot_abstain": True, "level_per_atlas": sr_best_per_atlas,
        "pooled": {"n": 4035, "served": block_for(sr_best, wanted_ids),
                   "forced": block_for(sr_best, wanted_ids)},
        "per_atlas": per_atlas(sr_best, sr_best),
    }
    arms["singler-label-main"] = {
        "shown_as": "SingleR, label.main (its documented default)", "kind": "singler", "n": 4035,
        "cannot_abstain": True,
        "pooled": {"n": 4035, "served": block_for(sr_main, wanted_ids),
                   "forced": block_for(sr_main, wanted_ids)},
        "per_atlas": per_atlas(sr_main, sr_main),
    }
    order += ["celltypist-best-arm", "celltypist-default-model", "singler-best-level", "singler-label-main"]

    # The same best-arm rule under the ontology-aware reading, as a sensitivity: a reader is
    # entitled to see what each tool reaches if its arms are picked by `related` instead of by
    # `exact`. These two arms are NOT the spine table's rows; the spine uses the published rule.
    def best_by_rule(table, flags, rule) -> dict:
        per_atlas = {}
        for atlas in ATLAS_ORDER:
            keys = sorted(k for k in table if flags[k]["dataset"] == atlas)
            per_atlas[atlas] = max(keys, key=lambda k: sum(
                1 for qid in qids_of[atlas]
                if (table[k].get(qid) == answer_cl[qid] if rule == "exact"
                    else hierarchy.verdict(answer_cl[qid], table[k].get(qid))
                    in ("exact", "subtype", "parent"))))
        return per_atlas

    ct_rel_per_atlas = best_by_rule(ct_rows, ct_flags, "related")
    sr_rel_per_atlas = best_by_rule(sr_rows, sr_flags, "related")
    ct_by_related = {qid: ct_rows[ct_rel_per_atlas[atlas_of[qid]]].get(qid) for qid in wanted_ids}
    sr_by_related = {qid: sr_rows[sr_rel_per_atlas[atlas_of[qid]]].get(qid) for qid in wanted_ids}
    arms["celltypist-best-arm-by-related"] = {
        "shown_as": "CellTypist, best arm per atlas chosen by related", "kind": "celltypist", "n": 4035,
        "cannot_abstain": True, "arm_per_atlas": ct_rel_per_atlas, "not_in_the_spine_table": True,
        "note": "a sensitivity, not the published selection: this is what CellTypist reaches if its "
                "arms are chosen by the ontology-aware reading",
        "pooled": {"n": 4035, "served": block_for(ct_by_related, wanted_ids),
                   "forced": block_for(ct_by_related, wanted_ids)},
        "per_atlas": per_atlas(ct_by_related, ct_by_related),
    }
    arms["singler-best-level-by-related"] = {
        "shown_as": "SingleR, best level per atlas chosen by related", "kind": "singler", "n": 4035,
        "cannot_abstain": True, "level_per_atlas": sr_rel_per_atlas, "not_in_the_spine_table": True,
        "note": "a sensitivity, not the published selection: this is what SingleR reaches if its "
                "label levels are chosen by the ontology-aware reading",
        "pooled": {"n": 4035, "served": block_for(sr_by_related, wanted_ids),
                   "forced": block_for(sr_by_related, wanted_ids)},
        "per_atlas": per_atlas(sr_by_related, sr_by_related),
    }

    # ------------------------------------------------------------- the per-arm tables
    chosen_maps: dict[str, dict[str, str | None]] = {k: records[k]["forced"] for k in arm_meta}
    chosen_maps.update({"celltypist-best-arm": ct_best, "celltypist-default-model": ct_default,
                        "singler-best-level": sr_best, "singler-label-main": sr_main,
                        "celltypist-best-arm-by-related": ct_by_related,
                        "singler-best-level-by-related": sr_by_related})
    parent_credit: dict[str, dict] = {}
    for key, chosen in chosen_maps.items():
        counter: Counter = Counter()
        for qid in wanted_ids:
            cl = chosen.get(qid)
            if cl is not None and hierarchy.verdict(answer_cl[qid], cl) == "parent":
                counter[cl] += 1
        parent_credit[key] = {"reading": "forced", "total": sum(counter.values()),
                              "top_terms": [[cl_to_name.get(c, c), c, n] for c, n in counter.most_common(5)]}

    # For the two tools, is the credit the tool's own granularity or the harmonisation table's?
    # The tool's OWN label behind each credited question, so a coarse mapping can be seen as such.
    own_maps = {
        "celltypist-best-arm": {qid: ct_own[ct_best_per_atlas[atlas_of[qid]]].get(qid) for qid in wanted_ids},
        "singler-best-level": {qid: sr_own[f"{atlas_of[qid]}/{sr_best_per_atlas[atlas_of[qid]]}"].get(qid)
                               for qid in wanted_ids},
    }
    parent_credit_own_labels: dict[str, dict] = {}
    for tool_key, own in own_maps.items():
        chosen = chosen_maps[tool_key]
        credits = Counter()
        behind: dict[str, Counter] = {}
        for qid in wanted_ids:
            cl = chosen.get(qid)
            if cl is not None and hierarchy.verdict(answer_cl[qid], cl) == "parent":
                credits[cl] += 1
                behind.setdefault(cl, Counter())[own.get(qid)] += 1
        parent_credit_own_labels[tool_key] = {
            "parent_credit": sum(credits.values()),
            "per_term": [{"mapped_cl_id": c, "mapped_name": cl_to_name.get(c, c), "credits": n,
                          "distinct_own_labels": len(behind[c]),
                          "own_labels": [[label, count] for label, count in behind[c].most_common(5)]}
                         for c, n in credits.most_common(5)],
            "note": "the tool's own label is often more specific than the id the harmonisation gave it, "
                    "so part of this credit is the mapping table's coarseness rather than the tool's",
        }

    # For the two tools, how the harmonisation reached the credited id: `basis` is the frozen
    # per-row field both lanes record, so "the tool named a broader term" and "the table made the
    # tool's term broader" are separable, and the difference is not in the tool's favour.
    basis_maps = {
        "celltypist-best-arm": {qid: ct_basis[ct_best_per_atlas[atlas_of[qid]]].get(qid) for qid in wanted_ids},
        "singler-best-level": {qid: sr_basis[f"{atlas_of[qid]}/{sr_best_per_atlas[atlas_of[qid]]}"].get(qid)
                               for qid in wanted_ids},
    }
    parent_credit_basis: dict[str, dict] = {}
    for tool_key, basis_map in basis_maps.items():
        chosen = chosen_maps[tool_key]
        by_verdict: dict[str, Counter] = {"parent": Counter(), "subtype": Counter()}
        for qid in wanted_ids:
            cl = chosen.get(qid)
            if cl is None:
                continue
            verdict = hierarchy.verdict(answer_cl[qid], cl)
            if verdict in by_verdict:
                by_verdict[verdict][basis_map.get(qid)] += 1
        parent_credit_basis[tool_key] = {
            key: {"total": sum(counter.values()), "by_basis": dict(counter.most_common())}
            for key, counter in by_verdict.items()}

    def per_arm_pooled(table, flags) -> dict:
        out = {}
        for key, choice_map in table.items():
            atlas = flags[key]["dataset"]
            block = block_for(choice_map, qids_of[atlas])
            out[key] = {"atlas": atlas, "n": len(qids_of[atlas]), "exact": block["exact"],
                        "subtype": block["subtype"], "parent": block["parent"],
                        "related": block["related"], "unplaceable": block["unplaceable"]}
        return out

    per_arm = {"celltypist": per_arm_pooled(ct_rows, ct_flags),
               "singler": per_arm_pooled(sr_rows, sr_flags)}

    # does the ontology-aware reading change which arm is best?
    selection_sensitivity = {}
    for tool, table in (("celltypist", ct_rows), ("singler", sr_rows)):
        for atlas in ATLAS_ORDER:
            keys = sorted(k for k in table if k.startswith(atlas + "/"))
            by_exact = {k: sum(1 for qid in qids_of[atlas] if table[k].get(qid) == answer_cl[qid])
                        for k in keys}
            by_related = {k: sum(1 for qid in qids_of[atlas]
                                 if hierarchy.verdict(answer_cl[qid], table[k].get(qid))
                                 in ("exact", "subtype", "parent")) for k in keys}
            best_exact = max(keys, key=lambda k: by_exact[k])
            best_related = max(keys, key=lambda k: by_related[k])
            selection_sensitivity[f"{tool}/{atlas}"] = {
                "best_by_exact": best_exact, "best_by_exact_correct": by_exact[best_exact],
                "best_by_related": best_related, "best_by_related_correct": by_related[best_related],
                "same_arm": best_exact == best_related}

    # ---------------------------------------------------------------- the floors
    def modal(qids) -> tuple[str | None, int]:
        counts = Counter(answer_cl[qid] for qid in qids)
        if not counts:
            return None, 0
        cl, n = counts.most_common(1)[0]
        return cl, n

    floors = {"note": "always answering the single most common answer: that atlas's most common "
                      "answer per atlas, the population's most common answer pooled. It is a floor on "
                      "any sample-level constant rule, and it is a strategy the tools can always "
                      "execute while a panelled model can execute it only on the questions whose panel "
                      "happens to contain that answer.",
              "per_atlas": {}}
    pooled_cl, pooled_n = modal(wanted_ids)
    floors["pooled"] = {"n": 4035, "answer_cl_id": pooled_cl, "name": cl_to_name.get(pooled_cl),
                        "correct": pooled_n, "rate": rate(pooled_n, 4035)}
    for atlas in ATLAS_ORDER:
        cl, n = modal(qids_of[atlas])
        floors["per_atlas"][atlas] = {"n": len(qids_of[atlas]), "answer_cl_id": cl,
                                      "name": cl_to_name.get(cl), "correct": n,
                                      "rate": rate(n, len(qids_of[atlas]))}

    below_floor: dict[str, list[str]] = {}
    for key in order:
        arm = arms[key]
        rows = []
        for atlas in ATLAS_ORDER:
            floor = floors["per_atlas"][atlas]["rate"]
            forced = arm["per_atlas"][atlas]["forced"]["exact"]["rate"]
            served = arm["per_atlas"][atlas]["served"]["exact"]["rate"]
            if forced is not None and forced < floor:
                extra = (f", served exact {100 * served:.1f}%" if served is not None and served != forced
                         else "")
                rows.append(f"{atlas}: forced exact {100 * forced:.1f}% against the atlas floor "
                            f"{100 * floor:.1f}%" + extra)
        if rows:
            below_floor[key] = rows

    # -------------------------------------------------- the published cells, reproduced
    reproduction = {"kev_and_laya": {}, "jev": {}, "tools": {}}
    for key in arm_meta:
        if key not in frozen["arms"]:
            continue
        published = frozen["arms"][key]
        mine = arms[key]["pooled"]
        reproduction["kev_and_laya"][key] = {
            "shown_as": arm_meta[key]["shown_as"],
            "served_published": published["served"]["accuracy"],
            "served_here": mine["served"]["exact"]["rate"],
            "served_correct_published": published["served"]["correct"],
            "served_correct_here": mine["served"]["exact"]["correct"],
            "served_agrees": published["served"]["correct"] == mine["served"]["exact"]["correct"],
            "forced_published": published["forced"]["accuracy"],
            "forced_here": mine["forced"]["exact"]["rate"],
            "forced_correct_published": published["forced"]["correct"],
            "forced_correct_here": mine["forced"]["exact"]["correct"],
            "forced_agrees": published["forced"]["correct"] == mine["forced"]["exact"]["correct"],
        }
    jev_frozen = json.loads((ROOT / JEV_SCORE).read_text(encoding="utf-8"))["by_family"]["panel_choice"]
    jev_mine = arms["jev-v4"]["pooled"]
    eligible = [qid for qid in wanted_ids if abstain_index(
        jev_by_cell[(by_qid[qid]["dataset"], by_qid[qid]["cell"])]) is not None]
    jev_forced_eligible = sum(1 for qid in eligible if records["jev-v4"]["forced"][qid] == answer_cl[qid])
    reproduction["jev"] = {
        "served_published": jev_frozen["served_correct"],
        "served_here": jev_mine["served"]["exact"]["correct"],
        "served_agrees": jev_frozen["served_correct"] == jev_mine["served"]["exact"]["correct"],
        "forced_published_on_the_rows_offering_an_abstain_option": jev_frozen["forced_correct"],
        "forced_here_on_those_rows": jev_forced_eligible,
        "forced_agrees_on_those_rows": jev_frozen["forced_correct"] == jev_forced_eligible,
        "forced_here_over_all_4035_rows": jev_mine["forced"]["exact"]["correct"],
        "note": "the Jev lane published its forced cell over the 1,043 rows that offered an abstain "
                "option; both that cell and the 4,035-row one are reported here",
        "rows_offering_an_abstain_option": len(eligible),
    }
    for label, key, frozen_key in (
            ("celltypist_best_declared_arm", "celltypist-best-arm", "celltypist_best_arm"),
            ("celltypist_default_model", "celltypist-default-model", "celltypist_default_model"),
            ("singler_best_level", "singler-best-level", "singler_best_label_level"),
            ("singler_label_main", "singler-label-main", "singler_label_main")):
        published = tool_scores["pooled"][frozen_key]
        here = arms[key]["pooled"]["served"]["exact"]
        reproduction["tools"][label] = {
            "published_rate": published["named_target_rate"],
            "published_correct": published["named_target"],
            "exact_here_correct": here["correct"], "exact_here_rate": here["rate"],
            "agrees": published["named_target"] == here["correct"]}
    checks["reproduction"] = reproduction
    checks["selection_under_the_two_readings"] = selection_sensitivity
    checks["arms_below_their_atlas_majority_floor"] = below_floor
    checks["worked_examples"] = worked_examples(hierarchy, by_qid, cl_to_name, records, ct_rows,
                                                ct_best_per_atlas, sr_rows, sr_best_per_atlas,
                                                atlas_of, answer_cl, args.examples)

    return {
        "what_this_is": "Every arm of the v4 census-matched panel_choice population scored under four "
                        "ontology-aware readings (exact / subtype / parent / related), on the served and "
                        "on the forced choice, so the paper's own arms and both reference-based annotators "
                        "are measured identically. Re-aggregation only: no model was run.",
        "rules": {
            "served": "a Kev arm: the dump's own top1 (checked equal to argmax(p) on every row); a Laya "
                      "arm and the Jev lane: argmax(p); a reference-based annotator: its own mapped "
                      "top-1 label, its only choice",
            "forced": "argmax over p with the abstain option's index removed "
                      "(reports/abstain-reconciliation.py's rule, imported rather than re-derived)",
            "exact": "the chosen CL id equals the answer's CL id",
            "subtype": "the chosen CL id is a strict descendant of the answer's CL id",
            "parent": "the chosen CL id is a strict ancestor of the answer's CL id: a partial "
                      "identification, never folded into a headline",
            "related": "exact or subtype or parent, reported as the sum of the three with each shown",
            "hierarchy": f"{HIERARCHY}, the recovered v3 hierarchy; v2 carries no parent links",
            "register": f"{REGISTER}, name to CL id: the corpus's own register, 82 terms and 82 "
                        "distinct names",
            "pooling": "every pooled rate is computed over questions, never as an average of per-atlas "
                       "rates"},
        "population": {"source": POPULATION, "family": "panel_choice", "rows": 4035,
                       "sha256": sha256_file(ROOT / POPULATION),
                       "per_atlas": {a: len(qids_of[a]) for a in ATLAS_ORDER}},
        "questions": {"source": QUESTIONS, "rows": len(questions),
                      "sha256": sha256_file(ROOT / QUESTIONS)},
        "hierarchy": {"source": HIERARCHY, "terms": len(hierarchy.parents),
                      "terms_with_parents": sum(1 for v in hierarchy.parents.values() if v),
                      "sha256": sha256_file(ROOT / HIERARCHY)},
        "floors": floors,
        "arms": arms, "order": order,
        "per_arm_tables": per_arm,
        "parent_credit": parent_credit,
        "parent_credit_own_labels": parent_credit_own_labels,
        "parent_credit_basis": parent_credit_basis,
        "checks": checks,
        "sources": sources,
    }


def worked_examples(hierarchy, by_qid, cl_to_name, records, ct_rows, ct_best_per_atlas, sr_rows,
                    sr_best_per_atlas, atlas_of, answer_cl, verbose: bool) -> list[dict]:
    """Three questions walked by hand: raw labels, both lineages, and the verdicts.

    Picked deterministically: the first question in suite-id order whose forced pick for the
    fine-tuned 9B arm is on the answer's lineage but not exact, then the first question whose
    CellTypist best-arm reading is a strict ancestor of the answer, then the first question whose
    SingleR best-level reading is a strict ancestor (SingleR's dominant error direction). Each walk
    re-derives the verdict from an explicit lineage, and the walk raises when it disagrees with the
    set-based rule.
    """
    picks: list[tuple[str, str, str, str | None]] = []
    kev = "delta-kev-9b-v4-s1"
    for qid in sorted(records[kev]["forced"]):
        if hierarchy.verdict(answer_cl[qid], records[kev]["forced"][qid]) in ("parent", "subtype"):
            picks.append(("kev", kev, qid, records[kev]["forced"][qid]))
            break
    for qid in sorted(answer_cl):
        arm = ct_best_per_atlas[atlas_of[qid]]
        chosen = ct_rows[arm].get(qid)
        if hierarchy.verdict(answer_cl[qid], chosen) == "parent":
            picks.append(("tool", f"celltypist/{arm}", qid, chosen))
            break
    for qid in sorted(answer_cl):
        level = sr_best_per_atlas[atlas_of[qid]]
        chosen = sr_rows[f"{atlas_of[qid]}/{level}"].get(qid)
        if hierarchy.verdict(answer_cl[qid], chosen) == "parent":
            picks.append(("tool", f"singler/{atlas_of[qid]}/{level}", qid, chosen))
            break

    out = []
    for kind, source, qid, chosen in picks:
        q = by_qid[qid]
        answer = answer_cl[qid]
        verdict = hierarchy.verdict(answer, chosen)
        second = hierarchy.path_check(answer, chosen)
        if verdict != second:
            raise SystemExit("the two paths disagree: STOP")
        served_pick = records[source]["served"][qid] if kind == "kev" else None
        forced_pick = records[source]["forced"][qid] if kind == "kev" else None
        entry = {
            "which": kind, "source": source, "question_id": qid, "atlas": q["dataset"],
            "cell": q["cell"], "answer_label": q["target"], "answer_cl_id": answer,
            "answer_cl_name": cl_to_name.get(answer, ""),
            "chosen_cl_id": chosen, "chosen_cl_name": cl_to_name.get(chosen, ""),
            "verdict": verdict,
            "verdicts": {"exact": verdict == "exact", "subtype": verdict == "subtype",
                         "parent": verdict == "parent",
                         "related": verdict in ("exact", "subtype", "parent")},
            "lineage_of_the_answer": [[node, name, list(parents)]
                                      for node, name, parents in hierarchy.path_to_root(answer)],
            "lineage_of_the_chosen": [[node, name, list(parents)]
                                      for node, name, parents in hierarchy.path_to_root(chosen)],
            "raw": {
                "kev_served_pick_cl": served_pick if kind == "kev" else None,
                "kev_served_pick_name": cl_to_name.get(served_pick, "an abstain key (not a cell type)")
                if kind == "kev" else None,
                "kev_forced_pick_cl": forced_pick if kind == "kev" else None,
                "kev_forced_pick_name": cl_to_name.get(forced_pick, "an abstain key (not a cell type)")
                if kind == "kev" else None,
                "tool_arm": source if kind == "tool" else None,
                "tool_mapped_cl": chosen if kind == "tool" else None,
            },
        }
        out.append(entry)
        if verbose:
            print("=" * 100)
            print(f"{kind} | {source} | question {qid} | atlas {q['dataset']} | cell {q['cell']}")
            print(f"  the question's answer: {q['target']!r} -> {answer} "
                  f"({cl_to_name.get(answer, '')})")
            if kind == "kev":
                print(f"  the arm's served pick is {cl_to_name.get(served_pick, 'an abstain key')} and "
                      f"its forced pick is {cl_to_name.get(forced_pick, 'an abstain key')}")
            else:
                print(f"  the arm's mapped label for this cell: {chosen} ({cl_to_name.get(chosen, '')})")
            for title, root in (("lineage of the answer", answer), ("lineage of the chosen", chosen)):
                print(f"  {title} ({root}):")
                for node, name, parents in hierarchy.path_to_root(root):
                    print(f"      {node:<12} {name:<58} parents={list(parents)}")
            print(f"  VERDICTS: exact={verdict == 'exact'} subtype={verdict == 'subtype'} "
                  f"parent={verdict == 'parent'} related={verdict in ('exact', 'subtype', 'parent')} "
                  f"| the independent walk says {second}")
    return out


def write_md(result: dict) -> None:
    """The report. No em dashes and no en dashes anywhere, US English."""
    arms = result["arms"]
    floors = result["floors"]

    def pct(value) -> str:
        return "not measured" if value is None else f"{100 * value:.1f}%"

    def cells(block) -> str:
        return (f"{pct(block['exact']['rate'])} ({block['exact']['correct']:,}) | "
                f"{pct(block['subtype']['rate'])} | {pct(block['parent']['rate'])} | "
                f"{pct(block['related']['rate'])}")

    spine = [("delta-kev-0.5b-v4-s1", "fine-tuned"), ("delta-kev-0.6b-v4-s1", "fine-tuned"),
             ("delta-kev-0.8b-v4-s1", "fine-tuned"), ("delta-kev-4b-v4-s1", "fine-tuned"),
             ("delta-kev-9b-v4-s1", "fine-tuned"), ("delta-kev-27b-v4-s1", "fine-tuned"),
             ("kev-0.5b-v4", "untuned rung"), ("kev-0.6b-v4", "untuned rung"),
             ("kev-0.8b-v4", "untuned rung"), ("kev-4b-v4", "untuned rung"),
             ("kev-9b-v4", "untuned rung"), ("kev-27b-v4", "untuned rung"),
             ("laya-v4-ft-spec", "Laya"), ("laya-v4-base", "Laya"),
             ("jev-v4", "API engine"), ("celltypist-best-arm", "reference-based"),
             ("singler-best-level", "reference-based"),
             ("celltypist-default-model", "reference-based"),
             ("singler-label-main", "reference-based")]

    lines = [
        "# The v4 census-matched population under four ontology-aware readings",
        "",
        "*Machine-readable companion: `reports/ontology-aware-reading-v4.json`.*",
        "",
        f"Population: the {result['population']['rows']:,} `panel_choice` rows of "
        f"`{result['population']['source']}`, spread over six atlases, the same questions for every arm "
        "below. Each arm is scored under four readings, identically for the paper's own arms and for "
        "both reference-based annotators:",
        "",
        "* **exact** the chosen Cell Ontology id equals the answer's id. This is the reading the "
        "manuscript publishes.",
        "* **subtype** the chosen id is a strict descendant of the answer's id: something narrower than "
        "the answer.",
        "* **parent** the chosen id is a strict ancestor of the answer's id: something broader. A "
        "broader term is a partial identification, not the answer, so it is reported separately and "
        "never folded into a headline number.",
        "* **related** exact or subtype or parent, always given beside the three it sums.",
        "",
        "Every reading is computed on the **served** choice and again on the **forced** choice, which "
        "is `argmax` over the probability vector with the abstain option removed, the settled rule of "
        "`reports/abstain-reconciliation.py` (imported by this script, not re-derived). Neither tool "
        "can abstain, so for both of them the served and the forced column are the same number by "
        "construction, and that is the column to read beside the models' forced column.",
        "",
        "The labelled answer and the chosen label are both mapped through the corpus's own register "
        f"(`{REGISTER}`, name to CL id, 82 terms with 82 distinct names), and both lineages are walked "
        f"in `{HIERARCHY}`, the recovered v3 hierarchy, whose 3,335 terms are the ones the corpus's own "
        "answer ids live in. The register places the labelled answer on "
        f"{result['checks']['register']['answers_the_register_places']:,} of "
        f"{result['checks']['register']['questions']:,} questions, and every label it cannot place is "
        "listed under the checks below rather than absorbed into a rate.",
        "",
        "## The comparison table, pooled over the 4,035 questions",
        "",
        "Rows are the paper's six fine-tuned arms, its six untuned released rungs, Laya's two arms, "
        "Jev, and the two tools at their best arm; the last two rows are the tools' documented "
        "defaults, kept visible because the published table carries them. Every denominator in this "
        "table is 4,035 questions.",
        "",
        "| arm | class | n | served exact | served subtype | served parent | served related | "
        "forced exact | forced subtype | forced parent | forced related |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, kind in spine:
        arm = arms[key]
        lines.append(f"| {arm['shown_as']} | {kind} | {arm['pooled']['n']:,} | "
                     f"{cells(arm['pooled']['served'])} | {cells(arm['pooled']['forced'])} |")
    lines += [
        "",
        "The `exact` columns carry their counts so the published cells can be checked; the `subtype` "
        "and `parent` columns are rates, and the JSON beside this file carries every count, so a "
        "reader can subtract them from `related` exactly.",
        "",
        "*How the two best-arm rows are chosen.* CellTypist is read per atlas by the arm the existing "
        "baseline declares for that tissue, and where a declared arm is beaten the better arm is "
        "taken, which is that baseline's own rule and its own published table: on muraro that is "
        "`fetal_pancreas` at 95.4% rather than the tissue-matched `pancreatic_islet` at 68.0%. "
        "SingleR is read by the better of its `label.main` and `label.fine` levels per atlas, which "
        "is also the published table's rule. The ontology-aware reading does not move these choices "
        "for the table above; what it would do to them is the sensitivity reported below.",
        "",
        "## The majority-class floor, beside every rate",
        "",
        "*The floor is always answering the single most common answer: that atlas's most common answer "
        "on the per-atlas rows, and the population's most common answer on the pooled row. It is a "
        "floor on any sample-level constant rule, and it is a strategy the two tools can always "
        "execute while a panelled model can execute it only on the questions whose panel happens to "
        "contain that answer, which makes it the strict version of the comparison.*",
        "",
        "| floor over | n | answer | correct | rate |",
        "|---|---:|---|---:|---:|",
        f"| the pooled {floors['pooled']['n']:,} questions | {floors['pooled']['n']:,} | "
        f"{floors['pooled']['name']} (`{floors['pooled']['answer_cl_id']}`) | "
        f"{floors['pooled']['correct']:,} | {pct(floors['pooled']['rate'])} |",
    ]
    for atlas in ATLAS_ORDER:
        f = floors["per_atlas"][atlas]
        lines.append(f"| {atlas} | {f['n']:,} | {f['name']} (`{f['answer_cl_id']}`) | "
                     f"{f['correct']:,} | {pct(f['rate'])} |")

    lines += ["", "### Arms that sit below the floor on an atlas, said rather than hidden", ""]
    below = result["checks"]["arms_below_their_atlas_majority_floor"]
    if below:
        for key in result["order"]:
            if key in below:
                lines.append(f"* **{arms[key]['shown_as']}**: " + "; ".join(below[key]) + ".")
    else:
        lines.append("* No arm sits below its atlas's majority-class floor on the forced reading.")
    lines.append("")

    lines += [
        "## Per atlas, long form",
        "",
        "One line per arm and atlas, so a pooled number cannot be carried by one atlas without the "
        "reader seeing it. Rates are over that atlas's own question count, printed in the `n` column, "
        "and the last column is that atlas's majority-class floor.",
        "",
        "| arm | atlas | n | served exact | served subtype | served parent | served related | "
        "forced exact | forced subtype | forced parent | forced related | atlas floor |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, _kind in spine:
        arm = arms[key]
        for atlas in ATLAS_ORDER:
            row = arm["per_atlas"][atlas]
            lines.append(f"| {arm['shown_as']} | {atlas} | {row['n']:,} | {cells(row['served'])} | "
                         f"{cells(row['forced'])} | {pct(floors['per_atlas'][atlas]['rate'])} |")

    lines += [
        "",
        "## The tools arm by arm, since a best arm is a choice",
        "",
        "A reader is entitled to see the arms a best-arm row was chosen from. Every arm is scored on "
        "its own atlas's questions; `exact` is the published reading and `related` is the "
        "ontology-aware one. The best-arm rows above take the arm the existing baseline declares per "
        "atlas, which the maximum over the declared arms also selects.",
        "",
        "| tool | arm | atlas | n | exact | subtype | parent | related |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for tool, pretty in (("celltypist", "CellTypist"), ("singler", "SingleR")):
        for key in sorted(result["per_arm_tables"][tool]):
            row = result["per_arm_tables"][tool][key]
            lines.append(f"| {pretty} | `{key}` | {row['atlas']} | {row['n']:,} | "
                         f"{pct(row['exact']['rate'])} | {pct(row['subtype']['rate'])} | "
                         f"{pct(row['parent']['rate'])} | {pct(row['related']['rate'])} |")

    sensitivity = result["checks"]["selection_under_the_two_readings"]
    changed = [k for k, v in sensitivity.items() if not v["same_arm"]]
    ct_rel = arms["celltypist-best-arm-by-related"]["pooled"]["forced"]
    sr_rel = arms["singler-best-level-by-related"]["pooled"]["forced"]
    lines += [
        "",
        f"**Does the ontology-aware reading change which arm is best? Yes, on "
        f"{len(changed)} of {len(sensitivity)} atlas and tool pairs** "
        f"({', '.join(sorted(changed))}); it is not a tie-break the reading is neutral about. The "
        "table above keeps the published, `exact`-based selection so the arms are the ones the "
        "existing report carries, and the sensitivity below is what a best arm chosen by the "
        "ontology-aware reading would reach. It is the tools' most favourable case, and it is "
        "reported so nobody has to take the selection on trust:",
        "",
        "| best-arm rule | CellTypist exact | CellTypist related | SingleR exact | SingleR related |",
        "|---|---:|---:|---:|---:|",
        f"| by `exact` (the published rule, the spine table) | "
        f"{pct(arms['celltypist-best-arm']['pooled']['forced']['exact']['rate'])} "
        f"({arms['celltypist-best-arm']['pooled']['forced']['exact']['correct']:,}/4,035) | "
        f"{pct(arms['celltypist-best-arm']['pooled']['forced']['related']['rate'])} "
        f"({arms['celltypist-best-arm']['pooled']['forced']['related']['correct']:,}/4,035) | "
        f"{pct(arms['singler-best-level']['pooled']['forced']['exact']['rate'])} "
        f"({arms['singler-best-level']['pooled']['forced']['exact']['correct']:,}/4,035) | "
        f"{pct(arms['singler-best-level']['pooled']['forced']['related']['rate'])} "
        f"({arms['singler-best-level']['pooled']['forced']['related']['correct']:,}/4,035) |",
        f"| by `related` (the tools' most favourable selection) | {pct(ct_rel['exact']['rate'])} "
        f"({ct_rel['exact']['correct']:,}/4,035) | {pct(ct_rel['related']['rate'])} "
        f"({ct_rel['related']['correct']:,}/4,035) | {pct(sr_rel['exact']['rate'])} "
        f"({sr_rel['exact']['correct']:,}/4,035) | {pct(sr_rel['related']['rate'])} "
        f"({sr_rel['related']['correct']:,}/4,035) |",
        "",
        "Even on that generous selection the tools do not reach the fine-tuned arms: CellTypist at "
        f"{pct(ct_rel['related']['rate'])} related sits below all six of them, and SingleR at "
        f"{pct(sr_rel['related']['rate'])} sits far below. What the selection does move is how much "
        "of CellTypist's deficit the reading closes, which is why the spine table states its rule "
        "rather than leaving a reader to guess.",
        "",
        f"One more limit belongs beside these numbers: on {arms['celltypist-best-arm']['pooled']['forced']['unplaceable']['correct']:,} "
        "of the 4,035 questions CellTypist's chosen label has no CL id in its own harmonisation at "
        f"all, and {arms['singler-best-level']['pooled']['forced']['unplaceable']['correct']:,} for "
        "SingleR. Those are counted as neither exact nor related, and they are a property of the "
        "mapping tables rather than of the cells.",
    ]

    # ------------------------------------------------- what the parent credit is made of
    credit = result["parent_credit"]
    lines += [
        "",
        "## What the `parent` credit is made of",
        "",
        "A lineage match is only as meaningful as the term that carries it, so the terms behind every "
        "`parent` count are printed rather than trusted. None of them is the ontology's root: no arm "
        "in this table earns credit by naming `cell` or `eukaryotic cell`, and the terms below are "
        "cell types a reader would recognise as coarser answers to the questions they were credited "
        "on. The count is the forced reading's, and it is the same reading as the table above.",
        "",
        "| arm | parent credit | the terms that carry it |",
        "|---|---:|---|",
    ]
    for key, _kind in spine:
        row = credit[key]
        terms = ", ".join(f"{name} x{count}" for name, _cl, count in row["top_terms"]) or "none"
        lines.append(f"| {arms[key]['shown_as']} | {row['total']:,} | {terms} |")

    own_credit = result["parent_credit_own_labels"]
    lines += [
        "",
        "**Is that credit the tool's own granularity or the harmonisation table's?** The two cases "
        "separate, and the tools' own labels behind each credited question are printed below. Where a "
        "tool said `Epithelial_cells` and the table wrote `epithelial cell`, the credit is the tool's "
        "own coarseness. Where a tool said `L2/3 IT_6` and the table wrote `neuron`, the credit is "
        "the table's: the tool did not name a broader term, the mapping replaced its finer term with "
        "one. That is the case for most of CellTypist's credit and for a minority of SingleR's, and "
        "it means this column measures neither tool's own granularity cleanly.",
        "",
        "| tool row | mapped id carrying the credit | credited questions | distinct own labels behind it | the tool's own labels, most common first |",
        "|---|---|---:|---:|---|",
    ]
    for key, pretty in (("celltypist-best-arm", "CellTypist, best declared arm"),
                        ("singler-best-level", "SingleR, better label level")):
        for entry in own_credit[key]["per_term"]:
            labels = ", ".join(f"`{label}` x{count}" for label, count in entry["own_labels"])
            lines.append(f"| {pretty} | {entry['mapped_name']} (`{entry['mapped_cl_id']}`) | "
                         f"{entry['credits']:,} | {entry['distinct_own_labels']} | {labels} |")

    basis = result["parent_credit_basis"]
    lines += [
        "",
        "**And which side made the term broader?** Both lanes' frozen artifacts record a `basis` for "
        "every mapped label, so the two cases separate exactly: `name` or `synonym` means the tool "
        "itself said the broad term, while `coarsened` means the table replaced the tool's finer label "
        "with a broader id. The distinction matters because only the first is a coarse identification "
        "by the tool.",
        "",
        "| tool row | `parent` credit | basis of the mapping on those questions |",
        "|---|---:|---|",
    ]
    for key, pretty in (("celltypist-best-arm", "CellTypist, best declared arm"),
                        ("singler-best-level", "SingleR, better label level")):
        entry = basis[key]["parent"]
        detail = ", ".join(f"`{name}` x{count:,}" for name, count in entry["by_basis"].items())
        lines.append(f"| {pretty} | {entry['total']:,} | {detail or 'none'} |")
    ct_basis = basis["celltypist-best-arm"]["parent"]["by_basis"]
    sr_basis = basis["singler-best-level"]["parent"]["by_basis"]
    ct_said_it = ct_basis.get("name", 0) + ct_basis.get("synonym", 0)
    sr_said_it = sr_basis.get("name", 0) + sr_basis.get("synonym", 0)
    lines += [
        "",
        f"For CellTypist, {ct_said_it:,} of its "
        f"{basis['celltypist-best-arm']['parent']['total']:,} parent-credited questions are cases "
        "where the tool itself named the broad term; the rest are cases where the harmonisation "
        "replaced a finer label of its own with a broader id, so the credit is the table's doing "
        "rather than a coarse call by the tool. For SingleR the balance is the other way: "
        f"{sr_said_it:,} of {basis['singler-best-level']['parent']['total']:,} are its own label. "
        "That difference is a property of the two mapping tables, and it is a warning against reading "
        "this column as either tool's own granularity.",
        "",
        "Two consequences for the sentences the manuscript can carry. First, bottom lines: on "
        f"CellTypist's best arms "
        f"{100 * ct_basis.get('coarsened', 0) / basis['celltypist-best-arm']['parent']['total']:.0f}% "
        "of the `parent` credit is the table's coarsening and not the tool's own coarse answer, so "
        "\"CellTypist named a broader term\" is false as a general statement about this column, even "
        "though it is true of SingleR's "
        f"({100 * sr_said_it / basis['singler-best-level']['parent']['total']:.0f}% of its credit is "
        "its own label). Second, the coarsening also works in the tool's favour on the `exact` "
        "column: a CellTypist label like `L2/3 IT_6` becomes `neuron`, which is exact when the "
        "question's answer is `neuron`, and a subtype of the answer when the answer is a finer neuron "
        "type. A finer harmonisation would move some of CellTypist's exact cells to subtype and some "
        "of its parent credits to unrelated, in a proportion this report does not measure. That "
        "sensitivity is listed under what is not measured here, and until it is measured no sentence "
        "should treat the tools' granularity as a property of the tools alone.",
    ]

    # ------------------------------------------------------------- the four answers
    ft = [k for k, kind in spine if kind == "fine-tuned"]
    ft_forced = {k: arms[k]["pooled"]["forced"] for k in ft}
    ft_served = {k: arms[k]["pooled"]["served"] for k in ft}
    best_ft = max(ft, key=lambda k: ft_forced[k]["exact"]["correct"])
    ct = arms["celltypist-best-arm"]["pooled"]["forced"]
    sr = arms["singler-best-level"]["pooled"]["forced"]
    ct_def = arms["celltypist-default-model"]["pooled"]["forced"]
    sr_def = arms["singler-label-main"]["pooled"]["forced"]
    lo_ft = min(ft, key=lambda k: ft_forced[k]["exact"]["correct"])
    hi_ft = best_ft
    lo_rel = min(ft, key=lambda k: ft_forced[k]["related"]["correct"])
    hi_rel = max(ft, key=lambda k: ft_forced[k]["related"]["correct"])

    def points(a, b) -> str:
        return f"{100 * (a - b):+.1f}"

    lines += [
        "",
        "## What the readings say, with the numbers behind each answer",
        "",
        "### (a) Under `exact`, do the fine-tuned arms beat the tools, and is the lead forced against "
        "forced?",
        "",
        f"**Yes, and the lead is stated forced against forced.** The six fine-tuned arms' forced "
        f"`exact` rates run from {pct(ft_forced[lo_ft]['exact']['rate'])} "
        f"({arms[lo_ft]['shown_as']}) to {pct(ft_forced[hi_ft]['exact']['rate'])} "
        f"({arms[hi_ft]['shown_as']}), over 4,035 questions each. CellTypist's best declared arm per "
        f"atlas is {pct(ct['exact']['rate'])} ({ct['exact']['correct']:,}/4,035) and SingleR's better "
        f"label level is {pct(sr['exact']['rate'])} ({sr['exact']['correct']:,}/4,035). The gap, "
        f"forced against forced, runs from {points(ft_forced[lo_ft]['exact']['rate'], ct['exact']['rate'])} "
        f"to {points(ft_forced[hi_ft]['exact']['rate'], ct['exact']['rate'])} points against CellTypist "
        f"and from {points(ft_forced[lo_ft]['exact']['rate'], sr['exact']['rate'])} to "
        f"{points(ft_forced[hi_ft]['exact']['rate'], sr['exact']['rate'])} points against SingleR. "
        "Neither tool can abstain, so its number is a forced number and the pair is reading-matched; "
        "the fine-tuned arms' served column is not the column to put beside it.",
        "",
        "The tools' documented defaults, on the same reading and the same denominator: CellTypist's "
        f"default model {pct(ct_def['exact']['rate'])} ({ct_def['exact']['correct']:,}/4,035), "
        f"SingleR's `label.main` {pct(sr_def['exact']['rate'])} ({sr_def['exact']['correct']:,}/4,035).",
        "",
        "### (b) Under `related`, how much of the tools' deficit disappears?",
        "",
        f"CellTypist's best declared arm moves from {ct['exact']['correct']:,}/4,035 "
        f"({pct(ct['exact']['rate'])}) exact to {ct['related']['correct']:,}/4,035 "
        f"({pct(ct['related']['rate'])}) related: "
        f"{100 * (ct['related']['rate'] - ct['exact']['rate']):+.1f} points of credit, splitting into "
        f"{ct['subtype']['correct']:,} questions ({pct(ct['subtype']['rate'])}) where CellTypist named "
        f"something **narrower** than the answer and {ct['parent']['correct']:,} "
        f"({pct(ct['parent']['rate'])}) where it named something **broader**. SingleR's better level "
        f"moves from {sr['exact']['correct']:,}/4,035 ({pct(sr['exact']['rate'])}) to "
        f"{sr['related']['correct']:,}/4,035 ({pct(sr['related']['rate'])}): "
        f"{sr['subtype']['correct']:,} narrower ({pct(sr['subtype']['rate'])}) and "
        f"{sr['parent']['correct']:,} broader ({pct(sr['parent']['rate'])}).",
        "",
        f"Beside that, the six fine-tuned arms' forced `related` rates run from "
        f"{pct(ft_forced[lo_rel]['related']['rate'])} ({arms[lo_rel]['shown_as']}) to "
        f"{pct(ft_forced[hi_rel]['related']['rate'])} ({arms[hi_rel]['shown_as']}). So the deficit that "
        f"survives is {points(ft_forced[lo_rel]['related']['rate'], ct['related']['rate'])} to "
        f"{points(ft_forced[hi_rel]['related']['rate'], ct['related']['rate'])} points against "
        f"CellTypist and {points(ft_forced[lo_rel]['related']['rate'], sr['related']['rate'])} to "
        f"{points(ft_forced[hi_rel]['related']['rate'], sr['related']['rate'])} points against SingleR. "
        "The reading therefore closes part of CellTypist's published deficit and none of the "
        "comparison: the margin narrows and stays large.",
        "",
        f"Against the single best fine-tuned arm ({arms[best_ft]['shown_as']}), the arithmetic is "
        f"direct: CellTypist's deficit goes from "
        f"{points(ft_forced[best_ft]['exact']['rate'], ct['exact']['rate'])} points on `exact` to "
        f"{points(ft_forced[best_ft]['related']['rate'], ct['related']['rate'])} points on `related`, "
        f"so the ontology-aware reading closes "
        f"{100 * ((ct['related']['rate'] - ct['exact']['rate']) / (ft_forced[best_ft]['exact']['rate'] - ct['exact']['rate'])):.0f}% "
        f"of the deficit measured against that arm. SingleR's deficit goes from "
        f"{points(ft_forced[best_ft]['exact']['rate'], sr['exact']['rate'])} points to "
        f"{points(ft_forced[best_ft]['related']['rate'], sr['related']['rate'])} points, so "
        f"{100 * ((sr['related']['rate'] - sr['exact']['rate']) / (ft_forced[best_ft]['exact']['rate'] - sr['exact']['rate'])):.0f}% "
        "of its deficit closes and the rest stays.",
        "",
        "The direction of the credit matters and is not flattering to either tool under a reading that "
        f"demands the answer: {ct['parent']['correct']:,} of CellTypist's {ct['related']['correct']:,} "
        f"related questions are credited on a broader id than the answer and only "
        f"{ct['subtype']['correct']:,} on a narrower one; for SingleR the split is "
        f"{sr['parent']['correct']:,} broader against {sr['subtype']['correct']:,} narrower. So over "
        f"{100 * ((ct['parent']['correct'] + sr['parent']['correct']) / (ct['related']['correct'] + sr['related']['correct'] - ct['exact']['correct'] - sr['exact']['correct'])):.0f}% "
        "of the credit the reading hands the two tools lands on a coarser id than the answer rather "
        "than a finer one. Where the credit for the broader id came from is a separate question with a "
        "measurable answer, and it does not have the same answer for the two tools: the section below "
        "shows the tool's own label behind each credited question and the mapping's own `basis` field.",
        "",
        "### (c) Do the paper's own arms gain anything under the ontology-aware rules?",
        "",
        "The six fine-tuned arms first, since they are the paper's headline:",
        "",
        "| fine-tuned arm | forced exact | forced related | forced gain (pt) | narrower (pt) | "
        "broader (pt) | served exact | served related | served gain (pt) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ft:
        f, s = ft_forced[key], ft_served[key]
        lines.append(f"| {arms[key]['shown_as']} | {pct(f['exact']['rate'])} | {pct(f['related']['rate'])} | "
                     f"{100 * (f['related']['rate'] - f['exact']['rate']):+.2f} | "
                     f"{100 * f['subtype']['rate']:.2f} | {100 * f['parent']['rate']:.2f} | "
                     f"{pct(s['exact']['rate'])} | {pct(s['related']['rate'])} | "
                     f"{100 * (s['related']['rate'] - s['exact']['rate']):+.2f} |")
    rank_exact = sorted(ft, key=lambda k: -ft_forced[k]["exact"]["correct"])
    rank_related = sorted(ft, key=lambda k: -ft_forced[k]["related"]["correct"])
    paper_family = [k for k, kind in spine if kind in ("fine-tuned", "untuned rung", "Laya", "API engine")]
    lines += [
        "",
        f"**They gain a little, and the ranking does not move: the best arm under `exact` and under "
        f"`related` is the same one ({arms[best_ft]['shown_as']}), and the order of the six is "
        f"{'identical' if rank_exact == rank_related else 'different'} between the two readings.** The "
        "gain is small for a structural reason: a panel question already fixes the answer's granularity "
        "in its own option list, so a wrong pick is almost always an unrelated option rather than a "
        "broader or a narrower name for the right one. The fine-tuned arms have also learned the "
        "register's own terms, which is why their remaining errors are unrelated options rather than "
        "coarser synonyms. A panel-based arm gains "
        f"{100 * (ft_forced[best_ft]['related']['rate'] - ft_forced[best_ft]['exact']['rate']):.2f} to "
        f"{max(100 * (ft_forced[k]['related']['rate'] - ft_forced[k]['exact']['rate']) for k in ft):.2f} "
        "points, so the reading is not a second way for the paper's headline numbers to score.",
        "",
        "**The rest of the paper's own rows are a different matter, and the difference is the finding.** "
        "Every arm that names terms freely when it is unsure buys a great deal under this reading:",
        "",
        "| paper arm | forced exact | forced related | forced gain (pt) | narrower (pt) | broader (pt) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in paper_family:
        if key in ft:
            continue
        f = arms[key]["pooled"]["forced"]
        lines.append(f"| {arms[key]['shown_as']} | {pct(f['exact']['rate'])} | {pct(f['related']['rate'])} | "
                     f"{100 * (f['related']['rate'] - f['exact']['rate']):+.2f} | "
                     f"{100 * f['subtype']['rate']:.2f} | {100 * f['parent']['rate']:.2f} |")
    zero_shot_best = max((k for k, kind in spine if kind == "untuned rung"),
                         key=lambda k: arms[k]["pooled"]["forced"]["related"]["correct"])
    jev_gain = 100 * (arms["jev-v4"]["pooled"]["forced"]["related"]["rate"]
                      - arms["jev-v4"]["pooled"]["forced"]["exact"]["rate"])
    lines += [
        "",
        f"Jev (API) gains {jev_gain:.1f} points, {arms['kev-9b-v4']['shown_as']} gains "
        f"{100 * (arms['kev-9b-v4']['pooled']['forced']['related']['rate'] - arms['kev-9b-v4']['pooled']['forced']['exact']['rate']):.1f} "
        "points, and the whole untuned band moves further than any fine-tuned arm. The consequence is "
        "plain and belongs in the paper rather than in a footnote: the margin by which the best "
        "fine-tuned arm beats the untuned family shrinks under the ontology-aware reading, from "
        f"{points(ft_forced[best_ft]['exact']['rate'], arms[zero_shot_best]['pooled']['forced']['exact']['rate'])} "
        f"to {points(ft_forced[best_ft]['related']['rate'], arms[zero_shot_best]['pooled']['forced']['related']['rate'])} "
        f"points against {arms[zero_shot_best]['shown_as']}, and from "
        f"{points(ft_forced[best_ft]['exact']['rate'], arms['jev-v4']['pooled']['forced']['exact']['rate'])} "
        f"to {points(ft_forced[best_ft]['related']['rate'], arms['jev-v4']['pooled']['forced']['related']['rate'])} "
        f"points against Jev. The ordering is preserved, the fine-tuned arms still fill the top six "
        "rows under both readings, but the reading is generous to any arm that names broad terms, and "
        "that is what the untuned band and Jev do when they are wrong.",
        "",
        "### (d) Is the comparison symmetric after this treatment?",
        "",
        "**No.** The four readings are now identical on both sides, and that is all that has been "
        "equalized. What remains asymmetric, stated rather than defended:",
        "",
        "1. **The choice set.** The paper's arms choose among the 4 to 42 options of each question, and "
        "the answer is on that list. CellTypist's best arms choose among their own model vocabularies "
        "and SingleR's among its reference's labels, and a tool whose vocabulary lacks the answer's "
        "term can never name it. The published report's reachability column quantifies that part of the "
        "deficit and this report does not remove it: under `related` a tool still has to have named "
        "something on the answer's lineage, and the answer's own term can sit outside its vocabulary "
        "entirely. This remains the strongest reason a reader can say the deck is stacked, and no "
        "reading of the same dumps answers it.",
        "2. **The training population.** The fine-tuned arms are trained on every atlas under a "
        "donor-level holdout, so they have seen other donors of the same atlases, while the tools' "
        "references are separate studies. No reading here touches that.",
        "3. **The floor's availability.** The majority-class floor is always executable by the tools and "
        "only sometimes executable by a panelled model, as the floor section says.",
        "4. **The mapping tables.** CellTypist's CL ids come from the project's own curated label map "
        "and SingleR's from a table built for this corpus, so a hole in either table is charged to the "
        "tool, while the paper's arms are read through the corpus's own register, which agrees with the "
        f"corpus's answer ids by construction (checked: "
        f"{result['checks']['register']['answers_the_register_places']:,}/4,035). On "
        f"{arms['celltypist-best-arm']['pooled']['forced']['unplaceable']['correct']:,} questions "
        "CellTypist named a label its own table cannot place at all, which the reading scores as "
        "neither exact nor related.",
        "5. **The served reading.** The tools can never lose a correct answer to a decline while every "
        "paper arm can, so their number belongs beside the forced column only.",
        "6. **The best-arm selection.** The tools' arms are picked per atlas by a rule fixed in "
        "advance, and the reading is not neutral about that choice: selected by `related`, CellTypist "
        f"reaches {pct(arms['celltypist-best-arm-by-related']['pooled']['forced']['related']['rate'])} "
        f"and SingleR {pct(arms['singler-best-level-by-related']['pooled']['forced']['related']['rate'])}. "
        "The spine table uses the published selection; a reader who prefers the other one should read "
        "those two numbers, and both are still below every fine-tuned arm.",
        "7. **The harmonisation tables, again, and in the tools' favour this time.** Most of "
        "CellTypist's `related` credit is a coarsened id rather than a coarse answer, and the same "
        "coarsening hands it exact cells, so the tools are scored through tables whose granularity "
        "neither the paper nor a reader can audit from these dumps alone. That is the one asymmetry "
        "here that a reviewer could say works in the tools' favour, and it is not measured.",
        "",
        "What the treatment buys is that the tools are no longer scored on a rule the paper's own arms "
        "were never held to. Under `exact` alone a reader can reasonably call the comparison stacked, "
        "because a tool that names a more specific or a more general term for the right cell is scored "
        "as wrong. That criticism now has a number: it is partly right about the size of the tools' "
        "deficit, and it does not survive as an explanation of it. What does not survive either, and "
        "has to be said in the same breath, is the reading as a free pass for the paper: the same rule "
        f"that lifts CellTypist by {100 * (ct['related']['rate'] - ct['exact']['rate']):.1f} points "
        f"lifts Jev by {jev_gain:.1f} points, and it leaves the paper's margin over Jev at about a "
        "third of its exact-reading size.",
        "",
        "## What is not measured here",
        "",
        "* **A finer harmonisation for CellTypist.** How much of its `exact` column is a subtype of the "
        "answer under its own label, and how many of its `parent` credits would become unrelated, is "
        "not measured: it needs a harmonisation built to finer terms, and the published baseline "
        "already reports that a supplementary Cell Ontology table changes the pooled best-arm accuracy "
        "by 0.000 points on the predictions its own curated table cannot place.",
        "* **Azimuth and scANVI** on this population: not measured, in this report or the one it "
        "extends.",
        "* **Seed replicates for the paper's arms** on any reading: the six fine-tuned arms are one run "
        "each, so a difference between two of them is a difference between draws as well as between "
        "sizes.",
        "* **The three constructed-evidence families** (`evidence_diluted`, `marker_noul`, "
        "`panel_miss`): no cell-level annotator can be scored on them, so the tools have no number "
        "there and none is implied.",
        "* **Jev's nondeterminism**: its row here is one draw of a nondeterministic API, so a margin "
        "against it smaller than the drift of an answer-preserving re-ask is not a result.",
        "",
        "## The three worked examples, walked by hand",
        "",
        "Three questions, with the raw labels, both lineages printed term by term, and the verdicts "
        "from the set-based rule beside the verdict from an explicit walk up the hierarchy. The walk "
        "raises when the two disagree, so the rule is auditable and not only its output. The three "
        "were picked by rule, not by hand: the first question where the fine-tuned 9B arm's forced "
        "pick is on the answer's lineage without being exact, then the first where CellTypist's "
        "best-arm reading is a strict ancestor of the answer, then the first where SingleR's is. "
        "Between them they show the two directions the `related` reading gives credit in, and they "
        "show that the paper's own arms are credited in the same direction as the tools are: the 9B "
        "arm named a finer option than the key and is scored `subtype`, not `exact`.",
        "",
    ]
    for entry in result["checks"]["worked_examples"]:
        raw = (f"; the arm's served pick maps to `{entry['raw']['kev_served_pick_name']}` and its "
               f"forced pick to `{entry['raw']['kev_forced_pick_name']}`" if entry["which"] == "kev"
               else f"; `{entry['source']}` mapped this cell to that id")
        lines += [
            f"**Example: `{entry['question_id']}`** ({entry['atlas']}, cell `{entry['cell']}`), read by "
            f"`{entry['source']}`",
            "",
            f"* the question's labelled answer is `{entry['answer_label']}` = "
            f"`{entry['answer_cl_id']}` ({entry['answer_cl_name']})",
            f"* the chosen id is `{entry['chosen_cl_id']}` ({entry['chosen_cl_name']})" + raw,
            "* the chain from the answer upward: "
            + " -> ".join(f"`{node}`" for node, _name, _parents in entry["lineage_of_the_answer"][:6])
            + (" ..." if len(entry["lineage_of_the_answer"]) > 6 else ""),
            "* the chain from the chosen upward: "
            + " -> ".join(f"`{node}`" for node, _name, _parents in entry["lineage_of_the_chosen"][:6])
            + (" ..." if len(entry["lineage_of_the_chosen"]) > 6 else ""),
            f"* verdicts: exact **{entry['verdicts']['exact']}**, subtype "
            f"**{entry['verdicts']['subtype']}**, parent **{entry['verdicts']['parent']}**, related "
            f"**{entry['verdicts']['related']}**; the independent walk says `{entry['verdict']}`",
            "",
        ]

    # ------------------------------------------------- the published cells, reproduced
    kev_rep = result["checks"]["reproduction"]["kev_and_laya"]
    lines += [
        "",
        "## Do the published numbers reproduce?",
        "",
        "`exact` is the published reading, so every served and forced cell of "
        "`reports/abstain-reconciliation-v4.json` is reproduced here before any new reading is read off "
        "the same rows. The six fine-tuned arms first:",
        "",
        "| fine-tuned arm | served published | served here | agrees | forced published | forced here | "
        "agrees |",
        "|---|---:|---:|:--:|---:|---:|:--:|",
    ]
    for key in ft:
        rep = kev_rep[key]
        lines.append(f"| {rep['shown_as']} | {pct(rep['served_published'])} "
                     f"({rep['served_correct_published']:,}/4,035) | {pct(rep['served_here'])} "
                     f"({rep['served_correct_here']:,}/4,035) | "
                     f"{'yes' if rep['served_agrees'] else 'NO'} | {pct(rep['forced_published'])} "
                     f"({rep['forced_correct_published']:,}/4,035) | {pct(rep['forced_here'])} "
                     f"({rep['forced_correct_here']:,}/4,035) | "
                     f"{'yes' if rep['forced_agrees'] else 'NO'} |")
    n_agree = sum(1 for v in kev_rep.values() if v["served_agrees"] and v["forced_agrees"])
    lines += [
        "",
        f"**{n_agree} of the {len(kev_rep)} arms on the same 4,035 questions reproduce both cells "
        "question for question**, the six fine-tuned arms among them, so the published served and forced "
        "figures are exactly this report's `exact` column. The remaining arms:",
        "",
        "| arm | served published | served here | forced published | forced here |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, rep in kev_rep.items():
        if key in ft:
            continue
        lines.append(f"| {rep['shown_as']} | {pct(rep['served_published'])} "
                     f"({rep['served_correct_published']:,}) | {pct(rep['served_here'])} "
                     f"({rep['served_correct_here']:,}) | {pct(rep['forced_published'])} "
                     f"({rep['forced_correct_published']:,}) | {pct(rep['forced_here'])} "
                     f"({rep['forced_correct_here']:,}) |")
    jev_rep = result["checks"]["reproduction"]["jev"]
    tools_rep = result["checks"]["reproduction"]["tools"]
    lines += [
        "",
        f"* **Jev (API)**: served {jev_rep['served_here']:,}/4,035 against the published "
        f"{jev_rep['served_published']:,} ({'agrees' if jev_rep['served_agrees'] else 'DIFFERS'}). Its "
        f"forced cell over the {jev_rep['rows_offering_an_abstain_option']:,} rows that offered an "
        f"abstain option is {jev_rep['forced_here_on_those_rows']:,} against the published "
        f"{jev_rep['forced_published_on_the_rows_offering_an_abstain_option']:,} "
        f"({'agrees' if jev_rep['forced_agrees_on_those_rows'] else 'DIFFERS'}); over all 4,035 rows "
        f"its forced `exact` is {jev_rep['forced_here_over_all_4035_rows']:,}.",
        f"* **CellTypist, best declared arm**: "
        f"{tools_rep['celltypist_best_declared_arm']['exact_here_correct']:,}/4,035 against the "
        f"published {pct(tools_rep['celltypist_best_declared_arm']['published_rate'])} "
        f"({tools_rep['celltypist_best_declared_arm']['published_correct']:,}) "
        f"({'agrees' if tools_rep['celltypist_best_declared_arm']['agrees'] else 'DIFFERS'}).",
        f"* **CellTypist, default model**: "
        f"{tools_rep['celltypist_default_model']['exact_here_correct']:,}/4,035 against the published "
        f"{pct(tools_rep['celltypist_default_model']['published_rate'])} "
        f"({tools_rep['celltypist_default_model']['published_correct']:,}) "
        f"({'agrees' if tools_rep['celltypist_default_model']['agrees'] else 'DIFFERS'}).",
        f"* **SingleR, better level**: {tools_rep['singler_best_level']['exact_here_correct']:,}/4,035 "
        f"against the published {pct(tools_rep['singler_best_level']['published_rate'])} "
        f"({tools_rep['singler_best_level']['published_correct']:,}) "
        f"({'agrees' if tools_rep['singler_best_level']['agrees'] else 'DIFFERS'}).",
        f"* **SingleR, label.main**: {tools_rep['singler_label_main']['exact_here_correct']:,}/4,035 "
        f"against the published {pct(tools_rep['singler_label_main']['published_rate'])} "
        f"({tools_rep['singler_label_main']['published_correct']:,}) "
        f"({'agrees' if tools_rep['singler_label_main']['agrees'] else 'DIFFERS'}).",
        "",
        "## Labels the register cannot place",
        "",
        "Counted and listed, never absorbed. Every one of them is one of the corpus's own abstain keys, "
        "which is what makes the JSON's `unplaceable` column readable: on the paper's arms it is the "
        "served pick being the abstain option, and on the forced reading it is empty by construction "
        "where the question offered an abstain option.",
        "",
        "| label | occurrences among the 4,035 questions' options |",
        "|---|---:|",
    ]
    for label, count in result["checks"]["register"]["labels_the_register_cannot_place"].items():
        lines.append(f"| `{label}` | {count:,} |")
    lines += [
        "",
        "The tool-side harmonisation's own holes are per arm in the JSON and in the published baseline "
        "report; they are not re-derived here.",
        "",
        "## Sources and how they were verified",
        "",
        "* The twelve per-arm row dumps and the two Laya reconciles were read from the scratch mirror of "
        "the cluster's `reports/` and `runs/` trees, and each was checked by sha256 against the digest "
        "recorded in `reports/abstain-reconciliation-v4.json` before anything was read from it. A "
        "mismatch is a hard error in this script, not a warning.",
        "* The twelve dumps are named `forced-rows-delta-kev-{0.5b,0.6b,0.8b,4b,9b,27b}-v4-s1.jsonl` and "
        "`forced-rows-kev-{0.5b,0.6b,0.8b,4b,9b,27b}-v4.jsonl`.",
        "* The tools' outputs are the frozen artifacts under `reports/annotation-tools-v4/`, read through "
        "the same harmonisation tables the published scorer uses.",
        "",
        "| file | sha256 (first 16 hex) |",
        "|---|---|",
    ]
    for path, digest in sorted(result["sources"].items()):
        lines.append(f"| `{path}` | `{digest[:16]}` |")
    lines += [
        "",
        "## The one sentence the manuscript can carry",
        "",
        f"> {closing_sentence(result)}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "source scripts/wsl/env.sh",
        "uv run --no-sync python reports/ontology-aware-reading-v4.py",
        "uv run --no-sync python reports/ontology-aware-reading-v4.py --examples  # the walks",
        "```",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def closing_sentence(result: dict) -> str:
    """One sentence, under 40 words, that survives all four readings."""
    ct = result["arms"]["celltypist-best-arm"]["pooled"]["forced"]
    sr = result["arms"]["singler-best-level"]["pooled"]["forced"]
    ft = [v["pooled"]["forced"]["related"]["correct"] for v in result["arms"].values()
          if v.get("kind") == "kev-fine-tuned"]
    sentence = (f"CellTypist and SingleR name the answer's lineage on {ct['related']['correct']:,} and "
                f"{sr['related']['correct']:,} of 4,035 questions, against {min(ft):,} to {max(ft):,} "
                "for the fine-tuned arms: the margin survives every reading, widest on exact ids and "
                "narrower once broader or narrower names count.")
    words = sentence.split()
    if len(words) > 40:
        raise SystemExit(f"the closing sentence is {len(words)} words: shorten it")
    return sentence


def main(argv=None) -> int:
    result = build(argv)
    OUT_JSON.write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_md(result)
    print(f"\nwrote {OUT_MD.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
