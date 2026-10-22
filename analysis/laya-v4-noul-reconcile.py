#!/usr/bin/env python3
"""Re-aggregate a **laya** rows dump onto the v4 census's ``marker_noul`` family -- one row is one
**question**.

Why this is not ``reports/laya-v4-reconcile.py``
-----------------------------------------------

That script (and the settled join it imports) reads **one row per record**: the three ``choice``
families ask one question per record, so ``(id, family)`` is the unit and ``served`` /
``abstain-pick`` / ``forced`` are per-record readings. ``marker_noul`` is a different question *type*:
4,035 records asking **63,851 binary ``noul`` questions** (4-41 each), whose option keys the reference
fixes to ``["false","true"]`` and whose label is a boolean. Keying those rows by ``(id, family)``
keeps the **last** question of every record and silently scores 4,035 of 63,851 questions -- a number
that reads like a family result and is not one. So here the unit is the **question**, the key is
``(id, question)``, and the acceptance is the population's own question count (63,851 over 4,035
records), never the record count.

The reading is taken, not re-derived: ``reports/laya-finetune-reconcile.py`` -- the settled join for
this lane -- supplies ``argmax``, ``forced`` and ``read_jsonl`` by path, exactly as
``reports/laya-v4-reconcile.py`` does. ``argmax(p)`` is ``served``; ``forced`` is the argmax with the
abstain option removed, and on a yes/no question there is no abstain option to remove, so
``forced == served`` -- **asserted and reported here rather than assumed**, because that equality is
what would break first if a noul row ever carried an abstain key.

The dump it writes (``--dump``) is the noul lane's own shape: none of this file's own formatting is
involved. It goes through ``sckevtype.eval.run.write_noul_row_dump`` -- ``FULL_DUMP_FIELDS``, one JSON
object per question, the same writer the kev passes use -- so the laya halves are read by the **shared
reader** instead of being a private number. That reading is then *performed*, not assumed: the file
just written is read back through ``scripts/noul-accuracy.py``'s own ``read_dump``/``summarise``/
``dump_path`` (imported by path, as the tests do) and the two sets of numbers must agree, with the
file's name pinned to the aggregator's own key for the arm. A disagreement is a refusal.

The arm's dump name is **composed, never typed in**, and ``--corpus`` is which census it belongs to
------------------------------------------------------------------------------

The aggregator keys an arm by ``forced-rows-<slug><suffix>.jsonl``, so the join derives that name from
the arm rather than accepting a path and hoping it matches. Two censuses write noul dumps and they do
**not** use the same name: the v4 lane writes ``forced-rows-<slug>-marker_noul.jsonl``, and the Baron
external-validation lane puts a ``-baron`` infix between the slug and the family
(``forced-rows-<slug>-baron-marker_noul.jsonl`` -- ``<cluster-job>`` names
every one of that lane's dumps that way). Deriving from the slug alone made the Baron lane's arms read
as **absent** while their dumps sat on disk complete, which is the failure this option exists to close;
the old guard refused rather than emitting the table, and it was right to.

So ``--corpus {v4,baron}`` selects the name rule, and the arm key the aggregator is asked for is
``<label>`` or ``<label>-baron`` accordingly. ``v4`` is the default and reproduces the previous
behaviour exactly. **A dump is never renamed or symlinked into the other census's name**: the v4 family
table reads ``reports/forced-rows-laya-<half>-marker_noul.jsonl``, so renaming a Baron dump onto that
path would silently replace the paper's v4 noul numbers with a different population's.

Usage::

    uv run python reports/laya-v4-noul-reconcile.py --rows runs/laya-v4-census-spec-marker_noul/rows \\
        --label laya-v4-ft-spec --out runs/laya-v4-census-spec-marker_noul/reconcile.json \\
        --dump reports/forced-rows-laya-v4-ft-spec-marker_noul.jsonl

    # the same arm on the Baron external-validation census: arm key laya-base-baron, expectation 39,197
    uv run python reports/laya-v4-noul-reconcile.py --corpus baron \\
        --rows runs/laya-baron-census-base-marker_noul/rows --label laya-base \\
        --expect-questions 39197 --out runs/laya-baron-census-base-marker_noul/reconcile.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CI = 0.95

#: The settled join for this lane: ``argmax``, ``forced``, ``read_jsonl`` and ``sha256_of`` come from
#: it, so a noul row and a ``panel_choice`` row cannot be read by two different rules.
SETTLED = ROOT / "analysis" / "laya-finetune-reconcile.py"
#: The shared noul aggregator. Used as a **reader** here (its module body has no side effects): the
#: dump this script writes is checked against the numbers that file would report from the same bytes.
AGGREGATOR = ROOT / "scripts" / "noul-accuracy.py"

DEFAULT_IDS = "reports/census-suite-ids-v4.jsonl"
DEFAULT_FAMILY = "marker_noul"
#: The reference's fixed option order for a yes/no question (``kev.api.question_keys``): index 0 is
#: *no*, index 1 is *yes*. A row that disagrees is refused by name rather than scored on two columns
#: that are not the ones the labels were written against.
NOUL_KEYS = ["false", "true"]

#: The census a dump belongs to, and the infix that census puts between the arm slug and the family.
#: ``v4`` writes ``forced-rows-<slug><suffix>.jsonl``; the Baron external-validation lane writes
#: ``forced-rows-<slug>-baron<suffix>.jsonl``. The infix is a **selector**, not something to smuggle in
#: through ``--label``: the aggregator keys an arm by its slug, so a hand-written ``laya-base-baron``
#: would compose the right name only by coincidence, and a typo either side of the dash reads as an
#: absent family rather than as a mistake.
DEFAULT_CORPUS = "v4"
CORPUS_INFIX = {"v4": "", "baron": "-baron"}

#: Each census's own ``marker_noul`` question count, used when ``--expect-questions`` is not given.
#: The acceptance is a *question* count and it differs per census: 63,851 over the v4 census's 4,035
#: records, 39,197 over the Baron suite's 2,600. Carrying v4's number into a Baron run would refuse a
#: complete dump as a prefix of a population it was never the prefix of.
EXPECTED_QUESTIONS = {"v4": 63851, "baron": 39197}


class ReconcileError(RuntimeError):
    """Something this join cannot score, named rather than defaulted."""


def module_from(path: Path, name: str, required: tuple[str, ...]):
    """A script in this repo, loaded by path, refusing by name when its API has moved."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReconcileError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [entry for entry in required if not hasattr(module, entry)]
    if missing:
        raise ReconcileError(f"{path} no longer defines {missing}; the rule this join applies has "
                            f"moved, and applying a copy of it would be a different measurement")
    return module


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="a laya rows directory (or a glob of .jsonl files)")
    parser.add_argument("--ids", default=DEFAULT_IDS, help=f"the population file (default {DEFAULT_IDS})")
    parser.add_argument("--family", default=DEFAULT_FAMILY, help="the population's family name")
    parser.add_argument("--label", default="laya", help="the arm slug (the aggregator's own key)")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=sorted(CORPUS_INFIX),
                        help="which census's dump-name rule to compose: 'v4' (default) is "
                             "forced-rows-<label><suffix>.jsonl, 'baron' inserts -baron between them "
                             "(forced-rows-<label>-baron<suffix>.jsonl), which is how the external "
                             "validation lane names its dumps; the arm key the aggregator is asked for "
                             "is <label> or <label>-baron accordingly")
    parser.add_argument("--out", default=None, help="the reconcile JSON to write")
    parser.add_argument("--dump", default=None,
                        help="write the rows as the noul lane's dump (the aggregator-readable shape)")
    parser.add_argument("--suffix", default="-marker_noul",
                        help="the dump name's trailing family tag (start it with a dash)")
    parser.add_argument("--expect-questions", type=int, default=None,
                        help="the population's own question count (the acceptance; 0 = whatever is on "
                             "disk, which is only right when nothing needs gating). Defaults to the "
                             f"selected census's own count: {EXPECTED_QUESTIONS}")
    parser.add_argument("--allow-missing", action="store_true",
                        help="report a population that is not fully covered instead of refusing; a "
                             "partial population is then NOT written out as a dump")
    args = parser.parse_args(argv)

    # The arm key is the *aggregator's* key, so the census infix is part of it: a Baron dump is
    # ``forced-rows-laya-base-baron-marker_noul.jsonl`` and the arm the shared table looks up is
    # ``laya-base-baron``. Composing it here (rather than letting a caller write ``--label
    # laya-base-baron``) is what keeps the v4 path byte-identical and refuses the double infix.
    infix = CORPUS_INFIX[args.corpus]
    if infix and args.label.endswith(infix):
        print(f"--label {args.label!r} already ends with the {args.corpus!r} corpus infix {infix!r}; "
              f"pass the arm slug without it (the infix is composed from --corpus) -- refusing",
              file=sys.stderr)
        return 2
    arm_key = f"{args.label}{infix}"
    expect_questions = (args.expect_questions if args.expect_questions is not None
                        else EXPECTED_QUESTIONS[args.corpus])

    settled = module_from(SETTLED, "laya_finetune_reconcile",
                          ("argmax", "forced", "read_jsonl", "sha256_of"))
    read_jsonl, argmax, forced = settled.read_jsonl, settled.argmax, settled.forced

    if os.path.isdir(args.rows):
        files = sorted(glob.glob(os.path.join(args.rows, "*.jsonl")))
    else:
        files = sorted(glob.glob(args.rows))
    if not files:
        print(f"no rows files under {args.rows}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ the population
    ids_path = Path(args.ids)
    if not ids_path.is_absolute():
        ids_path = ROOT / ids_path
    if not ids_path.is_file():
        print(f"no population file at {ids_path}", file=sys.stderr)
        return 2
    population = [row for row in read_jsonl(ids_path) if row.get("family") == args.family]
    records = {row["id"] for row in population}
    if not population or len(records) != len(population):
        print(f"{ids_path}: {len(population)} {args.family!r} rows, {len(records)} distinct ids -- a "
              f"repeated id would be counted twice; refusing", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ the rows
    answers: dict[tuple[str, str], dict] = {}
    skipped_ask = 0
    outside = Counter()
    for path in files:
        for number, row in enumerate(read_jsonl(Path(path)), start=1):
            where = f"{path}:{number}"
            if row.get("type") != "noul":
                raise ReconcileError(f"{where} is not a noul row (type {row.get('type')!r}); this join "
                                     f"scores yes/no questions and one row is one question")
            if row.get("ask", "base") != "base":
                skipped_ask += 1
                continue
            if list(row.get("keys") or []) != NOUL_KEYS:
                raise ReconcileError(f"{where} carries keys {row.get('keys')!r}, not {NOUL_KEYS} -- the "
                                     f"reference's own option order for a yes/no question")
            probabilities = row.get("p") or []
            if len(probabilities) != 2:
                raise ReconcileError(f"{where} carries {len(probabilities)} probabilities, not 2")
            if int(row.get("label", -1)) not in (0, 1):
                raise ReconcileError(f"{where} carries label {row.get('label')!r}, not 0/1")
            if row["id"] not in records:
                outside[row["id"]] += 1
                continue
            key = (row["id"], row["question"])
            if key in answers:
                raise ReconcileError(f"{where} repeats {key}, which would weight one question twice")
            answers[key] = row
    if outside:
        print(f"{sum(outside.values())} rows name {len(outside)} record(s) outside {args.ids}'s "
              f"{args.family!r} population (e.g. {sorted(outside)[:3]}); the denominator would not be "
              f"the population -- refusing", file=sys.stderr)
        return 2
    if not answers:
        print(f"{args.rows} holds no noul answers for {args.family!r}; refusing", file=sys.stderr)
        return 2

    questions = len(answers)
    covered = len({record for record, _ in answers})
    missing_questions = max(expect_questions - questions, 0) if expect_questions else 0
    missing_records = len(records) - covered
    complete = not missing_questions and not missing_records
    if not complete and not args.allow_missing:
        print(f"the rows cover {questions} of {expect_questions} questions over {covered} of "
              f"{len(records)} records; the pass writes its rows as it scores, so this is a prefix of a "
              f"run still in flight -- a number over a prefix is not the family's number. Pass "
              f"--allow-missing to report it as such (no dump is written then).", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ the reading
    served_correct = answered_yes = reference_yes = forced_correct = 0
    forced_differs = 0
    p_yes: list[float] = []
    p_label: list[float] = []
    per_record = Counter()
    per_record_correct = Counter()
    for key in sorted(answers):
        row = answers[key]
        pick = argmax(row["p"])
        label = int(row["label"])
        served_correct += int(pick == label)
        answered_yes += int(pick == 1)
        reference_yes += label
        hidden = forced(row)
        forced_correct += int(hidden == label)
        forced_differs += int(hidden != pick)
        p_yes.append(float(row["p"][1]))
        p_label.append(float(row["p"][label]))
        per_record[key[0]] += 1
        per_record_correct[key[0]] += int(pick == label)

    accuracy = served_correct / questions
    reference_yes_rate = reference_yes / questions
    floor = max(reference_yes_rate, 1.0 - reference_yes_rate)
    degenerate = reference_yes_rate in (0.0, 1.0)
    counts = sorted(per_record.values())
    result = {
        "what_this_is": f"A laya rows dump ({args.rows}) re-aggregated onto {args.ids}'s "
                        f"{len(population)} {args.family} records, **by question**: one row is one "
                        f"yes/no question and the unit of every number below is the question. The "
                        f"reading (served = argmax(p), forced = argmax(p) without the abstain option) is "
                        f"imported from {SETTLED.name}; the option order is the reference's own "
                        f"{NOUL_KEYS}.",
        "label": arm_key,
        "unit": "questions (a marker_noul record asks 4-41 of them)",
        "rows_source": {"glob": args.rows, "files": len(files), "answers": len(answers),
                        "rows_skipped_not_the_base_ask": skipped_ask},
        "population": {"source": args.ids, "family": args.family, "records": len(records),
                       "records_covered": covered, "questions_expected": expect_questions},
        "acceptance": {
            "unit": "questions", "questions": questions, "records": covered,
            "expected_questions": expect_questions, "expected_records": len(records),
            "passed": complete,
            "note": "a dump holding one row per record would be 4,035 questions and would have scored "
                    "1/15.8 of this family; the acceptance is the suite's own question count",
        },
        "coverage": {"questions_matched": questions, "missing_questions": missing_questions,
                     "records_matched": covered, "missing_records": missing_records,
                     "complete": complete},
        f"noul_questions_{questions}": {
            "n": questions,
            "records": covered,
            "served": {"correct": served_correct, "accuracy": accuracy},
            "says_yes": {"count": answered_yes, "rate": answered_yes / questions},
            "reference": {"yes": reference_yes, "yes_rate": reference_yes_rate},
            "majority_floor": floor,
            "margin_over_floor": None if degenerate else accuracy - floor,
            "degenerate": degenerate,
            "forced": {"correct": forced_correct, "accuracy": forced_correct / questions,
                       "note": "identical to served by construction: a yes/no question offers no "
                               "abstain option to remove, and the count of rows where the two readings "
                               "disagree is reported beside it"},
            "forced_differs_from_served": forced_differs,
            "diagnostics": {
                "mean_p_yes": sum(p_yes) / len(p_yes),
                "median_p_yes": percentile(p_yes, 0.5),
                "p90_p_yes": percentile(p_yes, 0.9),
                "answers_yes_over_half": sum(1 for value in p_yes if value > 0.5),
                "mean_p_of_the_reference": sum(p_label) / len(p_label),
                "median_p_of_the_reference": percentile(p_label, 0.5),
                "questions_per_record": {"min": counts[0] if counts else None,
                                         "median": percentile([float(v) for v in counts], 0.5),
                                         "max": counts[-1] if counts else None},
                "records_with_every_question_correct": sum(1 for record in per_record
                                                           if per_record_correct[record] == per_record[record]),
            },
        },
        "settled_rules_from": {"path": str(SETTLED.relative_to(ROOT)),
                               "sha256": settled.sha256_of([SETTLED])[str(SETTLED)]},
        "sha256": settled.sha256_of([Path(path) for path in files]),
    }
    cell = result[f"noul_questions_{questions}"]

    # Only the non-default census is recorded: with `--corpus` unused the JSON this script writes is
    # byte-for-byte what it wrote before the selector existed, which is what makes the v4 reading
    # provably untouched by this option.
    if args.corpus != DEFAULT_CORPUS:
        result["corpus"] = {
            "selector": args.corpus, "arm_key": arm_key, "label_requested": args.label,
            "dump_name_rule": f"forced-rows-{arm_key}{args.suffix}.jsonl",
            "expect_questions_from": "the corpus default" if args.expect_questions is None
                                     else "--expect-questions",
            "note": f"a {args.corpus} dump carries the {'-baron'!r} infix between arm and family; the "
                    f"arm key the shared aggregator looks up is {arm_key!r}, and no dump was renamed "
                    f"or linked into the other census's name",
        }

    # ------------------------------------------------------------------ the dump, and reading it back
    if args.dump and not complete:
        # Not an error with `--allow-missing` (the caller explicitly asked for a partial report), but
        # never a partial dump: a file under the aggregator's own name is read as a finished
        # population, and this one would be a prefix of it.
        print(f"NOTE no dump was written: the population is {missing_questions} question(s) and "
              f"{missing_records} record(s) short, and a partial dump under the lane's own name "
              f"cannot be told from a finished one")
        args.dump = None
    if args.dump:
        from sckevtype.eval.run import write_noul_row_dump

        dump_path = Path(args.dump)
        if not dump_path.is_absolute():
            dump_path = ROOT / dump_path
        info = write_noul_row_dump(dump_path, list(answers.values()))
        result["dump"] = {"path": str(dump_path), "rows": info["rows"], "sha256": info["sha256"],
                          "skipped": info["skipped"], "fields": "FULL_DUMP_FIELDS (17 fields)",
                          "writer": "sckevtype.eval.run.write_noul_row_dump"}

        # The shared reader, on the bytes just written. Both the numbers and the **name** are checked:
        # the aggregator keys an arm by `forced-rows-<slug><suffix>.jsonl`, so a dump under any other
        # name is a number the shared table would never look up.
        agg = module_from(AGGREGATOR, "noul_accuracy",
                          ("read_dump", "summarise", "dump_path", "arm_from_dump"))
        expected = agg.dump_path(dump_path.parent, arm_key, args.suffix)
        if expected != dump_path:
            raise ReconcileError(f"the dump is at {dump_path.name} but the aggregator looks for "
                                 f"{expected.name} when the arm is {arm_key!r}: the key would never "
                                 f"be looked up, and the family would read as absent")
        read_back = agg.summarise(arm_key, dump_path, agg.read_dump(dump_path))
        fields = ("questions", "records", "correct", "answered_yes", "majority_floor")
        agree = (read_back["questions"] == questions and read_back["records"] == covered
                 and read_back["correct"] == served_correct
                 and read_back["answered_yes"] == answered_yes
                 and abs(read_back["majority_floor"] - floor) < 1e-12)
        result["aggregator_check"] = {
            "reader": str(AGGREGATOR.relative_to(ROOT)), "arm_key": arm_key,
            "dump_path_the_aggregator_looks_for": str(expected), "agrees": agree,
            "agreed_fields": list(fields), "portrait": {name: read_back[name] for name in fields},
            "accuracy_from_the_aggregator": read_back["accuracy"],
            "note": "the file just written is read back through the shared reader; a disagreement is a "
                    "refusal, not a quieter row",
        }
        if not agree:
            raise ReconcileError(f"the reconcile and {AGGREGATOR.name} disagree on the same bytes: "
                                 f"{ {name: read_back[name] for name in fields} }")

    # ------------------------------------------------------------------ print
    margin = "n/a (degenerate)" if cell["margin_over_floor"] is None else f"{cell['margin_over_floor']:+.4f}"
    print(f"[{arm_key}]  {args.family} by QUESTION   n={questions} questions over {covered} records"
          f"{'' if complete else '  (INCOMPLETE)'}")
    print(f"  accuracy {cell['served']['correct']}/{questions} = {cell['served']['accuracy']:.4f}   "
          f"floor {cell['majority_floor']:.4f}   margin {margin}")
    print(f"  says yes {cell['says_yes']['count']}/{questions} = {cell['says_yes']['rate']:.4f}   "
          f"reference yes {cell['reference']['yes_rate']:.4f}   "
          f"mean P(yes) {cell['diagnostics']['mean_p_yes']:.4f}")
    print(f"  forced {cell['forced']['correct']}/{questions} = {cell['forced']['accuracy']:.4f} "
          f"(differs from served on {cell['forced_differs_from_served']} rows -- a yes/no question "
          f"offers no abstain option)")
    print(f"  questions/record: min {cell['diagnostics']['questions_per_record']['min']} "
          f"median {cell['diagnostics']['questions_per_record']['median']} "
          f"max {cell['diagnostics']['questions_per_record']['max']}   "
          f"records all-correct {cell['diagnostics']['records_with_every_question_correct']}")
    if skipped_ask:
        print(f"  NOTE {skipped_ask} row(s) carried an ask other than 'base' and were not scored")
    if "aggregator_check" in result:
        check = result["aggregator_check"]
        print(f"  aggregator {check['reader']} reads it back as arm {check['arm_key']}: "
              f"{'AGREES' if check['agrees'] else 'DISAGREES'} on "
              + ", ".join(f"{name}={check['portrait'][name]}" for name in check["agreed_fields"]))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".partial")
        tmp.write_text(json.dumps(result, indent=1) + "\n")
        os.replace(tmp, out)
        print(f"wrote {out}")
    print("LAYA_NOUL_RECONCILE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
