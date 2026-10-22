#!/usr/bin/env python3
"""Score **Jev's** noul rows on the v4 census population -- the sibling of ``score-census-v4.py``.

Why this file exists instead of a run of ``score-census-v4.py``. That script scores *choice* rows:
its rule chain has a forced-answer half and an always-abstain floor, and both are defined over the rows
that offer an **abstain option**. A noul question's ballot is the API's fixed ``["false","true"]`` with
no abstain position at all, so on this family every row would be counted as *excluded*, the eligible
denominator would be 0, and the forced / floor / delta columns would all be undefined -- the script
would in fact crash trying to format them. It also joins on ``(dataset, cell, family)``, which for
``marker_noul`` is a **record**, so it would keep one row of the 15.8 the record carries and report the
record count as if it were the question count. The noul family's unit is the question, and its own
instrument already exists.

So the *number-producing function is not re-implemented*: :func:`summarise` (and the ``sha256_file`` it
calls) is lifted **from ``scripts/noul-accuracy.py``'s source** by ``ast`` -- the same device that script
uses for ``binom_two_sided`` -- and applied here to Jev's rows. Whatever the Kev arms' noul column is
computed by is what computes Jev's, so the two are one measurement rather than two harnesses that agree
today. The rule it applies is the census scoring's own served rule (``argmax(p) == label``, which is the
same rule ``sckevtype.eval.metrics.top1_accuracy`` and ``score-census-v4.py``'s ``served`` column use),
taken against the reference's fixed ``["false","true"]`` keys.

Three checks run on every invocation and each can fail:

* **coverage** -- every ``marker_noul`` triple of the id list must have rows in the dump, and the rows'
  question count must equal ``--expect-questions`` (default 63,851, the count the ten Kev noul jobs are
  accepted on). A dump that is short is not the family's number;
* **shape** -- every selected row must be a noul row with keys ``["false","true"]``, two probabilities
  and a 0/1 label, and no ``(id, question)`` pair may repeat (a repeat would weight one question twice);
* **two implementations agree** -- the accuracy is recomputed here by a second, independent argmax loop
  and must match the lifted ``summarise`` exactly. Disagreement is a refusal, not a rounding note.

    uv run --no-sync python scripts/jev/score-noul-v4.py \
        --dump reports/jev-v4/rows.jsonl \
        --ids reports/census-suite-ids-v4.jsonl \
        --out reports/jev-v4/census-score-noul.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:                      # the package lives in the repo, not in the venv
    sys.path.insert(0, str(ROOT))

from sckevtype.contracts.freeze import file_digest, write_json

NOUL_KEYS = ["false", "true"]
FAMILY = "marker_noul"
#: The noul lane's own acceptance: ``marker_noul`` is 4,035 records carrying this many yes/no questions
#: (``reports/v4-noul.md``'s preflight line, and what ``scripts/noul-accuracy.py``'s dumps are held to).
EXPECT_QUESTIONS = 63851
#: The instrument whose number-producing function this script applies, by path -- named so a reader can
#: check that the two numbers really do come from one function.
INSTRUMENT = ROOT / "scripts" / "noul-accuracy.py"


def lift(path: Path, names: tuple[str, ...]) -> dict:
    """The named function definitions taken from another script's **source**, never from a copy."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    if missing:
        raise SystemExit(f"REFUSING: {path} no longer defines {missing} -- the function this script "
                         f"reports with has moved, and a copy of it would be a second instrument")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[*imports, *[definitions[name] for name in names]], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, filename=str(path), mode="exec"), namespace)      # noqa: S102 - the project's own source
    return namespace


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify(row: dict) -> str | None:
    """Why this row cannot be scored as a noul question, or ``None`` if it can."""
    if row.get("type") != "noul":
        return None                        # a choice row: not this family, and not an error
    if list(row.get("keys") or []) != NOUL_KEYS:
        return f"keys {row.get('keys')!r}, not {NOUL_KEYS}"
    if len(row.get("p") or []) != 2:
        return f"{len(row.get('p') or [])} probabilities, not 2"
    if int(row.get("label", -1)) not in (0, 1):
        return f"label {row.get('label')!r}, not 0/1"
    return None


def argmax_index(values: list[float]) -> int:
    """The census scoring's own argmax: the first maximum, no numpy, no tie-breaking surprises."""
    best = 0
    for index, value in enumerate(values):
        if value > values[best]:
            best = index
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="Jev's row dump (rows.jsonl, all four families)")
    ap.add_argument("--ids", required=True, help="the census id list naming the population")
    ap.add_argument("--family", default=FAMILY)
    ap.add_argument("--expect-questions", type=int, default=EXPECT_QUESTIONS,
                    help=f"the family's question count the dump is accepted on (default {EXPECT_QUESTIONS})")
    ap.add_argument("--out", default=None, help="JSON to write")
    args = ap.parse_args()

    dump_path, ids_path = Path(args.dump), Path(args.ids)
    for path in (dump_path, ids_path):
        if not path.exists():
            raise SystemExit(f"REFUSING: {path} does not exist")
    instrument = lift(INSTRUMENT, ("sha256_file", "summarise"))

    ids = read_jsonl(ids_path)
    population = [row for row in ids if row["family"] == args.family]
    if not population:
        raise SystemExit(f"REFUSING: {ids_path} names no {args.family} rows -- nothing to score")
    triples = {(row["dataset"], row["cell"], row["family"]) for row in population}
    if len(triples) != len(population):
        raise SystemExit(f"REFUSING: {ids_path} names a {args.family} triple twice")

    rows: dict[tuple[str, str], dict] = {}
    problems: list[str] = []
    seen_type: Counter = Counter()
    not_in_population = 0
    for number, row in enumerate(read_jsonl(dump_path), start=1):
        seen_type[str(row.get("type"))] += 1
        complaint = classify(row)
        if complaint is not None:
            problems.append(f"{dump_path}:{number} id={row.get('id')!r}: {complaint}")
            continue
        if row.get("type") != "noul":
            continue
        triple = (str(row.get("dataset")), str(row.get("cell")), str(row.get("family")))
        if triple not in triples:
            not_in_population += 1                  # a noul record outside the census: counted, not scored
            continue
        key = (row["id"], row["question"])
        if key in rows:
            problems.append(f"{dump_path}:{number} repeats {key}, which would weight it twice")
            continue
        rows[key] = row

    if problems:
        print(f"REFUSING: {len(problems)} row-level problem(s) in {dump_path}, e.g.")
        for line in problems[:5]:
            print(f"  {line}")
        return 2

    scored_triples = {(r["dataset"], r["cell"], r["family"]) for r in rows.values()}
    missing = sorted(f"{d}/{c}" for d, c, _ in (triples - scored_triples))
    if missing:
        print(f"REFUSING: {len(missing)} of the id list's {len(triples)} {args.family} records have no row "
              f"in {dump_path}, e.g.")
        for name in missing[:5]:
            print(f"  {name}")
        return 3
    if len(rows) != args.expect_questions:
        print(f"REFUSING: {dump_path} holds {len(rows)} {args.family} questions over the census "
              f"population, not the {args.expect_questions} this family is accepted on "
              f"({len(scored_triples)} records, {not_in_population} noul rows outside the population)")
        return 4

    summary = instrument["summarise"]("jev-v4", dump_path, rows)
    # the independent second read of the same rule: the census scoring's served rule, closed over here.
    correct = sum(1 for row in rows.values() if argmax_index(list(row["p"])) == int(row["label"]))
    agree = abs(correct / len(rows) - summary["accuracy"]) < 1e-12
    if not agree:
        print("REFUSING: the lifted summarise() and this script's own argmax disagree on the same rows:")
        print(f"  summarise : {summary['accuracy']} ({summary['correct']}/{summary['questions']})")
        print(f"  this file : {correct / len(rows)} ({correct}/{len(rows)})")
        return 5

    per_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows.values():
        per_dataset[str(row["dataset"])].append(row)
    by_dataset = {name: instrument["summarise"](f"jev-v4: {name}", dump_path, dict(
        zip(((r["id"], r["question"]) for r in group), group))) for name, group in sorted(per_dataset.items())}

    result = {
        "what_this_is": "Jev's noul (binary marker) column on the v4 census population: one row per "
                        "question, argmax(p) == label against the reference's keys ['false','true'].",
        "unit": f"questions (not records): {len(scored_triples)} census records carrying {len(rows)} questions",
        "instrument": {
            "rule": "argmax(p) == label  (the census scoring's served rule, sckevtype.eval.metrics.top1_accuracy)",
            "summarise_lifted_from": str(INSTRUMENT.relative_to(ROOT)),
            "instrument_sha256": file_digest(INSTRUMENT),
            "checks": {
                "coverage": "ok: every marker_noul triple of the id list has rows; nothing missing",
                "shape": "ok: every scored row is noul with keys ['false','true'], 2 probabilities, a 0/1 label",
                "duplicate_pairs": "ok: no (id, question) pair repeats",
                "question_count": f"ok: {len(rows)} == the {args.expect_questions} the noul lane accepts on",
                "two_implementations_agree": f"ok: accuracy {summary['accuracy']!r} from both",
            },
        },
        "dump": {"path": str(dump_path), "rows": sum(seen_type.values()), "sha256": file_digest(dump_path),
                 "rows_by_type": dict(sorted(seen_type.items())),
                 "noul_rows_scored": len(rows), "noul_rows_outside_the_population": not_in_population},
        "ids": {"path": str(ids_path), "rows": len(ids), "sha256": file_digest(ids_path),
                "family_rows": len(population)},
        "family_filter": args.family,
        "expected_questions": args.expect_questions,
        "primary": summary,
        "by_dataset": by_dataset,
    }
    print(f"dump        : {dump_path}  ({result['dump']['rows']} rows, {str(result['dump']['sha256'])[:16]})")
    print(f"rows/type   : {result['dump']['rows_by_type']}")
    print(f"population  : {len(scored_triples)} records, {len(rows)} questions, "
          f"{not_in_population} noul rows outside it (not scored)")
    print(f"checks      : coverage ok, shape ok, no duplicate pairs, "
          f"count == {args.expect_questions}, two implementations agree")
    print(f"accuracy    : {summary['correct']}/{summary['questions']} = {summary['accuracy']:.4f}"
          f"   (argmax p against the reference's ['false','true'])")
    print(f"floor       : majority-class {summary['majority_floor']:.4f}"
          f"   margin {summary['margin_over_floor']:+.4f}")
    print(f"says yes    : {summary['answered_yes_rate']:.4f}  against the reference's "
          f"{summary['reference_yes_rate']:.4f}   mean P(yes) {summary['mean_p_yes']:.4f}")
    for name, entry in by_dataset.items():
        print(f"    dataset {name:<20} n={entry['questions']:<6d} accuracy {entry['accuracy']:.4f}"
              f"  says-yes {entry['answered_yes_rate']:.4f}  floor {entry['majority_floor']:.4f}")
    if args.out:
        write_json(Path(args.out), result)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
