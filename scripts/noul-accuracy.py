#!/usr/bin/env python3
"""The noul column: binary marker-absence accuracy for the fine-tuned arms and their released rungs.

What this answers
-----------------

The v4 census carries four families. Three are ``choice`` questions and go through the forced-naming
lane; the fourth, ``marker_noul``, is 4,035 records asking **63,851 yes/no questions** (4-41 per
record) — "does this cell satisfy the candidate's definition?" — whose reference answers are booleans
and whose option keys the reference fixes to ``["false","true"]``. It is the one family with no
scored column, because the choice writer skipped every non-choice row and wrote an empty dump.

The rows this reads are the noul dumps ``scripts/score-forced-rows.py --question-type noul`` writes:
one line per **question**, in the full dump's own 17 fields, so a noul dump is read with the same
reader as a choice dump. One row is one question, which is the unit every number below is taken on —
the acceptance for the pass is the suite's own count over the same records, not the record count.
Because the row shape *is* the choice dump's shape, it carries no ``type`` field: a noul row is
recognised by the reference's own fixed keys ``["false","true"]``, two probabilities and a boolean
label, and the round trip "write with the writer, read with this reader" is pinned by
``tests/analysis/test_noul_accuracy.py``.

What it reports, per arm: the question count and the record count, accuracy (``argmax(p) == label``,
the same rule :func:`sckevtype.eval.metrics.top1_accuracy` applies), the share of questions the model
answered *yes* beside the reference's own yes-rate, and **the majority-class floor** — the accuracy of
always giving the more common answer — because an accuracy quoted without its floor is the mistake
this project has already made once on this family's sibling tables. A family whose reference answers
are all one value is marked degenerate and not scored.

Then the comparison the ask is about: for each size, the **fine-tune against the rung it started
from**, on the questions both answered. Our model is deterministic, so the comparison is *paired* —
the two arms answer the same 63,851 questions — and what it tests is the discordant pairs, with the
project's own exact two-sided binomial (``reports/abstain-reconciliation.py``'s ``binom_two_sided``,
the test ``reports/PAIRED-TESTS.md`` uses, imported from that file's source rather than re-derived).

The roster carries the same comparison for **Laya**, the independent sibling engine, whose two v4
census halves are its fine-tune (``laya-v4-ft-spec``) and the released base it started from
(``laya-base``). Their dumps are written by that lane's own reconcile
(``reports/laya-v4-noul-reconcile.py``, one row per question) through this lane's writer, so the rows
read here are the same shape as every kev dump's. One thing those two halves do **not** share is the
render geometry — the fine-tune answers at 8192/2048 and the released base at its shipped 1024/256 —
and the pair's ``note`` says so beside its row rather than leaving it to be inferred.

**A paired view keys its arms by slug, never by filename.** ``forced-rows-<slug><suffix>.jsonl`` is
turned back into ``<slug>`` by :func:`arm_from_dump` (the lane's own prefix and the family suffix
stripped **by name**), so the roster's names — ``delta-kev-0.8b-v4-s1`` / ``kev-0.8b-v4`` — are what
the pairing looks up. A key that kept the ``forced-rows-`` prefix missed on *both* sides and reported
two complete dumps as an untestable pair, which reads exactly like a missing dump; the round trip and
the end-to-end pairing are pinned in ``tests/analysis/test_noul_accuracy.py``. An arm whose other half
is still in flight is reported **as a pair** that is not testable yet, with each side's state, so a
dump that exists is never printed as absent.

The exact binomial is a float, and the noul family's discordant counts run into the tens of thousands:
a lopsided pair lands below what a float can hold (``5e-324``) and comes back ``0.0`` — correctly
rounded, and unreadable in a table. Such a ``p`` is reported beside its ``log10`` magnitude, which is
the same quantity taken from the same file (``reports/abstain-reconciliation.py``'s
``binom_two_sided_log10``, whose log-space arithmetic has no power of two as a float).

Usage
-----

::

    uv run --no-sync python scripts/noul-accuracy.py                  # every v4 arm that has a dump
    uv run --no-sync python scripts/noul-accuracy.py --suffix=-marker_noulv4

``--suffix`` takes a value that begins with a dash, so an argparse quirk applies: pass it as
``--suffix=-marker_noul`` (the ``=`` form). ``--suffix -marker_noul`` is read as an option, not a value,
and the run dies on ``expected one argument`` — the mistake this example exists to prevent.

``--expect-questions N`` is the pass's own acceptance (the suite's question count over the same records,
63,851 for the v4 census's ``marker_noul``) applied to the *dump*: a pass writes its dump as it scores,
so a file with fewer rows is a **prefix** — a real, correctly written dump of an unfinished run. A number
over a prefix is not the family's number, so those dumps are listed as ``incomplete`` with their counts
and are **not** scored. Without the flag every dump present is scored, which is only right for a
directory holding finished passes.

Reads only dumps already on disk; runs no model and submits no job. Writes
``reports/noul-accuracy-v4.json`` and ``reports/noul-accuracy-v4.md``. Every number is a function of
the bytes read, so re-running it on the same dumps reproduces the same report hash.
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

#: The reference's fixed keys for a noul question, in option order (``kev.api.question_keys``): index
#: 0 is *no*, index 1 is *yes*. A dump whose rows disagree with this is refused rather than scored.
NOUL_KEYS = ["false", "true"]

#: The kinds of disagreement a dump can have with this reader, each a refusal by name.
NOT_NOUL = "not a noul row"

#: The arms this report compares, as ``(key, fine-tune slug, released-rung slug)``. The kev slugs are the
#: ones the v4 lane writes (``armen`` of ``submit-v4-ladder.sh``): a fine-tune by its run directory, a
#: released rung as ``kev-<size>-v4`` through the offline HF cache. Both halves of a pair answer the
#: **same** census records, so a paired comparison between them is legitimate and a pooled one is not.
#: The key is the size for a kev pair and the engine for the Laya pair, whose two halves are the
#: fine-tune and the **released base it started from** -- the same comparison, one rung down.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("0.5b", "delta-kev-0.5b-v4-s1", "kev-0.5b-v4"),
    ("0.6b", "delta-kev-0.6b-v4-s1", "kev-0.6b-v4"),
    ("0.8b", "delta-kev-0.8b-v4-s1", "kev-0.8b-v4"),
    ("4b", "delta-kev-4b-v4-s1", "kev-4b-v4"),
    ("9b", "delta-kev-9b-v4-s1", "kev-9b-v4"),
    # The 27B pair landed last: its two v4 ``marker_noul`` dumps were still being written when the
    # first version of this roster was drawn up, so the table carried five sizes and the ladder
    # ended at 9B. Both halves are on disk and complete now (63,851 questions each), so the pair
    # joins the roster in parameter order -- after 9B, before the sibling engine, which is not a
    # rung of this ladder.
    ("27b", "delta-kev-27b-v4-s1", "kev-27b-v4"),
    ("laya", "laya-v4-ft-spec", "laya-base"),
)

#: What a pair's two halves do **not** share, stated where the pair is printed rather than left to be
#: inferred. The kev halves are the same architecture answered through the same prompts, so their
#: discordance is weights. Laya's two halves are rendered at *their own* geometry -- the fine-tune at
#: 8192/2048, the released base at its shipped 1024/256, where the option block is cut -- so the
#: discordance carries a prompt difference too. The pairing is still on the same 63,851 questions (the
#: two dumps cover the same population), which is what makes the test legitimate; it is not a test of
#: the weights alone.
PAIR_NOTES: dict[str, str] = {
    "laya": "the two halves answer the same 63,851 questions at their own render geometry (the "
            "fine-tune at 8192/2048, the released base at its shipped 1024/256), so the discordant "
            "pairs carry a prompt difference as well as a weights difference",
}

#: The v4 lane's dump name for an arm: ``<prefix><slug><suffix>.jsonl`` (``score-forced-rows.py``'s
#: ``--out``). One function derives a dump's path from an arm and one recovers the arm from a dump, so
#: no key in this report can be a filename.
DUMP_PREFIX = "forced-rows-"


def dump_path(reports: Path, arm: str, suffix: str) -> Path:
    """The dump the lane writes for ``arm`` -- the one path shape this report reads."""
    return reports / f"{DUMP_PREFIX}{arm}{suffix}.jsonl"


def arm_from_dump(path: Path, suffix: str) -> str:
    """The arm slug ``path`` carries: the lane prefix and the family suffix stripped **by name**.

    The inverse of :func:`dump_path`, and the reason a paired view can be keyed by arm: stripping only
    the suffix leaves ``forced-rows-delta-kev-0.8b-v4-s1``, which the roster never looks up, so both
    halves of a complete pair read as missing in the paired table and as "the other half" in the
    absent list. A name that is not the lane's is refused by name rather than keyed as-is.
    """
    tail = f"{suffix}.jsonl"
    if not path.name.startswith(DUMP_PREFIX) or not path.name.endswith(tail):
        raise ValueError(f"{path.name} is not a {DUMP_PREFIX!r}...{tail!r} dump: its arm slug cannot "
                         f"be recovered, and keying it as a filename would hide it from the pairing")
    return path.name[len(DUMP_PREFIX):-len(tail)]


def full_roster() -> tuple[str, ...]:
    """Every arm the roster names, fine-tune and rung, in the order the pairs are declared."""
    return tuple(name for _, arm, rung in PAIRS for name in (arm, rung))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rule_from(path: Path, names: tuple[str, ...]):
    """The named function definitions taken from another script's **source**, never from a copy.

    ``reports/abstain-reconciliation.py`` is the settled analysis that defines this project's exact
    two-sided binomial on discordant pairs (``reports/PAIRED-TESTS.md`` uses it). Importing the module
    would execute its body and rewrite ``reports/abstain-reconciliation.json``; re-implementing the
    rule here would let two analyses disagree. So the definitions are lifted by ``ast`` and executed
    on their own — the same device ``scripts/arms-forced.py`` uses.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    if missing:
        raise ValueError(f"{path} no longer defines {missing}: the rule this report applies has moved, "
                         f"and applying a copy of it would be a different test")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[*imports, *[definitions[name] for name in names]], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, filename=str(path), mode="exec"), namespace)     # noqa: S102 - the project's own source
    return namespace


def read_dump(path: Path) -> dict[tuple[str, str], dict]:
    """The dump's rows keyed by ``(record id, question id)``, refusing anything unexpected.

    The key is the pair, not the question id: a ``holds:CL:...`` question id repeats across records,
    so keying on it alone would silently drop all but one of a record's answers. A repeated pair is
    refused — it would weight one question twice — and so is any row that does not say *yes/no against
    a boolean reference*, because every number below would otherwise be taken on a question that is
    not the one it claims to be.

    **A noul row is recognised by its shape, not by a ``type`` field**, because the dump this reader
    is pointed at does not have one: ``write_noul_row_dump`` writes the **same 17 fields**
    (``sckevtype/eval/run.py``'s ``FULL_DUMP_FIELDS``) every choice rung dump carries, so that a
    consumer reads both with one reader — and ``type`` is not among them (``dump_lines`` *filters* on
    ``row["type"]``, it does not write it). Requiring it here refused five complete, correctly written
    dumps and is the mismatch this reader carried until the round-trip test below was added. A row
    that *does* declare a ``type`` must still declare ``noul``: an explicitly-typed choice row is
    refused by name rather than scored as a yes/no question.
    """
    rows: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "type" in row and row["type"] != "noul":
                raise ValueError(f"{path}:{number} is {NOT_NOUL} (type {row.get('type')!r}); this is "
                                 f"not a noul dump")
            if list(row.get("keys") or []) != NOUL_KEYS:
                raise ValueError(f"{path}:{number} carries keys {row.get('keys')!r}, not {NOUL_KEYS} "
                                 f"-- the reference's option order for a yes/no question")
            probabilities = row.get("p") or []
            if len(probabilities) != 2:
                raise ValueError(f"{path}:{number} carries {len(probabilities)} probabilities, not 2")
            if int(row.get("label", -1)) not in (0, 1):
                raise ValueError(f"{path}:{number} carries label {row.get('label')!r}, not 0/1")
            key = (row["id"], row["question"])
            if key in rows:
                raise ValueError(f"{path}:{number} repeats {key}, which would weight it twice")
            rows[key] = row
    if not rows:
        raise ValueError(f"{path} holds no rows: refusing to report a number over an empty dump")
    return rows


def summarise(name: str, path: Path, rows: dict[tuple[str, str], dict]) -> dict:
    """One arm's column: counts, accuracy, both rates, and the floor the accuracy has to clear."""
    answers = [max(range(2), key=lambda index: row["p"][index]) for row in rows.values()]
    labels = [int(row["label"]) for row in rows.values()]
    correct = sum(1 for answer, label in zip(answers, labels) if answer == label)
    yes_rate = sum(answers) / len(answers)
    reference_yes = sum(labels) / len(labels)
    floor = max(reference_yes, 1.0 - reference_yes)
    degenerate = reference_yes in (0.0, 1.0)
    return {
        "arm": name,
        "dump": str(path),
        "sha256": sha256_file(path),
        "questions": len(rows),
        "records": len({record for record, _ in rows}),
        "accuracy": correct / len(rows),
        "correct": correct,
        "answered_yes": sum(answers),
        "answered_yes_rate": yes_rate,
        "reference_yes_rate": reference_yes,
        "majority_floor": floor,
        "margin_over_floor": None if degenerate else correct / len(rows) - floor,
        "majority_floor_note": (
            f"the reference is {reference_yes:.4f} *yes*, so a rule that always answers "
            f"{'yes' if reference_yes >= 0.5 else 'no'} scores {floor:.4f} reading nothing"
        ),
        "degenerate": degenerate,
        "mean_p_yes": sum(float(row["p"][1]) for row in rows.values()) / len(rows),
    }


def compare(name: str, left: dict, right: dict, rows_left: dict, rows_right: dict, rule) -> dict:
    """The paired comparison of two arms on the questions they both answered.

    Paired because they are the same questions (same suite, same census records, same T = 1): what is
    left is which questions one arm got right and the other did not. ``b`` is the count one answered
    correctly and the other did not, ``c`` the reverse, and the test is the project's exact two-sided
    binomial on the discordant pairs — never a difference of two accuracies, which on a paired design
    throws away the pairing and the noise it cancels.
    """
    shared = sorted(set(rows_left) & set(rows_right))
    if not shared:
        raise ValueError(f"{left['arm']} and {right['arm']} share no questions: nothing to compare")
    only_left = len(set(rows_left) - set(rows_right))
    only_right = len(set(rows_right) - set(rows_left))
    b = c = 0
    for key in shared:
        left_right = max(range(2), key=lambda index: rows_left[key]["p"][index]) == int(rows_left[key]["label"])
        right_right = max(range(2), key=lambda index: rows_right[key]["p"][index]) == int(rows_right[key]["label"])
        b += int(left_right and not right_right)
        c += int(right_right and not left_right)
    return {
        "pair": name,
        "left": left["arm"], "right": right["arm"],
        "questions": len(shared),
        "only_in_left": only_left, "only_in_right": only_right,
        "accuracy_left": sum(1 for key in shared
                             if max(range(2), key=lambda i: rows_left[key]["p"][i]) == int(rows_left[key]["label"])) / len(shared),
        "accuracy_right": sum(1 for key in shared
                              if max(range(2), key=lambda i: rows_right[key]["p"][i]) == int(rows_right[key]["label"])) / len(shared),
        "left_right_only": b, "right_right_only": c,
        "p_value": rule["binom_two_sided"](b, c),
        #: The same number in log space. A lopsided pair on 63,851 questions has a p-value below
        #: ``5e-324``, which a float cannot hold: ``p_value`` is then ``0.0`` and this is the magnitude.
        "p_value_log10": rule["binom_two_sided_log10"](b, c),
        "rule": "exact two-sided binomial on discordant pairs (reports/PAIRED-TESTS.md's test)",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="noul-accuracy",
                                     description="Noul (binary) accuracy per arm, and fine-tune vs rung.")
    parser.add_argument("--reports", default="reports", help="directory the dumps live in")
    parser.add_argument("--suffix", default="-marker_noul", help="the dump name's trailing family tag")
    parser.add_argument("--expect-questions", type=int, default=0,
                        help="the population's own question count (63,851 for the v4 census's "
                             "marker_noul); a dump holding fewer rows is a prefix of a pass still "
                             "running and is reported as incomplete instead of scored (0 = score "
                             "whatever is on disk)")
    parser.add_argument("--out", default="reports/noul-accuracy-v4.json", help="JSON report to write")
    parser.add_argument("--markdown", default="reports/noul-accuracy-v4.md", help="markdown report to write")
    return parser.parse_args(argv)


def p_value_text(entry: dict) -> str:
    """The pair's p-value: the number a float holds, or its ``log10`` magnitude when it cannot.

    A paired test on 63,851 questions reaches p-values below ``5e-324``, where the float rule returns
    ``0.0`` -- correct for the type and unreadable in a table. The magnitude comes from the same file's
    ``binom_two_sided_log10``; it is never re-derived here.
    """
    if entry["p_value"] > 0.0:
        return f"p={entry['p_value']:.3g}"
    return f"p<5e-324 (log10 {entry['p_value_log10']:.1f})"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rule = rule_from(ROOT / "analysis" / "abstain-reconciliation.py",
                     ("binom_two_sided", "binom_two_sided_log10"))
    reports = Path(args.reports)

    # Discovery is by the lane's own name shape and the key is the **arm slug**, never the filename:
    # a key that kept `forced-rows-` would miss the roster's own names on both sides of every pair and
    # report two complete dumps as an untestable pair.
    on_disk = {arm_from_dump(path, args.suffix): path
               for path in sorted(reports.glob(f"{DUMP_PREFIX}*{args.suffix}.jsonl"))}

    summaries, rowsets, absent, incomplete = [], {}, [], []
    states: dict[str, str] = {}
    for name in full_roster():
        path = on_disk.get(name)
        if path is None or path.stat().st_size == 0:
            absent.append({"arm": name, "expected": str(dump_path(reports, name, args.suffix)),
                           "found_on_disk": path is not None and path.stat().st_size == 0,
                           "why": ("the pass wrote an empty file: no rows of this arm are on disk"
                                   if path is not None else "no such dump")})
            states[name] = "absent"
            continue
        rows = read_dump(path)
        if args.expect_questions and len(rows) < args.expect_questions:
            incomplete.append({
                "arm": name, "dump": str(path), "questions": len(rows),
                "records": len({record for record, _ in rows}),
                "expected_questions": args.expect_questions,
                "shortfall": args.expect_questions - len(rows),
                "sha256": sha256_file(path),
                "why": "the pass writes its dump as it scores, so this is a prefix of a run still "
                       "in flight; a number over a prefix is not the family's number",
            })
            states[name] = "incomplete"
            continue
        rowsets[name] = rows
        summaries.append(summarise(name, path, rows))
        states[name] = "complete"

    by_arm = {entry["arm"]: entry for entry in summaries}
    pairs, untestable = [], []
    for size, left, right in PAIRS:
        if left in by_arm and right in by_arm:
            entry = compare(size, by_arm[left], by_arm[right], rowsets[left], rowsets[right], rule)
            if size in PAIR_NOTES:
                entry["note"] = PAIR_NOTES[size]
            pairs.append(entry)
            continue
        untestable.append({
            "pair": size, "left": left, "right": right,
            "left_state": states[left], "right_state": states[right],
            "why": "a paired test needs both halves scored on the same questions; each half's own state "
                   "is named here (incomplete = a dump being written, absent = no dump at all), so a "
                   "dump that exists is never read as missing",
        })
    roster = set(full_roster())
    other_dumps = sorted(arm for arm in on_disk if arm not in roster)

    result = {
        "what_this_is": "The noul (binary marker) column of the v4 census: one row per question, "
                        "scored argmax(p) == label against keys ['false','true'], plus the paired "
                        "fine-tune-vs-rung comparison on the questions both answered.",
        "unit": "questions (not records): marker_noul is 4,035 records carrying 63,851 questions",
        "keys": NOUL_KEYS,
        "dumps": {"directory": str(reports), "suffix": args.suffix,
                  "expect_questions": args.expect_questions or None,
                  "arms_on_disk": sorted(on_disk),
                  "sha256": {entry["arm"]: entry["sha256"] for entry in summaries}},
        "arms": summaries,
        "pairs": pairs,
        "pairs_not_testable": untestable,
        "absent": absent,
        "incomplete": incomplete,
        "other_dumps": other_dumps,
    }
    payload = json.dumps(result, indent=1, sort_keys=True) + "\n"
    Path(args.out).write_text(payload, encoding="utf-8")
    Path(args.markdown).write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("unit", "keys")}, indent=1))
    for entry in summaries:
        print(f"  {entry['arm']:<22s} questions={entry['questions']:<6d} records={entry['records']:<5d} "
              f"accuracy={entry['accuracy']:.4f}  says-yes={entry['answered_yes_rate']:.4f}  "
              f"floor={entry['majority_floor']:.4f}")
    for pair in pairs:
        print(f"  {pair['pair']:<5s} {pair['left']} vs {pair['right']}: "
              f"{pair['accuracy_left']:.4f} vs {pair['accuracy_right']:.4f} on {pair['questions']} questions, "
              f"discordant {pair['left_right_only']}/{pair['right_right_only']}, {p_value_text(pair)}")
        if pair.get("note"):
            print(f"          note: {pair['note']}")
    for entry in incomplete:
        print(f"  INCOMPLETE {entry['arm']}: {entry['questions']}/{entry['expected_questions']} questions "
              f"({entry['records']} records) -- a prefix of a pass in flight, not scored")
    for entry in absent:
        print(f"  ABSENT {entry['arm']}: {entry['expected']}")
    for entry in untestable:
        print(f"  UNTESTABLE {entry['pair']}: {entry['left']} ({entry['left_state']}) vs "
              f"{entry['right']} ({entry['right_state']}) -- the paired test needs both halves scored")
    for arm in other_dumps:
        print(f"  ON DISK, NOT IN THE ROSTER: {arm} -- the pairing never looks this key up")
    print(f"wrote {args.out} and {args.markdown}  "
          f"json_sha256={hashlib.sha256(payload.encode()).hexdigest()[:16]}")
    return 0 if summaries else 2


def markdown(result: dict) -> str:
    """The same numbers, as the table the manuscript can carry and a reader can check."""
    lines = [
        "# Noul (binary marker) accuracy, v4 census",
        "",
        "One row per **question** — `marker_noul` is 4,035 records carrying 63,851 yes/no questions, so a",
        "record-count table would hide 15.8 questions per row. Answer = `argmax(p)` against the reference's",
        "fixed keys `[\"false\",\"true\"]`; the *says yes* column is the model's own answer rate beside the",
        "reference's, and `floor` is the majority-class accuracy — the number an accuracy has to clear",
        "before it means anything.",
        "",
        "| arm | questions | records | accuracy | floor | margin | says yes | reference yes | mean P(yes) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in result["arms"]:
        margin = "n/a (degenerate)" if entry["margin_over_floor"] is None else f"{entry['margin_over_floor']:+.4f}"
        lines.append(f"| {entry['arm']} | {entry['questions']} | {entry['records']} | "
                     f"{entry['accuracy']:.4f} | {entry['majority_floor']:.4f} | {margin} | "
                     f"{entry['answered_yes_rate']:.4f} | {entry['reference_yes_rate']:.4f} | "
                     f"{entry['mean_p_yes']:.4f} |")
    lines += ["", "## Fine-tune against the rung (or base) it started from", "",
              "Same questions on both sides, so the test is on the discordant pairs (exact two-sided",
              "binomial): `fine-tune right, rung wrong` / `rung right, fine-tune wrong`. A p-value below",
              "`5e-324` is not a float and is shown as its `log10` magnitude — the same number from the",
              "same rule (`reports/abstain-reconciliation.py`).", "",
              "| size | fine-tune | rung | accuracy | discordant (FT/rung) | p |", "|---|---|---|---|---|---|"]
    for pair in result["pairs"]:
        lines.append(f"| {pair['pair']} | {pair['left']} | {pair['right']} | "
                     f"{pair['accuracy_left']:.4f} vs {pair['accuracy_right']:.4f} | "
                     f"{pair['left_right_only']} / {pair['right_right_only']} | {p_value_text(pair)} |")
    for pair in result["pairs"]:
        if pair.get("note"):
            lines += ["", f"`{pair['pair']}` — {pair['note']}."]
    if result.get("pairs_not_testable"):
        lines += ["", "## Pairs not testable yet", "",
                  "A paired test needs both halves scored on the same questions. Each half's own state is",
                  "named here — `incomplete` is a dump the pass is still writing, `absent` is no dump at",
                  "all — so a dump that exists is never read as missing:", ""]
        lines += [f"* `{e['pair']}` — {e['left']} ({e['left_state']}) vs "
                  f"{e['right']} ({e['right_state']})" for e in result["pairs_not_testable"]]
    if result.get("incomplete"):
        lines += ["", "## In progress (not scored)", "",
                  "A dump the pass has not finished writing is a **prefix** — real, correctly written rows of",
                  "a run still in flight. A number over a prefix is not the family's number, so these arms",
                  "carry no accuracy until their pass lands all of the population's questions:", ""]
        lines += [f"* `{e['arm']}` — {e['questions']} of {e['expected_questions']} questions on disk "
                  f"({e['shortfall']} to go), {e['records']} records" for e in result["incomplete"]]
    if result["absent"]:
        lines += ["", "## Not measured", ""]
        lines += [(f"* `{entry['arm']}` — no dump at `{entry['expected']}`"
                   if not entry.get("found_on_disk") else
                   f"* `{entry['arm']}` — an **empty** dump at `{entry['expected']}` (0 bytes: the "
                   f"pass wrote no rows of this arm)") for entry in result["absent"]]
    if result.get("other_dumps"):
        lines += ["", "On disk under a key the roster does not name (the pairing never looks these up):", ""]
        lines += [f"* `{arm}`" for arm in result["other_dumps"]]
    lines += ["", "Dumps (sha256):", ""]
    lines += [f"* `{arm}` `{digest}`" for arm, digest in sorted(result["dumps"]["sha256"].items())]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
