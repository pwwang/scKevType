"""The noul column's arithmetic, its refusals, and the paired comparison it is built on.

`scripts/noul-accuracy.py` reports the binary marker-accuracy the v4 census was missing, per **question**
(``marker_noul`` is 4,035 records carrying 63,851 questions). Three of its properties are load-bearing
and are checked here on hand-built dumps, with no GPU and no cluster:

* the **rule is taken, not re-derived** — the exact two-sided binomial comes from the parsed source of
  ``reports/abstain-reconciliation.py``, the file that settled it (whose module body would rewrite
  ``reports/abstain-reconciliation.json``, so it is never executed);
* the **unit is the question and the key is ``(record, question)``** — a ``holds:CL:...`` question id
  repeats across records, so keying on it alone would drop answers in silence;
* **an accuracy is reported with its floor**, and a family whose reference is one value only is marked
  degenerate rather than scored — the family this one sits beside has a 100% floor three times over.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def noul():
    spec = importlib.util.spec_from_file_location("noul_accuracy", SCRIPTS / "noul-accuracy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(record: str, question: str, p, label: int, **extra) -> dict:
    row = {"id": record, "question": question, "type": "noul", "keys": ["false", "true"],
           "label": label, "p": list(p)}
    row.update(extra)
    return row


def _dump(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------- the rule is taken, not re-written
def test_the_paired_test_comes_from_the_file_that_settled_it(noul):
    """``binom_two_sided`` is executed from that file's source; the module body never runs."""
    rule = noul.rule_from(ROOT / "analysis" / "abstain-reconciliation.py", ("binom_two_sided",))
    assert rule["binom_two_sided"](0, 0) == 1.0, "no discordant pairs, no evidence"
    assert rule["binom_two_sided"](5, 5) == 1.0, "a symmetrical split is the null itself"
    assert rule["binom_two_sided"](10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert rule["binom_two_sided"](0, 10) == rule["binom_two_sided"](10, 0), "two-sided by construction"


def test_a_moved_rule_is_refused_rather_than_replaced(noul):
    """If the definitions this report applies disappear, it must fail — not silently apply a copy."""
    with pytest.raises(ValueError, match="no longer defines"):
        noul.rule_from(SCRIPTS / "noul-accuracy.py", ("binom_two_sided",))


# --------------------------------------------------------------------- readers and refusals
def test_read_dump_keys_on_the_record_and_question_pair(tmp_path, noul):
    """The same question id on two records is two answers, not a repeat."""
    path = _dump(tmp_path / "d.jsonl", [_row("test/custom/0", "holds:CL:0000001", [0.9, 0.1], 0),
                                        _row("test/custom/1", "holds:CL:0000001", [0.1, 0.9], 1)])
    rows = noul.read_dump(path)
    assert len(rows) == 2, "a question id alone is not unique across records"
    assert noul.summarise("arm", path, rows)["records"] == 2


@pytest.mark.parametrize("rows, message", [
    ([_row("r", "q", [0.9, 0.1], 0, type="choice")], "not a noul row"),
    ([_row("r", "q", [0.9, 0.1], 0, keys=["alpha", "beta"])], "not \\['false', 'true'\\]"),
    ([_row("r", "q", [0.9], 0)], "not 2"),
    ([_row("r", "q", [0.9, 0.1], 7)], "not 0/1"),
    ([_row("r", "q", [0.9, 0.1], 0), _row("r", "q", [0.1, 0.9], 1)], "repeats"),
    ([], "empty dump"),
])
def test_read_dump_refuses_anything_it_cannot_score(tmp_path, noul, rows, message):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        noul.read_dump(path)


# --------------------------------------------------------------------- the round trip that was missing
def _scored_row(index: int, p, label: int, *, record: str = "test/custom/0", type: str = "noul") -> dict:
    """A row in the shape ``score_partition`` returns: kev's fields plus the metadata the CLI adds.

    Every field here is one the scoring path really sets, ``type`` included — the writer *filters* on
    that field, which is exactly why a dump that forgets to write it is the failure this pair of tests
    exists to catch.
    """
    return {"id": record, "question": f"holds:CL:{index:07d}", "split": "test", "type": type,
            "dataset": "synthetic", "donor": "d0", "family": "marker_noul", "variant": "clean",
            "keys": ["false", "true"], "label": label, "size": 2, "abstain": None,
            "abstain_source": "factory", "abstain_key": None, "abstain_text": None,
            "abstain_position": None, "abstain_last": False, "none_p": None, "p": list(p)}


def test_a_dump_the_writer_writes_is_a_dump_this_reader_reads(tmp_path, noul):
    """**The mismatch this file was missing**: write with the writer, read with the aggregator.

    The reader required a ``type`` field the writer never writes (``FULL_DUMP_FIELDS`` is the choice
    dump's 17 fields, and ``type`` is not among them), so every dump the pass produced was refused at
    its first line and the noul column could not be aggregated from data that was already on disk.
    Both sides are exercised here on the same rows, so the pair cannot drift again: the writer's bytes
    are read back, the count survives, and the accuracy is the one the vectors say it is.
    """
    from sckevtype.eval import run as eval_run

    rows = [_scored_row(0, [0.9, 0.1], 0),                 # says no, reference no  -> right
            _scored_row(1, [0.2, 0.8], 0),                 # says yes, reference no -> wrong
            _scored_row(2, [0.1, 0.9], 1, record="test/custom/1")]   # says yes, reference yes -> right
    path = tmp_path / "forced-rows-delta-kev-0.8b-v4-s1-marker_noul.jsonl"
    info = eval_run.write_noul_row_dump(path, rows)

    written = [json.loads(line) for line in path.read_text().splitlines()]
    assert all("type" not in row for row in written), \
        "the dump is the choice dump's shape: the type is a filter, not a field of the row"
    assert info["rows"] == len(rows) and info["skipped"] == 0

    read = noul.read_dump(path)
    assert len(read) == 3, "one row per question, and none dropped by the reader"
    assert set(read) == {("test/custom/0", "holds:CL:0000000"), ("test/custom/0", "holds:CL:0000001"),
                         ("test/custom/1", "holds:CL:0000002")}
    entry = noul.summarise("delta-kev-0.8b-v4-s1", path, read)
    assert (entry["questions"], entry["records"], entry["correct"]) == (3, 2, 2)
    assert entry["accuracy"] == pytest.approx(2 / 3)
    assert entry["reference_yes_rate"] == pytest.approx(1 / 3)


def test_the_reader_refuses_the_choice_dump_of_the_same_lane(tmp_path, noul):
    """The negative control: the *other* dump of the same lane is not read as a noul dump.

    Without this, "the reader accepts the writer's output" would also be satisfied by a reader that
    accepts anything — and the two dumps are written to sibling names by one script.
    """
    from sckevtype.eval import run as eval_run

    choice = {"id": "test/custom/0", "question": "label", "split": "test", "type": "choice",
              "dataset": "synthetic", "donor": "d0", "family": "panel_choice", "variant": "clean",
              "keys": ["CL:0000057", "CL:0000182"], "label": 1, "size": 2, "abstain": None,
              "abstain_source": "factory", "abstain_key": None, "abstain_text": None,
              "abstain_position": None, "abstain_last": False, "none_p": None, "p": [0.4, 0.6]}
    path = tmp_path / "forced-rows-kev-0.8b-v4-marker_choice.jsonl"
    eval_run.write_full_row_dump(path, [choice])

    with pytest.raises(ValueError, match="not \\['false', 'true'\\]"):
        noul.read_dump(path)


# --------------------------------------------------------------------- the arithmetic
def test_accuracy_is_argmax_against_the_reference_and_carries_its_floor(tmp_path, noul):
    """Four questions, three of them *no*: naming yes twice is 1/4 against a 3/4 always-no floor."""
    rows = [_row("r0", "q0", [0.9, 0.1], 0),      # no   / no   -> right
            _row("r1", "q1", [0.2, 0.8], 0),      # yes  / no   -> wrong
            _row("r2", "q2", [0.9, 0.1], 1),      # no   / yes  -> wrong
            _row("r0", "q3", [0.1, 0.9], 0)]      # yes  / no   -> wrong
    entry = noul.summarise("arm", _dump(tmp_path / "d.jsonl", rows), noul.read_dump(tmp_path / "d.jsonl"))

    assert entry["questions"] == 4 and entry["records"] == 3
    assert entry["answered_yes"] == 2 and entry["correct"] == 1
    assert entry["accuracy"] == 0.25, "argmax(p) against the reference's own label"
    assert entry["reference_yes_rate"] == 0.25
    assert entry["majority_floor"] == 0.75, "always answering *no* reads nothing and scores 3/4"
    assert entry["margin_over_floor"] == pytest.approx(0.25 - 0.75), \
        "the margin is negative here, and it is reported as such rather than dropped"


def test_a_single_valued_reference_is_marked_degenerate(tmp_path, noul):
    """An all-*no* family has a 100% floor: a number over it is not a measurement."""
    rows = [_row("r0", "q0", [0.1, 0.9], 0), _row("r0", "q1", [0.2, 0.8], 0)]
    path = _dump(tmp_path / "d.jsonl", rows)
    entry = noul.summarise("arm", path, noul.read_dump(path))
    assert entry["degenerate"] is True and entry["majority_floor"] == 1.0
    assert entry["margin_over_floor"] is None and "reading nothing" in entry["majority_floor_note"]


def test_the_pair_comparison_counts_discordance_and_pairs_on_the_shared_questions(tmp_path, noul):
    """The comparison the ask is about: same questions both sides, so the test is on the discordant ones."""
    shared = [_row("r", f"q{i}", [0.9, 0.1], 0) for i in range(6)]        # the reference is *no* on all six
    left = shared + [_row("r", "q-extra", [0.9, 0.1], 0)]
    right = [row if index != 0 else _row("r", "q0", [0.1, 0.9], 0) for index, row in enumerate(shared)]
    rule = noul.rule_from(ROOT / "analysis" / "abstain-reconciliation.py",
                          ("binom_two_sided", "binom_two_sided_log10"))
    rows_left, rows_right = noul.read_dump(_dump(tmp_path / "l.jsonl", left)), noul.read_dump(_dump(tmp_path / "r.jsonl", right))
    left_entry = noul.summarise("delta", tmp_path / "l.jsonl", rows_left)
    right_entry = noul.summarise("rung", tmp_path / "r.jsonl", rows_right)

    pair = noul.compare("0.8b", left_entry, right_entry, rows_left, rows_right, rule)
    assert pair["questions"] == 6 and pair["only_in_left"] == 1, "the extra question is not compared"
    assert (pair["left_right_only"], pair["right_right_only"]) == (1, 0)
    assert pair["accuracy_left"] == 1.0 and pair["accuracy_right"] == pytest.approx(5 / 6)
    assert pair["p_value"] == rule["binom_two_sided"](1, 0) == 1.0, \
        "one discordant pair cannot separate two arms, and the report says so"


def test_the_report_lists_what_is_absent_instead_of_reporting_over_it(tmp_path, noul):
    """End to end over a directory holding one complete pair: the other four are named as absent."""
    name = "delta-kev-0.8b-v4-s1"
    rung = "kev-0.8b-v4"
    for slug, p in ((name, [0.9, 0.1]), (rung, [0.1, 0.9])):
        _dump(tmp_path / f"forced-rows-{slug}-marker_noul.jsonl",
              [_row("test/custom/0", "q0", p, 0), _row("test/custom/0", "q1", p, 0)])
    out, markdown = tmp_path / "noul.json", tmp_path / "noul.md"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(markdown)])
    report = json.loads(out.read_text())

    assert rc == 0
    assert {entry["arm"] for entry in report["arms"]} == {name, rung}
    assert [pair["pair"] for pair in report["pairs"]] == ["0.8b"]
    assert report["pairs"][0]["p_value"] == pytest.approx(2 * 0.5 ** 2), \
        "the fine-tune is right on both, the rung on neither: two discordant pairs"
    assert {entry["arm"] for entry in report["absent"]} >= {"delta-kev-0.5b-v4-s1", "kev-9b-v4"}
    assert report["unit"].startswith("questions")
    assert str(report["arms"][0]["sha256"]) == noul.sha256_file(Path(report["arms"][0]["dump"]))
    assert "| arm | questions |" in markdown.read_text(), "the markdown carries the same table"


def test_no_dumps_costs_nothing_and_reports_absence(tmp_path, noul):
    """A directory with nothing in it must not write a report that looks like a measurement."""
    out, markdown = tmp_path / "noul.json", tmp_path / "noul.md"
    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(markdown)])
    report = json.loads(out.read_text())

    assert rc == 2, "nothing measured is not success"
    assert report["arms"] == []
    # Derived from the roster, never typed: a pair added to PAIRS (the 27B's, once its dumps landed)
    # must not be able to make this pass by being forgotten, nor break it by existing.
    assert len(report["absent"]) == len(noul.full_roster()) == 2 * len(noul.PAIRS), \
        "every arm the roster names is reported absent -- one fine-tune and one rung per pair"
    assert "Not measured" in markdown.read_text()


def test_an_unfinished_dump_is_a_prefix_and_is_not_scored(tmp_path, noul):
    """``--expect-questions``: the pass's own acceptance applied to the dump, with its positive control.

    A pass writes its dump as it scores, so a file holding fewer rows than the population asks is a
    **prefix** of a run in flight — readable, correctly written, and not a result. It must be reported as
    incomplete rather than scored, and the complete dump beside it must still be scored, or "refuses a
    prefix" and "refuses everything" would be the same observation.
    """
    name, rung = "delta-kev-0.5b-v4-s1", "kev-0.5b-v4"
    prefix = [_row("test/custom/0", f"q{i}", [0.9, 0.1], 0) for i in range(3)]
    complete = [_row("test/custom/0", f"q{i}", [0.9, 0.1], 0) for i in range(6)]
    _dump(tmp_path / f"forced-rows-{name}-marker_noul.jsonl", prefix)
    _dump(tmp_path / f"forced-rows-{rung}-marker_noul.jsonl", complete)
    out, markdown = tmp_path / "noul.json", tmp_path / "noul.md"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(markdown),
                    "--expect-questions", "6"])
    report = json.loads(out.read_text())
    text = markdown.read_text()

    assert rc == 0, "one complete arm is a measurement"
    assert [entry["arm"] for entry in report["arms"]] == [rung], "only the complete dump is scored"
    assert report["arms"][0]["questions"] == 6
    assert [(entry["arm"], entry["questions"], entry["expected_questions"], entry["shortfall"])
            for entry in report["incomplete"]] == [(name, 3, 6, 3)]
    assert report["dumps"]["expect_questions"] == 6
    assert "In progress (not scored)" in text and name in text, "the prefix is named, not dropped"
    assert rung in text


def test_without_the_flag_every_dump_on_disk_is_scored(tmp_path, noul):
    """The flag is the gate, not a change of behaviour: a finished directory needs no flag."""
    name, rung = "delta-kev-0.5b-v4-s1", "kev-0.5b-v4"
    for slug in (name, rung):
        _dump(tmp_path / f"forced-rows-{slug}-marker_noul.jsonl",
              [_row("test/custom/0", f"q{i}", [0.9, 0.1], 0) for i in range(3)])
    out = tmp_path / "noul.json"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(tmp_path / "n.md")])
    report = json.loads(out.read_text())

    assert rc == 0 and report["incomplete"] == []
    assert {entry["arm"] for entry in report["arms"]} == {name, rung}


# ------------------------------------------------------- the keying the paired table is built on
def test_a_dumps_key_is_the_arm_slug_never_the_filename(tmp_path, noul):
    """``forced-rows-<slug><suffix>.jsonl`` ⇄ ``<slug>``: the paired view's key, and its inverse.

    Stripping only the suffix leaves ``forced-rows-delta-kev-0.8b-v4-s1``, which the roster's own names
    (``delta-kev-0.8b-v4-s1`` / ``kev-0.8b-v4``) never match — the pairing then misses on *both* sides
    and two complete dumps read as an untestable pair. The round trip is asserted rather than assumed,
    and a name outside the lane's shape is refused instead of keyed as-is.
    """
    arm = "delta-kev-0.8b-v4-s1"
    path = noul.dump_path(tmp_path, arm, "-marker_noul")

    assert path.name == f"forced-rows-{arm}-marker_noul.jsonl"
    assert noul.arm_from_dump(path, "-marker_noul") == arm, \
        "the lane's prefix is not part of the key: the roster looks up the slug"
    assert noul.dump_path(tmp_path, noul.arm_from_dump(path, "-marker_noul"), "-marker_noul") == path
    assert noul.arm_from_dump(tmp_path / "forced-rows-kev-4b-v4-marker_noulv4.jsonl", "-marker_noulv4") \
        == "kev-4b-v4", "the suffix is a parameter, and the prefix is still the lane's"
    with pytest.raises(ValueError, match="cannot be recovered"):
        noul.arm_from_dump(tmp_path / "kev-0.8b-v4.jsonl", "-marker_noul")


def test_a_pair_of_complete_dumps_is_actually_paired(tmp_path, noul):
    """End to end: two complete dumps under the lane's own names produce the paired row, with a p-value.

    The regression this pins is the *empty* paired table: the markdown carried the heading and no rows
    while both halves were complete on disk, which reads exactly like two missing dumps.
    """
    name, rung = "delta-kev-0.5b-v4-s1", "kev-0.5b-v4"
    right = [_row("test/custom/0", f"q{i}", [0.9, 0.1], 0) for i in range(6)]      # says no, reference no
    wrong = [_row("test/custom/0", f"q{i}", [0.1, 0.9], 0) for i in range(6)]      # says yes, reference no
    _dump(tmp_path / f"forced-rows-{name}-marker_noul.jsonl", right)
    _dump(tmp_path / f"forced-rows-{rung}-marker_noul.jsonl", wrong)
    out, markdown = tmp_path / "noul.json", tmp_path / "noul.md"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(markdown),
                    "--expect-questions", "6"])
    report = json.loads(out.read_text())
    text = markdown.read_text()

    assert rc == 0
    assert [pair["pair"] for pair in report["pairs"]] == ["0.5b"], \
        "both halves are complete, so the pair must be tested rather than reported as missing"
    pair = report["pairs"][0]
    assert (pair["left"], pair["right"]) == (name, rung), "the pair is keyed by slug, not by filename"
    assert (pair["left_right_only"], pair["right_right_only"]) == (6, 0)
    assert pair["accuracy_left"] == 1.0 and pair["accuracy_right"] == 0.0
    assert pair["p_value"] == pytest.approx(2 * 0.5 ** 6), \
        "six discordant pairs, all one way: the exact two-sided binomial on them"
    assert pair["p_value_log10"] == pytest.approx(math.log10(pair["p_value"])), \
        "the same number in log space, reported beside the float"
    assert f"| 0.5b | {name} | {rung} |" in text, "the markdown table carries the paired row"
    # Derived from the roster, never typed: every pair whose dumps are absent is named, and the one
    # pair that is complete is not.
    assert {entry["pair"] for entry in report["pairs_not_testable"]} == \
        {size for size, _left, _right in noul.PAIRS} - {"0.5b"}
    assert report["pairs_not_testable"][0]["left_state"] == "absent"


def test_the_laya_halves_are_a_row_of_this_table_and_their_pair_carries_its_note(tmp_path, noul):
    """The Laya halves are kev-shaped dumps under the lane's own names, so they pair like any other.

    Their dumps are written by the Laya lane's own reconcile through this lane's writer; nothing about
    the keying changes, which is the property this test pins: ``reports/forced-rows-laya-<half>-
    marker_noul.jsonl`` is read, scored and paired against the released base, and the one thing the two
    halves do not share (their render geometry) is printed with the row instead of being left implicit.
    """
    ft, base = "laya-v4-ft-spec", "laya-base"
    _dump(tmp_path / f"forced-rows-{ft}-marker_noul.jsonl",
          [_row("test/custom/0", f"q{i}", [0.9, 0.1], 0) for i in range(6)])
    _dump(tmp_path / f"forced-rows-{base}-marker_noul.jsonl",
          [_row("test/custom/0", f"q{i}", [0.1, 0.9], 0) for i in range(6)])
    out, markdown = tmp_path / "noul.json", tmp_path / "noul.md"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(markdown),
                    "--expect-questions", "6"])
    report = json.loads(out.read_text())

    assert rc == 0
    assert {entry["arm"] for entry in report["arms"]} == {ft, base}
    assert [entry["pair"] for entry in report["pairs"]] == ["laya"]
    pair = report["pairs"][0]
    assert (pair["left"], pair["right"]) == (ft, base)
    assert pair["note"] == noul.PAIR_NOTES["laya"] and "geometry" in pair["note"]
    assert pair["note"] in markdown.read_text(), "the caveat travels with the row, not only with the JSON"


def test_a_dump_in_flight_is_named_as_such_and_never_as_absent(tmp_path, noul):
    """One half complete, the other a prefix: the pair is untestable **because of the state**, said out loud.

    The cluster report of 09:56 listed ``kev-4b-v4`` — whose dump existed and was being written — under
    "no dump at *the other half of the pair*". Two different facts printed as one sentence, and the
    wrong one was the one that printed.
    """
    name, rung = "delta-kev-0.8b-v4-s1", "kev-0.8b-v4"
    _dump(tmp_path / f"forced-rows-{name}-marker_noul.jsonl",
          [_row("test/custom/0", f"q{i}", [0.9, 0.1], 0) for i in range(6)])
    _dump(tmp_path / f"forced-rows-{rung}-marker_noul.jsonl",
          [_row("test/custom/0", f"q{i}", [0.1, 0.9], 0) for i in range(2)])
    out, markdown = tmp_path / "noul.json", tmp_path / "noul.md"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(markdown),
                    "--expect-questions", "6"])
    report = json.loads(out.read_text())
    entry = next(item for item in report["pairs_not_testable"] if item["pair"] == "0.8b")

    assert rc == 0
    assert report["pairs"] == [], "a half that is still being written is not a half to compare against"
    assert (entry["left_state"], entry["right_state"]) == ("complete", "incomplete")
    assert not [item for item in report["absent"] if item["arm"] == rung], \
        "a dump that exists is not an absent dump, whatever the pair can do with it"
    assert f"{rung} (incomplete)" in markdown.read_text(), "the state is printed, not inferred"
    assert "| size | fine-tune | rung |" in markdown.read_text()


def test_an_empty_dump_is_reported_as_empty_rather_than_missing(tmp_path, noul):
    """The 9B rung's dump exists and holds no rows (the fp32 OOM): the report says which of the two it is."""
    (tmp_path / "forced-rows-kev-9b-v4-marker_noul.jsonl").write_text("", encoding="utf-8")
    out, markdown = tmp_path / "noul.json", tmp_path / "noul.md"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(markdown)])
    report = json.loads(out.read_text())
    entry = next(item for item in report["absent"] if item["arm"] == "kev-9b-v4")

    assert rc == 2, "an empty dump is not a measurement"
    assert entry["found_on_disk"] is True and entry["why"].startswith("the pass wrote an empty file")
    assert "**empty** dump" in markdown.read_text()


def test_a_dump_under_a_key_the_roster_does_not_name_is_listed(tmp_path, noul):
    """The shape the bug would produce: a key carrying the lane's prefix is *shown*, not silently skipped.

    ``forced-rows-<slug>`` as the key is what a key-stripping-only-the-suffix produces, and the roster
    never looks it up — so the dump must appear in the report rather than disappear from it.
    """
    _dump(tmp_path / "forced-rows-forced-rows-kev-4b-v4-marker_noul.jsonl",
          [_row("test/custom/0", "q0", [0.9, 0.1], 0)])
    out = tmp_path / "noul.json"

    rc = noul.main(["--reports", str(tmp_path), "--out", str(out), "--markdown", str(tmp_path / "n.md")])
    report = json.loads(out.read_text())

    assert rc == 2, "a dump nobody pairs is not a measurement"
    assert report["other_dumps"] == ["forced-rows-kev-4b-v4"], \
        "a file under a key the roster does not name is reported, never dropped"
    assert [pair["pair"] for pair in report["pairs"]] == []
