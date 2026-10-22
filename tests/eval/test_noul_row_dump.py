"""The noul row dump, and the acceptance that has to be taken on the QUESTION count.

``marker_noul`` is the one census family whose questions are all ``noul`` — 4,035 records carrying
63,851 questions — and it is the family the choice writer writes nothing for (it skips every
non-choice row). That failure shape is pinned here in both directions:

* the **choice** writer still writes only choice rows, and still counts the rest as ``skipped`` — the
  documented behaviour, not a bug this change papered over;
* the **noul** writer writes the binary rows in the *same* 17-field shape, one row per question, and
  refuses an empty selection rather than writing a file that reads like a population never scored;
* the re-score script's new surface: which writer a ``--question-type`` selects, that the acceptance
  is a question count, and that ``--append`` resumes a killed pass without inventing or losing a row.

No GPU, no checkpoint, no suite: the rows are built in the shape the scoring path produces, and the
one model-shaped call (``score_partition``) is replaced by a fake predictor that returns a fixed
probability vector per question. The assertions are about the *bytes* written and the *counts*
recorded, which is where a writer that silently drops a family damages a result.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from sckevtype.eval import run as eval_run

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
NOUL_KEYS = ["false", "true"]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rescore():
    return _load_script("score_forced_rows_noul", "score-forced-rows.py")


def _row(index: int, p=None, *, keys=tuple(NOUL_KEYS), label: int = 0, type: str = "noul",
         record: str | None = None, **extra) -> dict:
    """A row shaped like the scoring path's: kev's fields plus the metadata ``extended_row`` adds."""
    row = {"id": record or f"test/custom/{index}", "question": f"holds:CL:{index:07d}", "split": "test",
           "type": type, "dataset": "synthetic", "donor": "d0", "family": "marker_noul",
           "variant": "clean", "keys": list(keys), "label": label, "size": len(keys),
           "abstain": None, "abstain_source": "factory", "abstain_key": None, "abstain_text": None,
           "abstain_position": None, "abstain_last": False, "none_p": None}
    if p is not None:
        row["p"] = list(p)
    row.update(extra)
    return row


# ------------------------------------------------------------------ the noul writer
def test_the_noul_writer_writes_one_row_per_question_in_the_full_dump_shape(tmp_path):
    """A noul row is one question, in the same 17 fields a choice dump carries: keys, label, p, top1."""
    rows = [_row(1, [0.9, 0.1], label=0), _row(2, [0.2, 0.8], label=1)]
    info = eval_run.write_noul_row_dump(tmp_path / "noul.jsonl", rows)
    written = [json.loads(line) for line in (tmp_path / "noul.jsonl").read_text().splitlines()]

    assert [row["keys"] for row in written] == [NOUL_KEYS, NOUL_KEYS], "the reference's own key order"
    assert [(row["label"], row["top1"]) for row in written] == [(0, 0), (1, 1)]
    assert [row["top1_p"] for row in written] == [0.9, 0.8]
    assert [row["p"] for row in written] == [[0.9, 0.1], [0.2, 0.8]]
    assert all(set(row) == set(eval_run.FULL_DUMP_FIELDS) for row in written), \
        "a noul dump is read with the choice dump's reader or not at all"
    assert all(row["abstain_key"] is None and row["none_p"] is None for row in written), \
        "a yes/no question offers no abstain option, and says so"
    assert info["rows"] == 2, "rows are questions here: two records' worth is two questions"


def test_the_noul_writer_skips_choice_rows_and_counts_them(tmp_path):
    """The mirror of the choice writer's own rule, so neither can quietly absorb the other's rows."""
    rows = [_row(1, [0.6, 0.4]), _row(2, [0.5, 0.5], type="choice", keys=("alpha", "beta"),
                                          family="panel_choice")]
    info = eval_run.write_noul_row_dump(tmp_path / "noul.jsonl", rows)
    written = [json.loads(line) for line in (tmp_path / "noul.jsonl").read_text().splitlines()]

    assert [row["id"] for row in written] == ["test/custom/1"]
    assert info == {"path": str(tmp_path / "noul.jsonl"), "rows": 1, "skipped": 1,
                    "sha256": eval_run.file_digest(tmp_path / "noul.jsonl")}


@pytest.mark.parametrize("writer", ["write_full_row_dump", "write_noul_row_dump"])
def test_an_empty_selection_is_refused_by_both_writers(tmp_path, writer):
    """A zero-match filter must raise, not write a file that reads like an unscored population.

    The positive control is the other type's row: it must still write, or "refuses empties" would be
    indistinguishable from "refuses everything".
    """
    path = tmp_path / f"{writer}.jsonl"
    choice = _row(1, [0.6, 0.4], type="choice", keys=("alpha", "beta"), family="panel_choice")
    noul = _row(2, [0.6, 0.4])
    rows = [noul] if writer == "write_full_row_dump" else [choice]

    with pytest.raises(ValueError, match="empty dump"):
        getattr(eval_run, writer)(path, rows)
    assert not path.exists(), "a refused dump is not written at all"

    other = [choice] if writer == "write_full_row_dump" else [noul]
    assert getattr(eval_run, writer)(path, other)["rows"] == 1


def test_the_choice_writer_is_unchanged_by_the_noul_branch(tmp_path):
    """The family the diagnosis found: 4,035 records whose questions are all noul -> choice writes none."""
    rows = [_row(i, [0.3, 0.7]) for i in range(3)]
    with pytest.raises(ValueError, match="no 'choice' rows"):
        eval_run.write_full_row_dump(tmp_path / "choice.jsonl", rows)
    assert not (tmp_path / "choice.jsonl").exists(), \
        "the choice writer must not leave an empty dump for a family it cannot score"


def test_the_choice_dumps_bytes_are_the_ones_the_pending_27b_pass_reads(tmp_path):
    """The shared line builder must not have moved the choice dump: its bytes are pinned here.

    ``write_full_row_dump`` now delegates to :func:`dump_lines`, and the 27B lane's driver
    (``<cluster-job>``) imports that function while its job is queued — so "the
    refactor changed nothing" has to be a claim about bytes. The expected line below was produced by
    the writer before the noul branch existed, and the digest is of the whole file, so a change to the
    field list, the key order or the serialisation fails here rather than in a scored table.
    """
    expected = ('{"id": "test/custom/0", "question": "label", "split": "test", "dataset": "synthetic", '
                '"donor": "d0", "family": "panel_choice", "variant": "clean", "size": 2, '
                '"keys": ["alpha", "beta"], "label": 1, "top1": 1, "top1_p": 0.6, "abstain": null, '
                '"abstain_source": "factory", "abstain_key": null, "none_p": 0.25, "p": [0.4, 0.6]}\n')
    row = {"id": "test/custom/0", "question": "label", "split": "test", "type": "choice",
           "dataset": "synthetic", "donor": "d0", "family": "panel_choice", "variant": "clean",
           "keys": ["alpha", "beta"], "label": 1, "size": 2, "abstain": None,
           "abstain_source": "factory", "abstain_key": None, "abstain_text": None,
           "abstain_position": None, "abstain_last": False, "none_p": 0.25, "p": [0.4, 0.6]}

    info = eval_run.write_full_row_dump(tmp_path / "choice.jsonl", [row])
    assert (tmp_path / "choice.jsonl").read_text() == expected
    assert info["sha256"] == "f5f41adbfcf51a39d47918d857eec2109d321f33cb4d9aaeeb30968e7d00a09a"


# ------------------------------------------------------------------ the re-score surface
def test_the_question_type_selects_the_writer(rescore):
    """One map, keyed by the same two names the flag accepts: a third type has to be added to both."""
    assert set(rescore.WRITERS) == {eval_run.CHOICE, eval_run.NOUL}
    assert rescore.WRITERS[eval_run.NOUL] is eval_run.write_noul_row_dump
    assert rescore.WRITERS[eval_run.CHOICE] is eval_run.write_full_row_dump


# ------------------------------------------------------------------ the serving-split cap (the 9B OOM)
def test_the_row_cap_only_ever_makes_a_pass_smaller_and_leaves_kev_alone(rescore):
    """``--rows-per-pass``: kev's own function, called with a smaller budget, and nothing else touched.

    The 9B rung's noul pass died inside ``_rows_hidden`` on one 40 GB card; the budget that sizes a
    pass is kev's, so the cap must be a ``min`` against kev's rule rather than a second implementation
    of it — and at or above kev's default it must be a no-op.
    """
    import kev.model as kev_model

    before = kev_model.rows_per_pass
    rows = [[0] * 7_000, [0] * 6_900] + [[0] * 900] * 20            # two long rows, then short ones
    assert kev_model.rows_per_pass(rows) == 2, "kev's own budget takes two maximal rows"
    try:
        replaced = rescore.cap_rows_per_pass(4_096)
        assert replaced is before, "the cap must replace kev's function, and only that"
        assert kev_model.rows_per_pass(rows) == 1, "4096 tokens cannot carry a 7,000-token row twice"
        assert kev_model.rows_per_pass(rows) == min(before(rows), before(rows, 0, 4_096)), \
            "the cap is kev's own rule at a smaller budget, not a copy of its arithmetic"
        assert kev_model.rows_per_pass([[0] * 300] * 100) == before([[0] * 300] * 100, 0, 4_096) == 13, \
            "the cap is a budget, so it bites on short rows too -- exactly as kev's own rule says"
        assert kev_model.rows_per_pass([[0] * 300] * 100) <= before([[0] * 300] * 100), \
            "and never asks for a bigger pass than kev's own split"
        assert kev_model.rows_per_pass(rows, 20_000) == before(rows, 20_000), \
            "a budget at or above kev's default is a no-op"
        assert rescore.rows_per_pass_record(4_096)["capped"] is True
        assert rescore.rows_per_pass_record(4_096)["kev_default"] == kev_model.SERVE_MAX_PACKED
        assert rescore.rows_per_pass_record(0) == {**rescore.rows_per_pass_record(4_096),
                                                   "budget": None, "capped": False}, \
            "an uncapped pass says so rather than claiming a budget it did not use"
        with pytest.raises(ValueError, match="positive int"):
            rescore.cap_rows_per_pass(0)
    finally:
        kev_model.rows_per_pass = rescore.cap_rows_per_pass(0, restore=True)
    assert kev_model.rows_per_pass is before, "restore puts kev's own function back"


def test_the_rows_per_pass_flag_defaults_to_kev_s_own_budget(rescore):
    """The default path of every existing arm: no flag, no patch, the same command line."""
    assert rescore.parse_args(["--suite", "s", "--checkpoint", "c", "--ids", "i", "--out", "o"]).rows_per_pass == 0
    assert rescore.parse_args(["--suite", "s", "--checkpoint", "c", "--ids", "i", "--out", "o",
                               "--rows-per-pass", "4096"]).rows_per_pass == 4096


def test_the_question_count_is_read_from_the_suite_not_from_the_ids_file(tmp_path, rescore):
    """The acceptance unit: 2 noul questions on one record, 1 on another, and nothing counted twice."""
    suite = tmp_path / "suite"
    suite.mkdir()
    lines = [
        {"_meta": {"id": "test/custom/0"}, "questions": {"a": {"type": "noul"}, "b": {"type": "noul"}}},
        {"_meta": {"id": "test/custom/1"}, "questions": {"c": {"type": "noul"}}},
        {"_meta": {"id": "test/custom/2"}, "questions": {"d": {"type": "choice"}}},
    ]
    (suite / "test.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    records = [{"_meta": {"id": "test/custom/0"}}, {"_meta": {"id": "test/custom/1"}}]

    assert rescore.expected_questions(records, suite, "test") == 3
    assert rescore.expected_questions(records[:1], suite, "test") == 2
    assert rescore.expected_questions([{"_meta": {"id": "test/custom/2"}}], suite, "test") == 0


def test_resume_keeps_the_records_that_are_there_and_re_scores_the_last_one(tmp_path, rescore):
    """A dump written by a killed pass: 3 rows over 2 records, the second of which may be half done."""
    order = ["test/custom/0", "test/custom/1", "test/custom/2"]
    path = tmp_path / "partial.jsonl"
    lines = [json.dumps(_row(0, [0.9, 0.1], record="test/custom/0")),
             json.dumps(_row(1, [0.9, 0.1], record="test/custom/0")),
             json.dumps(_row(2, [0.4, 0.6], record="test/custom/1"))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    kept, start = rescore.resume_state(path, order)

    assert kept == lines[:2], "the last record's rows are dropped, because it may be half written"
    assert start == 1, "and that record is re-scored rather than trusted"
    assert rescore.resume_state(tmp_path / "absent.jsonl", order) == ([], 0)


def test_resume_drops_a_truncated_last_line(tmp_path, rescore):
    """A kill mid-write leaves a partial line, which is not a row and must not enter the count."""
    order = ["test/custom/0", "test/custom/1", "test/custom/2"]
    complete = [json.dumps(_row(0, [0.9, 0.1], record="test/custom/0")),
                json.dumps(_row(1, [0.9, 0.1], record="test/custom/1"))]
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("\n".join(complete) + "\n"
                         + json.dumps(_row(2, [0.4, 0.6], record="test/custom/2"))[:-4],
                         encoding="utf-8")
    whole = tmp_path / "whole.jsonl"
    whole.write_text("\n".join(complete) + "\n", encoding="utf-8")

    assert rescore.resume_state(truncated, order) == rescore.resume_state(whole, order) \
        == (complete[:1], 1), "the partial line is dropped, not parsed, and not counted"


@pytest.mark.parametrize("content, message", [
    ('{"id": "test/custom/9", "keys": ["false", "true"]}\n', "not in this population"),
    ('not json at all\n', "is not a dump row"),
])
def test_resume_refuses_a_dump_of_another_population(tmp_path, rescore, content, message):
    """Continuing someone else's dump would put two populations in one file, so it must fail loudly."""
    path = tmp_path / "foreign.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        rescore.resume_state(path, ["test/custom/0", "test/custom/1"])


def test_resume_refuses_rows_out_of_record_order(tmp_path, rescore):
    """Rows are in suite order; a re-ordered file cannot be a prefix of this population's dump."""
    order = ["test/custom/0", "test/custom/1", "test/custom/2"]
    path = tmp_path / "reordered.jsonl"
    path.write_text("\n".join([json.dumps(_row(1, [0.9, 0.1], record="test/custom/1")),
                               json.dumps(_row(0, [0.9, 0.1], record="test/custom/0"))]) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="suite's order"):
        rescore.resume_state(path, order)


def test_the_append_path_writes_the_same_bytes_as_the_single_shot_writer(tmp_path, rescore):
    """The claim ``--append`` makes: a chunked, resumed pass is byte-identical to a one-shot write.

    Both files are built here from the same rows through the library's own writers — the chunked one
    by concatenating ``dump_lines`` output per chunk through ``resume_state``, the other by
    ``write_noul_row_dump`` — so a divergence in either path fails this test rather than a run.
    """
    rows = [_row(record, [0.9, 0.1] if index % 2 == 0 else [0.1, 0.9], label=index % 2,
                 record=f"test/custom/{record}")
            for record in range(4) for index in range(3)]
    one_shot = tmp_path / "one_shot.jsonl"
    eval_run.write_noul_row_dump(one_shot, rows)

    chunked = tmp_path / "chunked.jsonl"
    chunked.write_text("", encoding="utf-8")
    for start in range(0, len(rows), 5):
        lines, _ = eval_run.dump_lines(rows[start:start + 5], eval_run.NOUL)
        with open(chunked, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    order = [f"test/custom/{record}" for record in range(4)]
    kept, resume_at = rescore.resume_state(chunked, order)
    chunked.write_text("\n".join(kept) + "\n", encoding="utf-8")          # the rewrite the script does
    for start in range(resume_at * 3, len(rows), 5):
        lines, _ = eval_run.dump_lines(rows[start:start + 5], eval_run.NOUL)
        with open(chunked, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    assert chunked.read_bytes() == one_shot.read_bytes()
    assert eval_run.file_digest(chunked) == eval_run.file_digest(one_shot)


# ------------------------------------------------------------------ the pass, end to end below the model
def _record(index: int, labels: Sequence[int], *, family: str = "marker_noul",
            question_type: str = "noul") -> dict:
    """A suite-shaped record: ``labels`` gives each question's reference answer (0 = no, 1 = yes)."""
    return {
        "state": "1. CD3D\n2. CD8A",
        "questions": {f"holds:CL:{j:07d}": {"type": question_type, "src": family,
                                            "criteria": {"true": "yes", "false": "no"},
                                            "label": bool(label)} for j, label in enumerate(labels)},
        "_meta": {"id": f"test/custom/{index}", "source": "real-v4", "variant": "clean",
                  "family": family, "dataset_id": "synthetic", "cell_id": f"cell-{index}",
                  "donor_id": "d0", "group_id": f"group-{index}", "panel_id": "panel-0"},
    }


def _predictor(*, naming_yes: bool = True, fail_after: int | None = None):
    """A stand-in for the forward pass: every question gets the *same* answer, at P = 0.9.

    A constant answerer is what makes the arithmetic checkable — naming *yes* on every question of
    the test population is 3 of its 6 questions. ``fail_after`` raises once that many records have
    been scored, which is how a wall-clock kill is simulated: the real pass is killed between chunks,
    with whatever it has already written left on disk.
    """
    state = {"calls": 0}

    def predict(record):
        state["calls"] += 1
        if fail_after is not None and state["calls"] > fail_after:
            raise RuntimeError("simulated wall-clock kill")
        probabilities = {}
        for qid in record["questions"]:
            yes = 0.9 if naming_yes else 0.1
            probabilities[qid] = {"false": 1.0 - yes, "true": yes}
        return {"probabilities": probabilities}

    return predict, state


@pytest.fixture(scope="module")
def population():
    return [_record(0, [0, 1]), _record(1, [1, 1]), _record(2, [0, 0])]


def test_the_noul_pass_writes_every_question_and_the_same_bytes_appended(tmp_path, rescore, population):
    """The whole path below the model: ``score_partition`` -> noul writer. 6 questions, 6 rows."""
    one_shot = tmp_path / "one.jsonl"
    info, rejected, resumed = rescore.rescore(population, _predictor()[0], "test", one_shot,
                                      question_type=eval_run.NOUL, chunk=2)

    assert info["rows"] == 6 and rejected == [] and resumed == 0, "one row per question, none rejected"
    written = [json.loads(line) for line in one_shot.read_text().splitlines()]
    assert len(written) == 6 and all(row["keys"] == NOUL_KEYS for row in written)
    assert {row["id"] for row in written} == {record["_meta"]["id"] for record in population}
    assert [row["label"] for row in written] == [0, 1, 1, 1, 0, 0], "the reference's answers, in order"

    appended = tmp_path / "appended.jsonl"
    info2, _, _ = rescore.rescore(population, _predictor()[0], "test", appended, question_type=eval_run.NOUL,
                          chunk=2, append=True)
    assert info2["rows"] == info["rows"] == 6
    assert appended.read_bytes() == one_shot.read_bytes(), \
        "chunked writing is the same dump, byte for byte"


def test_a_killed_pass_resumes_and_lands_on_the_same_dump(tmp_path, rescore, population):
    """The failure the append mode exists for: killed after 2 of 3 records, then resumed."""
    killed = tmp_path / "killed.jsonl"
    with pytest.raises(RuntimeError, match="simulated"):
        rescore.rescore(population, _predictor(fail_after=2)[0], "test", killed,
                question_type=eval_run.NOUL, chunk=1, append=True)
    partial = [json.loads(line) for line in killed.read_text().splitlines()]
    assert len(partial) == 4, "the two records it did score are on disk, not lost with the process"

    info, _, resumed = rescore.rescore(population, _predictor()[0], "test", killed,
                               question_type=eval_run.NOUL, chunk=1, append=True)
    assert resumed == 1, "the last record present is re-scored rather than trusted"
    assert info["rows"] == 6

    clean = tmp_path / "clean.jsonl"
    rescore.rescore(population, _predictor()[0], "test", clean, question_type=eval_run.NOUL, chunk=2)
    assert killed.read_bytes() == clean.read_bytes(), \
        "a resumed dump is the dump a single-shot pass would have written"


def test_the_pass_answers_the_binary_question_and_scores_against_the_reference(tmp_path, rescore, population):
    """The number itself, on known answers: naming *yes* everywhere is 3/6, and the default is the yes-option."""
    path = tmp_path / "noul.jsonl"
    rescore.rescore(population, _predictor(naming_yes=True)[0], "test", path, question_type=eval_run.NOUL)
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    correct = sum(1 for row in rows if row["top1"] == row["label"])
    assert correct == 3, "answers are [no, yes, yes, yes, no, no] and the model says yes every time"
    assert all(row["top1"] == 1 for row in rows), "top1 == 1 is the model answering yes (p[1] = P(yes))"


def test_the_pass_refuses_to_write_an_empty_noul_dump(tmp_path, rescore):
    """A population with no noul rows must fail, not leave a file that reads like an unscored family."""
    choice_only = [_record(0, [0], family="panel_choice", question_type="choice")]
    with pytest.raises(ValueError, match="empty dump"):
        rescore.rescore(choice_only, _predictor()[0], "test", tmp_path / "none.jsonl",
                question_type=eval_run.NOUL, append=True)
    assert not (tmp_path / "none.jsonl").read_text(), "nothing written, not an empty file"
