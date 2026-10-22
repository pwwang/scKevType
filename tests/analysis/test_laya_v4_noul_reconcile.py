"""The Laya noul join: one row per **question**, and the dump the shared aggregator reads.

`reports/laya-v4-noul-reconcile.py` is the join the Laya halves of the `marker_noul` family go through.
Four of its properties are load-bearing and are checked here on hand-built rows, with no GPU, no
cluster and no suite:

* the unit is the **question** -- keyed by `(record, question)`, because keying a noul family by
  `(record, family)` (the three choice families' join) keeps one question of every 15.8 and prints a
  number that reads like the family's;
* the acceptance is the population's own **question count**, and a population that is short is refused
  instead of scored -- and is **not** written out as a dump, because a partial dump under the lane's own
  name is a population that can be read as a finished one;
* the rows it writes are the shared lane's own shape (`FULL_DUMP_FIELDS`, the writer the kev passes
  use), which is what the read-back below checks: the file is read through `scripts/noul-accuracy.py`'s
  own reader and the two sets of numbers must agree;
* the dump's **name** is the aggregator's own key for the arm (`forced-rows-<slug>-marker_noul.jsonl`),
  so a dump under any other name is a number the shared table would never look up.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def join():
    spec = importlib.util.spec_from_file_location(
        "laya_v4_noul_reconcile", ROOT / "analysis" / "laya-v4-noul-reconcile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _laya_row(record: str, question: str, p, label: int, ask: str = "base", type: str = "noul") -> dict:
    """A row in the shape `sckevtype.laya.score infer` writes: the item's fields plus what came back."""
    return {"id": record, "question": question, "split": "test", "type": type, "ask": ask,
            "family": "marker_noul", "variant": "clean", "source": "census", "task": "holds",
            "qtype": 2, "keys": ["false", "true"], "label": label, "p": list(p),
            "logits": [0.0, 0.0], "dataset": "synthetic", "cell": "c0", "donor": "d0", "panel": "p0"}


_counter = itertools.count()


def _rows_dir(tmp_path: Path, rows) -> Path:
    """One rows directory per call: a second reconcile in a test must not read the first one's rows."""
    directory = tmp_path / f"rows{next(_counter)}"
    directory.mkdir()
    (directory / "census-000000.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return directory


def _population(tmp_path: Path, records) -> Path:
    path = tmp_path / "ids.jsonl"
    path.write_text("\n".join(json.dumps({"id": record, "family": "marker_noul"}) for record in records)
                    + "\n", encoding="utf-8")
    return path


def test_the_unit_is_the_question_and_the_dump_is_read_back_by_the_shared_reader(tmp_path, join):
    """Four questions over two records: one row per question, and the aggregator agrees on the bytes."""
    rows = [_laya_row("test/custom/0", "holds:CL:0000001", [0.9, 0.1], 0),   # says no,  reference no
            _laya_row("test/custom/0", "holds:CL:0000002", [0.2, 0.8], 0),   # says yes, reference no
            _laya_row("test/custom/0", "holds:CL:0000003", [0.9, 0.1], 1),   # says no,  reference yes
            _laya_row("test/custom/1", "holds:CL:0000001", [0.1, 0.9], 1)]   # says yes, reference yes
    out = tmp_path / "reconcile.json"
    dump = tmp_path / "forced-rows-laya-test-half-marker_noul.jsonl"

    rc = join.main(["--rows", str(_rows_dir(tmp_path, rows)),
                    "--ids", str(_population(tmp_path, ["test/custom/0", "test/custom/1"])),
                    "--label", "laya-test-half", "--expect-questions", "4",
                    "--out", str(out), "--dump", str(dump)])

    result = json.loads(out.read_text())
    cell = result["noul_questions_4"]
    assert rc == 0
    assert result["unit"].startswith("questions")
    assert (cell["n"], cell["records"], cell["served"]["correct"]) == (4, 2, 2)
    assert cell["served"]["accuracy"] == 0.5
    assert cell["reference"]["yes_rate"] == pytest.approx(0.5)
    assert cell["majority_floor"] == pytest.approx(0.5) and cell["margin_over_floor"] == pytest.approx(0.0)
    assert cell["says_yes"]["rate"] == pytest.approx(0.5)
    assert result["acceptance"]["passed"] is True

    # The dump: the shared lane's own shape, no `type` field, one object per question, and the
    # aggregator's own key for the arm.
    written = [json.loads(line) for line in dump.read_text().splitlines()]
    assert len(written) == 4
    assert all("type" not in row for row in written), "the type is the writer's filter, not a field"
    assert all(row["keys"] == ["false", "true"] and len(row["p"]) == 2 for row in written)
    assert result["dump"]["rows"] == 4 and result["dump"]["sha256"]

    check = result["aggregator_check"]
    assert check["agrees"] is True, "the file just written must be the file the shared reader scores"
    assert check["portrait"]["correct"] == 2 and check["portrait"]["questions"] == 4
    assert Path(check["dump_path_the_aggregator_looks_for"]) == dump


def test_a_short_population_is_refused_and_never_written_out_as_a_dump(tmp_path, join):
    """Two of four questions is a prefix: no number, and above all no dump under the lane's name."""
    rows = [_laya_row("test/custom/0", "holds:CL:0000001", [0.9, 0.1], 0),
            _laya_row("test/custom/0", "holds:CL:0000002", [0.9, 0.1], 1)]
    out = tmp_path / "reconcile.json"
    dump = tmp_path / "forced-rows-laya-test-half-marker_noul.jsonl"

    rc = join.main(["--rows", str(_rows_dir(tmp_path, rows)),
                    "--ids", str(_population(tmp_path, ["test/custom/0", "test/custom/1"])),
                    "--label", "laya-test-half", "--expect-questions", "4",
                    "--out", str(out), "--dump", str(dump)])

    assert rc == 2 and not out.exists()
    assert not dump.exists(), "a partial population must not appear under the aggregator's name"

    # ... and with the flag it is reported as incomplete, still without a dump.
    rc = join.main(["--rows", str(_rows_dir(tmp_path, rows)),
                    "--ids", str(_population(tmp_path, ["test/custom/0", "test/custom/1"])),
                    "--label", "laya-test-half", "--expect-questions", "4", "--allow-missing",
                    "--out", str(out), "--dump", str(dump)])
    result = json.loads(out.read_text())
    assert rc == 0 and result["coverage"]["complete"] is False
    assert result["acceptance"]["passed"] is False and "dump" not in result
    assert not dump.exists()


def test_a_choice_row_and_a_non_yes_no_row_are_refused_by_name(tmp_path, join):
    """This join scores yes/no questions: the two ways a row can be something else are named."""
    population = _population(tmp_path, ["test/custom/0"])
    with pytest.raises(join.ReconcileError, match="not a noul row"):
        join.main(["--rows", str(_rows_dir(tmp_path, [_laya_row("test/custom/0", "label", [0.9, 0.1], 0,
                                                               type="choice")])),
                   "--ids", str(population), "--expect-questions", "1", "--out", str(tmp_path / "x.json")])

    bad = _laya_row("test/custom/0", "holds:CL:0000001", [0.9, 0.1], 0)
    bad["keys"] = ["alpha", "beta"]
    with pytest.raises(join.ReconcileError, match="not \\['false', 'true'\\]"):
        join.main(["--rows", str(_rows_dir(tmp_path, [bad])), "--ids", str(population),
                   "--expect-questions", "1", "--out", str(tmp_path / "x.json")])


def test_a_record_outside_the_population_is_refused_rather_than_scored(tmp_path, join):
    """A question from outside the population would move the denominator of the family's number."""
    rows = [_laya_row("test/custom/0", "holds:CL:0000001", [0.9, 0.1], 0),
            _laya_row("test/elsewhere/9", "holds:CL:0000001", [0.9, 0.1], 0)]
    rc = join.main(["--rows", str(_rows_dir(tmp_path, rows)),
                    "--ids", str(_population(tmp_path, ["test/custom/0"])),
                    "--expect-questions", "2", "--out", str(tmp_path / "x.json")])
    assert rc == 2 and not (tmp_path / "x.json").exists()


def _four_questions(tmp_path: Path) -> tuple[Path, Path]:
    """Two records, four yes/no questions: the smallest population that is both complete and scored."""
    rows = [_laya_row("test/custom/0", "holds:CL:0000001", [0.9, 0.1], 0),
            _laya_row("test/custom/0", "holds:CL:0000002", [0.2, 0.8], 0),
            _laya_row("test/custom/0", "holds:CL:0000003", [0.9, 0.1], 1),
            _laya_row("test/custom/1", "holds:CL:0000001", [0.1, 0.9], 1)]
    return _rows_dir(tmp_path, rows), _population(tmp_path, ["test/custom/0", "test/custom/1"])


def test_the_baron_corpus_composes_the_infix_and_the_aggregator_key(tmp_path, join):
    """``--corpus baron`` is the whole fix: the same arm on the external-validation census.

    The Baron lane writes ``forced-rows-<slug>-baron-marker_noul.jsonl``, so an arm derived from the
    slug alone (``laya-base``) named a path that does not exist and the family read as absent while its
    dump sat on disk complete. Here the dump name, the arm the JSON reports and the key the shared
    aggregator is asked for are all composed from the selector -- and the two censuses' names are
    **different files**, neither derived from the other.
    """
    rows_dir, population = _four_questions(tmp_path)
    out = tmp_path / "reconcile.json"
    dump = tmp_path / "forced-rows-laya-base-baron-marker_noul.jsonl"

    rc = join.main(["--corpus", "baron", "--rows", str(rows_dir), "--ids", str(population),
                    "--label", "laya-base", "--expect-questions", "4",
                    "--out", str(out), "--dump", str(dump)])

    result = json.loads(out.read_text())
    assert rc == 0
    assert result["label"] == "laya-base-baron", "the arm key carries the census infix"
    assert result["corpus"]["selector"] == "baron" and result["corpus"]["arm_key"] == "laya-base-baron"
    check = result["aggregator_check"]
    assert check["agrees"] is True and check["arm_key"] == "laya-base-baron"
    assert Path(check["dump_path_the_aggregator_looks_for"]) == dump
    assert dump.exists()

    # The v4 name for the same arm is a different file, and the v4 JSON is what it always was: no
    # `corpus` block, the bare slug as its label.
    v4_out = tmp_path / "reconcile-v4.json"
    v4_dump = tmp_path / "forced-rows-laya-base-marker_noul.jsonl"
    rc = join.main(["--rows", str(rows_dir), "--ids", str(population), "--label", "laya-base",
                    "--expect-questions", "4", "--out", str(v4_out), "--dump", str(v4_dump)])
    v4 = json.loads(v4_out.read_text())
    assert rc == 0 and v4["label"] == "laya-base" and "corpus" not in v4
    assert v4_dump.exists() and v4_dump != dump
    assert v4["noul_questions_4"] == result["noul_questions_4"], "same rows, same census-independent reading"


def test_the_census_defaults_and_the_double_infix(tmp_path, join, capsys):
    """Each census's own acceptance, and a label that already carries the infix is refused by name."""
    assert join.EXPECTED_QUESTIONS == {"v4": 63851, "baron": 39197}

    rows_dir, population = _four_questions(tmp_path)
    rc = join.main(["--corpus", "baron", "--rows", str(rows_dir), "--ids", str(population),
                    "--label", "laya-base-baron", "--expect-questions", "4",
                    "--out", str(tmp_path / "x.json")])
    assert rc == 2 and not (tmp_path / "x.json").exists(), "laya-base-baron-baron is not an arm"
    assert "already ends with the 'baron' corpus infix" in capsys.readouterr().err

    # Without `--expect-questions` a baron run is gated on the Baron noul count, not the v4 one -- the
    # number the refusal names is what proves which default it took.
    rc = join.main(["--corpus", "baron", "--rows", str(rows_dir), "--ids", str(population),
                    "--label", "laya-base", "--out", str(tmp_path / "y.json")])
    assert rc == 2 and not (tmp_path / "y.json").exists()
    assert "4 of 39197 questions" in capsys.readouterr().err

