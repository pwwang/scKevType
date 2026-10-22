"""The forced-choice re-score: the vector the forced rule needs, and the population it is taken on.

Two things are checked here, both without a GPU, a checkpoint or a suite on disk:

* ``write_full_row_dump`` now carries ``p`` — the vector ``top1`` is the argmax of — so a rule that
  drops the abstain option can be applied to a row after the fact;
* ``scripts/score-forced-rows.py``'s population selection: which ids it takes, in which order, and
  its refusals. The refusals are the point: a requested id the partition does not carry must fail the
  run rather than quietly score a smaller population.

The rule itself is checked in ``tests/analysis/test_arms_forced.py`` (it is taken from
``reports/abstain-reconciliation.py``'s own source, so there is nothing here to compare it against).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sckevtype.eval import run as eval_run

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rescore():
    return _load_script("score_forced_rows", "score-forced-rows.py")


def _row(index: int, p=None, *, keys=("alpha", "beta"), label: int = 0, **extra) -> dict:
    row = {"id": f"row-{index}", "question": "label", "split": "test", "type": "choice",
           "dataset": "synthetic", "donor": "d0", "family": "panel_choice", "variant": "clean",
           "keys": list(keys), "label": label, "size": len(keys),
           "abstain": None, "abstain_source": "factory", "abstain_key": None, "none_p": None}
    if p is not None:
        row["p"] = list(p)
    row.update(extra)
    return row


# --------------------------------------------------------------------- the vector is retained
def test_the_full_dump_carries_the_vector_top1_is_the_argmax_of(tmp_path):
    """The one field that makes the forced rule re-appliable: `p`, aligned with `keys`."""
    probabilities = [0.05, 0.7, 0.25]
    info = eval_run.write_full_row_dump(
        tmp_path / "full.jsonl",
        [_row(0, probabilities, keys=("alpha", "beta", "gamma"), label=2)])
    line = json.loads((tmp_path / "full.jsonl").read_text().splitlines()[0])

    assert line["p"] == probabilities
    assert len(line["p"]) == len(line["keys"]), "the vector is aligned with the option list"
    assert max(range(len(line["p"])), key=lambda i: line["p"][i]) == line["top1"]
    assert line["p"][line["top1"]] == line["top1_p"]
    assert set(line) == set(eval_run.FULL_DUMP_FIELDS)
    assert info["rows"] == 1


def test_the_vector_is_written_as_plain_floats(tmp_path):
    """A numpy scalar must not reach ``json.dumps``; the line is JSON by construction."""
    numpy = pytest.importorskip("numpy")
    eval_run.write_full_row_dump(
        tmp_path / "full.jsonl", [_row(0, [numpy.float32(0.25), numpy.float32(0.75)])])
    line = json.loads((tmp_path / "full.jsonl").read_text().splitlines()[0])
    assert line["p"] == [0.25, 0.75]
    assert all(type(value) is float for value in line["p"]), "plain floats, not numpy scalars"


def test_a_row_with_no_abstain_option_has_nothing_to_drop(tmp_path):
    """Injected rows carry the question as served: `keys` has no abstain option, `p` is over it."""
    eval_run.write_full_row_dump(
        tmp_path / "full.jsonl",
        [_row(0, [0.4, 0.6], abstain=None, abstain_source="injected", abstain_key="other")])
    line = json.loads((tmp_path / "full.jsonl").read_text().splitlines()[0])
    assert line["abstain"] is None and "other" not in line["keys"]
    assert len(line["p"]) == len(line["keys"])


# --------------------------------------------------------------------- the population
def _ids_file(tmp_path, records) -> Path:
    path = tmp_path / "ids.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_read_ids_filters_on_the_family_field(tmp_path, rescore):
    path = _ids_file(tmp_path, [
        {"id": "test/custom/0", "family": "panel_choice"},
        {"id": "test/custom/1", "family": "panel_miss"},
        {"id": "test/custom/2", "family": "panel_choice"},
    ])
    assert rescore.read_ids(path, "panel_choice") == ["test/custom/0", "test/custom/2"]
    assert rescore.read_ids(path, None) == ["test/custom/0", "test/custom/1", "test/custom/2"]


def test_read_ids_refuses_a_repeated_id(tmp_path, rescore):
    """A repeated id would score one question twice and weight it double in every rate."""
    path = _ids_file(tmp_path, [{"id": "test/custom/0", "family": "panel_choice"},
                                {"id": "test/custom/0", "family": "panel_choice"}])
    with pytest.raises(ValueError, match="repeated"):
        rescore.read_ids(path, "panel_choice")


def test_read_ids_refuses_an_empty_selection(tmp_path, rescore):
    path = _ids_file(tmp_path, [{"id": "test/custom/0", "family": "panel_miss"}])
    with pytest.raises(ValueError, match="no ids"):
        rescore.read_ids(path, "panel_choice")


def test_select_records_keeps_the_suites_order_and_refuses_a_missing_id(rescore):
    records = [{"_meta": {"id": "test/custom/2"}}, {"_meta": {"id": "test/custom/0"}},
               {"_meta": {"id": "test/custom/1"}}]
    chosen = rescore.select_records(records, ["test/custom/1", "test/custom/0"], "test")
    assert [record["_meta"]["id"] for record in chosen] == ["test/custom/0", "test/custom/1"], \
        "the suite's order, not the id file's"

    with pytest.raises(ValueError, match="not in the test partition"):
        rescore.select_records(records, ["test/custom/0", "test/custom/9"], "test")


def test_select_records_covers_the_whole_population(rescore):
    """Every requested id is scored exactly once: the count out equals the count in."""
    records = [{"_meta": {"id": f"test/custom/{i}"}} for i in range(10)]
    chosen = rescore.select_records(records, [f"test/custom/{i}" for i in range(10)], "test")
    assert len(chosen) == 10
