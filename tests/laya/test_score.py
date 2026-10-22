"""The two parts of the scoring step that have logic: which rows exist, and what the gate says.

``_tensors`` and ``answer`` need a checkpoint and a GPU, so the cluster job is what exercises them.
What is checked here is what a wrong version would change silently: a choice row emitted without an
axis-1 statistic (which lowers the AUROC's denominator without saying so), and a criterion compared
with the wrong operator (which decides the pilot).
"""

from __future__ import annotations

import pytest

from sckevtype.data.families import NONE_OPTIONS
from sckevtype.laya import score

ABSTAIN = NONE_OPTIONS[0][0]


def _answer(ask: str, keys: list[str], p: list[float], **fields):
    answer = {"ask": ask, "id": "r0", "question": "q0", "keys": keys, "p": p, "logits": list(p),
              "label": 0, "qtype": 0, "type": "choice", "split": "test", "family": "panel_choice",
              "variant": "clean", "source": "panel_choice", "task": "panel_choice", "dataset": "d",
              "cell": "c", "donor": "n", "panel": "p", "group": "g"}
    return {**answer, **fields}


def test_a_question_that_already_offers_abstain_takes_its_statistic_from_the_base_ask():
    rows, rejected = score.rows_from_answers([_answer("base", ["A", ABSTAIN, "B"], [0.2, 0.5, 0.3])])
    assert not rejected
    assert (rows[0]["abstain"], rows[0]["none_p"]) == (1, 0.5)


def test_a_question_without_one_takes_the_variant_ask():
    rows, rejected = score.rows_from_answers([
        _answer("base", ["A", "B"], [0.6, 0.4]),
        _answer("abstain", ["A", "B", ABSTAIN], [0.3, 0.2, 0.5]),
    ])
    assert not rejected
    assert rows[0]["p"] == [0.6, 0.4]  # the scored distribution is the one without the abstain option
    assert rows[0]["abstain"] is None
    assert rows[0]["none_p"] == 0.5


def test_a_choice_row_with_no_variant_is_rejected_rather_than_emitted_without_a_statistic():
    rows, rejected = score.rows_from_answers([_answer("base", ["A", "B"], [0.6, 0.4])])
    assert rows == []
    assert len(rejected) == 1 and "abstain variant" in rejected[0]["error"]


def test_a_noul_row_needs_no_abstain_statistic():
    rows, rejected = score.rows_from_answers([_answer("base", ["A", "B"], [0.6, 0.4], type="noul", qtype=2)])
    assert not rejected and rows[0]["none_p"] is None and rows[0]["abstain"] is None


def test_the_gate_is_the_specs_three_criteria_in_order():
    assert [(entry["name"], entry["operator"], entry["threshold"]) for entry in score.criteria(
        {"accuracy": 0.9, "ece_after_temperature": 0.05, "axis1_auroc": 0.7})] == [
        ("accuracy", ">", 0.899), ("ECE", "≤", 0.059), ("axis-1 AUROC", ">", 0.611)]


@pytest.mark.parametrize("name, given, passed", [
    ("accuracy", {"accuracy": 0.899}, False),            # ">" is strict: the transfer number itself fails
    ("accuracy", {"accuracy": 0.8991}, True),
    ("ECE", {"ece_after_temperature": 0.059}, True),     # "≤" is not: step 11's own ECE passes
    ("ECE", {"ece_after_temperature": 0.0591}, False),
    ("axis-1 AUROC", {"axis1_auroc": 0.611}, False),
    ("axis-1 AUROC", {"axis1_auroc": 0.6111}, True),
])
def test_each_boundary_lands_the_way_the_sentence_reads(name, given, passed):
    by_name = {entry["name"]: entry for entry in score.criteria(
        {"accuracy": 0.9, "ece_after_temperature": 0.05, "axis1_auroc": 0.7, **given})}
    assert by_name[name]["passed"] is passed


def test_a_number_the_gate_names_but_the_run_lacks_raises_instead_of_passing():
    with pytest.raises(score.ScoreError, match="axis1_auroc"):
        score.criteria({"accuracy": 0.9, "ece_after_temperature": 0.05})


def test_the_verdict_needs_all_three():
    def entry(name, passed):
        return {"name": name, "operator": ">", "threshold": 0.0, "observed": 0.0, "passed": passed}

    assert score.verdict([entry("accuracy", True), entry("ECE", True), entry("axis-1 AUROC", True)]) == "PASS"
    assert score.verdict([entry("accuracy", True), entry("ECE", False), entry("axis-1 AUROC", True)]) == "FAIL"
