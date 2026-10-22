"""The Baron paired tests: the rule they apply, the arm list they share, and the headline they answer.

`reports/paired-tests-baron.py` is the Baron counterpart of `reports/paired-tests-v4.py`: point estimates
become exact paired tests on the same records. Seven properties are load-bearing, and each is checked here
on hand-built data — no GPU, no cluster, no dump on the network share:

* **the rule is taken, not re-derived** — the exact two-sided binomial comes from the parsed source of
  `reports/abstain-reconciliation.py`, the file that settled it, and a moved definition is refused by name;
* **the arm list has one home** — the roster is built from `reports/abstain-reconciliation-v4.py`'s own
  `KEV_ARMS` / `LAYA_ARMS` through `reports/baron-reconciliation.py`'s `--corpus` naming, so no second
  hand-written arm list or dump pattern exists;
* **the cells are the discordant pairs** — `both / A only / B only / both wrong` come from the two arms'
  verdicts record by record, and two arms that do not cover the same records are refused;
* **the forced reading really removes the abstain option** — a record where the model declined is wrong
  served and can be right forced; a record served with no abstain option has one verdict;
* **the correction is Holm in log space** — no adjusted value is below its raw value, the step-up is
  monotone, and the family size travels with each row;
* **the headline is oriented to the 27B, whichever side of the pair it sits on** — the ordering checks are
  emitted small arm first, so a lookup keyed on the 27B being the left side would report the headline as
  untestable while the numbers sat in the table;
* **the published table re-derives from its own counts** — every p-value in
  `reports/paired-tests-baron.json` is `binom_two_sided(b, c)` on the discordant counts printed beside it,
  so a hand-edit or a stale run cannot survive.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT_JSON = REPORTS / "paired-tests-baron.json"
REPORT_MD = REPORTS / "paired-tests-baron.md"

TWENTY_SEVEN = "delta-kev-27b-v4-s1"


@pytest.fixture(scope="module")
def paired():
    spec = importlib.util.spec_from_file_location("paired_tests_baron", ROOT / "analysis" / "paired-tests-baron.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rule(paired):
    return paired.rule_from(paired.RULE_SOURCE, ("binom_two_sided", "binom_two_sided_log10"))


@pytest.fixture(scope="module")
def committed():
    assert REPORT_JSON.is_file(), "reports/paired-tests-baron.json has not been generated"
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


def row(label: int, probabilities, abstain=None) -> dict:
    """A choice row in the dump's shape: the vector, the reference index, and the abstain position."""
    return {"id": f"r{label}{len(probabilities)}{abstain}", "family": "panel_choice", "label": label,
            "abstain": abstain, "keys": [f"opt{index}" for index in range(len(probabilities))],
            "p": list(probabilities)}


def vectors(*flags, label: int = 0) -> dict:
    """``{id: {"label", "served"}}`` from a sequence of served verdicts."""
    return {f"r{index}": {"label": label, "served": bool(flag)} for index, flag in enumerate(flags)}


# --------------------------------------------------------------- the rule is taken, not re-written
def test_the_paired_test_comes_from_the_file_that_settled_it(rule):
    assert rule["binom_two_sided"](0, 0) == 1.0, "no discordant pairs, no evidence"
    assert rule["binom_two_sided"](5, 5) == 1.0, "a symmetrical split is the null itself"
    assert rule["binom_two_sided"](10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert rule["binom_two_sided"](0, 10) == rule["binom_two_sided"](10, 0), "two-sided by construction"
    assert rule["binom_two_sided_log10"](10, 0) == pytest.approx(
        math.log10(rule["binom_two_sided"](10, 0)))


def test_a_moved_rule_is_refused_rather_than_replaced(paired):
    with pytest.raises(paired.Refused, match="no longer defines"):
        paired.rule_from(ROOT / "analysis" / "paired-tests-baron.py", ("binom_two_sided",))


# --------------------------------------------------------------- one arm list, one naming rule
def test_the_roster_comes_from_the_settled_arm_list_and_the_corpus_infix(paired):
    """Every arm is the settled tables' own, and Baron's dumps carry the corpus infix v4's do not."""
    settled = paired.settled_module()
    baron = paired.roster("baron", REPORTS, ROOT / "runs")
    v4 = paired.roster("v4", REPORTS, ROOT / "runs")
    for slug, _shown, _declared in settled.KEV_ARMS:
        assert slug in baron and slug in v4, f"{slug} is missing from the roster"
        assert Path(baron[slug]["declared"]).name == f"forced-rows-{slug}-baron.jsonl"
        assert Path(v4[slug]["declared"]).name == f"forced-rows-{slug}.jsonl"
    assert TWENTY_SEVEN in baron, "the 27B arm must be in the ladder this table tests"
    assert baron[TWENTY_SEVEN]["shown_as"] == "fine-tuned 27B"
    for slug, _shown, _declared in settled.LAYA_ARMS:
        assert baron[slug]["kind"] == "laya"
        assert baron[slug]["canonical"].endswith("rows")


def test_a_fine_tune_without_its_rung_is_refused_rather_than_dropped(paired):
    """A missing rung is not a missing test: the ladder must fail, not quietly omit the pair."""
    arms = {"delta-kev-27b-v4-s1": {"kind": "kev"}, "delta-kev-9b-v4-s1": {"kind": "kev"},
            "kev-9b-v4": {"kind": "kev"}}
    with pytest.raises(paired.Refused, match="no rung"):
        paired.ladder(arms)


def test_the_ladder_is_ordered_by_parameter_count(paired):
    roster = paired.roster("baron", REPORTS, ROOT / "runs")
    sizes = [size for size, _left, _right in paired.ladder(roster)]
    assert sizes[-1] == "27b", f"the 27B must close the ladder, got {sizes}"
    assert sizes == sorted(sizes, key=lambda size: float(size.rstrip("b")))


# --------------------------------------------------------------- the readings, per row
def test_the_forced_reading_removes_the_abstain_option(paired):
    """A decline is wrong served and right forced; the same row is the same verdict with no option."""
    # index 1 is the abstain option and it is what the arm named; the label is 0.
    declined = row(0, [0.1, 0.8, 0.1], abstain=1)
    assert paired.verdicts(declined, "served") is False, "it named the abstain option, not the label"
    assert paired.verdicts(declined, "forced") is True, "with the option removed it names the label"

    no_option = row(0, [0.7, 0.3], abstain=None)
    assert paired.verdicts(no_option, "served") is True
    assert paired.verdicts(no_option, "forced") is True, "nothing to remove, so the verdict does not move"

    with pytest.raises(ValueError, match="unknown reading"):
        paired.verdicts(no_option, "something-else")


def test_the_local_argmax_and_forced_index_are_the_settled_ones(paired):
    """This file carries a copy of two rules; they must agree with the settled module on real shapes."""
    settled = paired.settled_module()
    for probabilities, abstain in (([0.1, 0.9], 0), ([0.4, 0.4, 0.2], 1), ([0.5, 0.5], None),
                                   ([0.2, 0.3, 0.5], 2)):
        candidate = {"p": probabilities, "abstain": abstain, "label": 0}
        assert paired.argmax(probabilities) == settled.argmax(probabilities)
        assert paired.forced_index(candidate) == settled.forced_index(candidate)


# --------------------------------------------------------------- the statistic
def test_the_contingency_is_counted_record_by_record(paired, rule):
    left = vectors(True, True, False, False, True)
    right = vectors(True, False, True, False, False)
    block = paired.mcnemar(left, right, "served", rule)
    assert block["n"] == 5
    assert block["contingency"] == {"both_correct": 1, "left_only": 2, "right_only": 1, "both_wrong": 1}
    assert block["discordant"] == 3
    assert block["p_value"] == rule["binom_two_sided"](2, 1)
    assert block["accuracy_left"] == pytest.approx(3 / 5)
    assert block["accuracy_right"] == pytest.approx(2 / 5)
    assert block["difference_points"] == pytest.approx(20.0)


def test_two_arms_on_different_records_are_refused(paired, rule):
    with pytest.raises(paired.Refused, match="not on the same records"):
        paired.mcnemar(vectors(True, False), {"only-one": {"label": 0, "served": True}}, "served", rule)


def test_holm_never_lowers_a_p_value_and_keeps_its_family_size(paired):
    entries = [{"p_value": p, "p_value_log10": math.log10(p)}
               for p in (1e-12, 0.004, 0.02, 0.3, 0.9)]
    paired.holm(entries)
    assert [entry["holm"]["family_size"] for entry in entries] == [5] * 5
    for entry in entries:
        assert entry["holm"]["p_value_log10"] >= entry["p_value_log10"] - 1e-12, "Holm only inflates"
        assert entry["holm"]["p_value"] >= entry["p_value"] * (1 - 1e-12)
    ordered = sorted(entries, key=lambda entry: entry["holm"]["rank"])
    adjusted = [entry["holm"]["p_value_log10"] for entry in ordered]
    assert adjusted == sorted(adjusted), "the step-up is monotone in the ranked order"
    assert entries[0]["holm"]["p_value_log10"] == pytest.approx(math.log10(1e-12) + math.log10(5))


# --------------------------------------------------------------- the headline's orientation
def _entry(left: str, right: str, reading: str, *, left_only: int, right_only: int,
           left_accuracy: float, right_accuracy: float, p_value: float = 0.0458) -> dict:
    """One comparison entry in the shape ``comparisons()`` emits (counts on the left/right arms)."""
    return {"comparison": f"{left} vs {right}", "group": "ordering", "reading": reading,
            "exploratory": True,
            "left": {"arm": left, "shown_as": left, "correct": 0, "accuracy": left_accuracy},
            "right": {"arm": right, "shown_as": right, "correct": 0, "accuracy": right_accuracy},
            "n": left_only + right_only, "discordant": left_only + right_only,
            "contingency": {"both_correct": 0, "left_only": left_only, "right_only": right_only,
                            "both_wrong": 0},
            "accuracy_left": left_accuracy, "accuracy_right": right_accuracy,
            "difference_points": 100 * (left_accuracy - right_accuracy),
            "p_value": p_value, "p_value_log10": math.log10(p_value), "holm": None}


def test_the_headline_is_oriented_to_the_27b_whichever_side_it_takes(paired):
    """The ordering checks are emitted (9B, 27B) — the 27B on the *right*. The headline must still read."""
    entries = [
        _entry("delta-kev-9b-v4-s1", TWENTY_SEVEN, "served", left_only=84, right_only=113,
               left_accuracy=0.8988, right_accuracy=0.9100),
        _entry("delta-kev-4b-v4-s1", TWENTY_SEVEN, "served", left_only=17, right_only=175,
               left_accuracy=0.8492, right_accuracy=0.9100, p_value=3.13e-34),
    ]
    block = paired.headline(entries, "baron")
    nine = block["pairs"]["27b_vs_9b_served"]
    four = block["pairs"]["27b_vs_4b_served"]
    assert nine["accuracy_27b"] == pytest.approx(0.9100)
    assert nine["accuracy_other"] == pytest.approx(0.8988)
    assert nine["discordant_27b_wins"] == 113, "the 27B's wins are the right side's `right_only`"
    assert nine["discordant_other_wins"] == 84
    assert nine["difference_points"] == pytest.approx(100 * 29 / 197)
    assert nine["significant_at_0.05"] is True
    assert four["discordant_27b_wins"] == 175
    assert "27B" in block["answer"]
    assert "vs 9B" in block["answer"] and "vs 4B" in block["answer"]


def test_the_headline_reports_absence_rather_than_a_number_when_the_pair_is_missing(paired):
    block = paired.headline([], "baron")
    assert block["pairs"] == {}
    assert "no 27B ordering pair" in block["answer"]


# --------------------------------------------------------------- the committed report
def test_the_committed_report_re_derives_every_p_value_from_its_own_counts(committed, rule):
    """A hand-edit or a stale run cannot survive: each p is the rule applied to the printed counts."""
    comparisons = committed["comparisons"]
    assert comparisons, "the report has no comparisons"
    for entry in comparisons:
        cells = entry["contingency"]
        assert cells["both_correct"] + cells["left_only"] + cells["right_only"] + cells["both_wrong"] \
            == entry["n"] == committed["population"]["rows"]
        assert entry["p_value"] == rule["binom_two_sided"](cells["left_only"], cells["right_only"])
        assert entry["p_value_log10"] == rule["binom_two_sided_log10"](cells["left_only"],
                                                                       cells["right_only"])
        if entry["holm"] is None:
            assert entry["exploratory"] is True, "only the exploratory checks escape the family"
        else:
            assert entry["holm"]["family_size"] == committed["multiplicity"]["families"][entry["reading"]]


def test_the_committed_report_carries_the_headline_pair_and_the_two_controls(committed):
    assert committed["population"]["rows"] == 2_600
    pairs = committed["headline"]["pairs"]
    assert set(pairs) == {"27b_vs_9b_served", "27b_vs_4b_served", "27b_vs_9b_forced", "27b_vs_4b_forced"}
    for entry in pairs.values():
        assert entry["accuracy_27b"] >= 0 and entry["accuracy_other"] >= 0
        assert entry["discordant_27b_wins"] + entry["discordant_other_wins"] > 0
    controls = committed["controls"]
    assert controls["n_reproduced"] == controls["n_checks"], "a control did not reproduce"
    assert any(check["value"] == 0.9100 for check in controls["checks"]), \
        "the Baron 27B served control must be asserted here"
    assert committed["checks"]["side_effects"]["changed"] == []
    assert committed["checks"]["reproduces_the_frozen_table"]["mismatches"] == 0


def test_the_committed_report_says_the_llm_has_no_forced_reading(committed):
    blocks = committed["not_testable"]
    assert blocks, "the missing forced reading must be stated, not dropped"
    assert all("no option-probability vector" in block["why"] for block in blocks)
    for entry in committed["comparisons"]:
        if entry["group"] == "external-llm":
            assert entry["reading"] == "served", "the LLM is never tested under forced"
    for arm in committed["arms"]:
        if arm["kind"] == "baseline" and arm["arm"].startswith("llm"):
            assert arm["readings"] == ["served"]


def test_the_markdown_carries_the_headline_section():
    assert REPORT_MD.is_file(), "reports/paired-tests-baron.md has not been generated"
    text = REPORT_MD.read_text(encoding="utf-8")
    assert "is the 27B's Baron lead real?" in text
    assert "Conclusions" not in text
    for arm in ("fine-tuned 27B", "fine-tuned 9B", "fine-tuned 4B", "Jev"):
        assert arm in text, f"{arm} is missing from the markdown table"
