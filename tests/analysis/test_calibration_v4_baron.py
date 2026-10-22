"""The calibration table's arithmetic: the bins, the Brier score, and the consistency of the report.

`reports/calibration-v4-baron.py` is the first ECE/Brier report this project has, so the things that can
go wrong are arithmetic and scope, not a model run. Each property below is checked on hand-built rows and
on the committed artefact — no GPU, no cluster, no dump on the network share:

* **ECE is the bin deviation the report says it is** — 10 equal-width bins on the top-1 probability, each
  weighted by its share, and the contributions sum to the ECE exactly;
* **the bin edges are half-open and the last bin is closed** — 1.0 lands in bin 9, 0.1 in bin 1, and a
  probability outside [0, 1] is refused rather than clamped;
* **the Brier score is multiclass over the full option set** — a 4-option row is summed over 4 terms, so a
  two-class reduction (which would flatter every arm) is not what is computed;
* **the direction is not lost** — mean top-1 probability against accuracy is printed separately from the
  ECE, because a scalar ECE cannot distinguish over-confidence from a flat, uninformative vector;
* **an empty set is refused** — no rows is not "perfectly calibrated";
* **the committed report is internally consistent** — every arm's bin counts sum to its n, its ECE is the
  sum of its own bins' contributions, and its accuracy is the count-weighted mean of its bins' accuracies;
* **the scope is stated** — every family is present on both suites with its own unit, and the general LLM
  is excluded by name with a reason instead of being zero-filled.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SCRIPT = ROOT / "analysis" / "calibration-v4-baron.py"
REPORT_JSON = REPORTS / "calibration-v4-baron.json"
REPORT_MD = REPORTS / "calibration-v4-baron.md"


@pytest.fixture(scope="module")
def calib():
    spec = importlib.util.spec_from_file_location("calibration_v4_baron", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed():
    assert REPORT_JSON.is_file(), "reports/calibration-v4-baron.json has not been generated"
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


def row(probabilities, label: int) -> dict:
    """One served row in the shape the metric functions take."""
    return {"p": list(probabilities), "label": label}


# --------------------------------------------------------------- the bins
@pytest.mark.parametrize("confidence, expected", [
    (0.0, 0), (0.05, 0), (0.1, 1), (0.35, 3), (0.999999, 9), (1.0, 9), (0.9, 9), (0.89, 8),
])
def test_the_bin_edges_are_half_open_and_the_last_bin_is_closed(calib, confidence, expected):
    assert calib.bin_index(confidence) == expected


def test_a_probability_outside_the_unit_interval_is_refused(calib):
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="outside"):
            calib.bin_index(bad)


# --------------------------------------------------------------- ECE and Brier, by hand
def test_ece_is_the_share_weighted_bin_deviation(calib):
    """Two rows in bin 9 at 1.0 confidence, one right and one wrong: ECE = 2/3 * |0.5 - 1.0|."""
    rows = [row([1.0, 0.0], 0), row([1.0, 0.0], 1), row([0.6, 0.4], 0)]
    block = calib.calibration(rows)
    assert block["n"] == 3
    assert block["bin_counts"] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 2]
    assert block["ece"] == pytest.approx((2 / 3) * abs(0.5 - 1.0) + (1 / 3) * abs(1.0 - 0.6))
    assert sum(entry["contribution"] for entry in block["bins"]) == pytest.approx(block["ece"])
    assert block["mean_top1_probability"] == pytest.approx((1.0 + 1.0 + 0.6) / 3)
    assert block["accuracy"] == pytest.approx(2 / 3)


def test_a_perfectly_confident_arm_has_zero_ece_and_a_perfect_brier(calib):
    rows = [row([1.0, 0.0], 0)] * 5
    block = calib.calibration(rows)
    assert block["ece"] == pytest.approx(0.0)
    assert block["brier"] == pytest.approx(0.0)
    assert block["bins_nonempty"] == 1


def test_the_brier_score_sums_over_the_full_option_set(calib):
    """A 2-option row and a 4-option row are scored over their own vectors, not a two-class reduction."""
    confident_wrong_2 = row([0.0, 1.0], 0)
    confident_wrong_4 = row([0.0, 0.0, 0.0, 1.0], 0)
    assert calib.brier([confident_wrong_2]) == pytest.approx(2.0)
    assert calib.brier([confident_wrong_4]) == pytest.approx(2.0)
    # a flat 4-option vector is worse under the full-option Brier than a flat 2-option one, which a
    # right-against-everything reduction would report as the same number
    flat_2 = calib.brier([row([0.5, 0.5], 0)])
    flat_4 = calib.brier([row([0.25, 0.25, 0.25, 0.25], 0)])
    assert flat_2 == pytest.approx(0.5)
    assert flat_4 == pytest.approx(0.75)
    assert flat_4 > flat_2


def test_the_direction_is_kept_beside_the_magnitude(calib):
    over = calib.calibration([row([0.9, 0.1], 1)] * 4)      # confident and always wrong
    # a three-option row it gets right with only 0.40 on the named option: correct but under-confident
    under = calib.calibration([row([0.4, 0.35, 0.25], 0)] * 4)
    assert over["mean_top1_minus_accuracy"] > 0.5
    assert under["mean_top1_minus_accuracy"] < -0.5
    assert over["ece"] == pytest.approx(abs(over["mean_top1_minus_accuracy"]))
    assert under["ece"] == pytest.approx(abs(under["mean_top1_minus_accuracy"]))


def test_the_largest_contributing_bin_is_named_with_its_share(calib):
    rows = [row([0.95, 0.05], 1)] * 3 + [row([0.35, 0.65], 1)] * 1
    block = calib.calibration(rows)
    worst = block["largest_contributing_bin"]
    assert worst["bin"] == block["bins"][worst["bin"]]["bin"]
    assert worst["contribution"] == max(entry["contribution"] for entry in block["bins"])
    assert worst["share_of_ece"] == pytest.approx(worst["contribution"] / block["ece"])


def test_an_empty_set_is_refused_rather_than_reported_as_calibrated(calib):
    with pytest.raises(ValueError, match="empty"):
        calib.calibration([])
    with pytest.raises(ValueError, match="empty"):
        calib.brier([])


# --------------------------------------------------------------- the committed report
def test_the_committed_report_covers_both_suites_and_every_family(committed):
    assert set(committed["corpora"]) == {"v4", "baron"}
    for corpus, block in committed["corpora"].items():
        assert set(block["families"]) == {"panel_choice", "evidence_diluted", "panel_miss",
                                         "marker_noul"}, corpus
        for family, cell in block["families"].items():
            assert cell["unit"] in ("record", "question")
            assert cell["n_expected"] > 0
            assert cell["arms"], f"{corpus}/{family} carries no arms"


def test_every_committed_arm_is_internally_consistent(committed):
    for corpus, block in committed["corpora"].items():
        for family, cell in block["families"].items():
            for slug, arm in cell["arms"].items():
                where = f"{corpus}/{family}/{slug}"
                assert len(arm["bin_counts"]) == arm["n_bins"] == 10, where
                assert sum(arm["bin_counts"]) == arm["n"], where
                assert arm["n"] == cell["n_expected"], where
                assert sum(entry["contribution"] for entry in arm["bins"]) == pytest.approx(arm["ece"]), \
                    where
                assert arm["bins_nonempty"] == sum(1 for count in arm["bin_counts"] if count), where
                # accuracy is the count-weighted mean of the bins' accuracies
                weighted = sum(entry["count"] * entry["accuracy"] for entry in arm["bins"]
                               if entry["count"])
                assert weighted / arm["n"] == pytest.approx(arm["accuracy"], abs=1e-12), where
                for entry in arm["bins"]:
                    if not entry["count"]:
                        assert entry["mean_confidence"] is None and entry["contribution"] == 0.0, where
                assert arm["mean_top1_minus_accuracy"] == pytest.approx(
                    arm["mean_top1_probability"] - arm["accuracy"]), where
                assert arm["brier"] >= 0.0, where


def test_the_committed_report_excludes_the_llm_by_name_and_keeps_jev_in(committed):
    assert committed["excluded"], "the exclusion must be stated, not silent"
    for name, why in committed["excluded"].items():
        assert "no option-probability vector" in why, name
    for corpus, block in committed["corpora"].items():
        for family, cell in block["families"].items():
            assert "jev" in cell["arms"], f"{corpus}/{family} is missing Jev, which carries a vector"
            assert not any("llm" in slug for slug in cell["arms"]), \
                f"{corpus}/{family} carries an LLM row it cannot score"


def test_the_committed_report_reproduced_the_frozen_accuracies(committed):
    check = committed["checks"]["accuracy_matches_the_frozen_table"]
    assert check["mismatches"] == 0
    assert check["cells_compared"] >= 100, "the frozen-accuracy check must compare the whole grid"
    assert committed["checks"]["side_effects"]["changed"] == []


def test_the_committed_report_names_the_marker_noul_unit_separately(committed):
    for block in committed["corpora"].values():
        noul = block["families"]["marker_noul"]
        assert noul["unit"] == "question"
        assert noul["n_expected"] > block["families"]["panel_choice"]["n_expected"], \
            "marker_noul is more questions than the population has records"
        assert block["families"]["panel_choice"]["unit"] == "record"


def test_the_markdown_states_the_method_and_the_bin_counts():
    assert REPORT_MD.is_file(), "reports/calibration-v4-baron.md has not been generated"
    text = REPORT_MD.read_text(encoding="utf-8")
    for phrase in ("ECE", "Brier", "10 equal-width bins", "mean top-1 p", "bin counts"):
        assert phrase in text, phrase
    assert "*general LLM (variants a and b)*" in text, "the exclusion is not stated in the markdown"
