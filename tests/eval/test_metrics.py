"""The reported statistics: hand-computable cases, ties, and the one seeded step.

These are the numbers that end up in the paper's tables, so the ones with a closed form are
checked against it rather than against a previous run of themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from sckevtype.eval import metrics


def _row(p, label, **fields):
    row = {"id": "row", "keys": [f"k{i}" for i in range(len(p))], "label": int(label), "p": list(p),
           "logits": list(np.log(np.asarray(p, dtype=float))), "inference_temperature": 1.0,
           "variant": "clean", "source": "panel_choice", "task": "panel_choice", "type": "choice",
           "size": len(p), "family": "panel_choice", "split": "test", "group": "g0"}
    row.update(fields)
    return row


def test_top1_accuracy_counts_the_argmax(make_rows):
    rows = [_row([0.7, 0.2, 0.1], 0), _row([0.7, 0.2, 0.1], 1), _row([0.1, 0.2, 0.7], 2)]
    assert metrics.top1_accuracy(rows) == {"n": 3, "accuracy": 2 / 3}
    assert metrics.top1_accuracy(rows, temperature=4.0)["accuracy"] == 2 / 3   # softmax is monotone


def test_top1_accuracy_refuses_an_empty_population():
    with pytest.raises(ValueError, match="empty"):
        metrics.top1_accuracy([])


def test_ece_separates_a_confident_right_model_from_a_confident_wrong_one():
    right = [_row([0.95, 0.05], 0) for _ in range(50)] + [_row([0.05, 0.95], 1) for _ in range(50)]
    wrong = [_row([0.95, 0.05], 1) for _ in range(50)] + [_row([0.05, 0.95], 0) for _ in range(50)]
    assert metrics.ece(right)["ece"] == pytest.approx(0.05)
    assert metrics.ece(wrong)["ece"] == pytest.approx(0.95)
    assert metrics.ece(right)["mean_confidence"] == pytest.approx(0.95)


def test_auroc_matches_its_closed_form():
    assert metrics.auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert metrics.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert metrics.auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == pytest.approx(0.5)
    # The positive ties one negative and loses to the other: half of one of two pairs.
    assert metrics.auroc([0.5, 0.5, 0.9], [True, False, False]) == pytest.approx(0.25)
    assert metrics.auroc([0.9, 0.5, 0.5], [False, True, False]) == pytest.approx(0.25)


def test_auroc_refuses_one_class():
    with pytest.raises(ValueError, match="both classes"):
        metrics.auroc([0.1, 0.2], [True, True])


def test_detection_at_fpr_stays_within_the_target(make_rows):
    """The threshold is a `higher` quantile of the negatives, so the achieved rate is not above it."""
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(1.0, 1.0, 400), rng.normal(-1.0, 1.0, 400)]
    labels = np.r_[np.ones(400, dtype=bool), np.zeros(400, dtype=bool)]
    report = metrics.detection_at_fpr(scores, labels, fpr=0.05)
    assert report["achieved_fpr"] <= 0.05
    assert report["detection_rate"] > 0.5
    assert report["n_positive"] == 400 and report["n_negative"] == 400


def test_detection_at_fpr_needs_a_valid_target():
    with pytest.raises(ValueError, match="target FPR"):
        metrics.detection_at_fpr([0.1, 0.9], [False, True], fpr=1.5)


def test_panel_adequacy_scores_the_abstain_option():
    """Axis 1 is a detection problem over ``none_p``, with the abstain target read off the row."""
    report = metrics.panel_adequacy(_adequacy_population())
    assert report["n"] == 60 and report["n_abstain_targets"] == 30
    assert report["auroc"] == 1.0
    assert report["detection_rate"] == 1.0        # the target FPR's own behaviour is the next test's
    assert report["per_family"]["panel_choice"]["n"] == 60


def test_panel_adequacy_refuses_rows_without_the_statistic(make_rows):
    rows = make_rows(5, 1, "test", with_none=False)
    with pytest.raises(ValueError, match="none_p"):
        metrics.panel_adequacy(rows)


def test_set_sizes_summarise_the_masks():
    sets = [np.array([1, 0, 0], dtype=bool), np.array([1, 1, 0], dtype=bool), np.array([0, 0, 0], dtype=bool)]
    assert metrics.set_sizes(sets) == {"n": 3, "mean": 1.0, "median": 1.0, "max": 2.0,
                                       "singleton_share": pytest.approx(1 / 3), "empty_share": pytest.approx(1 / 3)}


def test_set_sizes_refuse_nothing():
    with pytest.raises(ValueError, match="no sets"):
        metrics.set_sizes([])


def test_risk_coverage_is_bounded_and_ordered(make_rows):
    rows = make_rows(200, 5, "test")
    report = metrics.risk_coverage(rows)
    assert 0.0 <= report["aurc"] <= 1.0
    assert report["coverage_at_1pct_error"] <= report["coverage_at_5pct_error"] <= 1.0
    assert report["curve"], "an empty curve would make the two summaries meaningless"
    assert all(0.0 <= point["risk"] <= 1.0 and 0.0 < point["coverage"] <= 1.0 for point in report["curve"])
    assert all({"threshold", "accepted", "errors"} <= set(point) for point in report["curve"])


def test_temperature_scaling_fits_on_calibration_and_reports_on_test(make_rows):
    """A model that says 0.9 and is right 80% of the time: the fitted temperature must soften."""
    calibration = [_row([0.9, 0.1], 0, split="calibration") for _ in range(80)]
    calibration += [_row([0.9, 0.1], 1, split="calibration") for _ in range(20)]
    test = [_row([0.9, 0.1], 0, split="test") for _ in range(40)]
    test += [_row([0.9, 0.1], 1, split="test") for _ in range(10)]
    report = metrics.temperature_scaling(calibration, test)
    assert report["fit_partition"] == "calibration"
    assert report["fit_rows"] == 100 and report["report_rows"] == 50
    assert report["temperature"] > 1.0                       # overconfident -> softened, not sharpened
    assert report["before"]["ece"] == pytest.approx(0.10)
    assert report["after"]["ece"] < report["before"]["ece"]
    assert report["after"]["nll"] < report["before"]["nll"]


def test_temperature_scaling_needs_logits(make_rows):
    calibration = make_rows(20, 3, "calibration")
    for row in calibration:
        del row["logits"]
    with pytest.raises(ValueError, match="no logits"):
        metrics.temperature_scaling(calibration, make_rows(5, 4, "test"))


def test_breakdown_covers_every_group(make_rows):
    rows = make_rows(60, 6, "test", n_options=4) + make_rows(40, 7, "test", n_options=8, family="panel_miss")
    report = metrics.breakdown(rows, "size")
    assert set(report) == {"4", "8"}
    assert sum(entry["n"] for entry in report.values()) == 100
    with pytest.raises(ValueError, match="unknown breakdown key"):
        metrics.breakdown(rows, "colour")


def test_every_statistic_is_deterministic_on_the_same_rows(make_rows):
    """Nothing here samples: two calls on the same rows are the same numbers, bit for bit."""
    rows = make_rows(150, 9, "test")
    calibration = make_rows(150, 10, "calibration")
    assert metrics.top1_accuracy(rows) == metrics.top1_accuracy(rows)
    assert metrics.ece(rows) == metrics.ece(rows)
    assert metrics.risk_coverage(rows) == metrics.risk_coverage(rows)
    assert metrics.per_stratum(rows) == metrics.per_stratum(rows)
    assert metrics.breakdown(rows, "family") == metrics.breakdown(rows, "family")
    assert metrics.temperature_scaling(calibration, rows) == metrics.temperature_scaling(calibration, rows)
    assert metrics.panel_adequacy(_adequacy_population()) == metrics.panel_adequacy(_adequacy_population())


def _adequacy_population() -> list[dict]:
    """Half the rows target the abstain option, half a real one — the shape the corpus has."""
    rows = []
    for index in range(60):
        row = _row([0.45, 0.35, 0.20], 2 if index % 2 else 0)
        row["keys"] = ["CL:1", "CL:2", "not_listed"]
        row["none_p"] = float(0.8 + 0.19 * (index % 5) / 4) if index % 2 else float(0.01 * (index % 5))
        row["id"] = f"row-{index}"
        rows.append(row)
    return rows


def test_chosen_partition_counts_the_split(make_rows):
    rows = make_rows(4, 1, "calibration") + make_rows(6, 2, "test")
    assert metrics.chosen_partition(rows) == {"calibration": 4, "test": 6}
    assert metrics.REPORT_PARTITION == "test"


def test_fit_rows_only_is_the_guard_the_metrics_expose(make_rows):
    from sckevtype.eval import PartitionError

    assert len(metrics.fit_rows_only(make_rows(3, 1, "calibration"))) == 3
    with pytest.raises(PartitionError):
        metrics.fit_rows_only(make_rows(3, 1, "test"))
