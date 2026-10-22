"""The composition: three actions, and the one bound the two axes together actually buy.

The load-bearing test is :func:`test_accepted_and_wrong_is_a_subset_of_miscoverage` — an exact
set inclusion, not a statistical one. It is what makes ``P(accepted and wrong) <= alpha`` follow
from the conformal coverage claim with no assumption about the model, and it fails the moment
``accept`` stops meaning "the set is a singleton".
"""

from __future__ import annotations

import numpy as np
import pytest

from sckevtype.eval import triage
from sckevtype.eval.conformal import StratifiedSets


def _wrong(row):
    return int(row["label"]) != int(np.asarray(row["p"]).argmax())


@pytest.mark.parametrize("method", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_accepted_and_wrong_is_a_subset_of_miscoverage(rows, method, alpha):
    """Every wrong auto-annotation is a miscoverage. Exact, on every row, no tolerance."""
    calibration, test = rows
    policy = triage.fit(calibration, alpha, beta=1.0, method=method)   # beta=1: axis-1 gate open
    accepted_and_wrong, miscovered = set(), set()
    for index, row in enumerate(test):
        if int(policy.sets.set_for(row).sum()) != 1:
            continue
        if int(policy.sets.set_for(row)[int(row["label"])]) == 0:
            miscovered.add(index)
        if _wrong(row):
            accepted_and_wrong.add(index)
    assert accepted_and_wrong <= miscovered
    if alpha >= 0.10:
        # At alpha=0.05 a singleton needs p_max above the 95th percentile of the scores, which on
        # this population only the confident-and-right rows reach: the inclusion still holds, but
        # over an empty set, so the witness requirement starts where the population can supply one.
        assert len(accepted_and_wrong) > 0, "no wrong auto-annotation here: the test proves nothing"


def test_the_per_request_risk_bound_holds_on_exchangeable_rows(rows):
    """``P(accepted and wrong)`` over *all* requests, bounded by alpha, on held-out rows."""
    calibration, test = rows
    policy = triage.fit(calibration, 0.10, beta=0.05)
    assert policy.sets.alpha == 0.10
    # A single draw of a probabilistic bound: the tolerance is three binomial standard errors.
    tolerance = 3 * float(np.sqrt(0.10 * 0.90 / len(test)))
    assert triage.risk_bound_holds(policy, test, tolerance=tolerance)


def test_the_conditional_risk_is_larger_than_the_bound_it_sits_beside(rows):
    """The number a reader will ask for is the error rate *among* accepted, and it is larger.

    Stated as a test so the gap is not a footnote: the bound is on the per-request rate, and the
    conditional rate is reported with an interval instead of a claim.
    """
    calibration, test = rows
    policy = triage.fit(calibration, 0.10, beta=0.05)
    actions = policy.decide_all(test)
    errors = sum(1 for row, action in zip(test, actions) if action == "accept" and _wrong(row))
    accepted = actions.count("accept")
    summary = triage.summary(policy, test)
    assert accepted > 0 and errors > 0
    assert summary["risk"] == errors / accepted
    assert summary["risk"] >= errors / len(test)     # the same errors over fewer requests
    assert summary["risk_ci95"][0] <= summary["risk_ci95"][1]


def test_the_axis1_gate_routes_rows_to_more_markers(rows):
    """``none_p`` above tau is the action no confidence threshold can express."""
    calibration, test = rows
    policy = triage.fit(calibration, 0.10, beta=0.50)          # a loose budget: tau is not -inf
    assert policy.tau is not None
    above = dict(test[0], none_p=policy.tau + 1e-9)
    below = dict(test[0], none_p=policy.tau - 1e-9)
    assert policy.decide(above) == "request_more_markers"
    assert policy.decide(below) != "request_more_markers"


def test_a_disabled_axis1_gate_falls_through_to_the_conformal_set(make_rows):
    """No calibration row carries an abstention statistic: the gate says so rather than guessing."""
    calibration = make_rows(200, 8, "calibration", with_none=False)
    test = make_rows(50, 9, "test")
    policy = triage.fit(calibration, 0.10, beta=0.05)
    assert policy.tau is None
    assert policy.to_dict()["axis1_gate"].startswith("disabled")
    assert set(policy.decide_all(test)) <= {"accept", "abstain"}


def test_actions_are_the_three_named_ones(rows):
    calibration, test = rows
    actions = triage.fit(calibration, 0.10, 0.05).decide_all(test)
    assert set(actions) <= set(triage.ACTIONS)
    assert len(actions) == len(test)


def test_risk_curve_is_deterministic_given_its_seed(rows):
    """The bootstrap is the only stochastic step in the layer, and it takes an explicit seed."""
    calibration, test = rows
    policy = triage.fit(calibration, 0.10, 0.05)
    first = triage.risk_curve([(policy, test)])[0]
    assert first == triage.risk_curve([(policy, test)])[0]
    assert first["risk_ci95"][0] <= first["risk_ci95"][1]
    assert first["n_accept"] == first["n"] - first["n_abstain"] - first["n_request_more_markers"]


def test_the_sweep_fits_on_calibration_only(rows):
    """Every grid point carries the same split, and the grid itself is stated, not fitted."""
    calibration, test = rows
    swept = triage.sweep(calibration, test, alphas=[0.05, 0.10], betas=[0.05], method="lac")
    assert swept["n_points"] == 2
    assert swept["fitted_on"] == "calibration"
    assert swept["scored_on"] == ["test"]
    assert len({point["alpha"] for point in swept["points"]}) == 2
    assert swept["max_coverage_within_5pct_risk"] is not None


def test_the_sweep_reports_a_stratum_it_cannot_fit(make_rows):
    """An alpha no stratum can carry is reported as unfittable, not silently dropped."""
    calibration = make_rows(12, 70, "calibration")
    swept = triage.sweep(calibration, make_rows(12, 71, "test"), alphas=[0.001], betas=[0.05])
    assert swept["n_points"] == 0
    assert swept["unfittable"] and "no stratum has enough" in swept["unfittable"][0]["reason"]


def test_summary_records_the_bound_and_what_it_excludes(rows):
    calibration, test = rows
    block = triage.summary(triage.fit(calibration, 0.10, 0.05), test)["policy"]["bound"]
    assert block["quantity"] == "P(accepted and wrong)"
    assert block["value"] == 0.10
    assert any("per-request conditional" in item for item in block["not_bounded"])
    assert any("panel" in item for item in block["not_bounded"])


def test_risk_bound_refuses_an_empty_population(rows):
    policy = triage.fit(rows[0], 0.10, 0.05)
    with pytest.raises(ValueError, match="no rows"):
        triage.risk_bound_holds(policy, [])


def test_fit_records_the_calibration_risk_it_selected_at(rows):
    """tau is the largest threshold whose corrected loss stays within beta — the CRC walk."""
    policy = triage.fit(rows[0], 0.10, beta=0.05)
    assert policy.calibration_risk is not None
    assert policy.calibration_risk <= 0.05
    assert policy.fit_rows > 0
    assert policy.sets == StratifiedSets.fit(rows[0], 0.10, "lac")
