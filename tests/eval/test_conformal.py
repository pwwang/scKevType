"""Axis 2: the conformal quantile, the sets it produces, and the coverage it actually delivers.

The statistical tests here are the ones that would catch a wrong rank constant or a set applied
at the wrong stratum. They are written against a synthetic population whose labels are drawn
independently of the model's probabilities, which is the strongest form of correct-but-unhelpful
model: coverage must still hit the nominal target, because the guarantee does not care whether
the probabilities are any good.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sckevtype.eval import conformal, stratum_of
from sckevtype.eval.conformal import (
    StratifiedSets,
    conformal_quantile,
    coverage,
    nonconformity,
    prediction_set,
)


def test_quantile_is_the_ceil_n_plus_1_rank(make_rows):
    """The rank is ``ceil((n+1)(1-alpha))`` — the ``+1`` is the whole finite-sample correction."""
    scores = [float(value) for value in range(1, 20)]        # n = 19
    assert conformal_quantile(scores, 0.10) == 18.0          # ceil(20 * 0.9) = 18
    assert conformal_quantile(scores, 0.50) == 10.0          # ceil(20 * 0.5) = 10
    assert conformal_quantile(scores, 0.90) == 2.0           # ceil(20 * 0.1) = 2


def test_quantile_is_infinite_when_the_sample_cannot_carry_alpha():
    """``n = 8, alpha = 0.1`` needs rank 9 of 8; the honest answer is a set of everything."""
    assert conformal_quantile([float(v) for v in range(1, 9)], 0.10) == math.inf
    assert conformal_quantile([float(v) for v in range(1, 10)], 0.10) == 9.0   # rank 9 of 9: just fits
    assert prediction_set([0.5, 0.5], math.inf, "lac").all()


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_alpha_must_be_a_probability(alpha):
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([0.1, 0.2], alpha)


def test_unknown_method_is_refused():
    with pytest.raises(ValueError, match="unknown conformal method"):
        nonconformity([0.5, 0.5], 0, "marginal")


@pytest.mark.parametrize("method", conformal.METHODS)
def test_coverage_hits_nominal_on_an_exchangeable_population(rows, method):
    """Calibration and test drawn from one distribution: coverage must land at ``1 - alpha``."""
    calibration, test = rows
    for alpha in (0.05, 0.10, 0.20):
        sets = StratifiedSets.fit(calibration, alpha, method).sets(test)
        observed = coverage(test, sets)["coverage"]
        assert observed >= 1 - alpha - 0.05, f"{method} at alpha={alpha}: coverage {observed}"
        assert observed <= 1 - alpha + 0.05, f"{method} at alpha={alpha}: coverage {observed}"


def test_coverage_holds_across_many_small_calibration_draws(make_rows):
    """Averaged over draws, coverage is at least nominal — the finite-sample rank, not the mean."""
    observed = []
    for seed in range(120):
        calibration = make_rows(24, 1000 + seed, "calibration")
        test = make_rows(24, 5000 + seed, "test")
        sets = StratifiedSets.fit(calibration, 0.10).sets(test)
        observed.append(coverage(test, sets)["coverage"])
    assert float(np.mean(observed)) >= 0.90, f"mean coverage {np.mean(observed)} over 120 draws"


@pytest.mark.parametrize("method", conformal.METHODS)
def test_set_size_is_monotone_in_alpha(rows, method):
    """Lower alpha means a stricter quantile means a larger set — and the sets nest."""
    calibration, test = rows
    mean_sizes, coverages = [], []
    for alpha in (0.01, 0.05, 0.10, 0.20, 0.40):
        sets = StratifiedSets.fit(calibration, alpha, method).sets(test)
        mean_sizes.append(float(np.mean([int(mask.sum()) for mask in sets])))
        coverages.append(coverage(test, sets)["coverage"])
    assert mean_sizes == sorted(mean_sizes, reverse=True), mean_sizes
    assert mean_sizes[0] > mean_sizes[-1]
    assert coverages == sorted(coverages, reverse=True), coverages      # lower alpha, higher coverage
    assert all(observed >= 1 - alpha - 0.05 for observed, alpha in zip(coverages, (0.01, 0.05, 0.10, 0.20, 0.40)))


@pytest.mark.parametrize("method", conformal.METHODS)
def test_a_singleton_set_is_the_argmax(rows, method):
    """The fact the composed bound rests on: one member means it is the top-1 decision.

    Checked at a quantile where a singleton is certain for both scores — at a fitted quantile on an
    uncertain model, APS sets are rarely singletons, which is itself worth knowing (it is why the
    composed gate auto-annotates less under APS at the same nominal coverage).
    """
    sharp = [0.9, 0.05, 0.05]
    mask = prediction_set(sharp, 0.9, method)            # LAC score 0.1 <= 0.9; APS score 0.9 <= 0.9
    assert list(mask) == [True, False, False]
    assert int(mask.argmax()) == int(np.asarray(sharp).argmax())

    calibration, test = rows
    fitted = StratifiedSets.fit(calibration, 0.10, method)
    for row in test:
        mask = fitted.set_for(row)
        if mask.sum() == 1:
            assert int(mask.argmax()) == int(np.asarray(row["p"]).argmax())


@pytest.mark.parametrize("method", conformal.METHODS)
def test_sets_and_scores_are_permutation_invariant(make_rows, method):
    """Reordering a panel's options must not change the set — the paper's invariance claim."""
    row = make_rows(1, 7, "calibration")[0]
    order = np.random.default_rng(0).permutation(len(row["keys"]))
    moved = dict(row, keys=[row["keys"][i] for i in order], p=[row["p"][i] for i in order],
                 label=int(np.flatnonzero(order == int(row["label"]))[0]))
    mask = prediction_set(row["p"], 0.9, method)
    moved_mask = prediction_set(moved["p"], 0.9, method)
    assert mask.any() and not mask.all(), "a trivial set would make the claim vacuous"
    assert list(moved_mask) == [mask[i] for i in order]      # moved_p[j] is p[order[j]]
    assert nonconformity(moved["p"], moved["label"], method) == nonconformity(row["p"], row["label"], method)


def test_quantile_is_taken_per_stratum_not_marginally(make_rows):
    """A thin stratum must not borrow a marginal quantile — it coarsens down its chain instead."""
    thin = make_rows(6, 21, "calibration")
    thick = make_rows(300, 22, "calibration", n_options=12)
    for row in thin:
        row["size"] = 3                                       # a 3-option question, 6 of them
    for row in thick:
        row["size"] = 12
    # At alpha=0.05 a 6-row stratum needs rank 7 of 6: unfittable, so it is left out of the map.
    fitted = StratifiedSets.fit(thin + thick, 0.05)
    assert (3, "choice") not in fitted.quantiles
    assert fitted.quantile_for(thin[0])[0] == ("choice",)      # coarsened, and it says so
    assert fitted.quantile_for(thick[0])[0] == (12, "choice")
    assert fitted.strata_used(thin + thick) == {"type": 6, "size+type": 300}
    assert fitted.to_dict()["support"] == {"12|choice": 300, "choice": 306}   # the thin stratum is absent, not zero


def test_fit_refuses_an_alpha_no_stratum_can_carry(make_rows):
    with pytest.raises(ValueError, match="no stratum has enough"):
        StratifiedSets.fit(make_rows(3, 30, "calibration"), 0.01)


def test_coverage_reports_per_stratum_and_per_family(make_rows):
    calibration = make_rows(300, 40, "calibration")
    test = make_rows(200, 41, "test", n_options=5) + make_rows(100, 42, "test", n_options=9, family="panel_miss")
    sets = StratifiedSets.fit(calibration, 0.10).sets(test)
    report = coverage(test, sets)
    assert report["n"] == 300
    assert set(report["per_stratum"]) == {f"{size}|choice" for size in (5, 9)}
    assert set(report["per_family"]) == {"panel_choice", "panel_miss"}
    assert all(0.0 <= entry["coverage"] <= 1.0 for entry in report["per_stratum"].values())


def test_coverage_rejects_a_length_mismatch(make_rows):
    rows = make_rows(4, 50, "test")
    with pytest.raises(ValueError, match="4 rows against 3 sets"):
        coverage(rows, np.ones((3, 4), dtype=bool))


def test_stratum_of_is_size_and_type(make_rows):
    row = make_rows(1, 60, "calibration")[0]
    assert stratum_of(row) == (5, "choice")
