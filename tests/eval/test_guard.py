"""The partition guard: fitting code must refuse anything that is not a calibration row.

This is the one invariant the paper's numbers stand on. A conformal quantile or a temperature
fitted on the rows it is then reported on produces numbers that look better and mean nothing, and
a reader cannot tell from the output — so the check has to be a raise, at the entry point, in
every fitting path.
"""

from __future__ import annotations

import pytest

from sckevtype.eval import (
    PartitionError,
    assert_fit_partition,
    conformal,
    metrics,
    rows_of,
    triage,
)


def test_guard_accepts_calibration(make_rows):
    assert_fit_partition(make_rows(80, 1, "calibration"), "test")     # does not raise


def test_guard_raises_on_a_single_test_record(make_rows):
    """One test record among calibration rows is enough — the required proof that it fires."""
    with pytest.raises(PartitionError) as raised:
        assert_fit_partition(make_rows(80, 1, "calibration") + make_rows(1, 2, "test"), "test")
    assert "'test': 1" in str(raised.value)
    assert "calibration" in str(raised.value)


def test_guard_raises_when_a_row_does_not_say_which_partition(make_rows):
    """A row with no split is refused too: "it did not say" must not read as "it is fine"."""
    row = dict(make_rows(1, 1, "calibration")[0])
    del row["split"]
    with pytest.raises(PartitionError, match="<no split>"):
        assert_fit_partition([row], "test")


def test_guard_raises_on_no_rows():
    with pytest.raises(PartitionError, match="no rows"):
        assert_fit_partition([], "test")


def test_development_rows_are_not_fit_partition_either(make_rows):
    """Anything not literally calibration is refused — including the dev partition."""
    with pytest.raises(PartitionError):
        assert_fit_partition(make_rows(5, 3, "development"), "test")


@pytest.mark.parametrize("fit", [
    lambda cal, test: conformal.StratifiedSets.fit(cal + test, 0.1),
    lambda cal, test: triage.fit(cal + test, 0.1, 0.05),
    lambda cal, test: triage.sweep(cal + test, test, alphas=[0.1], betas=[0.05]),
    lambda cal, test: metrics.temperature_scaling(cal + test, test),
])
def test_every_fitting_entry_point_raises(fit, make_rows):
    """Every path that fits something — quantiles, thresholds, a temperature — refuses test rows."""
    with pytest.raises(PartitionError):
        fit(make_rows(80, 1, "calibration"), make_rows(20, 2, "test"))


def test_reporting_paths_accept_test_rows(make_rows):
    """The guard is at the fit, not a blanket refusal: reporting on test rows is the point."""
    calibration, test = make_rows(200, 1, "calibration"), make_rows(60, 2, "test")
    sets = conformal.StratifiedSets.fit(calibration, 0.1).sets(test)
    assert conformal.coverage(test, sets)["n"] == len(test)
    assert len(triage.fit(calibration, 0.1, 0.05).decide_all(test)) == len(test)
    assert metrics.top1_accuracy(test)["n"] == len(test)


def test_rows_of_selects_the_partition(make_rows):
    """The selector the CLI uses to hand a fitting path calibration rows and only those."""
    rows = make_rows(10, 1, "calibration") + make_rows(4, 2, "test")
    picked = rows_of(rows, split="calibration")
    assert len(picked) == 10
    assert_fit_partition(picked, "test")                     # the selector's output is fit-safe
    assert len(rows_of(rows, split="calibration", family="panel_choice")) == 10
