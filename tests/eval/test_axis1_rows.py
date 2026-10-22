"""The axis-1 statistic's edges, and the dump it has to be recomputable from.

No GPU and no checkpoint: the rows are built here in the shape :func:`sckevtype.eval.run.
write_row_dump` writes and :func:`sckevtype.eval.metrics.panel_adequacy` reads, so every claim
about the dumps and about the stratification is checked against the library rather than against a
transcript. The step-12 check skips on a box that does not hold the three probe dumps.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sckevtype.eval import metrics, run

ROOT = Path(__file__).resolve().parents[2]
ABSTAIN_INDEX = 3


def _row(index: int, none_p: float | None, *, factory: bool, positive: bool = False,
         family: str = "evidence_diluted") -> dict:
    """A row shaped like the CLI's: the target is the abstain option exactly when ``positive``."""
    size = ABSTAIN_INDEX + 1
    return {
        "id": f"row-{index}", "question": "label", "split": "test", "dataset": "synthetic", "donor": "d0",
        "family": family, "variant": "clean", "size": size,
        "label": ABSTAIN_INDEX if positive else 0,
        # A factory question offers the abstain option at ``ABSTAIN_INDEX``; an injected one appends
        # it past the factory's options, which is what makes an injected row a negative by construction.
        "abstain": ABSTAIN_INDEX if factory else size,
        "abstain_source": metrics.FACTORY if factory else metrics.INJECTED,
        "abstain_key": "not_listed", "abstain_text": None,
        "abstain_position": ABSTAIN_INDEX if factory else size, "abstain_last": positive,
        "none_p": none_p,
    }


def _population() -> list[dict]:
    """Both classes, both conventions, and no family carrying a single class."""
    return [
        _row(0, 0.9, factory=True, positive=True),
        _row(1, 0.95, factory=True, positive=True),
        _row(2, 0.7, factory=True, positive=True),
        _row(3, 0.1, factory=True),
        _row(4, 0.2, factory=True),
        _row(5, 0.05, factory=False),
        _row(6, 0.3, factory=False),
    ]


def test_a_dump_alone_recomputes_the_statistic(tmp_path):
    """The point of the dump: a reader with this file and no GPU re-derives the same number."""
    rows = _population()
    info = run.write_row_dump(tmp_path / "rows.jsonl", rows)
    dump = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]

    assert info == {"path": str(tmp_path / "rows.jsonl"), "rows": len(rows),
                    "sha256": run.file_digest(tmp_path / "rows.jsonl")}
    assert all(row["positive"] == int(metrics.axis1_label(row)) for row in dump)
    assert all(row["abstain_source"] in (metrics.FACTORY, metrics.INJECTED) for row in dump)
    assert metrics.panel_adequacy(dump)["primary"] == metrics.panel_adequacy(rows)["primary"]


def test_a_row_without_the_statistic_is_written_and_excluded():
    """``none_p: null`` is a row the statistic had to drop, and the dump says how many."""
    rows = _population() + [_row(7, None, factory=True, positive=True)]
    report = metrics.panel_adequacy(rows)
    assert report["n_choice_rows_without_statistic"] == 1
    assert report["n"] == len(rows) - 1


def test_a_single_class_selection_refuses_rather_than_returns_a_number():
    """One class only is the finding, so ``primary`` is None with the reason beside it."""
    report = metrics.panel_adequacy([_row(0, 0.9, factory=True, positive=True), _row(1, 0.8, factory=True, positive=True)])
    assert report["primary"] is None
    assert "one axis-1 class only" in report["per_family"]["evidence_diluted"]["note"]
    assert any("primary statistic is undefined" in reason for reason in run._axis1_reasons(report))


def test_an_unstated_serving_convention_refuses_the_statistic():
    """A row whose convention is unrecorded cannot be attributed to an arm, so it is a reason."""
    rows = [{**_row(index, none_p, factory=True, positive=index == 0), "abstain_source": None}
            for index, none_p in ((0, 0.9), (1, 0.1))]
    report = metrics.panel_adequacy(rows)
    assert report["n_unstated_convention"] == 2
    assert report["primary"] is None
    assert "no row states a factory-given abstain option" in report["primary_note"]
    assert report["by_serving_convention"]["gap"]["auc_convention"] is None
    assert any("serving convention" in reason for reason in run._axis1_reasons(report))


def test_a_thin_stratum_is_flagged_not_silently_reported():
    report = metrics.panel_adequacy(_population())
    assert report["per_family"]["evidence_diluted"]["thin"] is True
    assert "THIN" in report["per_family"]["evidence_diluted"]["note"]


@pytest.mark.parametrize("name", ["transfer", "pair", "bypass"])
def test_the_new_definition_reproduces_the_step12_numbers(name, capsys):
    """Spec item 5: the numbers step 12 quotes must come back out of the new code path.

    A disagreement is a finding about this code, not something to reconcile by editing the claim —
    so the script prints the number it produced and exits non-zero, and this test fails with it.
    """
    if not (ROOT / "reports" / f"axis1-rows-{name}.jsonl").exists():
        pytest.skip(f"no step-12 dump for {name} on this box")

    spec = importlib.util.spec_from_file_location("axis1_recompute", ROOT / "scripts" / "axis1-recompute.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    previous, sys.argv = sys.argv, ["axis1-recompute.py", "--dump", name]
    try:
        status = module.main()
    finally:
        sys.argv = previous
    printed = capsys.readouterr().out
    assert "DIFFERS" not in printed, printed
    assert status == 0
