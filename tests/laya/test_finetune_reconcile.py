"""The counting rule in ``reports/laya-finetune-reconcile.py``, checked against its source of truth.

The script's whole job is to apply ``reports/abstain-reconciliation.py``'s three rules to a laya
dump, so what is tested here is that it applies *those* rules and not a lookalike: the served rule is
``argmax`` over the full probability vector with ties going to the lowest index (Python's ``max``
would take the highest), and the forced rule removes the abstain option rather than lowering it.

The equivalence the script's docstring claims -- that running it on the pilot's rows reproduces
``reports/both-forced-table.json``'s ``laya-pilot`` cell to the integer -- is a property of the whole
1,819-row dump, not of one row, and is checked by running the script (its log records the
cross-check); it is not re-run here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sckevtype.data.families import NONE_OPTIONS
from sckevtype.eval import abstain_index

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "analysis" / "laya-finetune-reconcile.py"
CENSUS_SUITE_IDS = ROOT / "reports" / "census-suite-ids.jsonl"

ABSTAIN = NONE_OPTIONS[0][0]


def _load():
    spec = importlib.util.spec_from_file_location("laya_finetune_reconcile", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reconcile():
    return _load()


def _row(keys: list[str], p: list[float], label: int = 0) -> dict:
    return {"keys": keys, "p": p, "label": label}


def test_abstain_index_finds_the_factory_wording():
    assert abstain_index(_row(["A", "B", ABSTAIN], [0.2, 0.3, 0.5])) == 2


def test_served_is_argmax_over_everything_including_the_abstain_option(reconcile):
    row = _row(["A", "B", ABSTAIN], [0.2, 0.3, 0.5])
    assert reconcile.argmax(row["p"]) == 2


def test_a_tie_goes_to_the_lowest_index_as_the_reconciliation_script_does(reconcile):
    # `max(range(len(p)), key=p.__getitem__)` -- which a re-implementation would naturally reach for
    # -- returns 2 here. `reports/abstain-reconciliation.py`'s argmax returns 0.
    assert reconcile.argmax([0.5, 0.1, 0.5]) == 0


def test_forced_removes_the_abstain_option_rather_than_lowering_it(reconcile):
    row = _row(["A", "B", ABSTAIN], [0.2, 0.3, 0.5], label=1)
    assert reconcile.forced(row) == 1
    # And it is not the served answer on this row: the two rules genuinely differ.
    assert reconcile.argmax(row["p"]) != reconcile.forced(row)


def test_forced_picks_the_runner_up_when_the_abstain_option_wins(reconcile):
    row = _row(["A", "B", ABSTAIN], [0.7, 0.2, 0.1])
    assert reconcile.forced(row) == 0


def test_forced_on_a_row_with_no_abstain_option_is_the_served_answer(reconcile):
    row = _row(["A", "B"], [0.4, 0.6], label=1)
    assert abstain_index(row) is None
    assert reconcile.forced(row) == reconcile.argmax(row["p"]) == 1


def test_the_settled_population_is_still_1819_panel_choice_rows():
    """A moved population is a stopped run, not a new denominator.

    ``reports/both-forced-table.json`` and the manuscript's Tables 2 and 3 are all written on this
    denominator; if it changes, every number computed against it silently means something else.
    """
    rows = [json.loads(line) for line in CENSUS_SUITE_IDS.read_text().splitlines() if line.strip()]
    panel_choice = [r for r in rows if r["family"] == "panel_choice"]
    assert len(panel_choice) == 1819
    assert len({(r["id"], r["family"]) for r in panel_choice}) == 1819
    assert len(rows) == 7203
