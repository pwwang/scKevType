"""The scope rule (Step 6 §2) and the coupling that keeps its claim true (§3).

These run off the committed histograms in ``configs/seed/v2/atlas_labels.json`` and
the committed ``artifacts/v2``, so they re-derive the scope with no atlas on disk and
no network — which is the point: a reader can check the rule against the numbers the
step reported.
"""

from __future__ import annotations

import json

import pytest
import yaml

from sckevtype.contracts import paths
from sckevtype.contracts.register import AtlasCounts, minimal_terms, plan_scope

LABELS = paths.root() / "configs" / "seed" / "v2" / "atlas_labels.json"
REPORT = paths.root() / "reports" / "step6-coverage.json"


@pytest.fixture(scope="module")
def histograms() -> list[AtlasCounts]:
    payload = json.loads(LABELS.read_text(encoding="utf-8"))
    return [AtlasCounts.model_validate(value) for _, value in sorted(payload.items())]


def test_minimal_terms_reaches_the_target_and_is_a_function_of_the_histogram():
    labels = {"CL:0000003": 60, "CL:0000001": 30, "CL:0000002": 10}
    assert minimal_terms(labels, 0.9) == ("CL:0000001", "CL:0000003")
    # A tie between two equally-sized terms must not depend on dict order.
    tied = {"CL:0000009": 5, "CL:0000001": 5}
    assert minimal_terms(tied, 0.5) == ("CL:0000001",) == minimal_terms(dict(reversed(list(tied.items()))), 0.5)


def test_a_greedy_that_stops_at_the_target_never_overshoots_by_a_whole_term():
    labels = {f"CL:{n:07d}": 1 for n in range(100)}
    chosen = minimal_terms(labels, 0.95)
    assert len(chosen) == 95


def test_the_scope_is_a_subset_of_what_is_defined(histograms):
    """§3's coupling: a demanded term with no grounded definition stays out of scope.

    This is the honest constraint — without it, widening the register to reach the
    gate would silently invent definitions.
    """
    defined = {t["cl_id"] for t in json.loads(paths.artifact("v2", "definitions.json").read_text(encoding="utf-8"))["definitions"]}
    plan = plan_scope(histograms, defined)
    assert set(plan.in_scope) <= defined
    assert not set(plan.in_scope) & set(plan.undefined)


def test_every_in_scope_term_has_a_definition_in_the_frozen_artifact(histograms):
    definitions = json.loads(paths.artifact("v2", "definitions.json").read_text(encoding="utf-8"))["definitions"]
    plan = plan_scope(histograms, {t["cl_id"] for t in definitions})
    for cl_id in plan.in_scope:
        term = next(t for t in definitions if t["cl_id"] == cl_id)
        assert term["pos_markers"], f"{cl_id} is in scope with no positive marker"


def test_v1_terms_stay_in_scope():
    """Two atlases already map against the v1 register; dropping a term is a regression."""
    v1 = {t["cl_id"] for t in json.loads(paths.artifact("v1", "cl_terms.json").read_text(encoding="utf-8"))["terms"]}
    v2 = {t["cl_id"] for t in json.loads(paths.artifact("v2", "cl_terms.json").read_text(encoding="utf-8"))["terms"]}
    assert v1 <= v2


def test_the_reported_table_is_what_the_rule_produces_now(histograms):
    """The report is re-derivable: same histograms, same scope, same rates."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    plan = plan_scope(histograms, report["defined"], required=report["required"])
    assert list(plan.in_scope) == report["in_scope"]
    produced = {atlas.atlas_id: atlas for atlas in plan.atlases}
    reported = {atlas["atlas_id"]: atlas for atlas in report["atlases"]}
    assert produced.keys() == reported.keys()
    for atlas_id, row in reported.items():
        assert round(produced[atlas_id].in_scope_rate, 6) == row["in_scope_rate"]
        assert produced[atlas_id].undefined_cells == row["undefined_cells"]


def test_the_seed_definitions_carry_their_provenance():
    """No marker without a named resource behind it (the step's hard rule)."""
    seed = yaml.safe_load((paths.root() / "configs" / "seed" / "v2" / "definitions.seed.yaml").read_text(encoding="utf-8"))
    assert "cellmarker2" in seed["source"].lower() or "CellMarker" in seed["source"]
    for definition in seed["definitions"]:
        assert definition["provenance"].startswith(("cellmarker2:", "sc-type:"))
        assert definition["text"], f"{definition['cl_id']} has no CL definition text"
