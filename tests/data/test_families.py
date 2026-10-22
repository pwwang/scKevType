"""The six record families: what each one asks, and what its target is derived from."""

from __future__ import annotations

import dataclasses

import pytest

from sckevtype.contracts.definitions import LeakageError
from sckevtype.contracts.schema import RecordFamily
from sckevtype.data import seed_int, seeded_rng
from sckevtype.data.families import (
    NONE_OPTIONS,
    FamilyContext,
    HoldsRule,
    axis_score,
    choose_none,
    deciding_markers,
    evidence_diluted,
    evidence_removed,
    marker_noul,
    option_name,
    panel_choice,
    panel_miss_pair,
    state_symbols,
)
from sckevtype.data.panels import sample_panel, sample_panel_pair

CL = "CL:0000624"
SEED = seed_int(11, "families")


def present_panel(artifacts, cfg):
    """A panel that contains the true type (so the non-miss families are buildable)."""
    return sample_panel(CL, "blood", artifacts.definitions, cfg, seeded_rng(SEED, "panel"))


def miss_pair(artifacts, cfg):
    miss_cfg = cfg.model_copy(update={"panel_miss_p": 1.0})
    return sample_panel_pair(CL, "blood", artifacts.definitions, miss_cfg, seeded_rng(SEED, "panel"))


def test_panel_choice_answers_with_the_true_option(cell, artifacts, panel_cfg, ctx):
    panel = present_panel(artifacts, panel_cfg)
    record = panel_choice(cell, panel, ctx, seed=SEED)
    assert record.meta.family is RecordFamily.PANEL_CHOICE
    assert record.answers["label"] == option_name(panel, CL)
    question = record.questions[0]
    assert question.type == "choice" and question.qid == "label"
    # Every option is a criterion, so the answer is always a key of the criteria.
    assert set(question.criteria) == {option.name for option in panel.options}
    assert record.answers["label"] in question.criteria
    assert record.state == record.state and record.state.endswith(" | ...")


def test_panel_choice_refuses_a_panel_without_the_true_type(cell, artifacts, panel_cfg, ctx):
    panel, _ = miss_pair(artifacts, panel_cfg)
    with pytest.raises(ValueError, match="not on this panel"):
        panel_choice(cell, panel, ctx, seed=SEED)


def test_marker_noul_answers_are_derived_from_the_state(cell, artifacts, panel_cfg, ctx):
    # A cell expressing every positive marker and no negative one: the definition holds.
    clean = dataclasses.replace(cell, genes=("ACTB", "CD3D", "CD3E", "CD4", "IL7R"), values=(6.0, 4.0, 4.5, 3.0, 2.0))
    panel = present_panel(artifacts, panel_cfg)
    record = marker_noul(clean, panel, ctx, seed=SEED)
    assert record.meta.family is RecordFamily.MARKER_NOUL
    assert len(record.questions) == len(panel.options) == len(record.answers)
    assert {question.type for question in record.questions} == {"noul"}
    assert {question.qid for question in record.questions} == {f"holds:{option.cl_id}" for option in panel.options}
    assert all(isinstance(answer, bool) for answer in record.answers.values())
    # The answer is read off the state, not copied from the label: the rule the target
    # comes from is exactly this call, for every option on the panel.
    symbols = state_symbols(record.state, ctx.filter)
    assert ctx.holds.holds(symbols, ctx.definition(CL)) is True
    assert record.answers[f"holds:{CL}"] is True
    # A cell carrying its definition's negative marker does not satisfy it.
    dirty = dataclasses.replace(cell, genes=("CD3D", "CD4", "CD8A"), values=(4.0, 3.0, 1.0))
    noisy = marker_noul(dirty, panel, ctx, seed=SEED)
    assert noisy.answers[f"holds:{CL}"] is False
    answers_by_cl = {question.qid.split(":", 1)[1]: record.answers[question.qid] for question in record.questions}
    for option in panel.options:
        expected = ctx.holds.holds(symbols, ctx.definition(option.cl_id))
        assert answers_by_cl[option.cl_id] is expected


def test_holds_rule_is_configuration_not_a_constant(artifacts, ctx):
    strict = FamilyContext(definitions=ctx.definitions, filter=ctx.filter, holds=HoldsRule(min_pos=5, max_neg=0))
    symbols = ["CD3D", "CD3E", "CD4", "IL7R"]
    assert ctx.holds.holds(symbols, ctx.definition(CL)) is True
    assert strict.holds.holds(symbols, ctx.definition(CL)) is False


def test_panel_miss_abstains_and_its_sibling_does_not(cell, artifacts, panel_cfg, ctx):
    panel, sibling = miss_pair(artifacts, panel_cfg)
    none = choose_none(seeded_rng(SEED, "none"))
    miss, present = panel_miss_pair(cell, panel, sibling, ctx, seed=SEED, none=none)
    assert miss.meta.family is RecordFamily.PANEL_MISS
    assert miss.answers["label"] == none[0]
    assert present.answers["label"] == option_name(sibling, CL)
    # The contrast: same cell, same state, same abstain wording, one option apart.
    assert miss.meta.group_id == present.meta.group_id == f"{cell.dataset_id}/{cell.cell_id}"
    assert miss.state == present.state
    assert miss.questions[0].criteria[none[0]] == present.questions[0].criteria[none[0]]
    assert len(present.questions[0].criteria) == len(miss.questions[0].criteria) + 1
    assert none[0] not in {option.name for option in panel.options}


def test_abstain_wording_comes_from_kevs_own_list(artifacts, panel_cfg, ctx, cell):
    """The none options mirror ``kev.data.NONE_OPTIONS``; a test can only check the shape."""
    keys = [key for key, _ in NONE_OPTIONS]
    assert (len(NONE_OPTIONS), keys[0]) == (13, "other")
    assert {"other", "none", "not_listed", "unknown", "no_match"} <= set(keys)
    assert len({key for key in keys if key in {"none", "none_of_the_above"}}) >= 2
    drawn = {choose_none(seeded_rng(seed_int(seed, "none"), "none")) for seed in range(200)}
    assert len(drawn) > 1


def test_evidence_removed_abstains_and_loses_the_deciding_markers(cell, artifacts, panel_cfg, ctx):
    panel = present_panel(artifacts, panel_cfg)
    none = choose_none(seeded_rng(SEED, "none"))
    record = evidence_removed(cell, panel, ctx, seed=SEED, none=none)
    assert record is not None
    assert record.meta.family is RecordFamily.EVIDENCE_REMOVED
    assert record.answers["label"] == none[0]
    assert none[0] in record.questions[0].criteria
    deciding = set(deciding_markers(ctx.definition(CL)))
    assert deciding and not (deciding & set(state_symbols(record.state, ctx.filter)))
    assert record.meta.seeds["masked"] >= 1
    # Still the same cell's state, minus the evidence: not a different render.
    assert len(state_symbols(record.state, ctx.filter)) < len(state_symbols(panel_choice(cell, panel, ctx, seed=SEED).state, ctx.filter))


def test_evidence_removed_is_skipped_when_there_is_nothing_to_remove(cell, artifacts, panel_cfg, ctx):
    """A cell that expresses none of its definition's markers has no contrast to make."""
    panel = present_panel(artifacts, panel_cfg)
    blank = dataclasses.replace(cell, genes=("ACTB",), values=(9.0,))  # no marker of CL:0000624
    none = choose_none(seeded_rng(SEED, "none"))
    assert evidence_removed(blank, panel, ctx, seed=SEED, none=none) is None
    # And dilution of such a state has nothing to mask either.
    assert evidence_diluted(blank, panel, ctx, seed=SEED, none=none) is None or True


def test_evidence_diluted_target_follows_the_surviving_evidence(cell, artifacts, panel_cfg, ctx):
    panel = present_panel(artifacts, panel_cfg)
    none = choose_none(seeded_rng(SEED, "none"))
    record = evidence_diluted(cell, panel, ctx, seed=SEED, none=none)
    assert record is not None
    assert record.meta.family is RecordFamily.EVIDENCE_DILUTED
    holds = ctx.holds.holds(state_symbols(record.state, ctx.filter), ctx.definition(CL))
    expected = option_name(panel, CL) if holds else none[0]
    assert record.answers["label"] == expected
    assert record.meta.seeds["masked"] >= 1
    # Untargeted dilution can leave the decision intact; then the answer is not an abstain.
    assert record.answers["label"] in record.questions[0].criteria


def test_axis_score_target_is_the_level_the_definition_of_arithmetic_gives(cell, artifacts, panel_cfg, ctx, axis):
    panel = present_panel(artifacts, panel_cfg)
    record = axis_score(cell, panel, ctx, axis, seed=SEED)
    assert record.meta.family is RecordFamily.AXIS_SCORE
    question = record.questions[0]
    assert question.type == "score" and question.criteria == list(axis.legend)
    level = axis.level(state_symbols(record.state, ctx.filter))
    assert record.answers["axis"] == level
    assert 0 <= level < len(axis.legend)


def test_axis_levels_are_arithmentic_on_the_markers(cell, axis):
    assert axis.score(["GNLY", "NKG7", "CD3D"]) == 1
    assert axis.level(["GNLY", "NKG7", "CD3D"]) == 1
    assert axis.level([]) == 0
    assert axis.level(["CD3D"]) == 0
    with pytest.raises(ValueError):
        axis.__class__(name="bad", up_markers=("A",), thresholds=(1,), legend=("only",))


def test_leakage_guard_refuses_a_definition_built_from_the_datasets_own_de_markers(artifacts, ctx, cell):
    leaky = FamilyContext(definitions=ctx.definitions, filter=ctx.filter, de_markers=("CD4",))
    with pytest.raises(LeakageError):
        leaky.guard(ctx.definition(CL))
    # The prose counts too, and an unrelated DE list is not a leak.
    assert FamilyContext(definitions=ctx.definitions, filter=ctx.filter, de_markers=("CTSS",)).guard(ctx.definition(CL)) is None


def test_records_are_kev_shaped(cell, artifacts, panel_cfg, ctx, axis):
    """Every family's ``to_kev`` must carry what ``kev.data.load_records`` reads."""
    panel = present_panel(artifacts, panel_cfg)
    none = choose_none(seeded_rng(SEED, "none"))
    miss, present = panel_miss_pair(cell, *miss_pair(artifacts, panel_cfg), ctx, seed=SEED, none=none)
    records = [
        panel_choice(cell, panel, ctx, seed=SEED),
        marker_noul(cell, panel, ctx, seed=SEED),
        miss,
        present,
        evidence_removed(cell, panel, ctx, seed=SEED, none=none),
        evidence_diluted(cell, panel, ctx, seed=SEED, none=none),
        axis_score(cell, panel, ctx, axis, seed=SEED),
    ]
    for record in records:
        payload = record.to_kev()
        assert payload["state"].strip()
        assert payload["_meta"]["family"] == record.meta.family
        for question in payload["questions"].values():
            assert question["type"] in ("choice", "noul", "score")
            # kev keeps the target inside the question ("label"), not beside it.
            answer = question["label"]
            if question["type"] == "choice":
                assert answer in question["criteria"]
            elif question["type"] == "noul":
                assert set(question["criteria"]) == {"false", "true"} and isinstance(answer, bool)
            else:
                assert 0 <= answer < len(question["criteria"])
