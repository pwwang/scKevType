"""Augmentations: pure functions of (record, seed) that never touch the answer."""

from __future__ import annotations

import random

from sckevtype.data import seed_int, seeded_rng
from sckevtype.data.augment import (
    CLEAN,
    PERMUTED,
    dilute_rendered,
    mask_record,
    mask_rendered,
    parse_state,
    permute_options,
    perturb_panel,
)
from sckevtype.data.families import state_symbols


def _choice_record(cell, artifacts, cfg, ctx, seed=3):
    from sckevtype.data.families import panel_choice
    from sckevtype.data.panels import sample_panel

    panel = sample_panel(cell.cl_id, cell.tissue, artifacts.definitions, cfg, seeded_rng(seed_int(seed), "panel"))
    return panel, panel_choice(cell, panel, ctx, seed=seed)


def test_permutation_reorders_the_options_and_leaves_the_answer(cell, artifacts, panel_cfg, ctx):
    _panel, record = _choice_record(cell, artifacts, panel_cfg, ctx)
    permuted = permute_options(record, seed_int(1, "permute"))
    assert permuted.state == record.state
    assert permuted.answers == record.answers
    assert permuted.meta.variant == PERMUTED and record.meta.variant == CLEAN
    assert permuted.meta.seeds["permute"] == seed_int(1, "permute") and "permute" not in record.meta.seeds
    assert set(permuted.questions[0].criteria) == set(record.questions[0].criteria)
    assert len(permuted.questions[0].criteria) == len(record.questions[0].criteria)
    assert permuted.meta.panel_id == record.meta.panel_id


def test_permutation_is_deterministic_and_actually_shuffles(cell, artifacts, panel_cfg, ctx):
    _panel, record = _choice_record(cell, artifacts, panel_cfg, ctx)
    orders = {tuple(permute_options(record, seed=seed).questions[0].criteria) for seed in range(12)}
    assert len(orders) > 1
    first = permute_options(record, seed=5)
    second = permute_options(record, seed=5)
    assert first.model_dump() == second.model_dump()


def test_permutation_keeps_the_abstain_option_and_its_wording(cell, artifacts, panel_cfg, ctx):
    """A shuffled option list must not shuffle the abstain option (or change its text)."""
    from sckevtype.data.families import choose_none, panel_miss_pair
    from sckevtype.data.panels import sample_panel_pair

    cfg = panel_cfg.model_copy(update={"panel_miss_p": 1.0})
    panel, sibling = sample_panel_pair(cell.cl_id, cell.tissue, artifacts.definitions, cfg, seeded_rng(seed_int(3), "panel"))
    none = choose_none(seeded_rng(seed_int(3, "none"), "none"))
    miss, present = panel_miss_pair(cell, panel, sibling, ctx, seed=3, none=none)
    for record in (miss, present):
        permuted = permute_options(record, seed=seed_int(3, "permute"))
        assert permuted.answers == record.answers
        assert set(permuted.questions[0].criteria) == set(record.questions[0].criteria)
        assert permuted.questions[0].criteria[none[0]] == record.questions[0].criteria[none[0]]
        assert len(permuted.questions[0].criteria) == len(record.questions[0].criteria)


def test_masking_re_renders_through_the_canonical_renderer(cell, artifacts, ctx):
    state = "CD3D 4 | CD4 3 | ACTB 6 | ..."
    masked, removed = mask_rendered(state, ("CD4",), ctx.filter)
    assert removed == ("CD4",)
    assert "CD4" not in state_symbols(masked, ctx.filter)
    assert "CD3D" in state_symbols(masked, ctx.filter)
    # Re-rendered, not string-patched: the result is a list the renderer would produce.
    assert masked.endswith(" | ...")
    assert mask_rendered(state, ("NOT_A_GENE",), ctx.filter) == (state, ())


def test_dilution_removes_exactly_k_genes_when_there_are_that_many(cell, artifacts, ctx):
    symbols = ["CD3D", "CD4", "IL7R", "MS4A1", "CD8A"]
    state = " | ".join(f"{symbol} {index + 1}" for index, symbol in enumerate(symbols)) + " | ..."
    diluted, removed = dilute_rendered(state, 3, random.Random(1), ctx.filter)
    assert len(removed) == 3
    assert len(state_symbols(diluted, ctx.filter)) == len(symbols) - 3
    assert not set(removed) & set(state_symbols(diluted, ctx.filter))
    # Deterministic under the seed, and the random draw is without replacement.
    again, _ = dilute_rendered(state, 3, random.Random(1), ctx.filter)
    assert again == diluted


def test_dilution_of_a_short_state_is_not_an_error(cell, artifacts, ctx):
    # A state with fewer than two symbols has nothing to dilute: unchanged, not an error.
    assert dilute_rendered("CD3D 1 | ...", 5, random.Random(0), ctx.filter) == ("CD3D 1 | ...", ())


def test_mask_record_returns_a_new_record(cell, artifacts, panel_cfg, ctx):
    _panel, record = _choice_record(cell, artifacts, panel_cfg, ctx)
    masked = mask_record(record, ("CD3D",), ctx.filter)
    assert masked is not record and record.state != masked.state
    assert masked.answers == record.answers
    assert "CD3D" not in state_symbols(masked.state, ctx.filter)


def test_perturbation_goes_through_the_contract_s_own_perturb(artifacts, panel_cfg):
    from sckevtype.data.panels import sample_panel

    panel = sample_panel("CL:0000624", "blood", artifacts.definitions, panel_cfg, seeded_rng(seed_int(2), "panel"))
    perturbed = perturb_panel(panel, artifacts.definitions, ("paraphrase",), seed_int(2, "perturb"))
    assert isinstance(perturbed, type(panel))
    assert [option.cl_id for option in perturbed.options] == [option.cl_id for option in panel.options]
    # The perturbation is reproducible, and it is in the definition text the model reads.
    assert perturbed.options == perturb_panel(panel, artifacts.definitions, ("paraphrase",), seed_int(2, "perturb")).options
    assert [option.definition for option in perturbed.options] != [option.definition for option in panel.options]


def test_parse_state_is_the_inverse_of_the_renderer(cell, artifacts):
    from sckevtype.contracts.render import render_rank_list

    state = render_rank_list(cell.genes, cell.values, artifacts.filter)
    parsed = parse_state(state, artifacts.filter)
    assert [symbol for symbol, _ in parsed] == state_symbols(state, artifacts.filter)
    # The second element is the 1-based rank the renderer wrote, not the expression value.
    assert [rank for _, rank in parsed] == list(range(1, len(parsed) + 1))
