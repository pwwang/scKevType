"""The finite panel vocabulary (Step 6 §5).

The property that matters is not that a variant is *sensible* — it is that a
variant **id** denotes one option set, for every cell. That is what makes a
panel-stratified quantile definable at all, so it is what is asserted here.
"""

from __future__ import annotations

import random

import pytest

from sckevtype.data.panels import PanelConfig, PanelVocabulary, sample_panel_pair


@pytest.fixture
def cfg(artifacts) -> PanelConfig:
    return PanelConfig.from_artifacts(artifacts.tissue_map, artifacts.cl_terms)


def test_the_same_variant_id_yields_the_same_option_set_across_cells(cfg, artifacts):
    definitions = artifacts.definitions
    vocabulary = PanelVocabulary(cfg, "blood", definitions)
    first = {variant: vocabulary.option_set(variant) for variant in vocabulary.variant_ids}
    # A second vocabulary, built independently, and read in the opposite order.
    again = PanelVocabulary(cfg, "blood", definitions)
    for variant in reversed(vocabulary.variant_ids):
        assert again.option_set(variant) == first[variant]
        assert again.panel(variant).panel_id == vocabulary.panel(variant).panel_id


def test_the_vocabulary_is_finite_and_the_variants_are_the_spec_s_kinds(cfg, artifacts):
    vocabulary = PanelVocabulary(cfg, "blood", artifacts.definitions)
    register = set(cfg.candidates("blood"))
    assert "blood:full" in vocabulary.variant_ids
    assert vocabulary.option_set("blood:full") == tuple(sorted(register))
    for variant in vocabulary.variant_ids:
        kind = variant.split(":", 1)[1]
        assert kind == "full" or kind.startswith(("minus", "plus"))
    # Finite, and small: 1 full + |minus| + |plus| at most, never one per cell.
    assert len(vocabulary.variant_ids) <= 1 + len(cfg.variant_minus) + len(cfg.variant_plus)


def test_minus_k_drops_k_and_plus_m_adds_m(cfg, artifacts):
    vocabulary = PanelVocabulary(cfg, "blood", artifacts.definitions)
    full = set(vocabulary.option_set("blood:full"))
    for variant in vocabulary.variant_ids:
        options = set(vocabulary.option_set(variant))
        kind = variant.split(":", 1)[1]
        if kind.startswith("minus"):
            assert len(full - options) == int(kind.removeprefix("minus"))
            assert not options - full
        elif kind.startswith("plus"):
            assert len(options - full) == int(kind.removeprefix("plus"))
            assert full <= options


def test_an_unknown_variant_is_an_error_not_a_default(cfg, artifacts):
    vocabulary = PanelVocabulary(cfg, "blood", artifacts.definitions)
    with pytest.raises(ValueError, match="no variant"):
        vocabulary.option_set("blood:minus99")


def test_panels_recur_across_cells_where_per_cell_sampling_does_not(cfg, artifacts):
    """The point of the vocabulary: two cells can name the same panel."""
    definitions = artifacts.definitions
    candidates = cfg.candidates("blood")
    variants = cfg.model_copy(update={"vocabulary": "variants", "panel_miss_p": 0.0})
    seen = {
        sample_panel_pair(cl_id, "blood", definitions, variants, random.Random(index))[0].panel_id
        for index, cl_id in enumerate(candidates * 3)
    }
    assert len(seen) <= len(PanelVocabulary(variants, "blood", definitions).variant_ids)

    per_cell = cfg.model_copy(update={"panel_miss_p": 0.0})
    fresh = {
        sample_panel_pair(cl_id, "blood", definitions, per_cell, random.Random(index))[0].panel_id
        for index, cl_id in enumerate(candidates * 3)
    }
    assert len(fresh) > len(seen)


def test_per_cell_sampling_is_still_the_default(cfg):
    """A v1 corpus must keep building exactly as it did."""
    assert PanelConfig().vocabulary == "per_cell"
