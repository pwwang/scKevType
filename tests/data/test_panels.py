"""Panel sampling: the invariants the supervision depends on."""

from __future__ import annotations

import pytest

from sckevtype.contracts.schema import panel_id, size_bin
from sckevtype.data import seed_int, seeded_rng
from sckevtype.data.panels import (
    PanelConfig,
    draw_size,
    sample_panel,
    sample_panel_pair,
)


def _panel(cl_id, cfg, artifacts, *, seed=1, tissue="blood"):
    return sample_panel(cl_id, tissue, artifacts.definitions, cfg, seeded_rng(seed_int(seed, cl_id), "panel"))


def test_true_type_is_on_the_panel_when_it_is_not_a_miss(artifacts, panel_cfg):
    for seed in range(20):
        panel = _panel("CL:0000624", panel_cfg, artifacts, seed=seed)
        assert "CL:0000624" in [option.cl_id for option in panel.options]


def test_panel_miss_genuinely_excludes_the_true_type(artifacts, panel_cfg):
    cfg = panel_cfg.model_copy(update={"panel_miss_p": 1.0})
    for seed in range(10):
        panel = _panel("CL:0000624", cfg, artifacts, seed=seed)
        assert "CL:0000624" not in [option.cl_id for option in panel.options]


def test_size_is_within_bounds_and_the_bin_matches_the_realised_size(artifacts, panel_cfg):
    cfg = panel_cfg.model_copy(update={"size_min": 3, "size_max": 7})
    for seed in range(30):
        panel = _panel("CL:0000623", cfg, artifacts, seed=seed)
        assert 3 <= len(panel.options) <= 7
        assert panel.size_bin == size_bin(len(panel.options))
        assert len({option.cl_id for option in panel.options}) == len(panel.options)


def test_sampling_is_deterministic_under_the_seed(artifacts, panel_cfg):
    first, second = _panel("CL:0000236", panel_cfg, artifacts, seed=5), _panel("CL:0000236", panel_cfg, artifacts, seed=5)
    assert first.model_dump() == second.model_dump()
    other = _panel("CL:0000236", panel_cfg, artifacts, seed=6)
    assert [option.name for option in other.options] != [option.name for option in first.options] or other.options != first.options


def test_panel_id_survives_option_reordering(artifacts, panel_cfg):
    """The panel id names the option *set*: training permutes the order, the id must not move."""
    panel = _panel("CL:0000860", panel_cfg, artifacts, seed=3)
    reversed_options = tuple(reversed(panel.options))
    assert panel_id(reversed_options) == panel.panel_id


def test_options_carry_their_definition_text(artifacts, panel_cfg):
    panel = _panel("CL:0000624", panel_cfg, artifacts, seed=2)
    assert all(option.definition.strip() for option in panel.options)
    true_option = next(option for option in panel.options if option.cl_id == "CL:0000624")
    definition = artifacts.definitions.get("CL:0000624")
    assert true_option.name == definition.name
    assert true_option.markers_pos == definition.pos_markers
    assert true_option.provenance


def test_the_pair_shares_a_cell_and_differs_only_by_the_missing_option(artifacts, panel_cfg):
    cfg = panel_cfg.model_copy(update={"panel_miss_p": 1.0})
    panel, sibling = sample_panel_pair("CL:0000624", "blood", artifacts.definitions, cfg, seeded_rng(seed_int(9), "panel"))
    assert sibling is not None
    miss_keys = {option.cl_id for option in panel.options}
    sibling_keys = {option.cl_id for option in sibling.options}
    assert "CL:0000624" not in miss_keys
    assert sibling_keys - miss_keys == {"CL:0000624"}
    # Exactly one option more: the contrast is "the true type is on the panel or it is not".
    assert len(sibling.options) == len(panel.options) + 1
    assert panel.panel_id != sibling.panel_id


def test_no_miss_means_no_sibling(artifacts, panel_cfg):
    panel, sibling = sample_panel_pair("CL:0000624", "blood", artifacts.definitions, panel_cfg, seeded_rng(seed_int(4), "panel"))
    assert sibling is None
    assert "CL:0000624" in [option.cl_id for option in panel.options]


def test_confusions_are_drawn_from_the_lineage_first(artifacts):
    """``confusion_frac=1`` must fill the panel from the true type's own tissue list."""
    cfg = PanelConfig.from_artifacts(artifacts.tissue_map, artifacts.cl_terms, size_min=4, size_max=4, confusion_frac=1.0, panel_miss_p=0.0)
    panel = _panel("CL:0000625", cfg, artifacts, seed=11)
    candidates = set(cfg.candidates("blood"))
    assert {option.cl_id for option in panel.options} <= candidates


def test_size_distributions_stay_inside_the_bounds(panel_cfg):
    loguniform = seeded_rng(seed_int(1), "size")
    uniform = seeded_rng(seed_int(1), "size")
    sizes = [draw_size(panel_cfg, loguniform) for _ in range(200)]
    uniforms = [draw_size(panel_cfg.model_copy(update={"size_dist": "uniform"}), uniform) for _ in range(200)]
    assert all(panel_cfg.size_min <= size <= panel_cfg.size_max for size in sizes + uniforms)
    assert len(set(sizes)) > 1 and len(set(uniforms)) > 1
    assert min(sizes) < max(uniforms)  # both are real draws, not a constant


def test_candidates_and_shared_depth_come_from_the_artifacts(panel_cfg):
    candidates = panel_cfg.candidates("blood")
    assert "CL:0000624" in candidates
    assert isinstance(candidates, tuple) and len(set(candidates)) == len(candidates)
    assert set(candidates) <= {cl_id for terms in panel_cfg.tissues.values() for cl_id in terms}
    assert panel_cfg.shared_depth("CL:0000624", "CL:0000624") >= panel_cfg.shared_depth("CL:0000624", "CL:0000236")


def test_tissue_candidates_are_required_not_guessed(panel_cfg):
    with pytest.raises(ValueError, match="mars"):
        panel_cfg.candidates("mars")
