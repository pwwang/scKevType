"""The state renderer: frozen behaviour, deterministic, ASCII, digest-traceable."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from sckevtype.contracts.render import (
    DEFAULT_EXCLUDE_PATTERNS,
    FilterConfig,
    parse_rank_list,
    render_digest,
    render_rank_list,
    verify_roundtrip,
)

GENES = ["CD3D", "CD3E", "CD8A", "MS4A1", "LYZ", "MT-CO1", "RPS6", "RPL13", "HBB", "MT-ND1", "MTRNR2L12", "ACTB"]
VALUES = [5.0, 5.0, 4.0, 3.0, 2.0, 9.0, 8.0, 7.0, 6.0, 6.0, 6.0, 9.0]
#: CD3D/CD3E tie at 5.0, everything above them is filtered out.
EXPECTED = "CD3D 1 | CD3E 2 | CD8A 3 | MS4A1 4 | LYZ 5 | ..."


def cfg(**overrides) -> FilterConfig:
    return FilterConfig(**{"ubiquitous": ("ACTB",), **overrides})


def test_default_patterns_are_the_frozen_ones():
    assert cfg().exclude_patterns == DEFAULT_EXCLUDE_PATTERNS


def test_renders_the_frozen_string():
    assert render_rank_list(GENES, VALUES, cfg()) == EXPECTED


def test_deterministic_and_input_order_independent():
    pairs = list(zip(GENES, VALUES))
    shuffled = list(pairs)
    random.Random(7).shuffle(shuffled)
    assert render_rank_list(*zip(*pairs), cfg()) == render_rank_list(*zip(*shuffled), cfg())


def test_ranks_are_descending_by_value():
    values = dict(zip(GENES, VALUES))
    pairs = parse_rank_list(render_rank_list(GENES, VALUES, cfg()), cfg())
    assert [rank for _, rank in pairs] == [1, 2, 3, 4, 5]
    assert [values[symbol] for symbol, _ in pairs] == sorted((values[symbol] for symbol, _ in pairs), reverse=True)


def test_equal_values_break_ties_by_symbol_ascending():
    assert render_rank_list(["ZZZ1", "AAA1"], [1.0, 1.0], cfg()) == "AAA1 1 | ZZZ1 2 | ..."


def test_excluded_families_are_dropped():
    rendered = render_rank_list(GENES, VALUES, cfg())
    for symbol in ("MT-CO1", "MT-ND1", "MTRNR2L12", "RPS6", "RPL13", "HBB", "ACTB"):
        assert symbol not in rendered, symbol


def test_max_genes_caps_the_list():
    rendered = render_rank_list(GENES, VALUES, cfg(max_genes=2))
    assert rendered == "CD3D 1 | CD3E 2 | ..."
    assert len(parse_rank_list(rendered, cfg(max_genes=2))) == 2


def test_delimiter_is_present_and_configurable():
    assert render_rank_list(GENES, VALUES, cfg()).endswith("| ...")
    assert render_rank_list(["CD3D"], [1.0], cfg(delimiter="| END")).endswith("| END")


def test_empty_and_all_filtered_input_render_the_delimiter_only():
    assert render_rank_list([], [], cfg()) == "| ..."
    assert render_rank_list(["MT-CO1", "RPS6"], [2.0, 1.0], cfg()) == "| ..."


def test_output_is_ascii_even_with_non_ascii_symbols():
    rendered = render_rank_list(["CD3D", "GÄNE1", "NAÏVE"], [1.0, 5.0, 4.0], cfg())
    assert rendered.isascii()
    assert rendered == "CD3D 1 | ..."


def test_duplicate_symbols_collapse_to_the_highest_value():
    rendered = render_rank_list(["CD3D", "CD3D", "LYZ"], [1.0, 9.0, 5.0], cfg())
    assert rendered == "CD3D 1 | LYZ 2 | ..."


def test_digest_changes_when_any_config_field_changes():
    base = cfg().digest()
    variants = [
        cfg(max_genes=201),
        cfg(delimiter="| etc"),
        cfg(exclude_patterns=("MT-",)),
        cfg(ubiquitous=("ACTB", "GAPDH")),
    ]
    assert len({base, *(variant.digest() for variant in variants)}) == len(variants) + 1
    assert cfg().digest() == base, "equal configs must digest equally"


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        render_rank_list(["CD3D"], [1.0, 2.0], cfg())
    with pytest.raises(ValueError, match="NaN"):
        render_rank_list(["CD3D"], [float("nan")], cfg())
    with pytest.raises(ValidationError):
        cfg(max_genes=0)


def test_render_digest_tracks_the_string():
    rendered = render_rank_list(GENES, VALUES, cfg())
    assert render_digest(rendered) == render_digest(rendered)
    assert render_digest(rendered) != render_digest(rendered + " ")
    assert len(render_digest(rendered)) == 64


def test_verify_roundtrip_passes_and_parsing_is_strict():
    assert verify_roundtrip(GENES, VALUES, cfg()) == EXPECTED
    with pytest.raises(ValueError, match="malformed"):
        parse_rank_list("CD3D| ...", cfg())
