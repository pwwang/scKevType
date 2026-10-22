"""Definitions: deterministic perturbation, no in-place mutation, and the leakage guard."""

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from sckevtype.contracts.definitions import (
    PARAPHRASE_TEMPLATES,
    Definition,
    DefinitionSet,
    LeakageError,
    assert_no_leakage,
    perturb,
)

SYNONYMS = {"MS4A1": ("CD20",), "CD8A": ("CD8",)}


def definition(**overrides) -> Definition:
    return Definition(
        **{
            "cl_id": "CL:0000625",
            "name": "CD8 T",
            "text": "Cytotoxic T cells",
            "pos_markers": ("CD3D", "CD8A", "MS4A1"),
            "neg_markers": ("CD4",),
            "provenance": "test",
            **overrides,
        }
    )


def test_render_text_is_the_exact_prompt_string():
    assert definition().render_text() == "Cytotoxic T cells. positive markers: CD3D, CD8A, MS4A1. negative markers: CD4."
    assert definition(text="", pos_markers=("CD3D",), neg_markers=()).render_text() == "positive markers: CD3D."
    assert definition(text="Only prose.", pos_markers=(), neg_markers=()).render_text() == "Only prose."
    assert definition(text="", pos_markers=(), neg_markers=()).render_text() == ""


def test_to_panel_option_carries_the_definition_and_the_markers():
    option = definition().to_panel_option()
    assert option.cl_id == "CL:0000625" and option.definition == definition().render_text()
    assert option.markers_pos == ("CD3D", "CD8A", "MS4A1") and option.markers_neg == ("CD4",)
    assert option.provenance == "test"


def test_perturbation_is_deterministic_for_a_seed():
    ops = ["paraphrase", "drop_marker", "reorder_markers", "synonym"]
    first = perturb(definition(), ops, random.Random(42), SYNONYMS)
    second = perturb(definition(), ops, random.Random(42), SYNONYMS)
    assert first == second
    assert first.digest() == second.digest()


def test_perturbation_never_mutates_the_input():
    original = definition()
    perturbed = perturb(original, ["drop_marker", "synonym", "paraphrase"], random.Random(1), SYNONYMS)
    assert original == definition()
    assert perturbed is not original
    assert perturbed.provenance != original.provenance


def test_drop_marker_removes_exactly_one_marker():
    original = definition()
    for seed in range(8):
        dropped = perturb(original, ["drop_marker"], random.Random(seed))
        assert len(dropped.markers) == len(original.markers) - 1
        assert set(original.markers) - set(dropped.markers) in [{"CD3D"}, {"CD8A"}, {"MS4A1"}, {"CD4"}]
        assert dropped.provenance.startswith("test+drop_marker(")


def test_drop_marker_on_a_markerless_definition_is_a_recorded_no_op():
    dropped = perturb(definition(pos_markers=(), neg_markers=()), ["drop_marker"], random.Random(0))
    assert dropped.markers == () and dropped.provenance == "test+drop_marker(none)"


def test_reorder_markers_keeps_the_marker_multiset():
    original = definition()
    reordered = perturb(original, ["reorder_markers"], random.Random(3))
    assert sorted(reordered.pos_markers) == sorted(original.pos_markers)
    assert sorted(reordered.neg_markers) == sorted(original.neg_markers)


def test_paraphrase_varies_the_prose_only():
    texts = set()
    for seed in range(10):
        paraphrased = perturb(definition(), ["paraphrase"], random.Random(seed))
        assert paraphrased.markers == definition().markers
        assert paraphrased.text.rstrip(".") in {template.format(text="Cytotoxic T cells") for template in PARAPHRASE_TEMPLATES}
        texts.add(paraphrased.text)
    assert len(texts) > 1, "paraphrase must actually rewrite the prose"


def test_synonym_swaps_in_a_curated_alias():
    swapped = perturb(definition(), ["synonym", "synonym"], random.Random(5), SYNONYMS)
    assert sorted(swapped.markers) == sorted(("CD3D", "CD20", "CD8", "CD4"))
    assert "synonym(" in swapped.provenance

    none_known = perturb(definition(pos_markers=("CD3D",), neg_markers=()), ["synonym"], random.Random(5), {})
    assert none_known.markers == ("CD3D",) and none_known.provenance == "test+synonym(none)"


def test_unknown_op_is_rejected():
    with pytest.raises(ValueError, match="unknown perturbation op"):
        perturb(definition(), ["summarise"], random.Random(0))


def test_leakage_guard_raises_on_overlap_and_passes_when_disjoint():
    with pytest.raises(LeakageError) as error:
        assert_no_leakage(definition(), ["CD8A", "GZMK"])
    assert "CD8A" in str(error.value)
    assert_no_leakage(definition(), ["GZMK", "NKG7", "EPCAM"])
    assert_no_leakage(definition(), [])


def test_leakage_guard_sees_markers_named_only_in_the_prose():
    smuggled = definition(text="Marked by MS4A1 and nothing else", pos_markers=("CD3D",), neg_markers=())
    with pytest.raises(LeakageError):
        assert_no_leakage(smuggled, ["MS4A1"])


def test_leakage_alpha_tolerates_a_shared_fraction():
    assert_no_leakage(definition(), ["CD4"], alpha=0.25)          # 1 of 4 markers
    with pytest.raises(LeakageError):
        assert_no_leakage(definition(), ["CD4", "CD3D"], alpha=0.25)   # 2 of 4


def test_definition_set_lookup_save_and_load(tmp_path):
    definitions = DefinitionSet(definitions=(definition(), definition(cl_id="CL:0000624", name="CD4 T")), synonyms=SYNONYMS, source="test")
    assert set(definitions.by_cl_id) == {"CL:0000624", "CL:0000625"}
    assert definitions.to_panel_option("CL:0000624").name == "CD4 T"
    with pytest.raises(KeyError):
        definitions.get("CL:9999999")

    path = tmp_path / "definitions.json"
    digest = definitions.save(str(path))
    assert digest == definitions.digest()
    assert DefinitionSet.load(str(path)) == definitions

    path.write_text(path.read_text(encoding="utf-8").replace("CD8 T", "CD8 cells"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        DefinitionSet.load(str(path))


def test_definition_set_requires_definitions():
    with pytest.raises(ValidationError):
        DefinitionSet(definitions=())
