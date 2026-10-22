"""CL joins: the coverage gate, the middleware expansion, and dataset-local types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sckevtype.contracts.definitions import Definition, DefinitionSet
from sckevtype.contracts.ontology import (
    ClTerm,
    LabelMap,
    TissueMap,
    load_cl_terms,
    local_type_id,
    save_cl_terms,
)


def definitions() -> DefinitionSet:
    return DefinitionSet(
        definitions=(
            Definition(cl_id="CL:0000624", name="CD4 T", text="Helper T cells", pos_markers=("CD3D", "CD4"), provenance="test"),
            Definition(cl_id="CL:0000625", name="CD8 T", text="Cytotoxic T cells", pos_markers=("CD3D", "CD8A"), provenance="test"),
            Definition(cl_id="CL:0000236", name="B cell", text="B lineage", pos_markers=("MS4A1",), provenance="test"),
        ),
        source="test",
    )


def label_map(**overrides) -> LabelMap:
    return LabelMap(
        **{
            "dataset_id": "seed_pbmc",
            "mapping": {"CD4 T": "CL:0000624", "CD8 T": "CL:0000625"},
            "source": "test",
            **overrides,
        }
    )


def test_min_coverage_gate_raises_and_names_the_unmapped_labels():
    labels = ["CD4 T", "CD8 T", "B", "Mystery"]
    with pytest.raises(ValueError) as error:
        label_map().require_min_coverage(labels, min_coverage=0.95)
    message = str(error.value)
    assert "Mystery" in message and "0.500" in message and "B" in message
    assert label_map().require_min_coverage(labels, min_coverage=0.5).rate == pytest.approx(0.5)


def test_unmappable_labels_are_dropped_or_kept_as_explicit_local_types():
    labels = ["CD4 T", "Mystery"]
    dropped, report = label_map().map_labels(labels)
    assert dropped == {"CD4 T": "CL:0000624"}
    assert report.unmapped == 1 and report.unmapped_items == ("Mystery",)

    kept, kept_report = label_map(allow_local_types=True).map_labels(labels)
    assert kept == {"CD4 T": "CL:0000624", "Mystery": "local:seed_pbmc:mystery"}
    assert kept_report.unmapped == 1, "a local type must still count as unmapped: it is not a CL term"
    assert local_type_id("seed_pbmc", "Mystery!") == "local:seed_pbmc:mystery"


def test_tissue_expansion_produces_a_valid_panel():
    tissue_map = TissueMap(tissues={"blood": ("CL:0000624", "CL:0000625")}, definitions=definitions(), source="test")
    spec = tissue_map.expand("blood")
    assert spec.source == "middleware" and spec.tissue == "blood" and spec.size_bin == "2-4"
    assert [option.cl_id for option in spec.options] == ["CL:0000624", "CL:0000625"]
    assert spec.options[0].definition == "Helper T cells. positive markers: CD3D, CD4."
    assert spec.options[0].markers_pos == ("CD3D", "CD4")
    assert len(spec.panel_id) == 64


def test_panel_id_is_stable_across_tissue_map_reordering():
    forward = TissueMap(tissues={"blood": ("CL:0000624", "CL:0000625")}, definitions=definitions(), source="test")
    reversed_ = TissueMap(tissues={"blood": ("CL:0000625", "CL:0000624")}, definitions=definitions(), source="test")
    assert forward.expand("blood").panel_id == reversed_.expand("blood").panel_id


def test_expanding_an_unknown_tissue_or_an_undefined_type_fails_loudly():
    tissue_map = TissueMap(tissues={"blood": ("CL:0000624",)}, definitions=definitions(), source="test")
    with pytest.raises(KeyError, match="no tissue"):
        tissue_map.expand("spleen")
    with pytest.raises(ValidationError, match="no definition"):
        TissueMap(tissues={"blood": ("CL:9999999",)}, definitions=definitions(), source="test")


def test_tissue_map_save_load_round_trips_and_refuses_tampering(tmp_path):
    tissue_map = TissueMap(tissues={"blood": ("CL:0000624", "CL:0000625")}, definitions=definitions(), source="test")
    path = tmp_path / "tissue_map.json"
    digest = tissue_map.save(str(path))
    assert TissueMap.load(str(path), definitions()) == tissue_map
    assert digest == tissue_map.digest()

    path.write_text(path.read_text(encoding="utf-8").replace("CL:0000625", "CL:0000236"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        TissueMap.load(str(path), definitions())


def test_cl_snapshot_round_trips_and_keeps_terms_sorted(tmp_path):
    terms = {
        "CL:0000625": ClTerm(cl_id="CL:0000625", name="CD8 T", parents=("CL:0000084",)),
        "CL:0000084": ClTerm(cl_id="CL:0000084", name="T cell"),
    }
    path = tmp_path / "cl_terms.seed.yaml"
    save_cl_terms(str(path), terms, source="test")
    loaded = load_cl_terms(str(path))
    assert loaded == terms and loaded["CL:0000625"].parents == ("CL:0000084",)
    assert path.read_text(encoding="utf-8").index("CL:0000084") < path.read_text(encoding="utf-8").index("CL:0000625")


def test_label_map_save_load_round_trips_and_refuses_tampering(tmp_path):
    path = tmp_path / "label_map.json"
    label_map().save(str(path))
    assert LabelMap.load(str(path)) == label_map()
    path.write_text(path.read_text(encoding="utf-8").replace("CL:0000624", "CL:0000625"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        LabelMap.load(str(path))


def test_label_map_requires_at_least_one_label():
    with pytest.raises(ValidationError):
        LabelMap(dataset_id="d", mapping={})
