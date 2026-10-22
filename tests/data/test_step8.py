"""Step 8: which candidate list a panel came from, and the cells the register cannot name.

§1 — a tissue the register maps is drawn against that tissue's list, and a tissue it
does not map (or one that did not resolve at all) is drawn against the register
itself, with the choice recorded per record. §2 — the cells whose true type has no
definition are *exactly* the cells the label map dropped, they are reused under no
family that needs a true type, and each says which of the two reasons applies to it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import h5py
import pytest

from sckevtype.contracts.schema import RecordFamily
from sckevtype.data import seeded_rng
from sckevtype.data.adapters import CellRecord
from sckevtype.data.build import build
from sckevtype.data.families import (
    NONE_OPTIONS,
    UNREGISTERED_TISSUE,
    UNREGISTERED_TYPE,
    FamilyContext,
    out_of_register,
)
from sckevtype.data.panels import GLOBAL_TISSUE, sample_miss_panel

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "configs" / "corpus" / "fixtures" / "smoke" / "data" / "cellxgene_lung.h5ad"
SPLITS = ("train", "calibration", "development", "test")
#: The identity map both test configs declare: a CELLxGENE label is already a CL id, so
#: the map's job is to say which terms are in scope. Everything else is dropped.
LABEL_MAP = {"CL:0000084": "CL:0000084", "CL:0000623": "CL:0000623", "CL:0000235": "CL:0000235", "CL:0000066": "CL:0000066"}
CONFIG = """\
name: {name}
seed: 20240917
version: v1
datasets:
  - id: {name}
    path: {path}
    registered: false
    format: cellxgene_h5ad
    tier: train
    tissue: {tissue}
{tissue_labels}    obs: {{label: cell_type_ontology_term_id, donor: donor_id, tissue: tissue_ontology_term_id}}
    label_map: {label_map}
    cap: 1000
    provenance: synthetic fixture (configs/corpus/fixtures/smoke/generate.py), not a real dataset
panel:
  size_min: 3
  size_max: 8
  panel_miss_p: 0.3
state:
  max_tokens: 384
"""


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _payloads(out: Path) -> list[dict]:
    return [
        json.loads(line) for split in SPLITS for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]


def _groups(payloads: list[dict], family: str | None = None) -> set[str]:
    return {p["_meta"]["group_id"] for p in payloads if family is None or p["_meta"]["family"] == family}


@pytest.fixture(scope="module")
def fixture_labels() -> dict[str, str]:
    """``cell_id -> label`` straight out of the h5ad's obs: the ground truth, not a count."""
    with h5py.File(FIXTURE, "r") as handle:
        obs = handle["obs"]
        index = "_index" if "_index" in obs else "index"
        return {_text(a): _text(b) for a, b in zip(obs[index][:], obs["cell_type_ontology_term_id"][:])}


def _build(tmp_path: Path, name: str, tissue: str | None, *, tissue_labels: str = "") -> tuple[dict, list[dict]]:
    """One-dataset corpus over the h5ad fixture whose cap never binds, so no cell is sampled away."""
    config = tmp_path / f"{name}.yaml"
    config.write_text(
        CONFIG.format(
            name=name,
            path=FIXTURE,
            tissue="null" if tissue is None else tissue,
            tissue_labels=tissue_labels,
            label_map="{" + ", ".join(f"{key}: {value}" for key, value in LABEL_MAP.items()) + "}",
        ),
        encoding="utf-8",
    )
    manifest = build(config, tmp_path / name)
    return manifest, _payloads(tmp_path / name)


def test_a_mapped_tissue_gets_its_own_list_and_an_unmapped_one_gets_the_register(artifacts, panel_cfg):
    assert panel_cfg.resolve("blood") == ("blood", panel_cfg.tissues["blood"])
    assert tuple(sorted(artifacts.definitions.by_cl_id)) == panel_cfg.register_terms
    for tissue in (None, "spleen"):  # never resolved / the register has no list for it
        assert panel_cfg.resolve(tissue) == (GLOBAL_TISSUE, panel_cfg.register_terms)
    with pytest.raises(ValueError, match="no tissue"):
        panel_cfg.candidates("spleen")


def test_a_panel_for_a_type_the_register_cannot_name_is_a_miss_by_construction(artifacts, panel_cfg):
    ctx = FamilyContext(definitions=artifacts.definitions, filter=artifacts.filter, corpus="unit")
    cell = CellRecord(
        cell_id="c1", dataset_id="unit", donor_id="d1", tissue="spleen", cl_id="CL:9999999", genes=("ACTB", "CD3D"), values=(6.0, 4.0)
    )
    panel = sample_miss_panel(GLOBAL_TISSUE, artifacts.definitions, panel_cfg, seeded_rng(1, "panel"))
    assert {option.cl_id for option in panel.options} <= set(panel_cfg.register_terms)  # the register's own types
    record = out_of_register(cell, panel, ctx, seed=1, none=NONE_OPTIONS[0], reason=UNREGISTERED_TYPE)
    assert record.meta.reason == UNREGISTERED_TYPE
    assert record.meta.panel_scope == GLOBAL_TISSUE
    assert record.answers["label"] == NONE_OPTIONS[0][0]  # it abstains
    # A panel that carries the true type is a panel_choice, not an out-of-register record.
    carried = sample_miss_panel("blood", artifacts.definitions, panel_cfg, seeded_rng(1, "panel"))
    with pytest.raises(ValueError, match="not out of register"):
        out_of_register(
            replace(cell, cl_id=carried.options[0].cl_id, tissue="blood"), carried, ctx, seed=1, none=NONE_OPTIONS[0],
            reason=UNREGISTERED_TYPE,
        )


def test_every_record_says_which_list_its_panel_came_from(smoke_build):
    payloads = [p for lines in smoke_build.lines.values() for p in lines]
    resolution = json.loads((smoke_build.out / "qc.json").read_text(encoding="utf-8"))["tissue_resolution"]
    assert set(resolution) == set(smoke_build.manifest["counts"]["per_dataset"])
    for dataset, entry in resolution.items():
        # The report counts cells, one scope each; the records are that cell's, repeated per family.
        scopes = {p["_meta"]["group_id"]: p["_meta"]["panel_scope"] for p in payloads if p["_meta"]["dataset_id"] == dataset}
        assert Counter(scopes.values()) == Counter(entry["cells_per_scope"]) != Counter()
    assert set(resolution["smoke_h5ad"]["cells_per_scope"]) == {"lung"}  # the declared tissue, in the map
    assert set(resolution["smoke_10x"]["cells_per_scope"]) == {"blood"}
    assert all(entry["tissue_list"] > 0 and entry["global"] == 0 for entry in resolution.values())


def test_out_of_register_is_exactly_the_cells_the_register_cannot_name(tmp_path, fixture_labels, artifacts):
    manifest, payloads = _build(tmp_path, "mapped", "lung")
    # The ground truth is the *register*: "cannot name" is a property of the artifact,
    # not of this config's label map. The map agrees with it here, which the second
    # assert pins down — otherwise "dropped by the map" and "unnamed by the register"
    # could differ and this test would be measuring the wrong set.
    register = set(artifacts.definitions.by_cl_id)
    assert set(LABEL_MAP) <= register
    dropped = {cell_id for cell_id, label in fixture_labels.items() if label not in register}
    assert dropped, "the fixture carries unmappable labels; without one this test proves nothing"
    counted = manifest["counts"]["per_dataset"]["mapped"]
    assert counted["cells_read"] == len(fixture_labels)
    assert counted["cells_dropped"] == len(dropped)
    assert counted["cells_in_scope"] + counted["cells_dropped"] == counted["cells_read"]
    assert _groups(payloads, RecordFamily.OUT_OF_REGISTER.value) == {f"mapped/{cell_id}" for cell_id in dropped}
    # A dropped cell is not quietly reused under a family that needs a true type.
    others = set().union(*(_groups(payloads, family.value) for family in RecordFamily if family is not RecordFamily.OUT_OF_REGISTER))
    assert not (others & _groups(payloads, RecordFamily.OUT_OF_REGISTER.value))
    # The register maps this dataset's tissue, so it is the *type* that is unknown.
    assert {p["_meta"]["reason"] for p in payloads if p["_meta"]["family"] == RecordFamily.OUT_OF_REGISTER.value} == {UNREGISTERED_TYPE}
    for payload in payloads:
        if payload["_meta"]["family"] == RecordFamily.OUT_OF_REGISTER.value:
            answer = payload["questions"]["label"]["label"]
            assert answer in {key for key, _ in NONE_OPTIONS}  # it abstains, and the option is on the panel
            assert answer in payload["questions"]["label"]["criteria"]


def test_a_tissue_the_register_does_not_map_is_the_other_reason(tmp_path, fixture_labels, artifacts):
    manifest, payloads = _build(tmp_path, "unmapped", None, tissue_labels='    tissue_labels: {"UBERON:0002048": spleen}\n')
    register = set(artifacts.definitions.by_cl_id)
    dropped = {cell_id for cell_id, label in fixture_labels.items() if label not in register}
    assert manifest["counts"]["per_dataset"]["unmapped"]["cells_dropped"] == len(dropped)
    assert _groups(payloads, RecordFamily.OUT_OF_REGISTER.value) == {f"unmapped/{cell_id}" for cell_id in dropped}
    assert {p["_meta"]["reason"] for p in payloads if p["_meta"]["family"] == RecordFamily.OUT_OF_REGISTER.value} == {UNREGISTERED_TISSUE}
    # §1 from the other end: no list exists for "spleen", so *every* record — including the
    # ones whose type the register can name — was drawn against the register.
    assert {p["_meta"]["panel_scope"] for p in payloads} == {GLOBAL_TISSUE}


def test_the_manifest_carries_per_atlas_and_per_family_counts(smoke_build, tmp_path):
    counts = smoke_build.manifest["counts"]
    labels = json.loads((smoke_build.out / "qc.json").read_text(encoding="utf-8"))["labels"]
    per_dataset = counts["per_dataset"]
    assert set(per_dataset) == {"smoke_h5ad", "smoke_10x"}
    assert sum(entry["records"] for entry in per_dataset.values()) == counts["records"]
    assert sum(counts["per_family"].values()) == counts["records"]
    for dataset, entry in per_dataset.items():
        assert entry["records"] == sum(entry["per_family"].values()), dataset
        assert set(entry["per_family"]) == {family.value for family in RecordFamily}
        assert (entry["cells_read"], entry["cells_in_scope"], entry["cells_dropped"]) == (
            labels[dataset]["total"], labels[dataset]["mapped"], labels[dataset]["dropped"]
        )
        assert 0.0 <= entry["in_scope_rate"] <= 1.0
    assert per_dataset["smoke_10x"]["per_family"][RecordFamily.OUT_OF_REGISTER.value] >= 1  # the "Doublet" labels


def test_the_family_is_additive_switching_it_off_changes_no_other_count(smoke_build, tmp_path):
    keep = [family.value for family in RecordFamily if family is not RecordFamily.OUT_OF_REGISTER]
    config = tmp_path / "smoke-without-out-of-register.yaml"
    config.write_text(Path(smoke_build.config).read_text(encoding="utf-8") + "\nfamilies: [" + ", ".join(keep) + "]\n", encoding="utf-8")
    off = build(config, tmp_path / "off")["counts"]
    assert off["per_family"][RecordFamily.OUT_OF_REGISTER.value] == 0
    assert {family: n for family, n in off["per_family"].items() if family != RecordFamily.OUT_OF_REGISTER.value} == {
        family: n for family, n in smoke_build.manifest["counts"]["per_family"].items() if family != RecordFamily.OUT_OF_REGISTER.value
    }
    for dataset, entry in smoke_build.manifest["counts"]["per_dataset"].items():
        assert off["per_dataset"][dataset]["per_family"][RecordFamily.OUT_OF_REGISTER.value] == 0
        assert off["per_dataset"][dataset]["cells_dropped"] == entry["cells_dropped"]  # dropped is a fact, not a family
