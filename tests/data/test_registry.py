"""``configs/datasets.yaml``: the real datasets, their access paths, and their tiers."""

from __future__ import annotations

from pathlib import Path

import pytest

from sckevtype.data import registry as registry_module
from sckevtype.data.registry import DatasetEntry, DatasetRegistry

REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"
TIERS = {"train", "T2", "T3"}
KINDS = {"file", "collection", "page"}


@pytest.fixture(scope="module")
def registry() -> DatasetRegistry:
    return registry_module.load(str(REGISTRY))


def test_every_entry_is_loadable_and_fully_specified(registry):
    assert len(registry.datasets) >= 8
    for entry in registry.datasets:
        assert entry.id and entry.tier in TIERS and entry.format and entry.provenance
        assert entry.licence
        assert entry.access, f"{entry.id} has no access path"
        for file in entry.access:
            assert file.kind in KINDS and file.url.startswith("https://")
            # An unpinned digest is a deliberate, reviewable state: it is pinned on fetch.
            assert file.sha256 is None or len(file.sha256) == 64


def test_the_verified_accessions_are_the_ones_the_literature_review_found(registry):
    """The ids and accessions here are the ones research/lit/08 verified — not guesses."""
    assert set(registry.by_id) >= {
        "pbmc3k",
        "zheng68k",
        "hao_cite",
        "dominguez_conde",
        "muraro",
        "baron",
        "tabula_sapiens",
        "tabula_muris_senis",
        "siletti",
    }
    assert registry.get("muraro").tier == "T3"
    assert registry.get("dominguez_conde").tier == "T2"
    assert registry.get("pbmc3k").tier == "train"
    assert registry.get("baron").obs.label == "assigned_cluster"


def test_the_benchmark_tiers_match_the_paper_plan(registry):
    tiers = {entry.id: entry.tier for entry in registry.datasets}
    assert tiers["hao_cite"] == tiers["zheng68k"] == tiers["pbmc3k"] == "train"
    assert tiers["muraro"] == tiers["baron"] == tiers["tabula_sapiens"] == "T3"


def test_unknown_keys_and_unknown_ids_are_errors(registry):
    with pytest.raises(KeyError):
        registry.get("not_a_dataset")
    with pytest.raises(ValueError):
        DatasetEntry.model_validate({"id": "x", "tier": "train", "format": "csv_expr", "surprise": True})


def test_a_dataset_marked_registered_must_be_in_the_registry(registry):
    from sckevtype.data.build import CorpusConfig, DatasetConfig, resolve_dataset

    corpus = CorpusConfig(
        name="x",
        seed=1,
        datasets=[DatasetConfig(id="placeholder", registered=False, format="csv_expr", tier="T3", path="/tmp/x", cap=1)],
        state={"max_tokens": 384},
    )
    with pytest.raises(KeyError):
        resolve_dataset(DatasetConfig(id="ghost", registered=True), corpus)
    resolved = resolve_dataset(DatasetConfig(id="baron", registered=True, path="/tmp/nowhere"), corpus)
    assert resolved["format"] == "csv_expr" and resolved["tier"] == "T3"
    assert resolved["cap"] and resolved["licence"]
