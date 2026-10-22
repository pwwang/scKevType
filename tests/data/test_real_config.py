"""The real corpus config: it parses, it agrees with the registry, it stays in scope.

Two claims are checked beyond parsing. First, `configs/corpus/real-v1.yaml` restates
fields the registry already owns (`format`, `tier`, `tissue`, `obs`, `cap`,
`genes_column`, and the label map) purely so the corpus is readable in one file — so
those restatements must never drift, and the registry stays the single source of truth.
Second, `label_map` is a *scope* over the frozen `artifacts/v1` vocabulary; a CL id
outside it must be impossible to write into this config.

Nothing here touches the network, and no test downloads an atlas: the fetch tests
drive the real registry entries with an injected opener, so what they prove is the
cache policy of `sckevtype.data.pull` without the bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sckevtype.contracts.freeze import file_digest
from sckevtype.data import fetch as fetch_module
from sckevtype.data import registry as registry_module
from sckevtype.data.build import Artifacts, CorpusConfig, load_config
from sckevtype.data.pull import is_cached, pull, resolve_cache

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "configs" / "corpus" / "real-v1.yaml"

#: Every dataset the registry describes, i.e. the corpus must cover all nine.
REGISTRY_IDS = sorted(entry.id for entry in registry_module.load().datasets)


@pytest.fixture(scope="module")
def config() -> CorpusConfig:
    return load_config(REAL)


@pytest.fixture(scope="module")
def artifacts() -> Artifacts:
    return Artifacts(ROOT / "artifacts" / "v1")


def test_the_corpus_parses_and_covers_every_registered_atlas(config):
    assert config.name == "real-v1"
    assert sorted(d.id for d in config.datasets) == REGISTRY_IDS
    assert all(d.registered for d in config.datasets)


@pytest.mark.parametrize("dataset_id", sorted(REGISTRY_IDS))
def test_the_restated_fields_do_not_drift_from_the_registry(config, dataset_id):
    """Every field below is copied from `configs/datasets.yaml`; a mismatch means one of them lied."""
    dataset = next(d for d in config.datasets if d.id == dataset_id)
    entry = registry_module.load().get(dataset_id)
    assert (dataset.format, dataset.tier, dataset.tissue) == (entry.format, entry.tier, entry.tissue)
    assert dataset.cap == entry.default_cap
    assert dataset.obs == entry.obs
    assert dataset.tissue_labels == entry.tissue_labels
    assert dataset.genes_column == entry.var_genes
    assert dataset.path is None, "a corpus dataset is fetched or staged, never a hard-coded path"
    # The registry's `label_map` is the artifact name (or None); the corpus may inline
    # the same scope as a dict, which is the case for every CL-carrying atlas below.
    if entry.label_map is not None:
        assert dataset.label_map == entry.label_map


@pytest.mark.parametrize("dataset_id", sorted(REGISTRY_IDS))
def test_no_label_map_reaches_outside_the_frozen_vocabulary(config, artifacts, dataset_id):
    """A `label_map` is the scope: nothing it names may leave the frozen Step-1 vocabulary.

    Two map kinds are in play and they are checked by the same rule. A *translation*
    map (`label_map.seed_pbmc`) has free-text keys — "CD4 T", not a CL id — whose
    targets must be in scope. A *scope* map has CL ids on both sides, so any key that
    is a CL id must be a frozen one. In both cases the targets are the claim.
    """
    dataset = next(d for d in config.datasets if d.id == dataset_id)
    if dataset.label_map is None:
        pytest.skip(f"{dataset_id}: no label map — see the comment in configs/corpus/real-v1.yaml")
    vocabulary = set(artifacts.cl_terms)
    mapping = artifacts.label_map(dataset).mapping
    assert set(mapping.values()) <= vocabulary, f"{dataset_id}: out-of-scope targets {sorted(set(mapping.values()) - vocabulary)}"
    terms = {key for key in mapping if key.startswith("CL:")}
    assert terms <= vocabulary, f"{dataset_id}: out-of-scope keys {sorted(terms - vocabulary)}"
    if isinstance(dataset.label_map, dict):
        assert set(mapping) == vocabulary, f"{dataset_id}: an inline scope names the whole frozen vocabulary, not a subset"


def test_the_registrys_own_access_kind_decides_fetch(config):
    """`fetch: true` iff the registry has a retrievable asset — never a choice made in the config."""
    for dataset in config.datasets:
        entry = registry_module.load().get(dataset.id)
        retrievable = all(file.kind in ("file", "collection") for file in entry.access)
        assert dataset.fetch is retrievable, f"{dataset.id}: fetch={dataset.fetch} but access kinds are {[f.kind for f in entry.access]}"


def test_a_page_only_dataset_is_reported_not_startable(config, capsys):
    """The four landing-page atlases must be reported, not attempted — `fetch` refuses to guess a URL."""
    from sckevtype.data.pull import main

    assert main(["--config", str(REAL), "--check", "--only", "baron"]) == 0
    record = json.loads(capsys.readouterr().out.strip())
    assert record["status"] == "not_startable"
    assert "GSE84133" in record["reason"]


def test_the_cache_root_is_documented_and_gitignored(config):
    """Fetched bytes live under one root, and that root is covered by .gitignore."""
    assert config.cache == "data/raw"
    cache = resolve_cache(config.cache)
    assert cache == ROOT / "data" / "raw" and cache.is_absolute()
    assert resolve_cache("/tmp/elsewhere") == Path("/tmp/elsewhere")
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert any(line.strip() in ("data/", "data") for line in ignored), "data/raw must never be committed"


def test_a_warm_pull_of_a_collection_makes_no_request(tmp_path, monkeypatch):
    """A cached collection dataset is a no-op *before* the asset URL is resolved."""
    entry = registry_module.load().get("muraro")
    assert entry.access[0].kind == "collection"
    target = tmp_path / entry.id
    target.mkdir(parents=True)
    cached = target / entry.access[0].name
    cached.write_bytes(b"already here")
    fetch_module._digest_sidecar(cached).write_text(file_digest(cached) + "\n", encoding="utf-8")

    def refuse(*args, **kwargs):
        raise AssertionError("a warm cache must not reach the network")

    monkeypatch.setattr(fetch_module, "_http_json", refuse)
    dataset = next(d for d in load_config(REAL).datasets if d.id == "muraro")
    record = pull(dataset, tmp_path, opener=refuse)
    assert record["status"] == "cached"
    assert record["seconds"] >= 0


def test_a_pull_resumes_a_partial_then_a_second_pull_is_a_no_op(tmp_path):
    """The whole file is never re-fetched: the partial continues, the complete copy is skipped."""
    dataset = next(d for d in load_config(REAL).datasets if d.id == "dominguez_conde")
    entry = registry_module.load().get("dominguez_conde")
    payload, head = b"A" * 64, b"A" * 16
    part = tmp_path / entry.id / (entry.access[0].name + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(head)
    seen: list[int] = []

    def opener(url, target, *, offset=0):
        seen.append(offset)
        with target.open("ab" if offset else "wb") as stream:
            stream.write(payload[offset:])

    assert pull(dataset, tmp_path, opener=opener)["status"] == "fetched"
    assert seen == [len(head)], "the partial must resume from its own length, not restart"
    assert (tmp_path / entry.id / entry.access[0].name).read_bytes() == payload

    def refuse(*args, **kwargs):
        raise AssertionError("a second pull must not re-download")

    assert pull(dataset, tmp_path, opener=refuse)["status"] == "cached"
    assert is_cached(entry, tmp_path / entry.id)
