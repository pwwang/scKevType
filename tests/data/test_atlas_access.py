"""Step 7 §1: the corpus-level `access:` override, proved without the network.

`configs/datasets.yaml` is the *inventory* of verified access paths: for baron,
hao_cite, pbmc3k and zheng68k that path is a landing page, and `fetch` correctly
refuses to guess a file URL from one. `configs/corpus/real-v2.yaml` therefore records
the probe-verified path for the one of the four that could be wired, and the corpus's
`access:` replaces the registry's for the pull, the check and the build alike — the
registry keeps saying what was verified, which is why it is not edited.

The collection API is stubbed with a payload of the shape the live endpoint returns
(flat ``datasets[].assets[]``, repaired by `pull.collection_assets`), so nothing here
touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sckevtype.contracts.freeze import file_digest
from sckevtype.data import fetch as fetch_module
from sckevtype.data import registry as registry_module
from sckevtype.data.build import load_config, resolve_dataset
from sckevtype.data.pull import is_cached, main, pull

ROOT = Path(__file__).resolve().parents[2]
REAL_V2 = ROOT / "configs" / "corpus" / "real-v2.yaml"

#: What the live curation API returns for this collection, abridged to the fields
#: `select_asset` reads. The URL and size are the ones `--check` printed on 2026-09-23.
HAO_ASSET = {
    "filetype": "H5AD",
    "url": "https://datasets.cellxgene.cziscience.com/example.h5ad",
    "filesize": 2606736974,
}


@pytest.fixture(scope="module")
def config():
    return load_config(REAL_V2)


@pytest.fixture(scope="module")
def hao_cite(config):
    return next(dataset for dataset in config.datasets if dataset.id == "hao_cite")


def _api_payload() -> dict:
    return {"datasets": [{"title": "nygc multimodal pbmc", "assets": [dict(HAO_ASSET)]}]}


def _staged(cache: Path, name: str, payload: bytes = b"already here") -> Path:
    """A complete, digest-pinned file in the cache — what a first fetch leaves behind."""
    target = cache / "hao_cite"
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    path.write_bytes(payload)
    fetch_module._digest_sidecar(path).write_text(file_digest(path) + "\n", encoding="utf-8")
    return path


def test_the_registry_still_says_page_and_the_corpus_records_the_probed_path(config, hao_cite):
    """The override is the corpus's record, not an edit to what the inventory verified."""
    entry = registry_module.load().get("hao_cite")
    assert [file.kind for file in entry.access] == ["page"], "configs/datasets.yaml is frozen"
    assert entry.obs.label is None and hao_cite.obs.label is not None, "the corpus supplies the columns the registry lacks"
    assert [file.kind for file in hao_cite.access] == ["collection"]
    assert hao_cite.access[0].selector == "nygc multimodal pbmc"


def test_every_wired_atlas_is_fetchable_and_every_url_is_a_probed_one(config):
    """A wired atlas must be `fetch: true` with no `kind: page` left in its paths."""
    for dataset in config.datasets:
        if dataset.access:
            assert dataset.fetch, f"{dataset.id}: an access override is only for a fetchable atlas"
            assert all(file.kind != "page" for file in dataset.access), f"{dataset.id}: a landing page is not a URL"


def test_the_state_tokenizer_pin_does_not_drift_from_the_registry(config):
    """The corpus's tokenizer is restated from `configs/models.yaml`, like every other restated field.

    It decides the number ``qc.json``'s ``tokens`` block reports states against, so a
    drifted pin would attribute the measurement to the wrong vocabulary.
    """
    import yaml

    registry = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]["0.8b"]
    assert config.state.tokenizer is not None, "a corpus that pins no tokenizer reports estimates, not measurements"
    assert config.state.tokenizer.id == registry["id"]
    assert config.state.tokenizer.revision == registry["revision"]


def test_check_reports_the_corpus_url_not_the_landing_page(monkeypatch, capsys):
    """`--check --only hao_cite` answers with the asset the corpus names, and its size."""
    monkeypatch.setattr(fetch_module, "_http_json", lambda url, **kwargs: _api_payload())
    assert main(["--config", str(REAL_V2), "--check", "--only", "hao_cite"]) == 0
    record = json.loads(capsys.readouterr().out.strip())
    assert record["status"] == "startable"
    assert record["bytes"] == HAO_ASSET["filesize"]
    assert record["files"][0]["url"] == HAO_ASSET["url"]


def test_a_warm_hao_cite_pull_resolves_nothing_and_reports_cached(tmp_path, monkeypatch, hao_cite):
    """Step 7 §3: a second run is a no-op *before* the collection URL is even resolved."""
    _staged(tmp_path, hao_cite.access[0].name)

    def refuse(*args, **kwargs):
        raise AssertionError("a warm cache must not reach the network")

    monkeypatch.setattr(fetch_module, "_http_json", refuse)
    record = pull(hao_cite, tmp_path, opener=refuse)
    assert record["status"] == "cached"
    assert is_cached(registry_module.with_access(registry_module.load(), "hao_cite", hao_cite.access).get("hao_cite"), tmp_path / "hao_cite")


def test_resuming_a_partial_does_not_refetch_the_whole_file(tmp_path, hao_cite):
    """§3's other half: a `.part` continues from its own length."""
    payload, head = b"A" * 64, b"A" * 16
    part = tmp_path / "hao_cite" / (hao_cite.access[0].name + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(head)
    seen: list[int] = []

    def opener(url, target, *, offset=0):
        seen.append(offset)
        with target.open("ab" if offset else "wb") as stream:
            stream.write(payload[offset:])

    assert pull(hao_cite, tmp_path, opener=opener)["status"] == "fetched"
    assert seen == [len(head)], "the partial must resume from its own length, not restart"
    assert (tmp_path / "hao_cite" / hao_cite.access[0].name).read_bytes() == payload


def test_the_override_reaches_the_build_not_just_the_pull(tmp_path, config, hao_cite):
    """The adapter is handed the corpus's file name, not the registry's landing page name."""
    name = hao_cite.access[0].name
    _staged(tmp_path, name)
    corpus = config.model_copy(update={"cache": str(tmp_path)})
    settings = resolve_dataset(hao_cite, corpus)
    assert settings["path"] == str(tmp_path / "hao_cite" / name)
    assert settings["format"] == "cellxgene_h5ad", "the corpus's format, not the registry's csv_expr"
