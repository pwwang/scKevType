"""The fetch layer: everything except the socket. Networked tests are marked and skipped."""

from __future__ import annotations

import hashlib
import json

import pytest

from sckevtype.data import fetch as fetch_module
from sckevtype.data import registry as registry_module
from sckevtype.data.fetch import FetchError, fetch, select_asset, stage, tree_digest
from sckevtype.data.registry import DatasetRegistry

#: The fields every registry entry must carry, so a test only writes the interesting ones.
BASE = {
    "tier": "train",
    "format": "tenx_mtx",
    "obs": {"label": "cell_type"},
    "default_cap": 10,
    "licence": "test",
    "provenance": "test",
}


def registry(entry: dict) -> DatasetRegistry:
    """A one-entry registry: these tests exercise fetch, not the registry's own rules."""
    return DatasetRegistry.model_validate({"version": "test", "source": "test", "datasets": [BASE | entry]})


def h5ad(title: str, url: str) -> dict:
    return {"title": title, "assets": [{"files": [{"filetype": "H5AD", "url": url}]}]}


def test_select_asset_matches_the_title_case_insensitively():
    payload = {"datasets": [h5ad("Global H5AD", "u1"), h5ad("Muraro pancreas", "u2")]}
    assert select_asset(payload, "muraro") == {"title": "Muraro pancreas", "url": "u2"}
    assert select_asset(payload, "MURARO")["url"] == "u2"
    with pytest.raises(FetchError, match="Global H5AD"):
        select_asset(payload, "siletti")
    # Only H5AD files are assets: a collection that also ships an RDS must not confuse it.
    mixed = {"datasets": [{"title": "Siletti", "assets": [{"files": [{"filetype": "RDS", "url": "r"}]}, *h5ad("Siletti", "u3")["assets"]]}]}
    assert select_asset(mixed, "siletti")["url"] == "u3"


def test_select_asset_without_a_selector_takes_the_only_asset_there_is():
    assert select_asset({"datasets": [h5ad("only", "u")]}, "")["url"] == "u"
    with pytest.raises(FetchError, match="matched 0 of 2"):
        select_asset({"datasets": [h5ad("a", "u1"), h5ad("b", "u2")]}, "")
    with pytest.raises(FetchError, match="no H5AD asset"):
        select_asset({"datasets": []}, "any")


def test_tree_digest_is_order_independent_and_content_sensitive(tmp_path):
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("y", encoding="utf-8")
    first = tree_digest(tmp_path)
    (tmp_path / "b.txt").write_text("z", encoding="utf-8")
    assert tree_digest(tmp_path) != first
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    assert tree_digest(tmp_path) == first


def test_download_writes_verifies_and_pins(tmp_path):
    payload = b"hello matrix"
    digest = hashlib.sha256(payload).hexdigest()
    entry = {
        "id": "tiny",
        "access": [{"name": "matrix.mtx", "kind": "file", "url": "https://example.invalid/matrix.mtx", "sha256": digest}],
    }
    calls = []

    def opener(url, target, *, offset=0):
        calls.append((url, offset))
        target.write_bytes(payload[offset:])

    path = fetch("tiny", tmp_path, registry=registry(entry), opener=opener)
    assert path == tmp_path / "tiny"
    assert (path / "matrix.mtx").read_bytes() == payload
    assert calls == [("https://example.invalid/matrix.mtx", 0)]
    # Cached and verified on the second call: nothing is downloaded again.
    fetch("tiny", tmp_path, registry=registry(entry), opener=opener)
    assert len(calls) == 1


def test_download_resumes_a_partial_file_and_discards_a_bad_one(tmp_path):
    payload = b"0123456789"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny"
    target.mkdir()
    (target / "data.bin.part").write_bytes(payload[:4])
    entry = {
        "id": "tiny",
        "format": "csv_expr",
        "access": [{"name": "data.bin", "kind": "file", "url": "https://example.invalid/d", "sha256": digest}],
    }
    seen = {}

    def opener(url, path, *, offset=0):
        seen["offset"] = offset
        path.write_bytes(payload[:offset] + payload[offset:])

    fetch("tiny", tmp_path, registry=registry(entry), opener=opener)
    assert seen["offset"] == 4  # resumed, not restarted
    assert (target / "data.bin").read_bytes() == payload
    assert not (target / "data.bin.part").exists()

    # A cache entry that no longer matches its pin is replaced, never trusted.
    (target / "data.bin").write_bytes(b"wrong")
    fetch("tiny", tmp_path, registry=registry(entry), opener=opener)
    assert (target / "data.bin").read_bytes() == payload

    # And bytes that do not match the pin are refused, leaving nothing behind.
    (target / "data.bin").unlink()
    with pytest.raises(FetchError, match="digest mismatch"):
        fetch("tiny", tmp_path, registry=registry(entry), opener=lambda url, path, *, offset=0: path.write_bytes(b"corrupt"))
    assert not (target / "data.bin").exists() and not (target / "data.bin.part").exists()


def test_an_unpinned_file_gets_a_digest_sidecar(tmp_path):
    entry = {
        "id": "tiny",
        "verify_on_fetch": True,
        "access": [{"name": "f.bin", "kind": "file", "url": "https://example.invalid/f"}],
    }
    fetch("tiny", tmp_path, registry=registry(entry), opener=lambda url, path, *, offset=0: path.write_bytes(b"abc"))
    sidecar = tmp_path / "tiny" / "f.bin.sha256"
    assert sidecar.read_text(encoding="utf-8").strip() == hashlib.sha256(b"abc").hexdigest()


def test_a_landing_page_access_path_is_refused_with_instructions(tmp_path):
    entry = {"id": "pbmc3k", "access": [{"name": "pbmc3k.tar.gz", "kind": "page", "url": "https://example.invalid/landing"}]}
    with pytest.raises(FetchError, match="landing page"):
        fetch("pbmc3k", tmp_path, registry=registry(entry))


def test_a_collection_resolves_through_the_api_then_downloads(tmp_path, monkeypatch):
    entry = {
        "id": "muraro",
        "tier": "T3",
        "format": "cellxgene_h5ad",
        "access": [{"name": "local.h5ad", "kind": "collection", "url": "https://api.example.invalid/c", "selector": "Muraro"}],
    }
    monkeypatch.setattr(fetch_module, "_http_json", lambda url: {"datasets": [h5ad("Muraro pancreas", "https://example.invalid/m.h5ad")]})
    seen = {}

    def opener(url, path, *, offset=0):
        seen["url"] = url
        path.write_bytes(b"h5ad")

    path = fetch("muraro", tmp_path, registry=registry(entry), opener=opener)
    assert seen["url"] == "https://example.invalid/m.h5ad"
    assert (path / "local.h5ad").read_bytes() == b"h5ad"


def test_stage_copies_a_directory_and_pins_its_digest(tmp_path, monkeypatch):
    # `stage` takes the directory the operator downloaded *into*: the registry names
    # the entry inside it (`bundle`), which is how a three-file 10x bundle is copied.
    source = tmp_path / "downloaded"
    (source / "bundle").mkdir(parents=True)
    (source / "bundle" / "matrix.mtx").write_text("1 1 1\n", encoding="utf-8")
    (source / "bundle" / "barcodes.tsv").write_text("b\n", encoding="utf-8")
    entry = {
        "id": "staged",
        "verify_on_fetch": True,
        "access": [{"name": "bundle", "kind": "file", "url": "file://ignored"}],
    }
    registry_file = tmp_path / "datasets.yaml"
    registry_file.write_text(json.dumps({"version": "test", "source": "test", "datasets": [BASE | entry]}), encoding="utf-8")
    monkeypatch.setattr(fetch_module, "_default_registry", lambda: registry_module.load(str(registry_file)))
    dest = stage("staged", tmp_path / "cache", source)
    assert (dest / "bundle" / "matrix.mtx").read_text(encoding="utf-8") == "1 1 1\n"
    # A directory cannot be pinned by file_digest, so the sidecar holds the tree digest.
    assert (dest / "bundle.sha256").read_text(encoding="utf-8").strip() == tree_digest(dest / "bundle")


@pytest.mark.network
def test_fetch_a_real_dataset_from_the_registry(tmp_path):
    """Downloads the smallest real dataset (pbmc3k). Skipped unless --run-network."""
    path = fetch("pbmc3k", tmp_path)
    assert (path / "matrix.mtx").exists() or any(path.iterdir())


class _RangeResponse:
    """A ``requests`` response reduced to what the resume check reads."""

    def __init__(self, status_code: int, content_range: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {"Content-Range": content_range} if content_range else {}


def test_a_416_past_the_end_is_a_finished_download_not_a_failure():
    """A part holding the whole object makes the resume range unsatisfiable — that is success.

    Observed for real: a 45 GB atlas's pull process died after the last byte landed
    (``data/raw/tabula_sapiens/tabula_sapiens.h5ad.part`` == the asset's full size), and
    every re-run failed with ``416`` while the complete file sat on disk. The caller
    still hashes the part, so accepting it is exactly as safe as any first fetch.
    """
    complete = _RangeResponse(416, "bytes */45013733785")
    assert fetch_module._resume_is_complete(complete, 45013733785)
    assert fetch_module._resume_is_complete(_RangeResponse(416, "bytes */100"), 120), "past the end is still complete"
    assert not fetch_module._resume_is_complete(_RangeResponse(416, "bytes */100"), 40), "short: a real error"
    assert not fetch_module._resume_is_complete(_RangeResponse(416), 40), "no Content-Range: a real error"
    assert not fetch_module._resume_is_complete(_RangeResponse(206, "bytes 0-1/100"), 40)
