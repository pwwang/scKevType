"""Pull a corpus's fetchable atlases into the cache.

    python -m sckevtype.data.pull --config configs/corpus/real-v1.yaml
    python -m sckevtype.data.pull --config configs/corpus/real-v1.yaml --check

This is the download side of a corpus config: it reads the config, keeps the
datasets marked ``fetch: true``, and hands each one to
:func:`sckevtype.data.fetch.fetch` — which owns the cache check, the ``Range``
resume, the digest verification and the ``<file>.sha256`` sidecar. Nothing about
*how* a file is downloaded is restated here.

Three things this module does add, because the config needs them:

**A destination.** ``resolve_cache`` turns a corpus's ``cache:`` value into an
absolute path. A relative value is anchored at the repo root, not at the current
directory, so ``cache: data/raw`` names the same tree no matter where the command
is run from.

**A pinned CELLxGENE asset title.** ``configs/datasets.yaml`` records each
collection's ``/dp/v1/`` URL and a *hint* at the asset title — Step 2's registry
says of that hint, "the title is the one thing the inventory did not pin". Three
of the five collection entries do not resolve with it: ``Tabula Sapiens`` matches
all 35 assets, ``Tabula Muris Senis`` and ``Siletti`` match none, because the live
titles are per-organ and per-dissection. ``configs/corpus/cellxgene-assets.yaml``
pins them, with the API response each pin was read from recorded alongside. The
selection *policy* stays in :func:`sckevtype.data.fetch.select_asset`; this module
only recovers the shape that function reads (the live API moved the asset from
``datasets[].assets[].files[]`` to a flat ``datasets[].assets[]``).

**A log.** One JSON object per line on stdout, flushed as it is written, because
the usual way to run this is detached:

    setsid nohup uv run python -m sckevtype.data.pull --config <cfg> \
        > reports/fetch.log.jsonl 2>&1 < /dev/null &

A warm cache is a no-op: a dataset whose files are already present and match
their sidecars is reported ``cached`` and never reaches the network at all.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..contracts import paths
from ..contracts.freeze import ENCODING, file_digest
from . import fetch as fetch_module
from . import registry as registry_module
from .build import CorpusConfig, DatasetConfig, load_config

#: The live CELLxGENE API. Step 2 recorded ``/dp/v1/`` collection URLs; that
#: endpoint no longer carries an asset URL, while the same collection id under
#: ``/curation/v1/`` does. Verified 2026-09-23 (see ``reports/step5.md``).
CURATION_API = "https://api.cellxgene.cziscience.com/curation/v1/collections"

#: Where a relative ``cache:`` is anchored when the caller names no root.
DEFAULT_CACHE = "data/raw"

STATUS_FETCHED = "fetched"
STATUS_CACHED = "cached"
STATUS_STARTABLE = "startable"          # ``--check`` only: a fetch would start
STATUS_NOT_STARTABLE = "not_startable"
STATUS_FAILED = "failed"


class AssetPins(BaseModel):
    """The CELLxGENE asset titles the registry could not pin, verified against the API.

    ``titles`` maps a dataset id to the exact title of the asset the corpus wants.
    Nothing else about the asset is pinned here: the URL and the size come from
    the live API on every cold fetch, and the *bytes* are pinned the way every
    other dataset's are — by the digest sidecar ``fetch`` writes after the first
    download. ``source`` and ``verified`` record where the titles came from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    source: str = Field(min_length=1)
    verified: str = Field(min_length=1)
    titles: dict[str, str] = Field(min_length=1)

    @classmethod
    def load(cls, path: str | Path | None = None) -> AssetPins:
        """Read the pin file (default: ``configs/corpus/cellxgene-assets.yaml``)."""
        path = Path(path) if path else paths.configs_dir() / "corpus" / "cellxgene-assets.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            return cls.model_validate(payload)
        except ValidationError as error:
            raise ValueError(f"{path}: invalid asset pins\n{error}") from None

    def selector(self, dataset_id: str, fallback: str | None) -> str:
        """The title to match for ``dataset_id``: the pin, else the registry's hint."""
        return self.titles.get(dataset_id) or (fallback or "")


def resolve_cache(cache: str | Path | None, *, root: str | Path | None = None) -> Path:
    """Where a corpus's datasets live: ``cache`` anchored at the repo root if relative.

    Returns an absolute path, so a corpus config names the same tree from any
    working directory.
    """
    path = Path(cache if cache else DEFAULT_CACHE).expanduser()
    if path.is_absolute():
        return path
    return Path(root) / path if root is not None else paths.root() / path


def collection_assets(payload: dict[str, Any]) -> dict[str, Any]:
    """Reshape a curation-API collection into the tree ``fetch.select_asset`` reads.

    The live endpoint returns ``datasets[].assets[]`` as a flat list of files
    (``{"filetype", "url", "filesize"}``); ``select_asset`` walks
    ``datasets[].assets[].files[]``. Only the nesting is repaired here — which
    asset wins is still decided by :func:`sckevtype.data.fetch.select_asset`.
    """
    return {
        "datasets": [
            {"title": str(entry.get("title", "")), "assets": [{"files": [dict(file) for file in entry.get("assets", [])]}]}
            for entry in payload.get("datasets", [])
        ]
    }


def resolve_collection(collection_url: str, title: str) -> tuple[str, int | None]:
    """``(file url, size in bytes)`` for the asset of ``collection_url`` titled ``title``."""
    collection_id = collection_url.rstrip("/").rsplit("/", 1)[-1]
    payload = collection_assets(fetch_module._http_json(f"{CURATION_API}/{collection_id}"))
    chosen = fetch_module.select_asset(payload, title)
    for entry in payload["datasets"]:
        for file in entry["assets"][0]["files"]:
            if file.get("url") == chosen["url"]:
                return chosen["url"], file.get("filesize")
    return chosen["url"], None


def resolve_access(registry: registry_module.DatasetRegistry, entry, pins: AssetPins):
    """``registry`` with ``entry``'s collection paths replaced by resolved file URLs.

    ``fetch`` then treats the entry exactly like a ``kind: file`` one, so the
    cache check, the resume and the digest sidecar are all still its job.
    """
    access = []
    for file in entry.access:
        if file.kind == "collection":
            url, _size = resolve_collection(file.url, pins.selector(entry.id, file.selector))
            file = file.model_copy(update={"url": url, "kind": "file", "selector": None})
        access.append(file)
    updated = entry.model_copy(update={"access": tuple(access)})
    return registry.model_copy(update={"datasets": tuple(updated if item.id == entry.id else item for item in registry.datasets)})


def is_cached(entry, target: Path) -> bool:
    """Whether every access path is present under ``target`` and matches its digest.

    ``fetch._check_cached``'s rule — unpinned files are taken as they are, a
    sidecar-pinned one must hash — asked *before* a collection URL is resolved, so
    a warm re-run makes no request at all.
    """
    for file in entry.access:
        path = target / file.name
        if not path.exists():
            return False
        expected = file.sha256
        sidecar = path.with_name(path.name + ".sha256")
        if expected is None and sidecar.exists():
            text = sidecar.read_text(encoding=ENCODING).strip()
            expected = text.split()[0] if text else None
        if expected and file_digest(path) != expected:
            return False
    return True


def pull(
    dataset: DatasetConfig,
    cache: str | Path,
    *,
    registry: registry_module.DatasetRegistry | None = None,
    pins: AssetPins | None = None,
    opener: fetch_module.Opener | None = None,
) -> dict[str, Any]:
    """Fetch one corpus dataset into ``cache``; return its report entry."""
    registry = registry if registry is not None else registry_module.load()
    if dataset.access:
        registry = registry_module.with_access(registry, dataset.id, dataset.access)
    entry = registry.get(dataset.id)
    target = Path(cache) / entry.id
    started = time.monotonic()
    cached = is_cached(entry, target)
    if not cached and any(file.kind == "collection" for file in entry.access):
        registry = resolve_access(registry, entry, pins if pins is not None else AssetPins.load())
    fetch_module.fetch(dataset.id, cache, registry=registry, opener=opener)
    return {
        "dataset": dataset.id,
        "tier": entry.tier,
        "status": STATUS_CACHED if cached else STATUS_FETCHED,
        "bytes": sum(item.stat().st_size for item in sorted(target.rglob("*")) if item.is_file() and item.suffix != ".sha256"),
        "seconds": round(time.monotonic() - started, 3),
        "files": [file.name for file in entry.access],
    }


def check(dataset: DatasetConfig, *, pins: AssetPins | None = None) -> dict[str, Any]:
    """Resolve one dataset's download URL without downloading it (``--check``).

    The verification this step is named for: it answers "would a fetch start, and
    on what?" for every dataset, in seconds rather than hours.
    """
    registry = registry_module.load()
    if dataset.access:
        registry = registry_module.with_access(registry, dataset.id, dataset.access)
    entry = registry.get(dataset.id)
    pages = [file for file in entry.access if file.kind == "page"]
    if pages:
        return {
            "dataset": dataset.id,
            "status": STATUS_NOT_STARTABLE,
            "reason": f"verified access path is a landing page, not a file URL ({pages[0].url})",
            "files": [file.name for file in entry.access],
        }
    pins = pins if pins is not None else AssetPins.load()
    resolved = []
    for file in entry.access:
        if file.kind == "collection":
            url, size = resolve_collection(file.url, pins.selector(entry.id, file.selector))
            resolved.append({"name": file.name, "title": pins.selector(entry.id, file.selector), "url": url, "bytes": size})
        else:
            resolved.append({"name": file.name, "title": None, "url": file.url, "bytes": file.bytes})
    return {
        "dataset": dataset.id,
        "status": STATUS_STARTABLE,
        "tier": entry.tier,
        "bytes": sum(item["bytes"] or 0 for item in resolved) or None,
        "files": resolved,
    }


def corpus_datasets(config: CorpusConfig, *, only: tuple[str, ...] = ()) -> list[DatasetConfig]:
    """The config's datasets, in config order (optionally restricted to ``only``).

    Every dataset is returned, fetchable or not: ``--check`` reports the landing-page
    ones too, and "not startable, and here is why" is a useful answer about them.
    """
    known = {dataset.id for dataset in config.datasets}
    unknown = sorted(set(only) - known)
    if unknown:
        raise ValueError(f"--only names datasets the corpus does not have: {unknown}")
    return [dataset for dataset in config.datasets if not only or dataset.id in only]


def pull_corpus(
    config_path: str | Path,
    *,
    cache: str | Path | None = None,
    pins_path: str | Path | None = None,
    only: tuple[str, ...] = (),
    opener: fetch_module.Opener | None = None,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fetch every ``fetch: true`` dataset of a corpus; return one entry per dataset.

    The datasets a corpus marks ``fetch: false`` are reported too, as
    ``not_startable`` with the reason, so the log accounts for all nine atlases
    rather than quietly covering five of them.
    """
    config = load_config(config_path)
    root = resolve_cache(cache if cache is not None else config.cache)
    registry = registry_module.load()
    pins = AssetPins.load(pins_path)
    emit = emit if emit is not None else _emit
    results: list[dict[str, Any]] = []
    for dataset in corpus_datasets(config, only=only):
        if dataset.access:
            registry = registry_module.with_access(registry, dataset.id, dataset.access)
        entry = registry.get(dataset.id)
        record: dict[str, Any] = {"dataset": dataset.id, "tier": entry.tier, "cache": str(root)}
        pages = [file for file in entry.access if file.kind == "page"]
        if not dataset.fetch:
            record |= {"status": STATUS_NOT_STARTABLE, "reason": "the corpus marks this dataset fetch: false (staged by hand)"}
        elif pages:
            record |= {
                "status": STATUS_NOT_STARTABLE,
                "reason": f"verified access path is a landing page, not a file URL ({pages[0].url})",
            }
        else:
            emit(record | {"event": "start"})
            try:
                record |= pull(dataset, root, registry=registry, pins=pins, opener=opener)
            except (fetch_module.FetchError, OSError, ValueError) as error:
                record |= {"status": STATUS_FAILED, "reason": f"{type(error).__name__}: {error}"}
        results.append(record)
        emit(record | {"event": "done"})
    return results


def _emit(record: dict[str, Any]) -> None:
    """One JSON object per line, flushed — a detached process must be readable while it runs."""
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sckevtype.data.pull", description="Fetch a corpus's datasets into its cache.")
    parser.add_argument("--config", required=True, help="corpus config YAML")
    parser.add_argument("--cache", default=None, help="cache root (default: the config's `cache:`)")
    parser.add_argument("--pins", default=None, help="CELLxGENE asset-title pins (default: configs/corpus/cellxgene-assets.yaml)")
    parser.add_argument("--only", action="append", default=[], help="restrict to one dataset id (repeatable)")
    parser.add_argument("--check", action="store_true", help="resolve the download URLs and exit without downloading")
    args = parser.parse_args(argv)
    try:
        if args.check:
            config = load_config(args.config)
            pins = AssetPins.load(args.pins)
            for dataset in corpus_datasets(config, only=tuple(args.only)):
                _emit(check(dataset, pins=pins))
            return 0
        results = pull_corpus(args.config, cache=args.cache, pins_path=args.pins, only=tuple(args.only))
    except (ValueError, KeyError, FileNotFoundError, fetch_module.FetchError) as error:
        _emit({"status": STATUS_FAILED, "reason": f"{type(error).__name__}: {error}"})
        return 1
    return 1 if any(record["status"] == STATUS_FAILED for record in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
