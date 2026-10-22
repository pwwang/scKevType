"""The only networked module in the package: fetch a registered dataset into the cache.

    fetch(dataset_id, dest, *, checksum=None) -> Path      # the dataset's cache directory

Three properties matter more than speed, because a corpus is reproducible only if
its inputs are:

* **cache** — a file that is already present and matches its digest is never
  re-downloaded, so a build re-runs offline over a warm cache;
* **resume** — a partial download is kept as ``<name>.part`` and continued with an
  HTTP ``Range`` request, and a server that ignores the range is detected (status
  ``200`` instead of ``206``) and restarted rather than appended to;
* **digest** — the completed file is hashed before it is put in place. A mismatch
  deletes the partial file and raises, so a corrupted download never becomes a
  cache hit. When the registry has no digest yet and ``verify_on_fetch`` is set,
  the digest actually seen is written to a ``<name>.sha256`` sidecar, which pins
  the cache from the first fetch onwards.

A dataset whose verified access path is a *landing page* (``kind: "page"``) is not
guessed at: the fetch fails with the page URL and the path to put the file at.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..contracts.freeze import ENCODING, file_digest
from .registry import DatasetEntry, DatasetRegistry, PinnedFile

#: ``opener(url, target, offset=0)`` writes the bytes of ``url`` from ``offset``
#: onwards into ``target``. Injected by the tests so the resume/verify logic is
#: exercised without a network.
Opener = Callable[..., None]

CELLXGENE_COLLECTION_API = "https://api.cellxgene.cziscience.com/dp/v1/collections"
#: The asset filetype every adapter reads.
CELLXGENE_ASSET_FILETYPE = "H5AD"


class FetchError(RuntimeError):
    """A dataset could not be fetched (offline, unpinned, wrong digest, no such asset)."""


def select_asset(payload: dict[str, Any], selector: str) -> dict[str, str]:
    """Pick one H5AD asset out of a CELLxGENE collection response.

    ``selector`` is matched case-insensitively against the dataset title (the
    registry's ``access[].name``); with a single candidate the selector is not
    needed. Pure, so the selection rule is tested without touching the network.
    """
    assets = []
    for dataset in payload.get("datasets", []):
        for asset in dataset.get("assets", []):
            for file in asset.get("files", []):
                if str(file.get("filetype", "")).upper() == CELLXGENE_ASSET_FILETYPE:
                    assets.append({"title": str(dataset.get("title", "")), "url": str(file.get("url", ""))})
    if not assets:
        raise FetchError(f"collection lists no {CELLXGENE_ASSET_FILETYPE} asset")
    wanted = selector.casefold().strip()
    matches = [asset for asset in assets if wanted and wanted in asset["title"].casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(assets) == 1:
        return assets[0]
    listed = ", ".join(f"{asset['title']!r}" for asset in assets)
    raise FetchError(f"asset selector {selector!r} matched {len(matches)} of {len(assets)} assets; candidates: {listed}")


def _resume_is_complete(response, offset: int) -> bool:
    """Whether a ``416`` means "the part already holds the whole object" rather than an error.

    A resume asks for ``bytes=<part size>-``. Ask for a range that begins at the object's
    own length and the server answers ``416`` with ``Content-Range: bytes */<total>`` —
    which is a *finished* download, not a failure (seen for real on a 45 GB atlas whose
    pull process died after the last byte landed). The caller still hashes what is on
    disk and pins it exactly as it would have after a completed stream.
    """
    if response.status_code != 416:
        return False
    total = response.headers.get("Content-Range", "").rpartition("/")[2]
    return total.isdigit() and offset >= int(total)


def _http_stream(url: str, target: Path, *, offset: int = 0) -> None:
    """Default opener: stream ``url`` into ``target``, appending from ``offset`` when resumed."""
    import requests

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as response:
            if offset and _resume_is_complete(response, offset):
                return
            response.raise_for_status()
            resumed = offset > 0 and response.status_code == 206
            if offset and not resumed:
                offset = 0          # the server ignored the range: restart rather than corrupt
            mode = "ab" if resumed else "wb"
            with target.open(mode) as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(block)
    except (OSError, ValueError) as error:   # requests' tree hangs off OSError; DNS/socket too
        raise FetchError(f"{url}: download failed ({type(error).__name__}: {error})") from None


def _http_json(url: str) -> dict[str, Any]:
    import requests

    try:
        response = requests.get(url, timeout=60, allow_redirects=True)
        response.raise_for_status()
        return response.json()
    except (OSError, ValueError) as error:   # ValueError covers a non-JSON body
        raise FetchError(f"{url}: cannot read collection metadata ({type(error).__name__}: {error})") from None


def tree_digest(path: Path) -> str:
    """Digest of a directory: sha256 over its sorted ``relative-path:digest`` lines.

    A 10x bundle is a directory, so ``file_digest`` cannot pin it. This is the
    same idea one level up, and the same discipline: the cache refuses to accept a
    directory whose contents changed.
    """
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(f"{item.relative_to(path)}:{file_digest(item)}\n".encode())
    return digest.hexdigest()


def _digest_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _cached_digest(path: Path) -> str | None:
    """The digest this cache entry is expected to have: its sidecar, or None."""
    sidecar = _digest_sidecar(path)
    if not sidecar.exists():
        return None
    recorded = sidecar.read_text(encoding=ENCODING).split()[0] if sidecar.read_text(encoding=ENCODING).strip() else None
    return recorded


def _check_cached(path: Path, expected: str | None) -> bool:
    """True when a cached file is usable: it matches ``expected`` (or its own sidecar)."""
    if not path.exists():
        return False
    known = expected or _cached_digest(path)
    if known is None:
        return True                 # unpinned and never verified: taking it is the documented risk
    return file_digest(path) == known


def _download(entry: DatasetEntry, file: PinnedFile, target_dir: Path, expected: str | None, opener: Opener | None) -> Path:
    path = target_dir / file.name
    url = file.url
    if file.kind == "collection":
        url = select_asset(_http_json(url), file.selector or "")["url"]
    elif file.kind == "page":
        raise FetchError(
            f"{entry.id}: the verified access path for {file.name!r} is a landing page, not a file URL "
            f"({file.url}). Download it there and place it at {path}, then re-run."
        )
    part = path.with_name(path.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    (opener or _http_stream)(url, part, offset=offset)
    digest = file_digest(part)
    if expected and digest != expected:
        part.unlink(missing_ok=True)
        raise FetchError(f"{entry.id}: {file.name} digest mismatch (expected {expected[:12]}…, got {digest[:12]}…) from {url}")
    os.replace(part, path)
    if expected is None and entry.verify_on_fetch:
        _digest_sidecar(path).write_text(digest + "\n", encoding=ENCODING)
    return path


def fetch(
    dataset_id: str,
    dest: str | Path,
    *,
    checksum: str | None = None,
    registry: DatasetRegistry | None = None,
    opener: Opener | None = None,
) -> Path:
    """Fetch every access path of ``dataset_id`` into ``dest``; return the dataset directory.

    ``checksum`` overrides the digest expected for the entry's *primary* file (the
    one the format keys on), which is how a corpus pins a dataset the registry has
    not pinned yet. Files already in the cache are verified and left alone.
    """
    registry = registry if registry is not None else _default_registry()
    entry = registry.get(dataset_id)
    target_dir = Path(dest) / entry.id
    target_dir.mkdir(parents=True, exist_ok=True)
    for index, file in enumerate(entry.access):
        expected = checksum if (checksum and index == 0) else file.sha256
        path = target_dir / file.name
        if _check_cached(path, expected):
            continue
        if path.exists():
            path.unlink()           # cached but unverifiable/stale: replace it
        _download(entry, file, target_dir, expected, opener)
    return target_dir


def stage(dataset_id: str, dest: str | Path, source: str | Path, *, registry: DatasetRegistry | None = None) -> Path:
    """Copy hand-downloaded files into the cache and pin them (for ``kind: "page"`` datasets).

    The operator downloads from the verified landing page, then this records the
    digest the cache must keep matching — the same discipline as a fetch, without
    pretending we know a file URL the inventory did not verify.
    """
    registry = registry if registry is not None else _default_registry()
    entry = registry.get(dataset_id)
    target_dir = Path(dest) / entry.id
    target_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    for file in entry.access:
        candidate = source / file.name if source.is_dir() else source
        if not candidate.exists():
            raise FetchError(f"{entry.id}: {candidate} is missing (expected {file.name})")
        path = target_dir / file.name
        if candidate.is_dir():
            shutil.copytree(candidate, path, dirs_exist_ok=True)
            digest = tree_digest(path)
        else:
            shutil.copyfile(candidate, path)
            digest = file_digest(path)
        if file.sha256 and digest != file.sha256:
            path.unlink(missing_ok=True)
            raise FetchError(f"{entry.id}: {file.name} digest mismatch (expected {file.sha256[:12]}…, got {digest[:12]}…)")
        if entry.verify_on_fetch:
            _digest_sidecar(path).write_text(digest + "\n", encoding=ENCODING)
    return target_dir


def _default_registry() -> DatasetRegistry:
    from . import registry as registry_module

    return registry_module.load()
