"""Digest discipline: canonical JSON, sha256 pinning, and manifest maintenance.

This module is the *only* place that knows how a byte becomes a digest, and the
only place that writes an artifact. It mirrors ``kev.suite`` where the two
overlap: JSON is UTF-8 with LF and ``indent=2, ensure_ascii=False,
allow_nan=False``, every file's sha256 goes into ``manifest.json``, and a frozen
file is never mutated in place — a change means a new version directory.

The digest helpers live here (rather than in a module of their own) because
every artifact in the package pins itself with them and freeze.py is the
digest-pinning module: paths -> freeze -> render -> schema -> the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ENCODING = "utf-8"  # frozen files are sha256-checked byte for byte; never rely on the locale
MANIFEST = "manifest.json"


# --------------------------------------------------------------------------- digests


def canonical_json(value: Any) -> str:
    """Key-sorted, compact, non-ASCII-preserving JSON: the byte string a digest is taken over.

    Two payloads that differ only in key order hash identically, which is what
    makes manifest merging and vocabulary comparison order-independent.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def payload_digest(value: Any) -> str:
    """sha256 of ``canonical_json(value)`` — the digest recorded *inside* an artifact."""
    return hashlib.sha256(canonical_json(value).encode(ENCODING)).hexdigest()


def text_digest(text: str) -> str:
    """sha256 of a UTF-8 string (e.g. a rendered state)."""
    return hashlib.sha256(text.encode(ENCODING)).hexdigest()


def file_digest(path: str | Path) -> str:
    """sha256 of a file's bytes, streamed so large partitions do not load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    """UTC timestamp for ``created_at``, second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit(directory: str | Path | None = None) -> str | None:
    """HEAD of the checkout, or None when the tree is not a git repository."""
    head = Path(directory or ".") / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding=ENCODING).strip()
    except OSError:
        return None
    match = re.match(r"ref:\s*(\S+)", ref)
    if not match:
        return ref or None  # detached HEAD: the file holds the commit itself
    try:
        return (head.parent / match.group(1)).read_text(encoding=ENCODING).strip()
    except OSError:
        return None


# --------------------------------------------------------------------------- JSON I/O


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding=ENCODING))


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON exactly the way ``kev.suite.write_json`` does (UTF-8, LF, indent=2)."""
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding=ENCODING)


def artifact_bytes(payload: Any) -> bytes:
    """The exact bytes an artifact payload becomes — the same bytes the digest is taken over.

    Exposed so a builder can decide whether writing would *change* a frozen file
    before touching it (``bytes``/``str`` verbatim, everything else as JSON).
    """
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode(ENCODING)
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(ENCODING)


def write_text(path: str | Path, text: str) -> None:
    """Write a UTF-8 text file with LF line endings, atomically."""
    _atomic_write(Path(path), text.encode(ENCODING))


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a sibling temp file + ``os.replace``, so a reader never sees a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- artifacts


def read_manifest(directory: str | Path) -> dict:
    """The manifest of an artifact version directory."""
    return read_json(Path(directory) / MANIFEST)


def update_manifest(manifest_path: str | Path, **fields: Any) -> dict:
    """Set top-level manifest fields (name, version, seed, source_digests, ...) and rewrite it.

    Existing keys survive, and ``created_at`` is only stamped on first creation:
    rebuilding identical artifacts leaves the manifest byte-stable.
    """
    path = Path(manifest_path)
    manifest = read_json(path) if path.exists() else {"files": {}}
    manifest.update({key: value for key, value in fields.items() if value is not None})
    manifest.setdefault("created_at", now())
    _atomic_write(path, (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode(ENCODING))
    return manifest


def write_artifact(
    directory: str | Path,
    name: str,
    payload: Any,
    manifest_path: str | Path | None = None,
    kind: str = "data",
    counts: int | None = None,
) -> str:
    """Write one artifact file and merge its sha256 into the manifest. Returns the digest.

    ``payload`` is JSON-serialised unless it is already ``str`` or ``bytes``
    (then it is written verbatim). ``manifest_path`` defaults to
    ``<directory>/manifest.json``. The merge is key-sorted, so two artifacts
    written in either order produce the same manifest.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    _atomic_write(path, artifact_bytes(payload))
    digest = file_digest(path)
    manifest_path = Path(manifest_path) if manifest_path else directory / MANIFEST
    manifest = read_json(manifest_path) if manifest_path.exists() else {"files": {}}
    entry = {"sha256": digest, "bytes": path.stat().st_size, "kind": kind}
    if counts is not None:
        entry["counts"] = counts
    manifest.setdefault("files", {})[name] = entry
    manifest.setdefault("created_at", now())   # stamped once, on the first write
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode(ENCODING))
    return digest


def verify_manifest(directory: str | Path) -> list[str]:
    """Re-hash every listed file. Returns a list of problems; empty means the artifacts are intact.

    A missing manifest, a missing file and a digest mismatch are all reported.
    Unlisted files (the manifest itself, editor scratch) are ignored.
    """
    directory = Path(directory)
    path = directory / MANIFEST
    if not path.exists():
        return [f"missing manifest: {path}"]
    manifest = read_json(path)
    problems: list[str] = []
    for name, entry in sorted(manifest.get("files", {}).items()):
        target = directory / name
        expected = entry["sha256"] if isinstance(entry, dict) else entry
        if not target.exists():
            problems.append(f"missing file: {name}")
        elif file_digest(target) != expected:
            problems.append(f"digest mismatch: {name} (manifest {expected[:12]}…, actual {file_digest(target)[:12]}…)")
    return problems


def new_version(directory: str | Path, src_version: str, new: str | None = None) -> Path:
    """Copy a frozen version forward (``v1`` -> ``v2``) and re-stamp its manifest.

    Files are copied byte for byte, so the copied version still verifies: the
    digests are the same content, under a new version label. Never edit a frozen
    version in place — this is how a constant changes.
    """
    root = Path(directory)
    source = root / src_version
    if not source.is_dir():
        raise FileNotFoundError(f"no such artifact version: {source}")
    match = re.fullmatch(r"v(\d+)", src_version)
    target_name = new or (f"v{int(match.group(1)) + 1}" if match else f"{src_version}_next")
    target = root / target_name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    shutil.copytree(source, target)
    update_manifest(target / MANIFEST, version=target_name, created_at=now(), previous_version=src_version)
    return target


def main(argv: list[str] | None = None) -> int:
    """``python -m sckevtype.contracts.freeze --verify artifacts/v1`` (exit 0 when intact)."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify (or branch) a frozen artifact version.")
    parser.add_argument("directory", nargs="?", help="artifact version directory, e.g. artifacts/v1")
    parser.add_argument("--verify", action="store_true", help="re-hash every listed file (default action)")
    parser.add_argument("--new-version", metavar="NAME", help="copy a version forward, e.g. --new-version v2")
    args = parser.parse_args(argv)

    if args.new_version:
        if not args.directory:
            parser.error("--new-version needs a version directory to read, e.g. artifacts/v1")
        target = new_version(Path(args.directory).parent, Path(args.directory).name, args.new_version)
        print(f"copied {args.directory} -> {target}", flush=True)
        return 0

    if not args.directory:
        parser.error("give an artifact version directory, e.g. artifacts/v1")
    problems = verify_manifest(args.directory)
    for problem in problems:
        print(problem, flush=True)
    print(f"{args.directory}: {'OK' if not problems else f'{len(problems)} problem(s)'}", flush=True)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
