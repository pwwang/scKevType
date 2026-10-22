"""Central path resolution for scKevType.

Every path the package touches is resolved here from the environment, so no
module hard-codes an absolute path. The three variables below are set by
``scripts/wsl/env.sh``; when unset, everything falls back to the in-repo
layout (which is what the CPU-only test suite uses).

    SCKEVTYPE_ROOT       source tree root          (default: the repo root)
    SCKEVTYPE_DATA       dataset cache (Linux fs)  (default: <root>/data)
    SCKEVTYPE_ARTIFACTS  frozen artifacts           (default: <root>/artifacts)

Resolvers are functions, not constants: tests and the CLIs may set a variable
in-process (``monkeypatch.setenv``) and expect it to take effect.
"""

from __future__ import annotations

import os
from pathlib import Path

# sckevtype/contracts/paths.py -> sckevtype/ -> repo root
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent


def root() -> Path:
    """Source-tree root (``SCKEVTYPE_ROOT`` or the checkout containing this file)."""
    return Path(os.environ.get("SCKEVTYPE_ROOT") or _REPO_ROOT)


def data_dir() -> Path:
    """Dataset cache root (``SCKEVTYPE_DATA`` or ``<root>/data``)."""
    return Path(os.environ.get("SCKEVTYPE_DATA") or root() / "data")


def artifacts_dir() -> Path:
    """Root of the frozen artifacts (``SCKEVTYPE_ARTIFACTS`` or ``<root>/artifacts``)."""
    return Path(os.environ.get("SCKEVTYPE_ARTIFACTS") or root() / "artifacts")


def runs_dir() -> Path:
    """Run directory root (``SCKEVTYPE_RUNS`` or ``<root>/runs``)."""
    return Path(os.environ.get("SCKEVTYPE_RUNS") or root() / "runs")


def configs_dir() -> Path:
    """Human-editable configuration and seed inputs (``<root>/configs``)."""
    return root() / "configs"


def seed_dir() -> Path:
    """The builder's seed inputs (``<root>/configs/seed``)."""
    return configs_dir() / "seed"


def artifact_version(version: str) -> Path:
    """Path of one artifact version directory, e.g. ``artifacts/v1``."""
    return artifacts_dir() / version


def artifact(version: str, name: str) -> Path:
    """Path of one file inside an artifact version."""
    return artifact_version(version) / name


def manifest_path(version: str) -> Path:
    """Path of an artifact version's manifest."""
    return artifact_version(version) / "manifest.json"
