"""The data factory: public annotated atlases in, kev-loadable supervision out.

This layer is what makes the paper's supervision reproducible: records are derived
by deterministic rules from public atlases, the candidate panel is a *request-time
variable* that is randomised during training, and every stochastic step takes an
explicit seed so that ``(seed, config, dataset digests)`` reproduces the corpus
byte for byte.

    registry  -> what a dataset is (access, format, obs columns, licence)
    fetch     -> the only networked module; caching, resume, digest verification
    adapters  -> cell tables from h5ad / 10x mtx / csv, and cells in CL terms
    panels    -> sample a candidate panel for one cell (size, composition, miss)
    families  -> the six record families of paper_plan.md §3
    augment   -> option permutation, definition perturbation, evidence masking
    splits    -> dataset tiers, donor-grouped splits, sampling caps
    build     -> the CLI that writes kev-compatible suites + a manifest
    qc        -> the probes the paper reports (confound, pairing, coverage)

Dependencies point one way (``data`` imports ``contracts`` and nothing else in
the package); nothing here imports torch, transformers or anndata at module
import time, so the factory runs on a CPU-only box.
"""

from __future__ import annotations

import hashlib
import random

__all__ = ["seed_int", "seeded_rng"]


def seed_int(seed: int, *parts: object) -> int:
    """A stable 64-bit seed derived from ``(seed, *parts)``.

    Every stochastic step keys its RNG this way, so a step's draws depend on
    *what it is drawing for* and not on how many draws happened before it. Same
    idea as ``kev.data.source_seed``: sha256 over the joined key, so a changed
    cell id or family name changes the stream rather than shifting it.
    """
    key = ":".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def seeded_rng(seed: int, *parts: object) -> random.Random:
    """The RNG for one seeded step (see :func:`seed_int`)."""
    return random.Random(seed_int(seed, *parts))
