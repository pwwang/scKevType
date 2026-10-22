"""Shared fixtures for the uncertainty layer.

Everything here is synthetic and CPU-only: no checkpoint, no GPU, no network. The rows the pure
modules consume are built by :func:`synthetic_rows` rather than loaded from the smoke suite,
because a test of a coverage guarantee needs a population whose distribution is *known* — the
point of the smoke corpus is that it is small and real-shaped, which is the opposite of what a
statistical property test wants. The CLI test that does use the corpus is marked ``slow``.

The rows are i.i.d. draws from one distribution. That is not decoration: the conformal guarantee
is a statement about exchangeable scores, so a test population that mixed two distributions would
test nothing but the tolerance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


def synthetic_rows(n: int, seed: int, split: str, *, n_options: int = 5, informed: float = 0.85,
                   family: str = "panel_choice", with_none: bool = True,
                   group_prefix: str = "g") -> list[dict[str, Any]]:
    """``n`` i.i.d. rows shaped like :mod:`sckevtype.eval.run`'s, with honest probabilities.

    ``informed`` is the share of rows on which the model concentrated on the **labelled** option;
    on the rest it concentrated on a wrong one, so the reported statistics have something to
    measure. At the default sharpness a confident row's set is a singleton and a wrong one's is
    the wrong singleton — which is what makes the triage gate's accept/abstain split non-trivial.

    The label is drawn **uniformly and independently of the probabilities**, so calibration and
    test are exchangeable by construction and the nominal coverage is the right target to compare
    an empirical one against — however good or bad the model's probabilities happen to be.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        label = int(rng.integers(n_options))
        concentration = np.ones(n_options)
        concentration[label if rng.random() < informed else int(rng.integers(n_options))] = 20.0
        p = rng.dirichlet(concentration)
        rows.append({
            "id": f"{split}/{index}", "question": "label", "group": f"{group_prefix}{index // 2}",
            "split": split, "type": "choice", "family": family, "panel": f"panel-{index}", "size": n_options,
            "dataset": "synthetic", "cell": f"cell-{index}", "donor": "d0", "variant": "clean", "source": family,
            "task": family, "keys": [f"option-{i}" for i in range(n_options)], "label": label,
            "p": p.tolist(), "logits": np.log(p).tolist(), "inference_temperature": 1.0,
            "abstain": None, "none_p": float(rng.random()) if with_none else None,
        })
    return rows


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``slow`` unless ``--run-slow`` was passed.

    The flag is read, not declared: ``tests/train/conftest.py`` declares it, and a second
    ``addoption`` for the same name is a collection error — so this file stays out of the way and
    reads it with a default, which is what makes ``pytest tests/eval`` runnable on its own.
    """
    if config.getoption("--run-slow", default=False):
        return
    skip = pytest.mark.skip(reason="slow test: re-run with --run-slow over the whole tests tree")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def make_rows() -> Callable[..., list[dict[str, Any]]]:
    """The row factory, as a fixture — so a test picks its own size, seed and partition."""
    return synthetic_rows


@pytest.fixture
def rows(make_rows: Callable[..., list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A calibration and a test population from the same distribution, 600 rows each."""
    return make_rows(600, 11, "calibration"), make_rows(600, 12, "test")
