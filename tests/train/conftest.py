"""Shared fixtures for the training layer.

Nothing here needs a GPU, a network or a downloaded weight. The suite builder is
exercised on the Step-2 corpus artifact, and the model guards are exercised on
stand-ins that quack like ``torch.nn.Module`` — so the frozen-backbone check, the
vision-tower check and the provenance schema are all testable on a CPU-only box.

The root conftest skips ``network`` by default but defines ``slow`` and leaves it
to the runner. A training test that quietly reaches for a card is worse than one
that is skipped, so this file applies the same default to ``slow``; pass
``--run-slow`` to run them.

The pinned tokenizer is the one exception: a suite's budget is only real if it
was measured with the tokenizer a run will use, so the test that checks that
asks for it and skips when the box has no copy.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: The Step-2 corpus artifact — the same one the acceptance command builds a suite from.
CORPUS = ROOT / "artifacts" / "corpus" / "smoke"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests marked slow (GPU, or the train extra's dependencies)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow test: re-run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    """The frozen Step-2 smoke corpus, read out of the artifact rather than rebuilt.

    Rebuilding it here would test the data factory again; what this layer has to
    get right is the conversion, so the input is the bytes the factory actually
    froze.
    """
    if not (CORPUS / "manifest.json").exists():
        pytest.skip(f"no corpus at {CORPUS}; build one with `python -m sckevtype.data.build`")
    return CORPUS


@pytest.fixture(scope="session")
def kev_installed() -> None:
    """Skip the kev-contract tests when the ``train`` extra is not installed."""
    pytest.importorskip("kev.suite", reason="the suite contract is kev's; requires the train extra")


class StandInTokenizer:
    """A tokenizer-shaped stand-in: one token per whitespace word, ``decode`` rejoins them.

    The suite builder measures its budget with whatever tokenizer it is handed,
    so the tests that assert *behaviour* (a measured budget, a build that fails,
    a truncation that records itself) hand it this one: the counts are then
    arithmetic a test can predict instead of Qwen's vocabulary, and the tests
    stay offline. The tests that assert the mechanism — that the pinned
    tokenizer is what a real suite is measured with — use the pinned one.
    """

    def __call__(self, text: str, add_special_tokens: bool = False) -> object:
        return SimpleNamespace(input_ids=text.split())

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        """``kev.model.encode`` frames the state and every branch with delimiter tokens.

        The ids themselves are opaque to the measurement — it counts them — so any
        stable integer will do; what matters is that the stand-in can be handed to
        kev's encoder without special-casing it in the code under test.
        """
        return abs(hash(token)) % 1000


@pytest.fixture(scope="session")
def stand_in_tokenizer() -> StandInTokenizer:
    return StandInTokenizer()


@pytest.fixture(scope="session")
def pinned_tokenizer() -> Any:
    """The tokenizer a real suite is measured with, or a skip when it is not on this box.

    ``sckevtype.train.suite.load_pinned_tokenizer`` reads ``configs/models.yaml``
    and loads that entry with kev's own loader. It needs the ``train`` extra and
    the weights' tokenizer in the HuggingFace cache; neither is required to test
    the code that uses it, so a box without them skips instead of failing.
    """
    from sckevtype.train.suite import load_pinned_tokenizer

    try:
        tokenizer, _ = load_pinned_tokenizer()
    except Exception as error:  # noqa: BLE001 - any load failure means "not on this box"
        pytest.skip(f"no pinned tokenizer available ({error}); it measures real suites")
    return tokenizer
