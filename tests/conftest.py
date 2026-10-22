"""Repository-wide pytest configuration.

Network tests exist (``fetch.py`` is the one module that can reach the internet) but
they never run by default: a test run has to be meaningful on a box with no network.
Pass ``--run-network`` to opt in.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-network", action="store_true", default=False, help="run tests marked network (off by default)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="network test: re-run with --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
