"""The training layer must import on a box that has never installed the train extra.

Every heavy dependency sits inside the function that needs it, and the package
``__init__`` re-exports lazily, so ``import sckevtype.train`` costs nothing. This
is checked in a subprocess because the test process itself may already have torch
imported by another test — the question is what *this* import pulls in, not what
is already resident.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_MODULES = [
    "sckevtype.train",
    "sckevtype.train.model",
    "sckevtype.train.suite",
    "sckevtype.train.compute",
]

#: ``yaml`` is a base dependency of this package and costs nothing to import, so it
#: is not on the list — the rule is that a *training* dependency must not be pulled
#: in by importing a module that only describes one.
_HEAVY = ("torch", "transformers", "peft", "kev")


def _import_and_report(modules: list[str]) -> list[str]:
    code = (
        "import json, sys\n"
        f"for name in {modules!r}:\n"
        "    __import__(name)\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("module", _MODULES)
def test_importing_a_train_module_pulls_no_heavy_dependency(module: str) -> None:
    loaded = _import_and_report([module])
    pulled = sorted(set(_HEAVY) & set(loaded))
    assert not pulled, f"import {module} pulled in {pulled}; move that import inside the function that needs it"


def test_the_whole_layer_imports_without_a_heavy_dependency() -> None:
    loaded = _import_and_report(_MODULES)
    pulled = sorted(set(_HEAVY) & set(loaded))
    assert not pulled, f"importing the training layer pulled in {pulled}"
