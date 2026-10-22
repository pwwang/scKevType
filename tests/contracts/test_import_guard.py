"""The contracts layer must import on a CPU-only box, with no model stack present.

Checked in a fresh interpreter: by the time these tests run, the test process
itself may have imported whatever it likes, and only a subprocess proves what
``import sckevtype.contracts`` actually pulls in.
"""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN = ("torch", "transformers", "datasets", "anndata", "sklearn", "jax")

PROGRAM = f"""
import sys
import sckevtype.contracts
import sckevtype.contracts.build
import sckevtype.contracts.definitions
import sckevtype.contracts.fetch
import sckevtype.contracts.freeze
import sckevtype.contracts.ontology
import sckevtype.contracts.paths
import sckevtype.contracts.render
import sckevtype.contracts.schema
import sckevtype.contracts.vocab

forbidden = [name for name in {FORBIDDEN!r} if name in sys.modules]
if forbidden:
    raise SystemExit("imported the model stack: " + ", ".join(sorted(forbidden)))
assert "sckevtype.contracts" in sys.modules
print("clean")
"""


def test_importing_contracts_pulls_in_no_model_stack():
    result = subprocess.run([sys.executable, "-c", PROGRAM], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "clean" in result.stdout


def test_the_contracts_public_surface_is_importable_without_kev_or_torch():
    program = """
import sckevtype.contracts as c
assert callable(c.render_rank_list) and callable(c.verify_manifest) and callable(c.build_vocabulary)
assert c.MAX_OPTIONS == 255
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
