"""scKevType — definition-conditioned cell-type annotation.

Layers, and the only direction dependencies point:

    contracts -> data -> train -> uncertainty -> eval

``sckevtype.contracts`` holds the frozen artifacts and the canonical state
renderer. It is importable on a CPU-only box: no torch, no transformers.
"""

__version__ = "0.1.0"
