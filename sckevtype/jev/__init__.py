"""The Jev probe: the frontier model of this class, asked our own questions.

`docs/PLAN-audit-steps-by-expected-result.md` E3 asks whether the paper's class claim survives the
frontier member of the class — TypeSafe's Jev, the closed model that Kev is an open reconstruction of
and that this paper's "System One" framing names. Nothing here is on the paper's own path: the probe
reads ``runs/real-v2``, sends a sample of it to a hosted endpoint, and writes its answers back into
the *same* row schema :mod:`sckevtype.eval` scores, so the frontier's numbers and ours are read by
one metric implementation rather than by two.

Three modules, split by what they are allowed to do:

:mod:`~sckevtype.jev.request`
    Render a corpus record into the System One request shape. Pure, offline, no key: it is the piece
    that can be read and checked without spending anything.
:mod:`~sckevtype.jev.client`
    The transport. Budgeted, retrying, and the only module that knows the API key exists.
:mod:`~sckevtype.jev.probe`
    The three-step run — ``sample``, ``ask``, ``evaluate`` — with every raw body saved, so
    interpreting the answers is a local step that costs nothing to repeat.

Imports nothing from the rest of the package at import time, and the rest of the package imports
nothing from here: the probe is a leaf, and deleting the directory would leave the paper's code path
untouched.
"""

from __future__ import annotations

from .client import KEY_ENV, BudgetExceeded, JevClient, JevError
from .request import DEFAULT_BASE_URL, DEFAULT_MODEL, RequestError, record_request

__all__ = [
           "DEFAULT_BASE_URL",
           "DEFAULT_MODEL",
           "KEY_ENV",
           "BudgetExceeded",
           "JevClient",
           "JevError",
           "RequestError",
           "record_request",
]
