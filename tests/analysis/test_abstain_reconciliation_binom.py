"""The exact binomial behind the reconciliations: identical where it ran, a number where it overflowed.

``reports/abstain-reconciliation.py``'s ``binom_two_sided`` is this project's exact two-sided binomial on
discordant pairs — ``reports/PAIRED-TESTS.md``'s test, which ``scripts/noul-accuracy.py``,
``scripts/arms-forced.py`` and ``scripts/9b-finetune-score.py`` lift out of that file's *source* rather
than re-implement (importing the file would rewrite ``reports/abstain-reconciliation.json``). Every
p-value the project has published comes from that function, and the v4 noul family — 63,851 questions
per arm, discordant counts in the tens of thousands — died in it on the float power ``2.0 ** n`` with
``OverflowError: int too large to convert to float``.

Three properties are pinned here, each on the rule **as its consumers lift it**:

* **the arithmetic it already did is untouched** — for every ``(b, c)`` the original expression could
  compute, byte-identical floats, the edge cases (``k = 0``, ``k = n``) and the explicit
  ``min(1.0, ...)`` cap included. The original is kept in this module as :func:`_settled`, verbatim,
  and is the oracle;
* **an ``n`` in the tens of thousands is a value, not an exception** — cross-checked against
  ``scipy.stats.binomtest`` (an independent implementation: the incomplete beta, not a tail sum) and,
  wherever the tail is affordable as integers, against exact integer arithmetic written again **in
  this file**;
* **the lift is real** — the source is executed the way ``arms-forced.py`` executes it (imports plus
  ``binom_two_sided`` *alone*, no module state, no helper) and the way ``noul-accuracy.py`` executes it.
  A ``NameError`` inside a live scoring run is the failure this pair of tests exists for, and the range
  that needs a second name has to refuse by name instead.

A p-value below ``5e-324`` cannot be a float, and a lopsided noul pair does reach that. The magnitude is
carried by ``binom_two_sided_log10``, pinned against ``log10`` of the exact integer ratio, because
``p = 0.0`` in a table is a number nobody can read.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
RULE_SOURCE = ROOT / "analysis" / "abstain-reconciliation.py"

#: What the noul aggregator lifts: the rule and the log-space magnitude its table carries.
NAMES = ("binom_two_sided", "binom_two_sided_log10")

#: Discordant-pair counts at and beyond the noul family's scale: the sizes ``2.0 ** n`` cannot hold.
LARGE = ((20000, 18723), (31700, 31000), (30000, 29999), (2000, 5000), (61000, 2), (62849, 5000))

#: Cheap inputs for the overflow control: the power ``2.0 ** n`` is the whole failure and it does not
#: care about ``k``, so the boundary ``n = 1024`` is one control and the shapes above are three more --
#: all with a small minority count, because the *settled* expression has to build its whole tail before
#: it reaches the power that kills it.
OVERFLOW_CONTROL = ((0, 1024), (20000, 1), (5000, 200), (61000, 2))

#: Sizes above the float power whose exact tail is still affordable: the exact-integer branch's range.
EXACT_TAIL = ((1000, 30), (700, 1400), (2000, 2000), (4096, 4096))


def _lift(names: tuple[str, ...]) -> dict:
    """The rule as its consumers execute it: this file's imports plus the named function bodies.

    A module-level constant, or a second function, that the named ones reached for would be absent from
    the namespace built here — which is exactly what these tests are for.
    """
    tree = ast.parse(RULE_SOURCE.read_text(encoding="utf-8"), filename=str(RULE_SOURCE))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    missing = [name for name in names if name not in definitions]
    assert not missing, f"{RULE_SOURCE} no longer defines {missing}: the rule this report applies moved"
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[*imports, *[definitions[name] for name in names]], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, filename=str(RULE_SOURCE), mode="exec"), namespace)      # noqa: S102
    return namespace


@pytest.fixture(scope="module")
def rule() -> dict:
    return _lift(NAMES)


@pytest.fixture(scope="module")
def rule_alone() -> dict:
    """The single name ``scripts/arms-forced.py`` and ``scripts/9b-finetune-score.py`` ask for."""
    return _lift(("binom_two_sided",))


def _settled(b: int, c: int) -> float:
    """The expression as it stood before the fix, **verbatim**: the oracle for "nothing moved"."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def _tail_exact_here(n: int, k: int) -> int:
    """``sum_{i<=k} C(n, i)`` written again here, so a cross-check is not the code under test."""
    return sum(math.comb(n, i) for i in range(k + 1))


def _log10_exact(n: int, k: int) -> float:
    """``log10`` of the exact ratio, from exact integers: the magnitude oracle for an underflowed ``p``."""
    return math.log10(2 * _tail_exact_here(n, k)) - n * math.log10(2)


def _grid() -> list[tuple[int, int]]:
    """Every ``(b, c)`` with ``n <= 60``, plus a spread of sizes up to the float power's own limit."""
    points = [(b, n - b) for n in range(61) for b in range(n + 1)]
    for n in (100, 199, 300, 511, 512, 700, 1000, 1022, 1023):
        for k in (0, 1, 2, 3, 17, 64, 255, n // 2 - 1, n // 2):
            points += [(k, n - k), (n - k, k)]
    return points


def test_the_rule_is_the_settled_expression_on_every_grid_point(rule):
    """Old == new, **exactly** (never ``approx``): every published p-value stays where it was.

    The grid carries its own controls: it is large enough that a change would land on it, at least one
    point exercises the ``min(1.0, ...)`` cap (so the cap is compared, not bypassed), and at least one
    is strictly below ``1.0`` (so the comparison is not a row of ones agreeing).
    """
    grid = _grid()
    values = {point: rule["binom_two_sided"](*point) for point in grid}
    differing = [(b, c, _settled(b, c), values[(b, c)]) for b, c in grid
                 if _settled(b, c) != values[(b, c)]]

    assert len(grid) > 2000, f"the grid is the control, and {len(grid)} points is too few to be one"
    assert differing == [], f"{len(differing)} of {len(grid)} points moved, first three: {differing[:3]}"
    assert [point for point in grid if values[point] == 1.0], "no point exercises the min(1.0, ...) cap"
    assert [point for point in grid if values[point] < 1.0], "every point saturates: nothing is compared"
    for edge in ((0, 0), (0, 5), (5, 0), (5, 5), (0, 1023), (1023, 0)):
        assert edge in grid, f"{edge} is one of the edge cases this test exists for"
    assert values[(0, 0)] == 1.0, "no discordant pairs is no evidence, and that is settled arithmetic"
    assert values[(5, 5)] == 1.0, "a symmetrical split is the null itself, through the cap"
    assert values[(10, 0)] == values[(0, 10)] == _settled(10, 0), "two-sided by construction"


def test_the_exact_integer_branch_is_the_exact_rule_above_the_float_power(rule):
    """The branch that replaced the ``OverflowError`` for choice-lane pairs is the exact tail, not a copy.

    ``n >= 1024`` with ``min(b, c) <= 4096`` is the range a 2,510-question population reaches, so it is
    checked against the tail built independently here (``math.comb``, in this file) — and the settled
    expression is required to die on each of those inputs, or the range would be untested.
    """
    for b, c in EXACT_TAIL:
        n, k = b + c, min(b, c)
        with pytest.raises(OverflowError, match="too large|out of range"):
            _settled(b, c)
        assert rule["binom_two_sided"](b, c) == min(1.0, 2 * _tail_exact_here(n, k) / (2 ** n)), \
            f"n={n} k={k}: the exact-integer branch is the exact rule"
    assert rule["binom_two_sided"](4096, 4096) == 1.0, \
        "the exact branch's own boundary is a balanced split, and the cap catches it"


def test_the_log_space_magnitude_is_the_settled_arithmetic_where_both_are_computable(rule):
    """``binom_two_sided_log10`` against ``log10`` of the *settled* expression, on the settled grid.

    Below ``n = 1024`` the float rule is the published arithmetic and the log-space rule is not the path
    the value takes, so this is where the two can be compared directly: they agree to ~1e-13 in the log,
    and a different tail would show up as a whole number of log units here, not as noise.
    """
    compared = 0
    for b, c in _grid():
        value = _settled(b, c)
        if not 0.0 < value < 1.0:
            continue
        assert rule["binom_two_sided_log10"](b, c) == pytest.approx(math.log10(value), abs=1e-9), \
            f"({b}, {c}): the log-space rule is not the settled rule"
        compared += 1
    assert compared > 100, f"only {compared} grid points had a representable p-value: too thin"


def test_the_settled_expression_really_overflows_at_these_sizes():
    """The positive control: without this, the cross-checks below could be testing a size that never broke.

    The old expression dies on ``2.0 ** n`` alone, so the control is the boundary (``n = 1024``) plus
    the shapes the cross-check uses, each with a small minority count so that the *settled* code reaches
    the power in reasonable time — it builds its whole tail first, and at ``k`` in the thousands that
    tail is minutes of big-integer arithmetic for a number nobody is going to use.
    """
    for b, c in OVERFLOW_CONTROL:
        # Two spellings of the same failure, and both are the one the run died of: "int too large to
        # convert to float" when the huge *sum* is multiplied by 2.0 (the noul family's case, k in the
        # thousands) and "Numerical result out of range" when the power itself is formed (small k).
        with pytest.raises(OverflowError, match="too large|out of range"):
            _settled(b, c)
    assert _settled(1023, 0) == 2.0 ** -1022, "one step below the boundary is an ordinary small number"


def test_large_n_agrees_with_scipy_binomtest(rule):
    """``scipy.stats.binomtest`` (incomplete beta) against the tail sum, at n in the tens of thousands.

    Scipy computes the two-sided p-value from the binomial CDF through the incomplete beta function —
    a different algorithm from summing ``C(n, i)`` — so agreement to ~1e-9 relative is two independent
    implementations of the same definition. Where scipy's own double underflows (a lopsided pair on
    60,000 questions has a p-value near ``1e-18000``), the case is not skipped: the log-space magnitude
    is required to be below ``-300``, and that magnitude is checked against exact integers below.
    """
    compared, underflowed = [], []
    for b, c in LARGE:
        n, k = b + c, min(b, c)
        mine = rule["binom_two_sided"](b, c)
        theirs = binomtest(k, n, 0.5, alternative="two-sided").pvalue
        if theirs > 0.0:
            assert mine == pytest.approx(theirs, rel=1e-9), f"n={n} k={k}: {mine!r} vs scipy {theirs!r}"
            compared.append((n, k))
        else:
            assert rule["binom_two_sided_log10"](b, c) < -300.0, \
                f"n={n} k={k}: the pair is not representable and not astronomically small either"
            underflowed.append((n, k))

    assert len(compared) >= 3, f"only {compared} were comparable as floats: the cross-check is thin"
    assert underflowed == [(61002, 2), (67849, 5000)], \
        f"the underflow cases changed: {underflowed} — this test's own positive control"


def test_a_p_value_a_float_cannot_hold_is_carried_in_log_space(rule):
    """``p`` underflows to ``0.0``; the magnitude is checked against the exact integer ratio."""
    for b, c in ((20000, 2000), (5000, 200), (61000, 2), (5000, 0)):
        n, k = b + c, min(b, c)
        assert rule["binom_two_sided"](b, c) == 0.0, f"n={n} k={k} is expected to underflow, and does not"
        assert rule["binom_two_sided_log10"](b, c) == pytest.approx(_log10_exact(n, k), abs=1e-9), \
            f"n={n} k={k}: the reported magnitude is not the exact one"


def test_the_two_forms_are_the_same_number_where_the_value_is_representable(rule):
    """``log10`` of the float against the log-space path: one rule, two renderings."""
    points = ((10, 0), (100, 923), (300, 723), (2000, 5000), (20000, 18723), (31700, 31000))
    for b, c in points:
        value = rule["binom_two_sided"](b, c)
        assert 0.0 < value < 1.0, f"({b}, {c}) is not a representable p-value to compare"
        assert rule["binom_two_sided_log10"](b, c) == pytest.approx(math.log10(value), abs=1e-9)

    capped = ((5, 5), (30000, 29999), (0, 0))
    for b, c in capped:
        assert rule["binom_two_sided"](b, c) == 1.0
        assert rule["binom_two_sided_log10"](b, c) == 0.0, \
            f"({b}, {c}): log10(1.0) is 0.0, and the cap must agree in both renderings"


def test_the_rule_runs_as_the_noul_aggregator_lifts_it(rule):
    """Imports and function bodies only, no module-level state -- and all three ranges reachable."""
    assert rule["binom_two_sided"](2, 5) == _settled(2, 5) == 58 / 128, "the settled float branch"
    assert rule["binom_two_sided"](700, 1400) == \
        min(1.0, 2 * _tail_exact_here(2100, 700) / (2 ** 2100)), "the exact-integer branch"
    assert rule["binom_two_sided"](20000, 18723) < 1.0, "the log-space branch"
    assert rule["binom_two_sided_log10"](20000, 18723) < 0.0
    assert rule["binom_two_sided_log10"](0, 0) == 0.0


def test_a_consumer_that_lifts_the_rule_alone_still_runs(rule_alone):
    """``arms-forced.py`` and ``9b-finetune-score.py`` lift ``binom_two_sided`` and nothing else.

    They are live scripts -- one is the re-score half of a cluster job -- and the populations they run on
    are 1,819 and 2,510 questions, so ``min(b, c)`` cannot exceed 1,255: every range they can reach has
    to work with no other name defined. The noul range, which does need ``binom_two_sided_log10``, has to
    refuse **by name** there rather than die of a ``NameError`` in the middle of a run.
    """
    assert "binom_two_sided_log10" not in rule_alone, "the point of this fixture is that it is absent"
    assert rule_alone["binom_two_sided"](2, 5) == 58 / 128, "the settled branch, unchanged"
    assert rule_alone["binom_two_sided"](700, 1400) == \
        min(1.0, 2 * _tail_exact_here(2100, 700) / (2 ** 2100)), \
        "n = 2,100 is a 2,510-question lane's own size, and it is past the float power"
    assert rule_alone["binom_two_sided"](1255, 1255) == 1.0, \
        "the choice lanes' largest possible minority count, at the cap"
    with pytest.raises(RuntimeError, match="lifted binom_two_sided alone"):
        rule_alone["binom_two_sided"](20000, 18723)
