"""Axis 2 — panel-stratified (Mondrian) conformal prediction sets.

A conformal set is valid in a precise, narrow sense, and the narrowness is the point:

    **Assumption (within-stratum exchangeability).** For each stratum ``s``, the calibration
    scores and the test score of any request in ``s`` are exchangeable — the calibration
    records are drawn from the same distribution as the record being annotated.

    **Claim.** Under that assumption, and with the finite-sample quantile below,
    ``P(Y in Gamma_alpha(X) | s) >= 1 - alpha`` for every stratum ``s``.

Neither half is checked here. Exchangeability is broken by anything that makes calibration and
test differ — a different atlas, a different platform, a different tissue — and no code can
tell that from the inside. So every nominal target this module produces is reported next to the
empirical coverage it actually got (:func:`coverage`), and :mod:`sckevtype.eval.run` records
both. A stratum whose calibration support is too thin to carry ``1 - alpha`` gets an infinite
quantile and therefore a set of every option: honest, useless, and visible in the set sizes
rather than hidden behind a marginal shortcut.

Two nonconformity scores, because they trade off differently and the paper compares them at
equal coverage:

``lac``  (least-ambiguous-set, a.k.a. threshold-on-score): ``1 - p(y)``. Sets are the options
         above a probability threshold, so they can be large when the model is merely unsure.
``aps``  (adaptive prediction sets): ``p(y)`` plus every probability above it. Sets adapt to how
         spread the mass is rather than to a fixed threshold.

Both are permutation-invariant over the options: a panel whose options are reordered gets the
same set, which the paper's permutation-invariance claim needs and which the rank-tie handling
below is written to preserve.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import CALIBRATION, assert_fit_partition, stratum_chain, stratum_of

#: The nonconformity scores this module implements.
METHODS = ("lac", "aps")


def _alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    return float(alpha)


def _method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown conformal method {method!r}; known: {METHODS}")
    return method


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The finite-sample conformal quantile of ``scores``: the ``ceil((n+1)(1-alpha))``-th smallest.

    The ``+1`` is not a choice. The rank ``ceil((n+1)(1-alpha))`` is what makes the coverage
    claim hold for a finite calibration set instead of only asymptotically; using
    ``np.quantile(scores, 1 - alpha)`` instead is the standard way to be quietly undercovered.
    When that rank exceeds ``n`` the sample cannot support ``1 - alpha`` at all and the quantile
    is ``inf`` — a set of every option, which covers trivially and automates nothing.
    """
    _alpha(alpha)
    ordered = np.sort(np.asarray(list(scores), dtype=float))
    if ordered.size == 0:
        raise ValueError("cannot take a conformal quantile of no scores")
    if not np.isfinite(ordered).all():
        raise ValueError("nonconformity scores must be finite")
    rank = math.ceil((ordered.size + 1) * (1 - alpha))
    return float("inf") if rank > ordered.size else float(ordered[rank - 1])


def nonconformity(probabilities: Sequence[float], label: int, method: str = "lac") -> float:
    """The score of one labelled option under ``method`` — small means "well explained by the model"."""
    _method(method)
    p = np.asarray(probabilities, dtype=float)
    if not 0 <= label < p.size:
        raise ValueError(f"label {label} is not one of {p.size} options")
    if method == "lac":
        return float(1.0 - p[label])
    return float(p[p >= p[label]].sum())


def _aps_scores(p: np.ndarray) -> np.ndarray:
    """The APS score of every option: its probability plus every probability above it.

    Ties get the whole tie group's cumulative mass, so the score of an option does not depend on
    where it happened to land in a tie. Sorting stably and reading the cumulative sum at each
    group's last index is the vectorised form of that.
    """
    order = np.argsort(-p, kind="stable")
    ordered = p[order]
    cumulative = np.cumsum(ordered)
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), ordered.size - 1]
    scores = np.empty_like(p)
    scores[order] = cumulative[ends][np.searchsorted(ends, np.arange(ordered.size), side="left")]
    return scores


def prediction_set(probabilities: Sequence[float], quantile: float, method: str = "lac") -> np.ndarray:
    """The conformal set at ``quantile``: a boolean mask over the options."""
    _method(method)
    p = np.asarray(probabilities, dtype=float)
    if not math.isfinite(quantile):
        return np.ones(p.size, dtype=bool)
    scores = 1.0 - p if method == "lac" else _aps_scores(p)
    return scores <= quantile


@dataclass(frozen=True)
class StratifiedSets:
    """Per-stratum conformal quantiles fitted on calibration, applied to anything else."""

    alpha: float
    method: str
    #: ``stratum -> quantile``, at each level of the chain that had enough calibration support.
    quantiles: dict[tuple[Any, ...], float]
    #: ``stratum -> calibration rows`` behind each fitted quantile.
    support: dict[tuple[Any, ...], int]

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]], alpha: float, method: str = "lac") -> StratifiedSets:
        """Fit one quantile per stratum, on ``calibration`` rows only.

        A stratum's quantile is only fitted when its calibration support can carry ``1 - alpha``
        (``ceil((n+1)(1-alpha)) <= n``); a thinner stratum is left out of ``quantiles`` so that a
        request landing there coarsens along its chain instead of silently receiving an infinite
        quantile it would then be reported as "covered" at.
        """
        _alpha(alpha)
        _method(method)
        assert_fit_partition(rows, "StratifiedSets.fit")

        scores: dict[tuple[Any, ...], list[float]] = {}
        for row in rows:
            label = row["label"]
            score = nonconformity(row["p"], int(label), method)
            for stratum in stratum_chain(row):
                scores.setdefault(stratum, []).append(score)

        quantiles, support = {}, {}
        for stratum, values in scores.items():
            if math.ceil((len(values) + 1) * (1 - alpha)) > len(values):
                continue
            quantiles[stratum] = conformal_quantile(values, alpha)
            support[stratum] = len(values)
        if not quantiles:
            raise ValueError(
                f"no stratum has enough {CALIBRATION} support to carry alpha={alpha}: every fitted "
                f"quantile would be infinite. Strata seen: "
                f"{sorted(((s, len(v)) for s, v in scores.items()), key=lambda item: [str(part) for part in item[0]])}"
            )
        return cls(alpha=alpha, method=method, quantiles=quantiles, support=support)

    def quantile_for(self, row: Mapping[str, Any]) -> tuple[tuple[Any, ...], float]:
        """``(stratum used, quantile)`` for ``row`` — the finest chain level that was fitted."""
        for stratum in stratum_chain(row):
            if stratum in self.quantiles:
                return stratum, self.quantiles[stratum]
        raise KeyError(
            f"no fitted quantile covers {stratum_of(row)} (chain {stratum_chain(row)}); "
            f"fitted strata: {sorted(self.quantiles)}"
        )

    def set_for(self, row: Mapping[str, Any]) -> np.ndarray:
        """The conformal set for ``row`` at the finest stratum it has calibration support at."""
        return prediction_set(row["p"], self.quantile_for(row)[1], self.method)

    def sets(self, rows: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
        return [self.set_for(row) for row in rows]

    def strata_used(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """How many rows were covered at each level of the chain, keyed by its repr.

        This is the honesty column of the stratified claim: rows on the coarser level are covered
        marginally over the question type, not per panel stratum, and the report says how many.
        """
        counts: dict[str, int] = {}
        for row in rows:
            stratum, _ = self.quantile_for(row)
            key = "size+type" if len(stratum) == 2 else "type"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "method": self.method,
            "nominal_coverage": 1.0 - self.alpha,
            # Sorted on the rendered key, not on the stratum tuple: the chain holds `(size, type)`
            # and `(type,)` side by side, and those tuples are not comparable.
            "quantiles": _by_rendered_stratum(self.quantiles),
            "support": _by_rendered_stratum(self.support),
        }


def _by_rendered_stratum(values: Mapping[tuple[Any, ...], Any]) -> dict[str, Any]:
    return {"|".join(str(part) for part in stratum): value
            for stratum, value in sorted(values.items(), key=lambda item: [str(part) for part in item[0]])}


def coverage(rows: Sequence[Mapping[str, Any]], sets: Sequence[Sequence[bool]]) -> dict[str, Any]:
    """Empirical coverage of ``sets`` on ``rows``, per stratum, per family, and marginal.

    Reported beside every nominal target, because the whole value of a stated exchangeability
    assumption is that a reader can see it fail. ``strata`` is keyed by the level the row was
    actually covered at, ``families`` by the record family it came from — the paper's
    panel-adequacy failures and its label-uncertainty failures come from the same records, and
    a marginal number averages them together.
    """
    if len(rows) != len(sets):
        raise ValueError(f"{len(rows)} rows against {len(sets)} sets")
    per_stratum: dict[str, list[int]] = {}
    per_family: dict[str, list[int]] = {}
    covered = 0
    for row, mask in zip(rows, sets):
        inside = int(bool(np.asarray(mask, dtype=bool)[int(row["label"])]))
        covered += inside
        per_stratum.setdefault("|".join(str(part) for part in stratum_of(row)), []).append(inside)
        per_family.setdefault(str(row.get("family", "unknown")), []).append(inside)
    return {
        "n": len(rows),
        "coverage": covered / len(rows) if rows else None,
        "per_stratum": {name: {"n": len(v), "coverage": sum(v) / len(v)} for name, v in sorted(per_stratum.items())},
        "per_family": {name: {"n": len(v), "coverage": sum(v) / len(v)} for name, v in sorted(per_family.items())},
    }


