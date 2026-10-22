"""The composition — a risk-controlled triage gate over the two axes.

Three actions, and the reason each exists:

``accept``                the axis-1 gate passed and the axis-2 conformal set is a **singleton**
                          ``{y_hat}``, so there is exactly one candidate the panel can support.
``request_more_markers``  the model's abstention statistic fired: the answer looks like it is not
                          on the panel (or the state does not carry it). The remedy is a wider
                          panel, not a human — this is the action no post-hoc confidence threshold
                          can express, because it names what is missing.
``abstain``               the gate passed but the conformal set has more than one member: the
                          answer is among the supplied candidates and the evidence does not pick
                          one. Route to a human.

**What the composed guarantee is.** Write ``A`` for the accepted set and ``Gamma`` for the
conformal set at level ``alpha``. Acceptance implies ``|Gamma| = 1``, so a wrong auto-annotation
implies ``Y not in Gamma``:

    errors among accepted  <=  miscoverage  =  the event the conformal set exists to bound.

Under the within-stratum exchangeability assumption stated in
:mod:`sckevtype.eval.conformal`, per stratum, ``P(Y not in Gamma) <= alpha``. Therefore

    **P(accepted AND wrong) <= alpha, per stratum**, with no assumption about the model's
    probabilities, its calibration, or the panel's composition.

That is a bound on the **per-request** error rate ``E[#errors]/n`` — every request counted,
including the ones routed to a human. The second gate (:func:`fit`) adds a conformal-risk-control
threshold on the axis-1 statistic at a budget ``beta``, and the accepted set is bounded by the
smaller of the two effects.

**What it is not.** This bound is *not*:
  * a bound on the error rate **among auto-annotated requests** — that is ``alpha / P(A)``, larger
    by the acceptance rate, and it is reported empirically with a cluster bootstrap
    (:func:`risk_curve`) rather than claimed;
  * a **per-request** conditional guarantee (coverage given *this* state, this panel). Barber et
    al. 2021: that is impossible without distributional assumptions. Only the stratum-level
    promise is made;
  * conditional on the answer being in the panel. ``P(Y in C)`` is axis 1, which is a *measured*
    statistic and enters no bound here;
  * valid under **shift**. The assumption is within-stratum exchangeability of calibration and
    test records; a different atlas, platform or tissue breaks it, and the report's nominal-vs-
    empirical coverage pair is where that becomes visible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import assert_fit_partition
from .conformal import StratifiedSets

#: The three actions a triage decision can produce.
ACTIONS = ("accept", "request_more_markers", "abstain")


@dataclass(frozen=True)
class Policy:
    """A fitted triage gate: a conformal level, an abstention threshold, and the axis it came from."""

    alpha: float
    beta: float
    sets: StratifiedSets
    #: The largest ``P(abstain)`` still accepted (``None`` = the axis-1 gate is disabled, which
    #: happens when the calibration partition carries no choice rows with an abstention statistic).
    tau: float | None
    fit_rows: int
    calibration_risk: float | None

    def decide(self, row: Mapping[str, Any]) -> str:
        """The action for one row — axis-1 gate first, then the conformal set."""
        if self.tau is not None and row.get("none_p") is not None and float(row["none_p"]) > self.tau:
            return "request_more_markers"
        if int(self.sets.set_for(row).sum()) == 1:
            return "accept"
        return "abstain"

    def decide_all(self, rows: Sequence[Mapping[str, Any]]) -> list[str]:
        return [self.decide(row) for row in rows]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha, "beta": self.beta, "tau": self.tau, "fit_rows": self.fit_rows,
            "calibration_risk": self.calibration_risk,
            "axis2": self.sets.to_dict(),
            "axis1_gate": "disabled (no calibration row carried an abstention statistic)" if self.tau is None else "p(abstain) <= tau",
            "bound": {"quantity": "P(accepted and wrong)", "value": self.alpha,
                      "scope": "per stratum, per request; assumes within-stratum exchangeability",
                      "not_bounded": ["risk among auto-annotated requests", "per-request conditional coverage",
                                      "P(answer is in the panel) — axis 1 is measured, not bounded",
                                      "behaviour under dataset or platform shift"]},
        }


def fit(calibration: Sequence[Mapping[str, Any]], alpha: float, beta: float = 0.05,
        method: str = "lac") -> Policy:
    """Fit the gate on ``calibration``: the conformal quantiles at ``alpha``, the axis-1 threshold at ``beta``.

    ``tau`` is chosen by **conformal risk control** over the accepted prefix. Order the calibration
    rows by ``P(abstain)`` ascending and walk the prefixes; the loss of a prefix is 1 on a row that
    would be both accepted and wrong, 0 otherwise, and it is non-increasing in the threshold
    because raising the threshold only ever adds rows to the accepted set. Take the largest
    threshold whose corrected empirical loss ``(errors + 1) / (n + 1)`` stays within ``beta``.

    The ``(errors + 1) / (n + 1)`` form is the finite-sample correction that makes the bound hold
    at the stated ``beta`` for a sample of this size rather than only asymptotically
    (Angelopoulos et al., conformal risk control). Note what is being bounded: the loss counts
    **every request**, so the bound is on the per-request error rate — see the module docstring
    for what that does and does not cover.
    """
    assert_fit_partition(calibration, "triage.fit")
    # Every row that carries an axis-1 statistic, which is more than the rows whose question
    # offered an abstain option: the CLI asks the others a second, abstain-bearing variant. The
    # threshold is chosen on all of them — a row is not less informative about "would accepting
    # this have been wrong" for having been asked the option only at serving time.
    rows = [row for row in calibration if row.get("none_p") is not None]
    sets = StratifiedSets.fit(calibration, alpha, method)
    if not rows:
        return Policy(alpha=alpha, beta=beta, sets=sets, tau=None, fit_rows=len(calibration), calibration_risk=None)

    ordered = sorted(rows, key=lambda row: float(row["none_p"]))
    losses = [int(bool(np.asarray(sets.set_for(row)).sum() == 1) and int(row["label"]) != int(np.asarray(row["p"]).argmax()))
              for row in ordered]
    # `errors` is over the whole request population, of which the calibration rows are the sample
    # the threshold is chosen on; the correction below uses that population's size.
    n = len(rows)
    errors, tau, risk = 0, None, None
    for index, loss in enumerate(losses):
        errors += loss
        if (errors + 1) / (n + 1) <= beta:
            tau, risk = float(ordered[index]["none_p"]), errors / n
    return Policy(alpha=alpha, beta=beta, sets=sets, tau=tau, fit_rows=n, calibration_risk=risk)


def risk_curve(policy_rows: Sequence[tuple[Policy, Sequence[Mapping[str, Any]]]]) -> list[dict[str, Any]]:
    """One point per fitted policy: what it accepts, and how wrong it is on the rows it was scored on.

    ``coverage`` is the share of requests auto-annotated; ``risk`` is the error rate **among**
    those, with a 95% cluster-bootstrap interval resampling the original record (kev's unit, so
    sibling questions of one cell move together). Both are empirical: :func:`fit`'s bound is on
    the per-request rate, this is the conditional quantity a reader will ask for, and the report
    gives it an interval rather than a claim.
    """
    points = []
    for policy, rows in policy_rows:
        actions = policy.decide_all(rows)
        accepted = [row for row, action in zip(rows, actions) if action == "accept"]
        wrong = [int(accepted_row["label"]) != int(np.asarray(accepted_row["p"]).argmax()) for accepted_row in accepted]
        points.append({
            "alpha": policy.alpha, "beta": policy.beta, "tau": policy.tau,
            "n": len(rows), "n_accept": len(accepted),
            "n_request_more_markers": actions.count("request_more_markers"),
            "n_abstain": actions.count("abstain"),
            "coverage": len(accepted) / len(rows) if rows else None,
            "risk": float(np.mean(wrong)) if wrong else None,
            "risk_ci95": _cluster_bootstrap(wrong, [row["group"] for row in accepted]) if accepted else None,
        })
    return points


def _cluster_bootstrap(wrong: Sequence[int], groups: Sequence[str], samples: int = 1000, seed: int = 0) -> list[float] | None:
    """95% percentile interval for the error rate, resampling whole original records.

    Sibling questions of one cell are not independent — a cell that is hard is hard in every
    question asked about it — so the resampling unit is the record, not the question. Resampling
    questions would produce an interval that is too narrow in exactly the direction that flatters
    the method.
    """
    if len(wrong) < 2 or len(set(groups)) < 2:
        return None
    by_group: dict[str, list[int]] = {}
    for value, group in zip(wrong, groups):
        by_group.setdefault(group, []).append(value)
    names = sorted(by_group)
    stacked = [np.asarray(by_group[name], dtype=float) for name in names]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        picked = rng.integers(0, len(names), size=len(names))
        pooled = np.concatenate([stacked[index] for index in picked])
        draws.append(pooled.mean())
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def sweep(calibration: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], *,
          alphas: Sequence[float], betas: Sequence[float], method: str = "lac") -> dict[str, Any]:
    """The risk–coverage figure's data: fit at every ``(alpha, beta)`` on calibration, score on ``rows``.

    ``alphas`` and ``betas`` are the grid; they are hyperparameters stated in advance, not fitted
    on ``rows``, which is why the whole grid can be reported against the same held-out partition.
    Every point carries the same split: fitted on calibration, scored on ``rows``.
    """
    assert_fit_partition(calibration, "triage.sweep")
    points: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for alpha in alphas:
        for beta in betas:
            try:
                policy = fit(calibration, alpha, beta, method)
            except ValueError as error:      # a stratum too thin to carry this alpha: reported, not hidden
                failed.append({"alpha": alpha, "beta": beta, "reason": str(error)})
                continue
            points.extend(risk_curve([(policy, rows)]))
    summary = {
        "n_points": len(points),
        "alphas": [float(value) for value in alphas],
        "betas": [float(value) for value in betas],
        "method": method,
        "unfittable": failed,
        "fitted_on": "calibration",
        "scored_on": sorted({str(row.get("split")) for row in rows}),
        "no_point_is_fitted_on_the_scored_rows": True,
    }
    if points:
        summary["risk_at_50pct_coverage"] = _risk_at_coverage(points, 0.5)
        summary["max_coverage_within_5pct_risk"] = _max_coverage_within_risk(points, 0.05)
    return {"points": points, **summary}


def _risk_at_coverage(points: Sequence[Mapping[str, Any]], target: float) -> float | None:
    """The lowest empirical risk among points that auto-annotate at least ``target`` of the requests."""
    eligible = [point["risk"] for point in points if (point["coverage"] or 0.0) >= target and point["risk"] is not None]
    return min(eligible) if eligible else None


def _max_coverage_within_risk(points: Sequence[Mapping[str, Any]], budget: float) -> float | None:
    """The largest share auto-annotated while the empirical risk among them stays within ``budget``."""
    eligible = [point["coverage"] for point in points if point["risk"] is not None and point["risk"] <= budget]
    return max(eligible) if eligible else None


def summary(policy: Policy, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The headline block for one fitted policy on one partition: actions, coverage, risk, composition."""
    actions = policy.decide_all(rows)
    accepted = [row for row, action in zip(rows, actions) if action == "accept"]
    wrong = [int(row["label"]) != int(np.asarray(row["p"]).argmax()) for row in accepted]
    strata = _action_strata(rows, actions)
    return {
        "policy": policy.to_dict(),
        "partition": sorted({str(row.get("split")) for row in rows}),
        "n": len(rows),
        "n_accept": len(accepted),
        "n_request_more_markers": actions.count("request_more_markers"),
        "n_abstain": actions.count("abstain"),
        "coverage": len(accepted) / len(rows) if rows else None,
        "risk": float(np.mean(wrong)) if wrong else None,
        "risk_ci95": _cluster_bootstrap(wrong, [row["group"] for row in accepted]) if accepted else None,
        "per_stratum": strata,
        "empirical_minus_bound": (float(np.mean(wrong)) - policy.alpha) if wrong else None,
    }


def _action_strata(rows: Sequence[Mapping[str, Any]], actions: Sequence[str]) -> dict[str, Any]:
    """Accepted/error counts per family, so a stratum-wide average cannot hide one bad family."""
    per_family: dict[str, dict[str, int]] = {}
    for row, action in zip(rows, actions):
        entry = per_family.setdefault(str(row.get("family", "unknown")), {"n": 0, "accept": 0, "request_more_markers": 0, "abstain": 0, "errors": 0})
        entry["n"] += 1
        entry[action] += 1
        if action == "accept" and int(row["label"]) != int(np.asarray(row["p"]).argmax()):
            entry["errors"] += 1
    return per_family


def risk_bound_holds(policy: Policy, rows: Sequence[Mapping[str, Any]], tolerance: float = 0.0) -> bool:
    """Whether the per-request error rate on ``rows`` stayed inside the bound ``alpha``.

    The bound is on ``P(accepted AND wrong)`` over all requests, not on the error rate among the
    accepted ones — so this is the check that matches the claim. It is a single-sample check of a
    probabilistic statement: passing does not prove the bound, and a corpus small enough to be
    cheap is a corpus where one error moves the number by more than ``alpha``.
    """
    if not math.isfinite(policy.alpha):
        raise ValueError("alpha must be finite")
    if not rows:
        raise ValueError("cannot check a bound on no rows")
    wrong = sum(1 for row, action in zip(rows, policy.decide_all(rows))
                if action == "accept" and int(row["label"]) != int(np.asarray(row["p"]).argmax()))
    return wrong / len(rows) <= policy.alpha + tolerance
