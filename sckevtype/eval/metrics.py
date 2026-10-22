"""The statistics Step 4 reports, as plain numbers.

Scoring is :mod:`kev.metrics`' — the pinned reference implementation of exactly these
quantities (ECE, NLL, Brier, risk–coverage, AURC, coverage at an error budget, temperature
fitting) — reused rather than restated, so a number here means what the same number means in
kev. What this module adds is the three things kev's metric set has no concept of:

* **the partition split**, enforced: the temperature is fitted on calibration
  (:func:`temperature_scaling` calls :func:`sckevtype.eval.assert_fit_partition` first) and
  every reported number is computed on whatever rows it is handed;
* **axis 1 as a detection problem** (:func:`panel_adequacy`) — AUROC and the detection rate at
  a target false-positive rate, over the model's trained abstention statistic;
* **per-family and per-stratum breakdowns** (:func:`breakdown`), because a marginal number is
  exactly what hides the failure this paper is about.

Every function returns JSON-serialisable Python scalars, lists and dicts. Nothing here plots,
imports torch, or needs a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import TEST, abstain_index, assert_fit_partition, stratum_of


def _p(row: Mapping[str, Any], temperature: float = 1.0) -> np.ndarray:
    """The row's probabilities at ``temperature``, restoring raw logits when it recorded any."""
    from kev.metrics import probabilities_at_temperature

    return np.asarray(probabilities_at_temperature(row, temperature), dtype=float)


def _check(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot score an empty population")


def top1_accuracy(rows: Sequence[Mapping[str, Any]], temperature: float = 1.0) -> dict[str, Any]:
    """Share of questions whose highest-probability option is the labelled one.

    Temperature-invariant by construction (softmax is monotone), which is why the temperature
    column of every table can be read without re-reading the accuracy column.
    """
    _check(rows)
    correct = [int(_p(row, temperature).argmax() == int(row["label"])) for row in rows]
    return {"n": len(rows), "accuracy": float(np.mean(correct))}


def ece(rows: Sequence[Mapping[str, Any]], temperature: float = 1.0, bins: int = 10) -> dict[str, Any]:
    """Expected calibration error and mean confidence, with kev's binning."""
    from kev.metrics import ece as kev_ece

    _check(rows)
    confidence = [_p(row, temperature).max() for row in rows]
    correct = [int(_p(row, temperature).argmax() == int(row["label"])) for row in rows]
    return {"n": len(rows), "ece": float(kev_ece(confidence, correct, bins=bins)),
            "mean_confidence": float(np.mean(confidence)), "accuracy": float(np.mean(correct)),
            "bins": int(bins)}


def temperature_scaling(calibration: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One temperature fitted by NLL on **calibration**, and what it does to the **test** rows.

    kev's fitter (a grid search minimising mean NLL over raw logits), kev's definition of which
    rows are eligible. The fit is on calibration and the report is on test; the two are separate
    arguments so that a caller cannot accidentally pass the same rows twice and call the result
    a calibration result. ``before``/``after`` are both on ``test``.
    """
    from kev.metrics import fit_temperature

    assert_fit_partition(calibration, "temperature_scaling")
    _check(test)
    missing = [row["id"] for row in calibration if "logits" not in row]
    if missing:
        raise ValueError(f"{len(missing)} calibration rows recorded no logits (e.g. {missing[0]}); a temperature cannot be fitted on probabilities")
    temperature = float(fit_temperature(list(calibration)))
    return {
        "temperature": temperature,
        "fit_partition": "calibration",
        "fit_rows": len(calibration),
        "report_rows": len(test),
        "before": {"nll": _nll(test, 1.0), "ece": ece(test, 1.0)["ece"]},
        "after": {"nll": _nll(test, temperature), "ece": ece(test, temperature)["ece"]},
    }


def _nll(rows: Sequence[Mapping[str, Any]], temperature: float) -> float:
    from kev.metrics import nll_at_temperature

    return float(np.mean([nll_at_temperature(row, temperature) for row in rows]))


def risk_coverage(rows: Sequence[Mapping[str, Any]], temperature: float = 1.0) -> dict[str, Any]:
    """The selective-prediction curve of the top-1 decision, and the summaries of it.

    ``curve`` is kev's (whole confidence groups, so ties move together); ``aurc`` is its
    right-step integral; ``coverage_at_*_error`` is the largest share of decisions that can be
    accepted in descending-confidence order while the error among the accepted stays inside the
    budget. The last one is an **in-sample maximum** over thresholds, not a deployed guarantee —
    :func:`sckevtype.eval.triage.fit` is where a threshold gets a risk argument behind it.
    """
    from kev.metrics import (
        area_under_risk_coverage,
        coverage_at_error,
        risk_coverage_curve,
    )

    _check(rows)
    confidence = np.asarray([_p(row, temperature).max() for row in rows])
    correct = np.asarray([_p(row, temperature).argmax() == int(row["label"]) for row in rows])
    return {
        "n": len(rows),
        "aurc": float(area_under_risk_coverage(confidence, correct)),
        "coverage_at_5pct_error": float(coverage_at_error(confidence, correct, 0.05)),
        "coverage_at_1pct_error": float(coverage_at_error(confidence, correct, 0.01)),
        "curve": risk_coverage_curve(confidence, correct),
    }


def set_sizes(sets: Sequence[Sequence[bool]]) -> dict[str, Any]:
    """Mean, median and maximum conformal set size, and the share of singleton sets.

    Set size is the price of coverage, and the singleton share is the share of requests the
    composed gate can auto-annotate at all — which is why it sits next to the coverage it buys.
    """
    if len(sets) == 0:
        raise ValueError("cannot summarise no sets")
    sizes = np.asarray([int(np.asarray(mask, dtype=bool).sum()) for mask in sets], dtype=float)
    return {"n": int(sizes.size), "mean": float(sizes.mean()), "median": float(np.median(sizes)),
            "max": float(sizes.max()), "singleton_share": float((sizes == 1).mean()),
            "empty_share": float((sizes == 0).mean())}


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank AUROC (Mann–Whitney), with tied scores sharing a mid-rank.

    Written out rather than imported so that the tie rule is visible: half of a tie group's mass
    counts as a hit. Ties are not a corner case here — a model that puts identical probability
    on two options produces them, and an AUROC that broke ties by position would be reading the
    option order.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be equal-length vectors")
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(f"AUROC needs both classes; got {positives} positive and {negatives} negative")
    order = np.argsort(s, kind="stable")
    ordered = s[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), ordered.size - 1]
    starts = np.r_[0, ends[:-1] + 1]
    mid_rank = (starts + ends) / 2.0 + 1.0     # ranks are 1-based
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = mid_rank[np.searchsorted(ends, np.arange(ordered.size), side="left")]
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def detection_at_fpr(scores: Sequence[float], labels: Sequence[bool], fpr: float = 0.05) -> dict[str, Any]:
    """The operating point a practitioner actually asks for: at this false-alarm rate, what do we catch?

    The threshold is the ``1 - fpr`` quantile of the negative scores taken with ``method="higher"``,
    so the *empirical* false-positive rate lands at or below the target rather than around it.
    Both the achieved rate and the token count are reported, because on a small corpus the
    quantile is coarse and the difference between the target and what was achieved is the whole
    reason to distrust the number.
    """
    if not 0.0 < fpr < 1.0:
        raise ValueError(f"target FPR must be in (0, 1); got {fpr}")
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape or s.ndim != 1:
        raise ValueError("scores and labels must be equal-length vectors")
    negatives, positives = s[~y], s[y]
    if negatives.size == 0 or positives.size == 0:
        raise ValueError(f"a detection rate needs both classes; got {positives.size} positive and {negatives.size} negative")
    threshold = float(np.quantile(negatives, 1.0 - fpr, method="higher"))
    flagged = s >= threshold
    return {
        "target_fpr": float(fpr),
        "threshold": threshold,
        "achieved_fpr": float(flagged[~y].mean()),
        "detection_rate": float(flagged[y].mean()),
        "n_positive": int(positives.size),
        "n_negative": int(negatives.size),
    }


#: How a row's abstain option got into the prompt it was served. These are the two conventions
#: that exist; anything else is a row whose convention its producer did not state.
FACTORY = "factory"      #: the data factory wrote the option into the question (build.py's `none=`)
INJECTED = "injected"    #: the evaluation CLI added one the factory had not (run.with_abstain)

#: A class this small makes an AUROC arithmetic rather than evidence: one row moves it by more
#: than the spread it is meant to resolve. Reported as a flag beside the number, not as a refusal —
#: the refusal conditions are the ones :func:`sckevtype.eval.run.plumbing_check` computes.
THIN_CLASS = 25


def _convention(row: Mapping[str, Any]) -> str | None:
    """``FACTORY``, ``INJECTED``, or ``None`` when the row does not say."""
    stated = str(row.get("abstain_source") or "")
    return stated if stated in (FACTORY, INJECTED) else None


def axis1_label(row: Mapping[str, Any]) -> bool:
    """Is the row's own answer the abstain option?

    Read off the question **as asked**: the label against the row's own option keys, so a question
    that offers no abstain option has no positive to detect and its row is definitionally a
    negative however its ``none_p`` was filled in. That is not a defect — there is no positivity in
    a question that never offered the option — but it is why the negative class is a mixture of two
    serving conventions, which is what the stratification below exists to take apart.

    A dumped row need not carry its keys: the scoring path recorded the abstain index it found, and
    ``abstain`` *is* that index (``None`` when the question offered none). The two readings are the
    same number; ``scripts/axis1-recompute.py`` checks the dumps' own ``positive`` column against
    this function rather than trusting it.
    """
    index = row["abstain"] if "abstain" in row else abstain_index(row)
    return index is not None and int(row["label"]) == index


def _arm(rows: Sequence[Mapping[str, Any]], fpr: float) -> dict[str, Any]:
    """One selection of axis-1 rows: its support, its AUROC and its operating point.

    A selection with one class gets ``auroc: None`` and a note instead of a number, because
    :func:`auroc` cannot be computed on it and silently dropping it would read as a zero.
    """
    labels = [axis1_label(row) for row in rows]
    positives = int(sum(labels))
    block: dict[str, Any] = {"n": len(rows), "n_positive": positives, "n_negative": len(rows) - positives}
    if not rows:
        return {**block, "auroc": None, "note": "no rows"}
    scores = [float(row["none_p"]) for row in rows]     # row order, so scores stay paired with labels
    block["mean_score"] = sum(scores) / len(scores)
    block["median_score"] = float(np.median(scores))
    if not 0 < positives < len(rows):
        return {**block, "auroc": None, "note": "n/a (one axis-1 class only)"}
    block.update(detection_at_fpr(scores, labels, fpr), auroc=auroc(scores, labels))
    block["thin"] = min(positives, len(rows) - positives) < THIN_CLASS
    block["note"] = (f"THIN: the smaller class holds {min(positives, len(rows) - positives)} rows, so one row "
                     f"moves this AUROC by more than it resolves" if block["thin"] else None)
    return block


def _by_convention(rows: Sequence[Mapping[str, Any]], fpr: float) -> dict[str, Any]:
    """The two serving conventions side by side, and how far apart they score.

    ``auc_convention`` is the AUROC of ``none_p`` read as "which convention served this row":
    0.5 means the model cannot tell them apart, and anything above that is the share of the
    statistic that is a measurement of prompt shape rather than of panel adequacy.
    """
    arms = {name: [row for row in rows if _convention(row) == name] for name in (FACTORY, INJECTED)}
    table = {name: _arm(group, fpr) for name, group in arms.items()}
    factory, injected = arms[FACTORY], arms[INJECTED]
    gap: dict[str, Any] = {"auc_convention": None, "note": "needs both arms to be non-empty"}
    if factory and injected:
        scores = [float(row["none_p"]) for row in factory + injected]
        gap = {"auc_convention": auroc(scores, [False] * len(factory) + [True] * len(injected)),
               "note": "0.5 = the model cannot tell which convention served the row"}
    return {**table, "gap": gap,
            "why": ("the two conventions are different prompts, so their rows are not comparable; "
                    "the factory-given arm is the one the primary statistic is taken from")}


def _primary(rows: Sequence[Mapping[str, Any]], fpr: float) -> tuple[dict[str, Any] | None, str]:
    """The reported axis-1 number, or ``None`` and the reason it cannot be taken.

    **Factory-given rows only**: a row the CLI injected an abstain option into was served a prompt
    the factory never wrote, and the two conventions are measurably different (see
    :func:`_by_convention`). **Families carrying both classes only**: a family with a single
    axis-1 class contributes to an AUROC solely by being ranked against another family — which is
    the model recognising which family a question came from, not panel adequacy. The n and both
    class counts travel with the number, because that is what makes it readable.
    """
    factory = [row for row in rows if _convention(row) == FACTORY]
    kept, excluded = {}, {}
    for family, group in sorted(_group_by(factory, "family").items()):
        summary = _arm(group, fpr)
        (kept if summary["n_positive"] and summary["n_negative"] else excluded)[str(family)] = group
    if not kept:
        return None, ("no family among the factory-given rows carries both axis-1 classes"
                      if factory else "no row states a factory-given abstain option")
    block = _arm([row for group in kept.values() for row in group], fpr)
    block["selection"] = ("factory-given rows, restricted to the families that carry both axis-1 classes")
    block["families"] = sorted(kept)
    block["excluded_families"] = {name: _arm(group, fpr)["note"] + f" (n={len(group)})" for name, group in excluded.items()}
    block["pooled_over_families"] = len(kept)
    return block, ""


def panel_adequacy(rows: Sequence[Mapping[str, Any]], fpr: float = 0.05) -> dict[str, Any]:
    """Axis 1, stratified by serving convention and by family.

    A choice row is a **positive** when its target is the abstain option — the panel does not
    contain the answer, or the state carries no evidence for it — and a **negative** when its
    target is a real option on the panel (see :func:`axis1_label`). The score is ``none_p``, the
    probability the model put on the abstain option when it was offered. There is no guarantee
    attached: this is the *measured* axis.

    The number this function exists to *not* report is the one pooled over every choice row. Four
    of the five families carry a single axis-1 class by construction, so that pool is largely
    cross-family base-rate separation, and the negative class additionally mixes two serving
    conventions. What is reported instead:

    * ``primary`` — **the** axis-1 statistic: factory-given rows of families that carry both
      classes, with n and both class counts. ``None`` (with ``primary_note``) when no such
      stratum exists, which is the refusal, not a zero.
    * ``per_family`` — every family, one-class families printed ``n/a (one axis-1 class only)``
      rather than omitted: the absences are the finding.
    * ``by_serving_convention`` — factory-given and injected arms side by side, so the gap between
      them stays visible instead of being averaged away.
    * the pooled keys (``auroc``, ``n``, ...) — kept for readers of earlier reports, always
      accompanied by ``mixed_family_pool`` and ``pooled_qualifier``. When the pool mixes
      single-class families the qualifier says so and the number is not panel adequacy.
    """
    without = sum(1 for row in rows if row.get("none_p") is None)
    rows = [row for row in rows if row.get("none_p") is not None]
    if not rows:
        raise ValueError("no rows carry a none_p statistic; axis 1 needs the abstain option to have been offered")

    report: dict[str, Any] = {"score": "p(abstain option)", **_arm(rows, fpr)}
    # Counted, not silent: a choice row whose abstain ask did not encode is absent from every number
    # below, and a statistic computed on whichever rows happened to fit is a sample nobody chose.
    report["n_choice_rows_without_statistic"] = without
    report["n_abstain_targets"] = report["n_positive"]          # the name earlier reports used

    per_family = {}
    for family, group in sorted(_group_by(rows, "family").items()):
        entry = {**_arm(group, fpr),
                 **{name: sum(1 for row in group if _convention(row) == name) for name in (FACTORY, INJECTED)}}
        entry["unstated"] = len(group) - entry[FACTORY] - entry[INJECTED]
        per_family[str(family)] = entry
    report["per_family"] = per_family

    one_class = sorted(name for name, entry in per_family.items() if entry["auroc"] is None)
    report["mixed_family_pool"] = bool(one_class) and len(per_family) > 1
    report["pooled_qualifier"] = (
        f"pooled over {len(per_family)} families, {len(one_class)} of which carry one axis-1 class only "
        f"({', '.join(one_class)}): the two classes are then drawn from different families, so this number is "
        f"in large part cross-family base-rate separation and is NOT a panel-adequacy estimate"
        if report["mixed_family_pool"] else
        "every family pooled here carries both axis-1 classes, so this pool is a within-family measure")

    report["by_serving_convention"] = _by_convention(rows, fpr)
    report["n_unstated_convention"] = sum(1 for row in rows if _convention(row) is None)
    report["primary"], report["primary_note"] = _primary(rows, fpr)
    return report


def breakdown(rows: Sequence[Mapping[str, Any]], key: str, temperature: float = 1.0) -> dict[str, Any]:
    """The same headline numbers per group — ``key`` is ``"family"``, ``"panel"``, ``"dataset"``, ``"size"``.

    A marginal accuracy and a marginal ECE are the two numbers that hide a panel-shaped failure;
    this is how they get taken apart. Panels are grouped by ``panel`` (the drawn panel's digest)
    and by ``size`` (its option count) because panels are drawn per cell and do not recur, so
    "per panel" is only meaningful at the size level.
    """
    if key not in ("family", "panel", "dataset", "size", "donor", "type", "variant"):
        raise ValueError(f"unknown breakdown key {key!r}")
    return {
        str(name): {**top1_accuracy(group, temperature), "ece": ece(group, temperature)["ece"],
                    "mean_confidence": ece(group, temperature)["mean_confidence"]}
        for name, group in sorted(_group_by(rows, key).items(), key=lambda item: str(item[0]))
    }


def _group_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[Any, list[Mapping[str, Any]]]:
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(key), []).append(row)
    return groups


def per_stratum(rows: Sequence[Mapping[str, Any]], temperature: float = 1.0) -> dict[str, Any]:
    """Accuracy and ECE per ``(option count, question type)`` — the strata conformal sets are keyed on."""
    groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(stratum_of(row), []).append(row)
    return {
        "|".join(str(part) for part in stratum): {**top1_accuracy(group, temperature), "ece": ece(group, temperature)["ece"]}
        for stratum, group in sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
    }


def chosen_partition(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many rows came from each partition — recorded in the report so a reader can see it."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("split") or "<no split>")] = counts.get(str(row.get("split") or "<no split>"), 0) + 1
    return counts


def fit_rows_only(rows: Sequence[Mapping[str, Any]], where: str = "metrics") -> Sequence[Mapping[str, Any]]:
    """The guard, exposed for a caller that wants to check its rows before a fit rather than after."""
    assert_fit_partition(rows, where)
    return rows


#: The testing partition's name, re-exported so a report can name it without importing the package.
REPORT_PARTITION = TEST
