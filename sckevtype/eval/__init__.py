"""Step 4 — the two-axis uncertainty layer: calibration, conformal sets, triage, metrics.

The claim under test is that two uncertainties a cell-type annotator faces are *different*
and separately measurable:

**Axis 1 — panel adequacy.** Is the true type inside the panel the user supplied? The model
carries a trained abstention answer, so this is a *detection* problem over its ``P(abstain)``
statistic, reported as an AUROC and as a detection rate at a target FPR
(:mod:`sckevtype.eval.metrics`). It is a **measured statistic and carries no guarantee** — the
probability that the answer is in the panel is not something this package bounds.

**Axis 2 — label uncertainty.** Among the supplied candidates, which one? A Mondrian
(panel-stratified) conformal set with coverage ``>= 1 - alpha`` per stratum
(:mod:`sckevtype.eval.conformal`). Its validity is distribution-free **given within-stratum
exchangeability** of the calibration and test records. That is an assumption this package
states rather than tests: every nominal target is reported beside its empirical coverage so a
violation is visible rather than hidden.

The composed policy is :mod:`sckevtype.eval.triage` — accept / request more markers / abstain,
parameterised by a target risk so the risk–coverage figure can sweep it. :mod:`sckevtype.eval.run`
is the CLI that scores a suite with a checkpoint and writes the report.

Partition hygiene is the invariant the rest of this package leans on
---------------------------------------------------------------

Every threshold, temperature and conformal quantile is fitted on the ``calibration`` partition
and reported on ``test``. :func:`assert_fit_partition` is called at each fit entry point and
*raises* when it is handed anything else, including a row that does not say which partition it
came from — an unlabelled row is not a licence to fit. The report paths deliberately do not
call it: they are supposed to see ``test`` rows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

#: The partition every fitted quantity in this package comes from.
CALIBRATION = "calibration"

#: The partition every reported number comes from.
TEST = "test"

#: What a stratum is keyed on. ``size`` is the number of options the question offered, which is
#: the panel attribute that survives from calibration to test — see :func:`stratum_chain`. The
#: panel *identity* does not: panels are drawn per cell by the data factory, so no ``panel_id``
#: in the calibration partition ever recurs in test, and a per-panel_id conformal quantile would
#: have no calibration support to be taken from.
SIZE, QTYPE = "size", "type"


class PartitionError(RuntimeError):
    """A fit path was handed a record it is not allowed to fit on."""


def split_of(row: Mapping[str, Any]) -> str:
    """The partition a row came from, or ``""`` when the row does not say."""
    return str(row.get("split") or "")


def assert_fit_partition(rows: Sequence[Mapping[str, Any]], where: str) -> None:
    """Raise unless *every* row is a calibration row.

    Called at the top of every fitting function in this package. The failure it guards is
    silent and flattering: fitting a conformal quantile or a temperature on the test partition
    makes every reported number an in-sample one, and nothing downstream can tell. Rows that
    carry no partition at all are refused too — "it did not say" must not read as "it is fine".
    """
    counts = Counter(split_of(row) or "<no split>" for row in rows)
    offending = {name: n for name, n in sorted(counts.items()) if name != CALIBRATION}
    if offending:
        raise PartitionError(
            f"{where}: refusing to fit on {offending}; every threshold, temperature and conformal "
            f"quantile in sckevtype.eval is fitted on the {CALIBRATION!r} partition only and reported "
            f"on {TEST!r}. Fitting and reporting on the same rows makes every number an in-sample one."
        )
    if not counts:
        raise PartitionError(f"{where}: no rows to fit on")


def stratum_of(row: Mapping[str, Any]) -> tuple[Any, Any]:
    """``(option count, question type)`` — the finest stratum a fitted quantile can be keyed on."""
    return (row[SIZE], row[QTYPE])


def stratum_chain(row: Mapping[str, Any]) -> tuple[tuple[Any, Any], ...]:
    """The strata a row may be covered at, finest first: ``(size, type)`` then ``(type,)``.

    A stratum whose calibration support cannot carry the requested ``1 - alpha`` yields an
    infinite quantile and therefore a set of every option. Coarsening to the marginal
    ``(type,)`` stratum is the documented alternative, and which level a row was actually
    covered at is recorded per row so a reader can see how much of the conditional claim
    survived. It is a coarsening, not a fallback to nothing: the two levels are different
    claims, and the report states the split between them.
    """
    return (stratum_of(row), (row[QTYPE],))


@lru_cache(maxsize=1)
def abstain_tokens() -> tuple[frozenset[str], frozenset[str]]:
    """The abstain option's ``(keys, wordings)``, read from the factory that writes them.

    Duplicating them here would be a second source of truth for the one option the whole of
    axis 1 is about, so they are imported from :mod:`sckevtype.data.families` — the module that
    also decides when an abstain option is offered at all.
    """
    from ..data.families import NONE_OPTIONS

    return frozenset(key for key, _ in NONE_OPTIONS), frozenset(text for _, text in NONE_OPTIONS if text)


def abstain_key(keys: Sequence[str]) -> str | None:
    """The key of the abstain option among ``keys``, or ``None`` when there is not one.

    Matched by key: the factory's abstain keys are not cell-type names and a panel option's key is
    a CL name, so the two cannot collide. How many matched is checked anyway — two matches would
    mean a panel had taken an abstain key, and axis 1 would then be scoring the wrong option.
    """
    known, _ = abstain_tokens()
    found = [key for key in keys if key in known]
    if len(found) > 1:
        raise ValueError(f"{len(found)} options claim to be the abstain option: {found}")
    return found[0] if found else None


def abstain_index(row: Mapping[str, Any]) -> int | None:
    """The index of the abstain option among the row's ``keys``, or ``None`` when it offers none.

    Note what this is *not*: it is a property of the question as asked, so a choice question that
    offers no abstain option has no index even though :mod:`sckevtype.eval.run` asks a second,
    abstain-bearing variant of it to obtain the axis-1 statistic.
    """
    key = abstain_key(row["keys"])
    return None if key is None else list(row["keys"]).index(key)


def rows_of(rows: Iterable[Mapping[str, Any]], **fields: Any) -> list[Mapping[str, Any]]:
    """The rows whose fields equal every given value — ``rows_of(rows, split=TEST, type="choice")``."""
    return [row for row in rows if all(row.get(name) == value for name, value in fields.items())]
