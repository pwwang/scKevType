"""Register v2: the in-scope CL term set, *derived from measured coverage*.

Step 6's rule (docs/SPEC-step6-register-v2.md §2) is the plan's own week-1 gate
(``paper_plan.md``: *"every dataset maps >=95% of cells to CL"*) turned into a
computation rather than a taste judgement:

    the smallest set of CL terms such that every benchmark atlas reaches
    ``target`` of its cells, constrained by "scope >= defined".

The second half of that sentence is the honest one. A term the rule demands but
no external resource can ground **stays out of scope and is counted** — that is
:class:`ScopePlan.undefined`, and it is why :func:`plan_scope` returns a plan
instead of raising: the shortfall is a number this step reports, not a lie it
hides. ``paper_plan``'s gate is a target, and a target that cannot be met is
evidence.

This module is pure. The histograms it plans over are measured elsewhere (once,
by ``scripts/curate_register_v2.py``, which also pins the digest of the bytes it
read) and are committed as ``configs/seed/atlas_labels.v2.json``, so the scope is
re-derivable offline and a test can re-run the rule without the atlases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: ``paper_plan.md`` week 1: "every dataset maps >=95% of cells to CL".
DEFAULT_TARGET = 0.95


class AtlasCounts(BaseModel):
    """One atlas's measured ``CL id -> cells`` histogram, and the bytes it came from.

    ``dataset_digest`` is what makes a coverage number attributable: a histogram
    without the digest of the file it was counted from is a number nobody can
    re-derive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    atlas_id: str = Field(min_length=1)
    dataset_digest: str = ""
    cells: int = Field(ge=1)
    labels: dict[str, int] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> AtlasCounts:
        counted = sum(self.labels.values())
        if counted != self.cells:
            raise ValueError(f"{self.atlas_id}: {counted} labelled cells but {self.cells} cells")
        return self

    def as_mapping(self) -> dict[str, int]:
        return dict(self.labels)


class TermPlan(BaseModel):
    """One term the 95% rule demands, and how much of the benchmark leans on it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cl_id: str
    cells: dict[str, int] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.cells.values())


class AtlasPlan(BaseModel):
    """One atlas, before and after the scope is applied — the row of the table."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atlas_id: str
    cells: int
    distinct_terms: int
    #: The minimal set the target needs, and what it buys (the "before" column is
    #: `required_*`: v1's register, which is what the atlases map against today).
    required_terms: int
    required_cells: int
    #: The scope after the "scope >= defined" constraint, and what it costs.
    in_scope_terms: int
    in_scope_cells: int
    dropped_cells: int
    #: Demanded terms this scope cannot carry because nothing grounds them, and the
    #: cells they hold: the shortfall, attributed to the terms that caused it.
    undefined: tuple[str, ...] = ()
    undefined_cells: int = 0

    @property
    def required_rate(self) -> float:
        return self.required_cells / self.cells if self.cells else 0.0

    @property
    def in_scope_rate(self) -> float:
        return self.in_scope_cells / self.cells if self.cells else 0.0

    @property
    def dropped_rate(self) -> float:
        return self.dropped_cells / self.cells if self.cells else 0.0

    @property
    def undefined_rate(self) -> float:
        """The share of this atlas's cells the *grounding* is responsible for losing."""
        return self.undefined_cells / self.cells if self.cells else 0.0

    def meets(self, target: float = DEFAULT_TARGET) -> bool:
        """Whether this atlas maps `target` of its cells under this scope."""
        return self.in_scope_rate >= target


class ScopePlan(BaseModel):
    """The whole derivation: what the rule demands, what is definable, what falls out."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: float = DEFAULT_TARGET
    #: Union over atlases of the minimal sets that reach `target`.
    demanded: tuple[str, ...]
    #: Terms that stay in scope regardless of the rule (v1's register: two atlases
    #: already map against it, so dropping them would be a regression, not a fix).
    required: tuple[str, ...]
    #: ``(demanded | required) & defined`` — the scope, by construction a subset of
    #: what has a grounded definition. This identity is the §3 coupling.
    in_scope: tuple[str, ...]
    #: Demanded or required terms with no grounded definition: counted, never coerced.
    undefined: tuple[str, ...]
    atlases: tuple[AtlasPlan, ...]

    @property
    def meets_target(self) -> bool:
        return all(atlas.in_scope_rate >= self.target for atlas in self.atlases)

    def payload(self) -> dict:
        return {
            "target": self.target,
            "demanded": list(self.demanded),
            "required": list(self.required),
            "in_scope": list(self.in_scope),
            "undefined": list(self.undefined),
            "atlases": [
                {
                    "atlas_id": atlas.atlas_id,
                    "cells": atlas.cells,
                    "distinct_terms": atlas.distinct_terms,
                    "required_terms": atlas.required_terms,
                    "required_cells": atlas.required_cells,
                    "required_rate": round(atlas.required_rate, 6),
                    "in_scope_terms": atlas.in_scope_terms,
                    "in_scope_cells": atlas.in_scope_cells,
                    "in_scope_rate": round(atlas.in_scope_rate, 6),
                    "dropped_cells": atlas.dropped_cells,
                    "dropped_rate": round(atlas.dropped_rate, 6),
                    "undefined": list(atlas.undefined),
                    "undefined_cells": atlas.undefined_cells,
                    "undefined_rate": round(atlas.undefined_rate, 6),
                }
                for atlas in self.atlases
            ],
        }


def minimal_terms(labels: Mapping[str, int], target: float = DEFAULT_TARGET) -> tuple[str, ...]:
    """The fewest terms whose cells reach ``target`` of the histogram.

    Ties break on the CL id so the set is a function of the histogram alone: a
    greedy that could pick either of two equally-sized terms would make the scope
    depend on the iteration order of a dict.
    """
    total = sum(labels.values())
    if total == 0:
        return ()
    chosen, seen = [], 0
    for cl_id, cells in sorted(labels.items(), key=lambda item: (-item[1], item[0])):
        chosen.append(cl_id)
        seen += cells
        if seen >= target * total:
            break
    return tuple(sorted(chosen))


def plan_scope(
    counts: Sequence[AtlasCounts],
    defined: Sequence[str] | Mapping[str, object],
    *,
    target: float = DEFAULT_TARGET,
    required: Sequence[str] = (),
) -> ScopePlan:
    """Apply the §2 rule to measured histograms, then the §3 constraint.

    ``defined`` is the set of terms with a grounded definition (a definition set
    or a tissue map's keys will do — anything iterable of CL ids).

    The rule runs twice per atlas: once over everything (``demanded``: what the
    target would take if the register could carry any term at all) and once over the
    definable terms (the scope). The two differ exactly where grounding is the
    binding constraint, which ``undefined``/``undefined_cells`` attributes to the
    terms responsible. When the scope still misses the target, the cause is the
    grounding — and that is worth saying out loud rather than averaging away.
    """
    defined_set = set(defined)
    required_set = set(required)
    demanded: set[str] = set()
    in_scope: set[str] = set()
    per_atlas: list[tuple[AtlasCounts, tuple[str, ...], tuple[str, ...]]] = []
    for atlas in counts:
        labels = atlas.as_mapping()
        demand = minimal_terms(labels, target)
        # The same greedy, over the terms that can actually be carried: a term with
        # no grounded definition is not a candidate, and the register compensates
        # with the next-best terms it *can* define rather than stopping short of the
        # target for a reason that is not about the register's reach.
        feasible = minimal_terms({cl: n for cl, n in labels.items() if cl in defined_set}, target)
        demanded |= set(demand)
        in_scope |= set(feasible)
        per_atlas.append((atlas, demand, feasible))
    in_scope |= required_set & defined_set
    plans = []
    for atlas, demand, feasible in per_atlas:
        labels = atlas.as_mapping()
        undefined = tuple(sorted((set(demand) - defined_set) & set(labels)))
        plans.append(
            AtlasPlan(
                atlas_id=atlas.atlas_id,
                cells=atlas.cells,
                distinct_terms=len(labels),
                required_terms=len(required_set & set(labels)),
                required_cells=sum(cells for cl_id, cells in labels.items() if cl_id in required_set),
                in_scope_terms=len(in_scope & set(labels)),
                in_scope_cells=sum(cells for cl_id, cells in labels.items() if cl_id in in_scope),
                dropped_cells=sum(cells for cl_id, cells in labels.items() if cl_id not in in_scope),
                undefined=undefined,
                undefined_cells=sum(labels[cl_id] for cl_id in undefined),
            )
        )
    return ScopePlan(
        target=target,
        demanded=tuple(sorted(demanded)),
        required=tuple(sorted(required_set)),
        in_scope=tuple(sorted(in_scope)),
        undefined=tuple(sorted((demanded | required_set) - defined_set)),
        atlases=tuple(plans),
    )
