"""Panel sampling: the request-time variable, drawn deterministically per cell.

The method conditions on a candidate panel that is *not* fixed: at training time
its size, its composition and whether it even contains the true type are drawn
per cell, and at serving time the middleware (or the user) supplies one. This
module is that draw. Three properties make it usable as supervision:

* **deterministic** — everything comes from the ``rng`` the caller seeds with
  ``(seed, cl_id, cell_id)``, so a cell's panel is the same on every rebuild;
* **order-free identity** — option order is shuffled, and ``PanelSpec.panel_id``
  is invariant to it (asserted here, because the training pipeline permutes
  options constantly and a panel whose identity depended on order would make two
  identical requests look like two different panels);
* **honest misses** — with probability ``panel_miss_p`` the true type is *not* on
  the panel, and then the paired correct-panel sibling gives the same cell a
  panel where it is. Without the sibling, "is the true type on the panel?" would
  be answerable from the option list alone (design-notes §2).
"""

from __future__ import annotations

import math
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.definitions import DefinitionSet
from ..contracts.ontology import TissueMap
from ..contracts.schema import PanelSpec, panel_id

#: Largest panel the serving API accepts (kev.api.MAX_OPTIONS); the sampler never
#: exceeds it even when a tissue lists more types than that.
MAX_OPTIONS = 255
#: The scope name for "the whole register" — what a cell whose tissue the register
#: does not map is drawn against. Not a sixth tissue list: it is the register.
GLOBAL_TISSUE = "global"


class PanelConfig(BaseModel):
    """How panels are drawn. Frozen, digested with the corpus config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    size_min: int = Field(default=4, ge=2)
    size_max: int = Field(default=40, ge=2)
    #: ``loguniform`` draws small panels far more often than large ones — a
    #: log-uniform-ish size distribution is what makes the size strata usable.
    size_dist: Literal["loguniform", "uniform"] = "loguniform"
    #: Probability that the true type is genuinely absent from the panel.
    panel_miss_p: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Fraction of the non-true options drawn from the same lineage (the rest are
    #: random distractors from the tissue's type list).
    confusion_frac: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Literal["user", "middleware", "random"] = "random"
    #: Whether a panel option's *definition text* carries the register's marker
    #: lists (``… . positive markers: A, B. negative markers: C.``). True is the
    #: register's own rendering, and what every corpus up to and including
    #: ``real-v2`` was built with. False renders the prose alone: the lists stay
    #: on the option as ``markers_pos``/``markers_neg``, so the holds rule, the
    #: leakage guard and every derived target are unchanged — only the prompt a
    #: model reads loses the gene list that a state's top genes could match.
    option_markers: bool = True
    #: ``per_cell`` — a fresh panel for every cell (Step 1's sampler, what a v1
    #: corpus was built with). ``variants`` — a panel drawn from the finite
    #: vocabulary below, so the *same* panel recurs across cells and a
    #: panel-stratified conformal quantile has calibration support (Step 6 §5).
    vocabulary: Literal["per_cell", "variants"] = "per_cell"
    #: How many variants of each kind the vocabulary declares, per tissue: replace
    #: the register-minus-k and register-plus-m distractors of the spec.
    variant_minus: tuple[int, ...] = (1, 2, 4, 8)
    variant_plus: tuple[int, ...] = (2, 4, 8)
    #: tissue -> candidate CL ids (the middleware's key, from the tissue map).
    tissues: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    #: Every type the register can name. This is what a cell whose tissue is not
    #: one of ``tissues`` is drawn against; empty means "as many as this config
    #: declares", i.e. the union of the tissue lists.
    register_terms: tuple[str, ...] = ()
    #: cl_id -> (self, parent, grandparent, …): the lineage used to rank
    #: confusions by how specific the shared ancestor is.
    lineage: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> PanelConfig:
        if self.size_max < self.size_min:
            raise ValueError(f"size_max {self.size_max} < size_min {self.size_min}")
        return self

    @classmethod
    def from_artifacts(cls, tissue_map: TissueMap, cl_terms: dict, **overrides) -> PanelConfig:
        """Build the config from the frozen tissue map and CL snapshot.

        The tissue lists are copied from the tissue map (the middleware spec), and
        the lineage is walked from the CL snapshot's ``parents`` links — self
        first, root last, cycle-safe.
        """
        tissues = {tissue: tuple(cl_ids) for tissue, cl_ids in tissue_map.tissues.items()}
        lineage = {}
        for cl_ids in tissues.values():
            for cl_id in cl_ids:
                lineage.setdefault(cl_id, _lineage(cl_id, cl_terms))
        return cls(register_terms=tuple(sorted(tissue_map.definitions.by_cl_id)), tissues=tissues, lineage=lineage, **overrides)

    def candidates(self, tissue: str) -> tuple[str, ...]:
        """The candidate list of one panel *scope*: a tissue's, or the register."""
        if tissue == GLOBAL_TISSUE:
            return self.register_terms or tuple(sorted({cl_id for cl_ids in self.tissues.values() for cl_id in cl_ids}))
        try:
            return self.tissues[tissue]
        except KeyError:
            raise ValueError(f"no tissue {tissue!r} in the panel config (have: {sorted(self.tissues)})") from None

    def resolve(self, tissue: str | None) -> tuple[str, tuple[str, ...]]:
        """``(scope, candidates)`` for one cell's tissue — the register's own rule.

        A tissue the register maps gets that tissue's list; anything else (a
        tissue it does not map, or a cell whose tissue did not resolve at all)
        gets the whole register. The scope name is what a record records, so
        "drawn from a tissue list" and "drawn from the register" are separable
        after the fact (Step 8 §1).
        """
        scope = tissue if tissue in self.tissues else GLOBAL_TISSUE
        return scope, self.candidates(scope)

    def shared_depth(self, left: str, right: str) -> int:
        """How specific the most specific shared ancestor of two types is.

        Returns the 1-based depth of the shared ancestor in ``left``'s lineage, or
        0 when they share only the root (or nothing). Higher = closer relatives,
        which is what makes a *confusion* of the true type interesting.
        """
        shared = set(self.lineage.get(left, ())) & set(self.lineage.get(right, ()))
        depths = [self.lineage.get(left, ()).index(cl_id) + 1 for cl_id in shared]
        return max(depths) if depths else 0


class PanelVocabulary:
    """The finite set of panels a tissue offers, each under a stable id.

    Per-cell sampling makes every request a panel of its own, so two cells never
    share one — and a quantile conditioned on the panel then has nothing to
    condition on. This is the smallest thing that fixes that: a *declared* set of
    variants keyed by an id that names how the variant was derived, so the id --
    not the draw -- is what a stratifier keys on.

    The three kinds are the spec's, and none is a new fact about biology:
    ``full`` is the tissue map's register, ``minus{k}`` drops its last k (the
    register is ordered, so "last" is reproducible), ``plus{m}`` adds m
    distractors drawn from *other* tissues' registers.
    """

    def __init__(self, cfg: PanelConfig, tissue: str, definitions: DefinitionSet) -> None:
        self.cfg = cfg
        self.definitions = definitions
        register = tuple(dict.fromkeys(cfg.candidates(tissue)))
        distractors = tuple(
            dict.fromkeys(cl_id for other, cl_ids in sorted(cfg.tissues.items()) if other != tissue for cl_id in cl_ids)
        )
        sets: dict[str, tuple[str, ...]] = {f"{tissue}:full": register[:MAX_OPTIONS]}
        for drop in cfg.variant_minus:
            if len(register) - drop >= cfg.size_min:
                sets[f"{tissue}:minus{drop}"] = register[: len(register) - drop]
        for add in cfg.variant_plus:
            extra = tuple(cl_id for cl_id in distractors if cl_id not in register)[:add]
            if extra:
                sets[f"{tissue}:plus{add}"] = (register + extra)[:MAX_OPTIONS]
        # Sorted, so a variant's option *set* does not depend on the register's order.
        self._ids = {variant: tuple(sorted(cl_ids)) for variant, cl_ids in sets.items()}

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._ids))

    def option_set(self, variant_id: str) -> tuple[str, ...]:
        """The CL ids this variant carries — a pure function of the id and the config."""
        try:
            return self._ids[variant_id]
        except KeyError:
            raise ValueError(f"no variant {variant_id!r} (have: {list(self.variant_ids)})") from None

    def panel(self, variant_id: str) -> PanelSpec:
        options = tuple(self.definitions.to_panel_option(cl_id, markers=self.cfg.option_markers) for cl_id in self.option_set(variant_id))
        return PanelSpec(options=options, tissue=variant_id.split(":", 1)[0], source="random")

    def pair(self, cl_id: str, rng: random.Random) -> tuple[PanelSpec, PanelSpec | None]:
        """A variant panel for a cell, plus the correct-panel sibling on a miss.

        Same contract as :func:`sample_panel_pair`, but which panel the cell gets is
        chosen from the vocabulary rather than composed, so a panel id recurs.
        """
        with_true = [variant for variant in self.variant_ids if cl_id in self._ids[variant]]
        without = [variant for variant in self.variant_ids if cl_id not in self._ids[variant]]
        miss = bool(without) and rng.random() < self.cfg.panel_miss_p
        choices = without if miss else with_true
        if not choices:                       # every variant carries the true type
            return self.panel(self.variant_ids[0]), None
        panel = self.panel(rng.choice(choices))
        if not miss:
            return panel, None
        present = sorted({option.cl_id for option in panel.options} | {cl_id})[:MAX_OPTIONS]
        options = tuple(self.definitions.to_panel_option(option, markers=self.cfg.option_markers) for option in _shuffled(present, rng))
        return panel, PanelSpec(options=options, tissue=panel.tissue, source="random")


def _lineage(cl_id: str, cl_terms: dict, *, max_depth: int = 32) -> tuple[str, ...]:
    """``(self, parent, …, root)`` from the CL snapshot's parent links (cycle-safe)."""
    chain, seen, current = [cl_id], {cl_id}, cl_id
    for _ in range(max_depth):
        term = cl_terms.get(current)
        parents = tuple(term.parents) if term else ()
        nxt = next((parent for parent in parents if parent not in seen), None)
        if not nxt:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    return tuple(chain)


def draw_size(cfg: PanelConfig, rng: random.Random) -> int:
    """One panel size from ``cfg.size_dist``, clamped to the configured bounds."""
    low, high = math.log(cfg.size_min), math.log(cfg.size_max)
    if cfg.size_dist == "uniform":
        return rng.randint(cfg.size_min, cfg.size_max)
    # log-uniform, rounded: small panels stay likely, huge ones stay possible
    return max(cfg.size_min, min(cfg.size_max, round(math.exp(rng.uniform(low, high)))))


def sample_panel(cl_id: str, tissue: str, definitions: DefinitionSet, cfg: PanelConfig, rng: random.Random) -> PanelSpec:
    """One panel for ``(cl_id, tissue)`` — see the module docstring.

    The true type is on the panel unless the draw is a miss; a miss is *really* a
    miss (the sampler never quietly puts the true type back, which would make the
    panel-miss family unlearnable noise).
    """
    candidates = cfg.candidates(tissue)
    if cl_id not in candidates:
        raise ValueError(f"{cl_id} is not a candidate type for tissue {tissue!r} (have: {list(candidates)})")
    size = min(draw_size(cfg, rng), len(candidates), MAX_OPTIONS)
    miss = rng.random() < cfg.panel_miss_p
    chosen = _compose(cl_id, candidates, size, cfg, rng, include_true=not miss)
    options = tuple(definitions.to_panel_option(option, markers=cfg.option_markers) for option in _shuffled(chosen, rng))
    spec = PanelSpec(options=options, tissue=tissue, source=cfg.source)
    assert spec.panel_id == panel_id(tuple(reversed(spec.options))), "panel order changed the panel id"
    return spec


def sample_panel_pair(
    cl_id: str, tissue: str, definitions: DefinitionSet, cfg: PanelConfig, rng: random.Random
) -> tuple[PanelSpec, PanelSpec | None]:
    """A panel plus, on a miss, the correct-panel sibling for the same cell.

    Returns ``(panel, None)`` when the true type is on the panel, and
    ``(panel, sibling)`` when it is not — ``sibling`` is that same panel with the
    true option added and the order redrawn. Both carry the same tissue and
    source; ``group_id`` (set by the caller) is what links them as one contrast.
    """
    if cfg.vocabulary == "variants":
        return PanelVocabulary(cfg, tissue, definitions).pair(cl_id, rng)
    panel = sample_panel(cl_id, tissue, definitions, cfg, rng)
    if any(option.cl_id == cl_id for option in panel.options):
        return panel, None
    candidates = cfg.candidates(tissue)
    present = {option.cl_id for option in panel.options} | {cl_id}
    chosen = _compose(cl_id, candidates, len(present), cfg, rng, include_true=True, forced=present)
    options = tuple(definitions.to_panel_option(option, markers=cfg.option_markers) for option in _shuffled(chosen, rng))
    sibling = PanelSpec(options=options, tissue=tissue, source=cfg.source)
    assert sibling.panel_id == panel_id(tuple(reversed(sibling.options))), "panel order changed the sibling panel id"
    return panel, sibling


def sample_miss_panel(scope: str, definitions: DefinitionSet, cfg: PanelConfig, rng: random.Random) -> PanelSpec:
    """A panel for a cell whose true type the register cannot name.

    The true type is not one of the register's, so it cannot be a candidate: the
    drawn panel is a genuine miss *by construction*, and there is no sibling to
    pair it with — no panel of register types contains the type, so the
    correct-panel contrast does not exist for it. Same draw, same scope name and
    same option surface as :func:`sample_panel`.
    """
    candidates = cfg.candidates(scope)
    if not candidates:
        raise ValueError(f"scope {scope!r} has no candidates — a panel needs at least one option")
    size = min(draw_size(cfg, rng), len(candidates), MAX_OPTIONS)
    chosen = _compose(None, candidates, size, cfg, rng, include_true=False)
    options = tuple(definitions.to_panel_option(cl_id, markers=cfg.option_markers) for cl_id in _shuffled(chosen, rng))
    spec = PanelSpec(options=options, tissue=scope, source=cfg.source)
    assert spec.panel_id == panel_id(tuple(reversed(spec.options))), "panel order changed the panel id"
    return spec


def _shuffled(items, rng: random.Random) -> list:
    order = list(items)
    rng.shuffle(order)
    return order


def _compose(
    cl_id: str | None,
    candidates: tuple[str, ...],
    size: int,
    cfg: PanelConfig,
    rng: random.Random,
    *,
    include_true: bool,
    forced: set[str] | None = None,
) -> list[str]:
    """Pick the option CL ids: lineage-nearest confusions first, then distractors.

    ``cl_id`` is None when the true type is not in the register at all: then no
    candidate can be a confusion of it, so the pool is drawn from unranked.
    """
    already = set(forced or ())
    if include_true:
        already.add(cl_id)
    # The true type is out of the pool either way: when it is included it is already
    # chosen, and when the draw is a miss it must stay out — a "miss" that can put the
    # true type back is not a miss, and the panel-miss family would be unlearnable.
    pool = [candidate for candidate in candidates if candidate != cl_id and candidate not in already]
    confusions = round(cfg.confusion_frac * max(size - len(already), 0))
    ranked = sorted(pool, key=lambda other: (-(cfg.shared_depth(cl_id, other) if cl_id else 0), other))
    chosen = ranked[:confusions]
    rest = [candidate for candidate in pool if candidate not in set(chosen)]
    chosen += rng.sample(rest, min(len(rest), max(size - len(already) - len(chosen), 0)))
    return sorted(already | set(chosen))
