"""The six record families of `paper_plan.md` §3 — this module is the Methods section.

Each builder turns one ``CellRecord`` (a cell, its expression, its CL label) plus
the panel that was sampled for it into a kev-loadable :class:`Record`:

===================  ===========================================================
family               question                                  target
===================  ===========================================================
``panel_choice``     which candidate does the state support?    the true option
``marker_noul``      does this definition hold? (per option)    derived per option
``panel_miss``       which candidate, when the true one is gone the abstain option
``evidence_removed`` same, with the true type's markers masked  the abstain option
``evidence_diluted`` same, with k random genes masked           true option, else abstain
``axis_score``       which level of an expression axis?         the level index
===================  ===========================================================

Three rules hold across all six, and they are what makes the corpus defensible:

1. **Targets are derived, never asserted.** ``marker_noul`` and the two evidence
   families read the answer off the state with :class:`HoldsRule` — a definition
   the state still satisfies is a "true", one it does not is a "false", and the
   derivations are recomputable by anyone with the artifact and the seed.
2. **The prompt does not say which family it is.** All choice questions share one
   instruction template; the family lives in ``_meta``. A model that could read
   the family off the wording would be answering a different question than the one
   the paper claims to evaluate.
3. **The leakage guard runs on every definition that enters a record** — the true
   type's and every candidate's — against the dataset's own DE hits
   (:func:`~sckevtype.contracts.definitions.assert_no_leakage`), so a definition
   that was itself derived from this dataset's differential expression cannot
   quietly turn the task into a lookup.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..contracts.definitions import Definition, DefinitionSet, assert_no_leakage
from ..contracts.render import FilterConfig, parse_rank_list, render_rank_list
from ..contracts.schema import (
    ChoiceQuestion,
    NoulQuestion,
    PanelOption,
    PanelSpec,
    Record,
    RecordFamily,
    RecordMeta,
    ScoreQuestion,
)
from .adapters import CellRecord
from .augment import dilute_rendered, mask_rendered

#: The abstain option, mirroring ``kev.data.NONE_OPTIONS``: several wordings, so
#: the *presence* of an option meaning "none of these" (and its phrasing) never
#: becomes the cue for the answer — including for the panel-miss sibling pair,
#: which shares one wording so the pair differs only in whether the true type is
#: on the panel. kev is not importable here (the training stack is an extra), so
#: the list is restated and the tests assert its shape.
NONE_OPTIONS: tuple[tuple[str, str | None], ...] = (
    ("other", "None of the above"),
    ("other", "A reason that fits none of the above"),
    ("none", "None of these"),
    ("other", "Something else"),
    ("not_listed", "Not listed here"),
    ("none_of_the_above", None),
    ("other", "A category that fits none of the above"),
    ("other", "None of the listed options apply"),
    ("unknown", "Cannot be determined from the options given"),
    ("other", "Other"),
    ("none", None),
    ("other", "An answer not covered by the other options"),
    ("no_match", "No option matches"),
)

#: The one instruction template every choice question uses (rule 2 above).
CHOICE_INSTRUCTIONS = (
    "The state is one cell from a single-cell RNA-seq experiment: its expressed genes, "
    "highest first, as a rank list. Each option is a candidate cell type with the definition "
    "it is annotated by. Which candidate does the state support? If the state supports none "
    "of them, choose the option that says so."
)
#: The same template for the marker questions, which ask about one candidate at a time.
NOUL_INSTRUCTIONS = (
    "The state is one cell from a single-cell RNA-seq experiment: its expressed genes, "
    "highest first, as a rank list. Does this cell satisfy the candidate's definition?"
)
NOUL_CRITERIA = {"true": "yes, the state satisfies the definition", "false": "no, it does not"}
AXIS_INSTRUCTIONS = (
    "The state is one cell from a single-cell RNA-seq experiment: its expressed genes, "
    "highest first, as a rank list. Which level of the named expression axis does this cell "
    "sit at? The options are the levels, lowest first."
)

NoneOption = tuple[str, str | None]

#: Why the register cannot name a cell. Checked tissue-first: a cell whose tissue
#: the register does not map is out of register whatever its label says, and that
#: is the more informative of the two facts.
UNREGISTERED_TISSUE = "unregistered_tissue"
UNREGISTERED_TYPE = "unregistered_type"


class HoldsRule(BaseModel):
    """When a definition "holds" of a state: enough positive markers, no negative ones.

    The default (at least one positive marker, no negative marker) is the
    definition-satisfaction rule the paper states; making it configuration rather
    than code means a corpus can tighten it and say so.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_pos: int = Field(default=1, ge=0)
    max_neg: int = Field(default=0, ge=0)

    def holds(self, symbols: Sequence[str] | set[str], definition: Definition) -> bool:
        present = set(symbols)
        positives = sum(1 for marker in definition.pos_markers if marker in present)
        negatives = sum(1 for marker in definition.neg_markers if marker in present)
        return positives >= self.min_pos and negatives <= self.max_neg


class AxisSpec(BaseModel):
    """One expression axis to score a cell on: up-markers, down-markers, level edges.

    ``thresholds`` are the score edges and ``legend`` the level descriptions; the
    level of a cell is the number of edges its ``up - down`` score clears, so the
    mapping from evidence to target is arithmetic and checkable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    up_markers: tuple[str, ...] = Field(min_length=1)
    down_markers: tuple[str, ...] = ()
    thresholds: tuple[int, ...] = ()
    legend: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> AxisSpec:
        if len(self.legend) != len(self.thresholds) + 1:
            raise ValueError(f"axis {self.name!r}: {len(self.legend)} levels need {len(self.legend) - 1} thresholds")
        if list(self.thresholds) != sorted(self.thresholds):
            raise ValueError(f"axis {self.name!r}: thresholds must be ascending")
        return self

    def score(self, symbols: Sequence[str] | set[str]) -> int:
        present = set(symbols)
        return sum(1 for marker in self.up_markers if marker in present) - sum(
            1 for marker in self.down_markers if marker in present
        )

    def level(self, symbols: Sequence[str] | set[str]) -> int:
        score = self.score(symbols)
        return sum(1 for threshold in self.thresholds if score >= threshold)


@dataclass(frozen=True)
class FamilyContext:
    """Everything the builders need that is not the cell or its panel."""

    definitions: DefinitionSet
    filter: FilterConfig
    corpus: str = ""
    holds: HoldsRule = field(default_factory=HoldsRule)
    leakage_alpha: float = 0.0
    #: The dataset's own DE hits, used by the leakage guard only (never a target).
    de_markers: tuple[str, ...] = ()
    #: How many genes ``evidence_diluted`` masks out of the state.
    dilution_k: int = 3

    def definition(self, cl_id: str) -> Definition:
        return self.definitions.get(cl_id)

    def option_definition(self, option: PanelOption) -> Definition:
        """The option as a :class:`Definition` for the leakage guard and the rule.

        ``text`` keeps the option's rendered string: the guard looks for a leaked
        marker *in the text the model reads*, so a definition whose prose names a
        marker its lists omit is still caught. The re-render the guard performs is
        therefore redundant prose — harmless, and conservative in the right
        direction.
        """
        return Definition(
            cl_id=option.cl_id,
            name=option.name,
            text=option.definition,
            pos_markers=option.markers_pos,
            neg_markers=option.markers_neg,
            provenance=option.provenance,
        )

    def guard(self, definition: Definition) -> None:
        assert_no_leakage(definition, self.de_markers, self.leakage_alpha)

    def guard_panel(self, panel: PanelSpec) -> None:
        for option in panel.options:
            self.guard(self.option_definition(option))

    def registered(self, cl_id: str) -> bool:
        """Can the register name this type? (Every definition is a register term.)"""
        return any(definition.cl_id == cl_id for definition in self.definitions.definitions)


def deciding_markers(definition: Definition) -> tuple[str, ...]:
    """The markers a definition depends on: its positives and its negatives."""
    return definition.pos_markers + definition.neg_markers


def state_symbols(state: str, cfg: FilterConfig) -> list[str]:
    """The symbols of a rendered state, in rank order."""
    return [symbol for symbol, _ in parse_rank_list(state, cfg)]


def render_state(cell: CellRecord, ctx: FamilyContext) -> str:
    """The cell's state, through the one canonical renderer."""
    return render_rank_list(cell.genes, cell.values, ctx.filter)


def choose_none(rng: random.Random) -> NoneOption:
    """One abstain option (key, wording) — see :data:`NONE_OPTIONS`."""
    return NONE_OPTIONS[rng.randrange(len(NONE_OPTIONS))]


def option_name(panel: PanelSpec, cl_id: str) -> str | None:
    """The option name for a CL id on this panel, or None when it is not on it."""
    for option in panel.options:
        if option.cl_id == cl_id:
            return option.name
    return None


def _criteria(panel: PanelSpec, ctx: FamilyContext, none: NoneOption | None = None) -> dict[str, str | None]:
    """The choice criteria: one entry per option, plus the abstain option if asked."""
    criteria: dict[str, str | None] = {option.name: option.definition for option in panel.options}
    if len(criteria) != len(panel.options):
        raise ValueError(f"two options on this panel share a name: {[option.name for option in panel.options]}")
    if none is not None:
        criteria[none[0]] = none[1]
    return criteria


def _meta(
    cell: CellRecord,
    panel: PanelSpec,
    family: RecordFamily,
    seed: int,
    ctx: FamilyContext,
    *,
    variant: str,
    extra_seeds: Mapping[str, int] | None = None,
    reason: str | None = None,
) -> RecordMeta:
    """Provenance for one record: the cell, the panel, the family, and the seeds used.

    ``group_id`` is the cell (``dataset/cell_id``): it is the bootstrap unit and
    the link between the families built for one cell — including the panel-miss
    pair, where the two records differ only in whether the true type is on the
    panel. ``panel_scope`` is read off the panel (the list it was drawn from), so
    no builder has to thread it through by hand.
    """
    return RecordMeta(
        dataset_id=cell.dataset_id,
        cl_id=cell.cl_id,
        panel_id=panel.panel_id,
        family=family,
        seeds={"record": seed, **dict(extra_seeds or {})},
        source=ctx.corpus or None,
        cell_id=cell.cell_id,
        donor_id=cell.donor_id,
        group_id=f"{cell.dataset_id}/{cell.cell_id}",
        variant=variant,
        panel_scope=panel.tissue,
        reason=reason,
    )


def panel_choice(
    cell: CellRecord,
    panel: PanelSpec,
    ctx: FamilyContext,
    *,
    seed: int,
    none: NoneOption | None = None,
    variant: str = "clean",
) -> Record:
    """The base family: pick the candidate whose definition the state satisfies."""
    ctx.guard(ctx.definition(cell.cl_id))
    ctx.guard_panel(panel)
    answer = option_name(panel, cell.cl_id)
    if answer is None:
        raise ValueError(f"{cell.cl_id} is not on this panel — build a panel_miss record instead")
    question = ChoiceQuestion(
        qid="label",
        instructions=CHOICE_INSTRUCTIONS,
        criteria=_criteria(panel, ctx, none),
        src=RecordFamily.PANEL_CHOICE,
    )
    return Record(
        state=render_state(cell, ctx),
        questions=(question,),
        answers={"label": answer},
        meta=_meta(cell, panel, RecordFamily.PANEL_CHOICE, seed, ctx, variant=variant),
    )


def marker_noul(
    cell: CellRecord,
    panel: PanelSpec,
    ctx: FamilyContext,
    *,
    seed: int,
    variant: str = "clean",
) -> Record:
    """One "does this definition hold of this state?" question per candidate.

    The answer is read off the state (:class:`HoldsRule`), not copied from the
    cell's label: a distractor whose markers happen to be expressed is a true
    "yes", and a true type whose markers are not expressed is a "false". That is
    the grounding signal — the model learns the definitions, not the label prior.
    """
    state = render_state(cell, ctx)
    symbols = state_symbols(state, ctx.filter)
    questions, answers = [], {}
    for option in panel.options:
        definition = ctx.option_definition(option)
        ctx.guard(definition)
        qid = f"holds:{option.cl_id}"
        questions.append(
            NoulQuestion(
                qid=qid,
                instructions={
                    "question": NOUL_INSTRUCTIONS,
                    "candidate": definition.name,
                    "definition": option.definition,
                },
                criteria=dict(NOUL_CRITERIA),
                src=RecordFamily.MARKER_NOUL,
            )
        )
        answers[qid] = ctx.holds.holds(symbols, definition)
    return Record(
        state=state,
        questions=tuple(questions),
        answers=answers,
        meta=_meta(cell, panel, RecordFamily.MARKER_NOUL, seed, ctx, variant=variant),
    )


def panel_miss(
    cell: CellRecord,
    panel: PanelSpec,
    ctx: FamilyContext,
    *,
    seed: int,
    none: NoneOption,
    variant: str = "clean",
) -> Record:
    """Axis-1 supervision: the true type is *not* on the panel, so abstaining is right."""
    ctx.guard(ctx.definition(cell.cl_id))
    ctx.guard_panel(panel)
    if option_name(panel, cell.cl_id) is not None:
        raise ValueError("this panel contains the true type — a panel_miss record needs a genuine miss")
    question = ChoiceQuestion(
        qid="label",
        instructions=CHOICE_INSTRUCTIONS,
        criteria=_criteria(panel, ctx, none),
        src=RecordFamily.PANEL_MISS,
    )
    return Record(
        state=render_state(cell, ctx),
        questions=(question,),
        answers={"label": none[0]},
        meta=_meta(cell, panel, RecordFamily.PANEL_MISS, seed, ctx, variant=variant),
    )


def out_of_register(
    cell: CellRecord,
    panel: PanelSpec,
    ctx: FamilyContext,
    *,
    seed: int,
    none: NoneOption,
    reason: str,
    variant: str = "clean",
) -> Record:
    """The abstention record for a cell the register cannot name (Step 8 §2).

    Identical evidence and answer surface to :func:`panel_miss` — one choice over
    the panel plus the abstain option, the abstain option being the target. What
    differs is *why* the true type is absent: a ``panel_miss`` is a draw that left
    it out, this is a type no panel of register types could ever contain. So there
    is no definition of the true type to guard (the register has none), and the
    leakage guard runs on the panel's options — the definitions the model reads —
    exactly as it does everywhere else.

    ``reason`` is the register's own account: its tissue is not one the register
    maps (:data:`UNREGISTERED_TISSUE`), or its type is not one of its terms
    (:data:`UNREGISTERED_TYPE`).
    """
    ctx.guard_panel(panel)
    if option_name(panel, cell.cl_id) is not None:
        raise ValueError(f"{cell.cl_id} is on this panel — it is not out of register")
    question = ChoiceQuestion(
        qid="label",
        instructions=CHOICE_INSTRUCTIONS,
        criteria=_criteria(panel, ctx, none),
        src=RecordFamily.OUT_OF_REGISTER,
    )
    return Record(
        state=render_state(cell, ctx),
        questions=(question,),
        answers={"label": none[0]},
        meta=_meta(cell, panel, RecordFamily.OUT_OF_REGISTER, seed, ctx, variant=variant, reason=reason),
    )


def panel_miss_pair(
    cell: CellRecord,
    panel: PanelSpec,
    sibling: PanelSpec,
    ctx: FamilyContext,
    *,
    seed: int,
    none: NoneOption,
    variant: str = "clean",
) -> tuple[Record, Record]:
    """The miss record and its correct-panel sibling, linked by ``group_id``.

    The pair is kev's ``none_pair`` contrast, rebuilt on real cells: same cell,
    same abstain wording, same everything except whether the true type is on the
    panel. Neither record alone tells the model whether abstaining is right; the
    pair together makes the two cases differ only in the option list.
    """
    miss = panel_miss(cell, panel, ctx, seed=seed, none=none, variant=variant)
    present = panel_choice(cell, sibling, ctx, seed=seed, none=none, variant=variant)
    if miss.meta.group_id != present.meta.group_id:
        raise ValueError("a panel-miss pair must share one group_id")
    return miss, present


def evidence_removed(
    cell: CellRecord,
    panel: PanelSpec,
    ctx: FamilyContext,
    *,
    seed: int,
    none: NoneOption,
    variant: str = "clean",
) -> Record | None:
    """Mask the true type's deciding markers; abstaining is the target.

    Returns ``None`` when the cell does not express any of them: there is no
    evidence to remove, so the record would be a copy of its sibling rather than a
    contrast, and the caller counts it as skipped instead of shipping it.
    """
    definition = ctx.definition(cell.cl_id)
    ctx.guard(definition)
    ctx.guard_panel(panel)
    answer = option_name(panel, cell.cl_id)
    if answer is None:
        raise ValueError(f"{cell.cl_id} is not on this panel — evidence_removed needs the true option present")
    state = render_state(cell, ctx)
    masked, removed = mask_rendered(state, deciding_markers(definition), ctx.filter)
    if not removed:
        return None
    question = ChoiceQuestion(
        qid="label",
        instructions=CHOICE_INSTRUCTIONS,
        criteria=_criteria(panel, ctx, none),
        src=RecordFamily.EVIDENCE_REMOVED,
    )
    return Record(
        state=masked,
        questions=(question,),
        answers={"label": none[0]},
        meta=_meta(
            cell,
            panel,
            RecordFamily.EVIDENCE_REMOVED,
            seed,
            ctx,
            variant=variant,
            extra_seeds={"masked": len(removed)},
        ),
    )


def evidence_diluted(
    cell: CellRecord,
    panel: PanelSpec,
    ctx: FamilyContext,
    *,
    seed: int,
    none: NoneOption,
    variant: str = "clean",
) -> Record | None:
    """Mask ``ctx.dilution_k`` random genes; the target depends on what survives.

    Unlike removal this is untargeted, so the answer is not fixed: if the state
    still satisfies the definition the true option is the answer, and if the
    definition no longer holds the answer is the abstain option. The pair of
    outcomes is exactly the graded signal — "the evidence got thinner" is only a
    reason to abstain when the definition stopped holding.
    """
    definition = ctx.definition(cell.cl_id)
    ctx.guard(definition)
    ctx.guard_panel(panel)
    answer = option_name(panel, cell.cl_id)
    if answer is None:
        raise ValueError(f"{cell.cl_id} is not on this panel — evidence_diluted needs the true option present")
    state = render_state(cell, ctx)
    diluted, removed = dilute_rendered(state, ctx.dilution_k, random.Random(seed), ctx.filter)
    if not removed:
        return None
    holds = ctx.holds.holds(state_symbols(diluted, ctx.filter), definition)
    question = ChoiceQuestion(
        qid="label",
        instructions=CHOICE_INSTRUCTIONS,
        criteria=_criteria(panel, ctx, none),
        src=RecordFamily.EVIDENCE_DILUTED,
    )
    return Record(
        state=diluted,
        questions=(question,),
        answers={"label": answer if holds else none[0]},
        meta=_meta(
            cell,
            panel,
            RecordFamily.EVIDENCE_DILUTED,
            seed,
            ctx,
            variant=variant,
            extra_seeds={"masked": len(removed)},
        ),
    )


def axis_score(
    cell: CellRecord,
    panel: PanelSpec,
    ctx: FamilyContext,
    axis: AxisSpec,
    *,
    seed: int,
    variant: str = "clean",
) -> Record:
    """Score the cell on an expression axis (a graded, non-categorical target)."""
    ctx.guard(ctx.definition(cell.cl_id))
    ctx.guard_panel(panel)
    level = axis.level(state_symbols(render_state(cell, ctx), ctx.filter))
    question = ScoreQuestion(
        qid="axis",
        instructions={"question": AXIS_INSTRUCTIONS, "axis": axis.name},
        criteria=list(axis.legend),
        src=RecordFamily.AXIS_SCORE,
    )
    return Record(
        state=render_state(cell, ctx),
        questions=(question,),
        answers={"axis": level},
        meta=_meta(cell, panel, RecordFamily.AXIS_SCORE, seed, ctx, variant=variant),
    )
