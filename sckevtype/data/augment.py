"""Augmentations: option permutation, definition perturbation, evidence masking.

Every function here is pure — it takes a record (or a state) and a seed and
returns a new object, never mutating its input. The train/test contract is the
reason: a training corpus is *always* permuted (a model that has seen the true
option in a fixed position learns the position, not the definition), while the
development and test splits carry both the clean and the permuted twin of the
same record. So the permutation has to be reproducible from the seed and has to
leave the answer untouched — it is a different presentation of the same question,
which is why it is recorded (``variant``/``seeds``) rather than hidden.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..contracts.definitions import DefinitionSet, perturb
from ..contracts.render import FilterConfig, parse_rank_list, render_rank_list
from ..contracts.schema import ChoiceQuestion, PanelSpec, Record

#: Record variant names. ``clean`` is the unaugmented presentation.
CLEAN = "clean"
PERMUTED = "permuted"


def permute_options(record: Record, seed: int) -> Record:
    """Reorder each choice question's options; the answer is unchanged.

    The permutation is drawn from ``seed`` and recorded as ``variant="permuted"``
    plus ``seeds["permute"]``: a corpus built twice with the same seed permutes
    the same way, and a reader can tell a permuted record from a clean one without
    diffing the criteria.
    """
    rng = random.Random(seed)
    questions = []
    for question in record.questions:
        if isinstance(question, ChoiceQuestion):
            names = list(question.criteria)
            rng.shuffle(names)
            questions.append(question.model_copy(update={"criteria": {name: question.criteria[name] for name in names}}))
        else:
            questions.append(question)
    meta = record.meta.model_copy(
        update={"variant": PERMUTED, "seeds": {**record.meta.seeds, "permute": seed}}
    )
    return record.model_copy(update={"questions": tuple(questions), "meta": meta})


def perturb_panel(panel: PanelSpec, definitions: DefinitionSet, ops: Sequence[str], seed: int) -> PanelSpec:
    """A panel whose option *definitions* are perturbed (paraphrase, marker drop, …).

    The perturbation is applied to the source definition and re-rendered through
    :meth:`~sckevtype.contracts.definitions.Definition.to_panel_option`, so the
    option text always comes from the one canonical renderer. The perturbed panel
    has a different ``panel_id`` than the original — which is the point: the
    identity of a panel includes how its definitions read.
    """
    rng = random.Random(seed)
    options = []
    for option in panel.options:
        source = definitions.get(option.cl_id)
        options.append(perturb(source, ops, rng, definitions.synonyms).to_panel_option())
    return PanelSpec(options=tuple(options), tissue=panel.tissue, source=panel.source)


def parse_state(rendered: str, cfg: FilterConfig) -> list[tuple[str, int]]:
    """The state as ``(symbol, rank)`` pairs — the renderer's own inverse."""
    return parse_rank_list(rendered, cfg)


def mask_rendered(rendered: str, markers: Sequence[str], cfg: FilterConfig) -> tuple[str, tuple[str, ...]]:
    """Drop ``markers`` from a rendered state; return ``(new state, dropped symbols)``.

    Reproducible (same input, same output) and reversible in the sense that
    matters: the dropped symbols are returned, and the surviving symbols keep
    their relative order, so the masked state is exactly what the model would have
    read had those genes not been expressed. Re-rendering goes through
    ``render_rank_list`` with the old rank as the ordering value, so the canonical
    renderer stays the only thing that writes a state.
    """
    pairs = parse_rank_list(rendered, cfg)
    dropping = set(markers)
    kept = [(symbol, rank) for symbol, rank in pairs if symbol not in dropping]
    dropped = tuple(symbol for symbol, _ in pairs if symbol in dropping)
    if not dropped:
        return rendered, ()
    symbols = [symbol for symbol, _ in kept]
    values = [float(-rank) for _, rank in kept]     # rank 1 first, ties impossible
    return render_rank_list(symbols, values, cfg), dropped


def dilute_rendered(rendered: str, k: int, rng: random.Random, cfg: FilterConfig) -> tuple[str, tuple[str, ...]]:
    """Drop ``k`` symbols drawn uniformly from the state (evidence dilution).

    Unlike :func:`mask_rendered` this is not targeted: it removes whatever the RNG
    picked, which is what makes it a *dilution* rather than an ablation.
    """
    pairs = parse_rank_list(rendered, cfg)
    if k <= 0 or len(pairs) < 2:
        return rendered, ()
    drawn = rng.sample([symbol for symbol, _ in pairs], min(k, len(pairs) - 1))
    return mask_rendered(rendered, drawn, cfg)


def mask_record(record: Record, markers: Sequence[str], cfg: FilterConfig) -> Record:
    """A record whose state has ``markers`` masked out (questions and answers kept).

    Used to build the evidence-removed training twin of a record without
    re-deriving its panel: the questions stay the same, so the contrast is exactly
    "same question, same panel, missing evidence".
    """
    state, _ = mask_rendered(record.state, markers, cfg)
    meta = record.meta.model_copy(update={"variant": f"{record.meta.variant}+masked"})
    return record.model_copy(update={"state": state, "meta": meta})
