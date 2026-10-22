"""The definition resource: what each type *means*, and how it may be perturbed.

A definition is the prompt text of one panel option, so it is a frozen artifact
with per-record provenance (design-notes §5/§6). Two rules follow from that and
are enforced here rather than trusted:

* **perturbation is deterministic** — every op draws from an explicit
  ``random.Random``, so the same seed reproduces the same definition;
* **no leakage** — a definition seeded from a dataset's own differential
  expression is the answer, not a definition, so :func:`assert_no_leakage`
  raises when the two overlap (paper_plan §2, design-notes §6).

Nothing here mutates in place: perturbations return new ``Definition`` objects
and leave the frozen original untouched.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .freeze import payload_digest, read_json, write_json
from .schema import PanelOption

#: Frozen paraphrase templates. Paraphrase has to be deterministic and offline,
#: so it is template substitution — no model in the loop, in training or in the
#: paper's paraphrase-robustness experiment.
PARAPHRASE_TEMPLATES: tuple[str, ...] = (
    "{text}",
    "Definition: {text}",
    "Cell type characterised as follows: {text}",
    "{text} (curated marker definition)",
    "Markers identify this population — {text}",
)
#: What separates the prose from the marker list in the rendered definition text.
TEXT_JOIN = ". "
POS_LABEL = "positive markers"
NEG_LABEL = "negative markers"


class LeakageError(ValueError):
    """Raised when a definition's markers come from the dataset it will be evaluated on."""


class Definition(BaseModel):
    """One cell type's definition, with the markers it is built from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cl_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    text: str = ""
    pos_markers: tuple[str, ...] = ()
    neg_markers: tuple[str, ...] = ()
    #: Which curated resource this came from (recorded per record; design-notes §6).
    provenance: str = ""
    #: Digest of the seed input this definition was built from.
    source_digest: str = ""

    @property
    def markers(self) -> tuple[str, ...]:
        return self.pos_markers + self.neg_markers

    def render_text(self, *, markers: bool = True) -> str:
        """The exact string placed in a panel option's ``definition``.

        Prose first, then the marker lists, single-sentence separated and
        terminated: ``<text>. positive markers: A, B. negative markers: C.``
        Frozen because it is prompt text — the same string must reach the model
        at training time and at serving time.

        ``markers=False`` renders the prose alone — no ``positive markers: …``
        and no ``negative markers: …`` clause. It is the *prompt* that changes,
        nothing else: the lists stay on the :class:`Definition` and on the
        :class:`~.schema.PanelOption` it builds, so :class:`HoldsRule`, the
        leakage guard and every derived target read exactly what they read
        before. A corpus turns it off when an option's text naming its own
        markers would answer the question by string-matching the state's top
        genes instead of by knowing the definition (``real-v3``); the default
        keeps the register's full rendering, which is what every earlier corpus
        was built with.
        """
        prose = self.text.strip().rstrip(".")
        parts = [prose] if prose else []
        if markers and self.pos_markers:
            parts.append(f"{POS_LABEL}: {', '.join(self.pos_markers)}")
        if markers and self.neg_markers:
            parts.append(f"{NEG_LABEL}: {', '.join(self.neg_markers)}")
        return TEXT_JOIN.join(parts) + "." if parts else ""

    def payload(self) -> dict:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return payload_digest(self.payload())

    def to_panel_option(self, *, markers: bool = True) -> PanelOption:
        """This definition as a panel option — the only way one should be built."""
        return PanelOption(
            cl_id=self.cl_id,
            name=self.name,
            definition=self.render_text(markers=markers),
            markers_pos=self.pos_markers,
            markers_neg=self.neg_markers,
            provenance=self.provenance,
        )


class DefinitionSet(BaseModel):
    """A frozen collection of definitions plus the synonym map its perturbations use."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    definitions: tuple[Definition, ...] = Field(min_length=1)
    #: marker symbol -> accepted aliases, used by the ``synonym`` perturbation.
    synonyms: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    source: str = ""
    version: str = "v1"

    @property
    def by_cl_id(self) -> dict[str, Definition]:
        return {definition.cl_id: definition for definition in self.definitions}

    def payload(self) -> dict:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return payload_digest(self.payload())

    def get(self, cl_id: str) -> Definition:
        try:
            return self.by_cl_id[cl_id]
        except KeyError:
            raise KeyError(f"no definition for {cl_id}") from None

    def to_panel_option(self, cl_id: str, *, markers: bool = True) -> PanelOption:
        """The panel option for one type."""
        return self.get(cl_id).to_panel_option(markers=markers)

    def save(self, path: str) -> str:
        digest = self.digest()
        write_json(path, {**self.payload(), "digest": digest})
        return digest

    @classmethod
    def load(cls, path: str) -> DefinitionSet:
        payload = read_json(path)
        digest = payload.pop("digest", None)
        definitions = cls.model_validate(payload)
        if digest != definitions.digest():
            raise ValueError(f"{path}: definition-set digest mismatch (recorded {digest}, actual {definitions.digest()})")
        return definitions


def perturb(
    definition: Definition,
    ops: Sequence[str],
    rng: random.Random,
    synonyms: Mapping[str, Sequence[str]] | None = None,
) -> Definition:
    """Apply definition perturbations, deterministically, returning a new Definition.

    Supported ops: ``paraphrase`` (frozen template), ``drop_marker`` (exactly one
    marker), ``reorder_markers`` (marker order only), ``synonym`` (one marker
    swapped for a curated alias). An op that cannot apply (nothing to drop, no
    alias known) is recorded in the provenance and leaves the definition
    otherwise unchanged rather than raising, so a perturbation schedule is
    reproducible across types with different marker coverage.
    """
    current = definition
    provenance = definition.provenance
    for op in ops:
        if op == "paraphrase":
            current = current.model_copy(update={"text": rng.choice(PARAPHRASE_TEMPLATES).format(text=current.text.strip().rstrip("."))})
            provenance = f"{provenance}+paraphrase"
        elif op == "drop_marker":
            marker, dropped = _drop_marker(current, rng)
            current = dropped
            provenance = f"{provenance}+drop_marker({marker})" if marker else f"{provenance}+drop_marker(none)"
        elif op == "reorder_markers":
            current = current.model_copy(
                update={"pos_markers": _shuffled(current.pos_markers, rng), "neg_markers": _shuffled(current.neg_markers, rng)}
            )
            provenance = f"{provenance}+reorder_markers"
        elif op == "synonym":
            current, swap = _synonym(current, rng, synonyms or {})
            provenance = f"{provenance}+synonym({swap})" if swap else f"{provenance}+synonym(none)"
        else:
            raise ValueError(f"unknown perturbation op: {op!r} (have: paraphrase, drop_marker, reorder_markers, synonym)")
        current = current.model_copy(update={"provenance": provenance})
    return current


def _shuffled(items: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    order = list(items)
    rng.shuffle(order)
    return tuple(order)


def _drop_marker(definition: Definition, rng: random.Random) -> tuple[str | None, Definition]:
    """Remove exactly one marker (the first occurrence, in the list it was drawn from)."""
    pool = definition.pos_markers + definition.neg_markers
    if not pool:
        return None, definition
    marker = rng.choice(pool)
    if marker in definition.pos_markers:
        pos = list(definition.pos_markers)
        pos.remove(marker)
        return marker, definition.model_copy(update={"pos_markers": tuple(pos)})
    neg = list(definition.neg_markers)
    neg.remove(marker)
    return marker, definition.model_copy(update={"neg_markers": tuple(neg)})


def _synonym(
    definition: Definition, rng: random.Random, synonyms: Mapping[str, Sequence[str]]
) -> tuple[Definition, str | None]:
    """Swap one marker for a curated alias of it."""
    pool = [marker for marker in definition.markers if synonyms.get(marker)]
    if not pool:
        return definition, None
    marker = rng.choice(pool)
    alias = rng.choice(tuple(synonyms[marker]))
    if marker in definition.pos_markers:
        pos = tuple(alias if item == marker else item for item in definition.pos_markers)
        return definition.model_copy(update={"pos_markers": pos}), f"{marker}->{alias}"
    neg = tuple(alias if item == marker else item for item in definition.neg_markers)
    return definition.model_copy(update={"neg_markers": neg}), f"{marker}->{alias}"


def assert_no_leakage(definition: Definition, dataset_de_markers: Sequence[str], alpha: float = 0.0) -> None:
    """Raise if the definition is (partly) derived from this dataset's own DE markers.

    ``dataset_de_markers`` is the dataset's differential-expression hit list, used
    for coverage QA only. ``alpha`` is the largest fraction of the definition's
    own markers allowed to overlap it; the default 0.0 means any overlap is
    leakage. Both the marker lists and the rendered text are checked, because
    prose can name a marker that the lists omit.
    """
    derived = {marker for marker in dataset_de_markers if marker}
    if not derived:
        return
    own = set(definition.markers)
    in_text = {
        marker for marker in derived if re.search(rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])", definition.render_text())
    }
    overlap = sorted(own & derived | in_text)
    if not overlap:
        return
    limit = alpha * max(1, len(own))
    if len(overlap) > limit:
        raise LeakageError(
            f"definition {definition.cl_id!r} ({definition.provenance}) shares {len(overlap)} marker(s) with this dataset's "
            f"differential expression (limit {alpha:.2f} of {len(own)}): {overlap}. Definitions must come from an external "
            f"curated resource; per-dataset DE is for coverage QA only."
        )
