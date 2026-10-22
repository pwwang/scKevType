"""Validated shapes for panels, records, cells, and manifests (pydantic v2).

The on-disk record format is kev's *labelled request* format — the one
``kev.data.load_records`` reads and ``kev.train --suite`` therefore consumes:

    {"state": str, "questions": {qid: {type, instructions, criteria, label, src}},
     "_meta": {...}}

:meth:`Record.to_kev` emits it and :meth:`Record.from_kev` reads it back, so
every record family in `paper_plan.md` §3 is a kev-loadable suite partition with
no adapter in between. Our own fields live inside ``_meta`` and are additive:
kev ignores keys it does not know, and ``kev.data.load_records`` preserves them.

The role split inside a record: ``questions`` say *what the panel is* (semantics
that reach the model), ``answers`` hold the target, ``meta`` holds provenance.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .freeze import payload_digest, text_digest

#: Mirrors ``kev.api.MAX_OPTIONS``: the serving API refuses more options. Importing
#: kev here would drag the training stack into the contracts layer, so the cap is
#: restated as a contract constant and checked against kev in the tests.
MAX_OPTIONS = 255
#: Frozen panel-size strata used by Mondrian (panel-stratified) conformal
#: calibration, design-notes §1: a bin is [edge_i, edge_{i+1}), the last is open.
SIZE_BIN_EDGES: tuple[int, ...] = (2, 5, 10, 20, 50, 100)

JSONContent = str | dict | list | int | float | bool | None


def _set(model: BaseModel, **fields: Any) -> None:
    """Assign fields from inside a ``mode="after"`` validator.

    These models are frozen, and an after-validator that *returns* a different
    instance has its return value dropped when validation runs through
    ``__init__`` — so derived fields (panel ids, normalized answers) must be set
    in place. ``object.__setattr__`` is the supported way to do that on a frozen
    pydantic v2 model.
    """
    for name, value in fields.items():
        object.__setattr__(model, name, value)


class RecordFamily(StrEnum):
    """The record families of `paper_plan.md` §3. Frozen: these names reach manifests."""

    PANEL_CHOICE = "panel_choice"
    MARKER_NOUL = "marker_noul"
    PANEL_MISS = "panel_miss"
    EVIDENCE_REMOVED = "evidence_removed"
    EVIDENCE_DILUTED = "evidence_diluted"
    AXIS_SCORE = "axis_score"
    #: A cell whose true type the register cannot name. Same semantics as
    #: ``panel_miss`` (the true type is not on the panel, so abstaining is right),
    #: but the reason is the register's scope rather than the draw — see
    #: :func:`sckevtype.data.families.out_of_register`. Added last so the families
    #: that were already frozen keep their order.
    OUT_OF_REGISTER = "out_of_register"


def size_bin(n_options: int) -> str:
    """The panel-size stratum of a panel with ``n_options`` options (frozen ladder)."""
    if n_options < 1:
        raise ValueError("a panel needs at least one option")
    lo = 1
    for edge in SIZE_BIN_EDGES:
        if n_options < edge:
            return f"{lo}-{edge - 1}"
        lo = edge
    return f"{lo}+"


def panel_id(options: list[PanelOption] | tuple[PanelOption, ...]) -> str:
    """Stable identity of a panel: sorted option CL ids + each option's definition digest.

    Deliberately independent of option *order* (panels are always permuted at
    training time) and of option *names* (rename-invariance is a capability the
    paper tests), so the same panel is the same stratum however it is presented.
    """
    payload = sorted(
        ({"cl_id": option.cl_id, "definition": text_digest(option.definition)} for option in options),
        key=lambda item: item["cl_id"],
    )
    return payload_digest(payload)


class PanelOption(BaseModel):
    """One candidate type on a panel, carrying the definition the model reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cl_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    definition: str
    markers_pos: tuple[str, ...] = ()
    markers_neg: tuple[str, ...] = ()
    provenance: str = ""


class PanelSpec(BaseModel):
    """A candidate panel: the request-time argument the whole method conditions on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    options: tuple[PanelOption, ...] = Field(min_length=1, max_length=MAX_OPTIONS)
    size_bin: str = ""
    tissue: str | None = None
    panel_id: str = ""
    source: Literal["user", "middleware", "random"]

    @model_validator(mode="after")
    def _fill_and_check(self) -> PanelSpec:
        cl_ids = [option.cl_id for option in self.options]
        if len(set(cl_ids)) != len(cl_ids):
            raise ValueError(f"a panel lists the same CL id twice: {sorted(cl_ids)}")
        expected_bin, expected_id = size_bin(len(self.options)), panel_id(self.options)
        if self.size_bin and self.size_bin != expected_bin:
            raise ValueError(f"size_bin {self.size_bin!r} does not match {len(self.options)} options ({expected_bin})")
        if self.panel_id and self.panel_id != expected_id:
            raise ValueError("panel_id does not match these options; rebuild the spec instead of editing it")
        _set(self, size_bin=expected_bin, panel_id=expected_id)
        return self


class ChoiceQuestion(BaseModel):
    """Options keyed by name, exactly as ``kev.api.Choice`` sees them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    qid: str = Field(min_length=1)
    type: Literal["choice"] = "choice"
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]
    src: str | None = None
    #: kev soft target: option name -> weight (used by unknowable records).
    target: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> ChoiceQuestion:
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options, got {len(self.criteria)}")
        if any(not str(key) for key in self.criteria):
            raise ValueError("every option needs a non-empty name")
        if self.target is not None:
            unknown = set(self.target) - set(self.criteria)
            if unknown:
                raise ValueError(f"target names options that are not on the panel: {sorted(unknown)}")
        return self


class NoulQuestion(BaseModel):
    """A yes/no question. kev's answer keys are ``["false", "true"]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    qid: str = Field(min_length=1)
    type: Literal["noul"] = "noul"
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None
    src: str | None = None


class ScoreQuestion(BaseModel):
    """An ordered-level question: ``criteria`` is the level legend, lowest first."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    qid: str = Field(min_length=1)
    type: Literal["score"] = "score"
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)
    src: str | None = None


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion
Answer = str | bool | int

_TRUE = {"yes", "true", "1"}
_FALSE = {"no", "false", "0"}


def normalize_answer(question: Question, answer: Answer) -> Answer:
    """Check an answer against its question and normalise it to kev's label convention.

    choice -> the option name; noul -> bool (kev encodes it as ``int(y)``);
    score -> the level index. Anything that does not reference a real option or
    level is rejected here rather than becoming a silently unscorable record.
    """
    if isinstance(question, ChoiceQuestion):
        if answer not in question.criteria:
            raise ValueError(f"{question.qid}: answer {answer!r} is not one of the panel options {sorted(question.criteria)}")
        return str(answer)
    if isinstance(question, NoulQuestion):
        if isinstance(answer, str):
            if answer.casefold() in _TRUE:
                return True
            if answer.casefold() in _FALSE:
                return False
        elif isinstance(answer, bool):
            return answer
        elif isinstance(answer, int) and answer in (0, 1):
            return bool(answer)
        raise ValueError(f"{question.qid}: a noul answer must be true/false, 0/1, or yes/no, got {answer!r}")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(question.criteria):
        raise ValueError(f"{question.qid}: a score answer must be a level index in [0, {len(question.criteria)}), got {answer!r}")
    return answer


class RecordMeta(BaseModel):
    """Provenance that makes a record reproducible from ``(seed, config, dataset digest)``.

    The last five fields are kev's ``_meta`` keys: they are what the suite
    builder groups bootstrap units by, so they travel with the record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    cl_id: str = Field(min_length=1)
    panel_id: str = Field(min_length=1)
    family: RecordFamily
    seeds: dict[str, int] = Field(default_factory=dict)
    source: str | None = None
    cell_id: str | None = None
    donor_id: str | None = None
    group_id: str | None = None
    variant: str = "clean"
    #: Which candidate list the panel was drawn from: a register tissue's name, or
    #: ``global`` for the whole register (a cell whose tissue the register does not
    #: map). Every record that carries a panel records it (Step 8 §1).
    panel_scope: str | None = None
    #: Why the register cannot name this cell — set by ``out_of_register`` only.
    reason: str | None = None


class Record(BaseModel):
    """One labelled request: what the model reads, what the target is, where it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: str
    questions: tuple[Question, ...] = Field(min_length=1)
    answers: dict[str, Answer]
    meta: RecordMeta

    @model_validator(mode="after")
    def _check(self) -> Record:
        qids = [q.qid for q in self.questions]
        if len(set(qids)) != len(qids):
            raise ValueError(f"duplicate question ids: {sorted(qids)}")
        missing, extra = set(qids) - set(self.answers), set(self.answers) - set(qids)
        if missing or extra:
            raise ValueError(f"answers must cover every question exactly: missing {sorted(missing)}, unexpected {sorted(extra)}")
        _set(self, answers={q.qid: normalize_answer(q, self.answers[q.qid]) for q in self.questions})
        return self

    def to_kev(self) -> dict[str, Any]:
        """The kev labelled-request dict ``kev.data.load_records`` reads."""
        questions = {}
        for question in self.questions:
            payload = question.model_dump(exclude={"qid"}, exclude_none=True)
            payload["label"] = self.answers[question.qid]
            questions[question.qid] = payload
        return {"state": self.state, "questions": questions, "_meta": self.meta.model_dump(mode="json", exclude_none=True)}

    @classmethod
    def from_kev(cls, record: dict[str, Any]) -> Record:
        """Inverse of :meth:`to_kev` for records this package wrote."""
        questions, answers = [], {}
        for qid, payload in record["questions"].items():
            label = payload["label"]
            # `qtype`/`keys` are added by kev.data.materialize; they are derived, not input.
            question = {key: value for key, value in payload.items() if key not in ("label", "qtype", "keys")}
            questions.append(_QUESTION_TYPES[question.get("type", "choice")].model_validate({**question, "qid": qid}))
            answers[qid] = label
        return cls(state=record["state"], questions=tuple(questions), answers=answers, meta=record["_meta"])


_QUESTION_TYPES = {"choice": ChoiceQuestion, "noul": NoulQuestion, "score": ScoreQuestion}


class CellState(BaseModel):
    """A rendered cell: the exact string the model reads, plus where it came from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    donor_id: str | None = None
    rendered: str
    n_genes: int = Field(ge=0)
    render_digest: str = ""

    @model_validator(mode="after")
    def _fill_and_check(self) -> CellState:
        expected = text_digest(self.rendered)
        if self.render_digest and self.render_digest != expected:
            raise ValueError("render_digest does not match the rendered state")
        _set(self, render_digest=expected)
        return self


class CoverageReport(BaseModel):
    """How much of a label or gene list a frozen resource can express.

    Shared by the vocabulary (out-of-vocabulary symbols) and the label maps
    (unmapped dataset labels), because both answer the same question: what does
    the model structurally *cannot* see, before any evaluation number is quoted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0)
    mapped: int = Field(ge=0)
    unmapped: int = Field(ge=0)
    rate: float
    unmapped_items: tuple[str, ...] = ()

    def ok(self, min_coverage: float) -> bool:
        return self.rate >= min_coverage


class FileEntry(BaseModel):
    """One pinned file inside a manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = 0
    kind: str = "data"
    counts: int | None = None


class SuiteManifest(BaseModel):
    """``artifacts/<version>/manifest.json``. Digest-pinned, sorted keys, atomically written."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    created_at: str
    git_commit: str | None = None
    seed: int | None = None
    files: dict[str, FileEntry] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    source_digests: dict[str, str] = Field(default_factory=dict)
    code_hashes: dict[str, str] = Field(default_factory=dict)
    previous_version: str | None = None
