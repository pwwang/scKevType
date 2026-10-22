"""Records, panels and cells: what is rejected, and what survives a JSON round trip."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sckevtype.contracts.schema import (
    MAX_OPTIONS,
    CellState,
    ChoiceQuestion,
    NoulQuestion,
    PanelOption,
    PanelSpec,
    Record,
    RecordFamily,
    RecordMeta,
    ScoreQuestion,
    panel_id,
    size_bin,
)


def meta(**overrides) -> RecordMeta:
    return RecordMeta(
        **{
            "dataset_id": "seed_pbmc",
            "cl_id": "CL:0000624",
            "panel_id": "p" * 8,
            "family": RecordFamily.PANEL_CHOICE,
            "seeds": {"panel": 7},
            **overrides,
        }
    )


def record(**overrides) -> Record:
    fields = {
        "state": "CD3D 1 | IL7R 2 | ...",
        "questions": (
            ChoiceQuestion(qid="type", instructions="Which type?", criteria={"a": "T cell", "b": "B cell"}),
            NoulQuestion(qid="holds", instructions="Does the definition hold?"),
        ),
        "answers": {"type": "a", "holds": "yes"},
        "meta": meta(),
        **overrides,
    }
    return Record(**fields)


def test_choice_options_must_be_non_empty_and_within_the_serving_cap():
    with pytest.raises(ValidationError):
        ChoiceQuestion(qid="q", criteria={})
    with pytest.raises(ValidationError):
        ChoiceQuestion(qid="q", criteria={f"opt{i}": None for i in range(MAX_OPTIONS + 1)})
    with pytest.raises(ValidationError):
        ChoiceQuestion(qid="q", criteria={"": "unnamed"})
    assert len(ChoiceQuestion(qid="q", criteria={f"opt{i}": None for i in range(MAX_OPTIONS)}).criteria) == MAX_OPTIONS


def test_choice_target_must_name_options_on_the_panel():
    with pytest.raises(ValidationError, match="not on the panel"):
        ChoiceQuestion(qid="q", criteria={"a": None}, target={"b": 1.0})


def test_answers_must_reference_a_real_option():
    with pytest.raises(ValidationError, match="not one of the panel options"):
        record(answers={"type": "c", "holds": True})


def test_score_answers_must_be_a_level_index():
    question = ScoreQuestion(qid="state", criteria=["naive", "effector"])
    assert record(questions=(question,), answers={"state": 1}).answers == {"state": 1}
    with pytest.raises(ValidationError, match="level index"):
        record(questions=(question,), answers={"state": 2})


def test_noul_answers_accept_kevs_conventions_and_normalise_to_bool():
    for given, expected in (("yes", True), ("no", False), (1, True), (0, False), (True, True), ("FALSE", False)):
        assert record(answers={"type": "a", "holds": given}).answers["holds"] is expected
    with pytest.raises(ValidationError, match="noul answer"):
        record(answers={"type": "a", "holds": "maybe"})


def test_answers_must_cover_every_question_exactly():
    with pytest.raises(ValidationError, match="cover every question"):
        record(answers={"type": "a"})
    with pytest.raises(ValidationError, match="cover every question"):
        record(answers={"type": "a", "holds": True, "extra": "a"})


def test_record_round_trips_through_json():
    original = record()
    restored = Record.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.answers == {"type": "a", "holds": True}


def test_to_kev_emits_the_labelled_request_shape():
    payload = record().to_kev()
    assert isinstance(payload["state"], str)
    assert payload["questions"]["type"]["label"] == "a"
    assert payload["questions"]["holds"]["label"] is True
    assert payload["questions"]["type"]["criteria"] == {"a": "T cell", "b": "B cell"}
    assert payload["_meta"]["family"] == "panel_choice"
    assert Record.from_kev(payload).answers == {"type": "a", "holds": True}


def test_records_are_loadable_by_kevs_own_request_model():
    """The compatibility claim, checked against kev itself when it is installed."""
    kev_api = pytest.importorskip("kev.api", reason="kev is an optional (train extra) dependency")
    payload = record().to_kev()
    request = {
        "state": payload["state"],
        "questions": {
            qid: {key: value for key, value in question.items() if key in ("type", "instructions", "criteria")}
            for qid, question in payload["questions"].items()
        },
    }
    parsed = kev_api.SystemOneRequest.model_validate(request)
    assert [q.type for q in parsed.questions.values()] == ["choice", "noul"]


def test_panel_size_bins_are_the_frozen_ladder():
    assert [size_bin(n) for n in (1, 2, 4, 5, 19, 20, 255)] == ["1-1", "2-4", "2-4", "5-9", "10-19", "20-49", "100+"]
    with pytest.raises(ValueError):
        size_bin(0)


def test_panel_id_ignores_order_and_names_but_tracks_definitions():
    options = [
        PanelOption(cl_id="CL:0000624", name="CD4 T", definition="CD3D, CD4"),
        PanelOption(cl_id="CL:0000625", name="CD8 T", definition="CD3D, CD8A"),
    ]
    spec = PanelSpec(options=tuple(options), source="user")
    reordered = PanelSpec(options=tuple(reversed(options)), source="user")
    renamed = PanelSpec(
        options=tuple(option.model_copy(update={"name": f"type-{i}"}) for i, option in enumerate(options)), source="user"
    )
    redefined = PanelSpec(
        options=(options[0].model_copy(update={"definition": "CD3D, CD4, IL7R"}), options[1]), source="user"
    )
    assert spec.panel_id == reordered.panel_id == renamed.panel_id == panel_id(options)
    assert spec.panel_id != redefined.panel_id
    assert spec.size_bin == "2-4"


def test_panel_rejects_duplicate_types_and_stale_ids():
    option = PanelOption(cl_id="CL:0000624", name="CD4 T", definition="CD3D")
    with pytest.raises(ValidationError, match="twice"):
        PanelSpec(options=(option, option), source="user")
    with pytest.raises(ValidationError, match="panel_id does not match"):
        PanelSpec(options=(option,), source="user", panel_id="0" * 64)


def test_panel_option_requires_a_named_type():
    with pytest.raises(ValidationError):
        PanelOption(cl_id="", name="x", definition="d")


def test_cell_state_pins_its_rendered_string():
    state = CellState(cell_id="c1", dataset_id="seed_pbmc", rendered="CD3D 1 | ...", n_genes=1)
    assert len(state.render_digest) == 64
    assert state.render_digest == CellState(
        cell_id="c2", dataset_id="seed_pbmc", rendered=state.rendered, n_genes=1, render_digest=state.render_digest
    ).render_digest
    with pytest.raises(ValidationError, match="render_digest does not match"):
        CellState(cell_id="c3", dataset_id="seed_pbmc", rendered="CD3D 1 | ...", n_genes=1, render_digest="0" * 64)
