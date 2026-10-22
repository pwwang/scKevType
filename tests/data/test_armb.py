"""Arm B: the target, the supervision that reaches it, and the refusal to re-label.

Two layers. The rule layer is tested on records built here from the frozen register, so the
five families' cases are exercised one at a time and the test says which case it is testing.
The build layer is tested on a two-partition corpus written into ``tmp_path``, because the
properties that matter there are about *which bytes change*: the transformed partition is
rewritten, the scored partition is copied byte for byte, and a record whose label is not the
evidence rule's answer stops the build instead of being silently relabelled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sckevtype.contracts.freeze import read_json, text_digest
from sckevtype.contracts.render import FilterConfig, render_rank_list
from sckevtype.data import armb
from sckevtype.data.families import NONE_OPTIONS

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
UBIQUITOUS = frozenset(FilterConfig(**read_json(ARTIFACTS / "v2" / "render_config.json")).ubiquitous)


@pytest.fixture(scope="module")
def index() -> armb.NameIndex:
    return armb.NameIndex.load("v2")


@pytest.fixture(scope="module")
def register() -> dict:
    """The v2 register, keyed by name — the definitions' real markers, for real states."""
    payload = read_json(ARTIFACTS / "v2" / "definitions.json")
    return {entry["name"]: entry for entry in payload["definitions"]}


def first_usable(register: dict, *, exclude: str = "") -> dict:
    """A register term whose markers a hand-built state can actually carry.

    Chosen rather than assumed: its markers must survive the renderer's filters (a
    ubiquitous or excluded marker would be dropped from the state, and the test would be
    measuring the filter instead of the rule), and no positive marker may also be a negative
    one.
    """
    for entry in register.values():
        if entry["name"] == exclude:
            continue
        pos = [m for m in entry["pos_markers"] if m not in UBIQUITOUS]
        if pos and len(pos) == len(entry["pos_markers"]) and not set(pos) & set(entry["neg_markers"]):
            return {**entry, "usable_pos": pos, "usable_neg": [m for m in entry["neg_markers"]
                                                              if m not in UBIQUITOUS]}
    pytest.skip("no register term has markers the renderer will keep")


@pytest.fixture(scope="module")
def subject(register) -> dict:
    return first_usable(register)


@pytest.fixture(scope="module")
def other(subject, register) -> dict:
    """A second register term: a panel needs more than one option, and a tamper needs one."""
    return first_usable(register, exclude=subject["name"])


def abstain_at(rank: int = 0) -> dict:
    """The abstain option, as an option list carries it: ``{key: wording}``.

    A record can only answer "none of these" if abstaining is on the list, and in the frozen
    corpus every withheld row has one — so a hand-built withheld record has to have one too.
    """
    return {NONE_OPTIONS[rank][0]: NONE_OPTIONS[rank][1]}


def state_of(genes: tuple[str, ...]) -> str:
    """A rendered state carrying exactly those symbols, at descending values."""
    values = tuple(9.0 - 0.5 * i for i in range(len(genes)))
    return render_rank_list(genes, values, armb.render_config("v2"))


def option(entry: dict) -> str:
    """The option text a panel would carry for a register term — the renderer's own format."""
    return (f"{entry['text'].rstrip('.')}. positive markers: {', '.join(entry['pos_markers'])}. "
            f"negative markers: {', '.join(entry['neg_markers'])}.")


def record(family: str, criteria: dict, label, state: str, cl_id: str,
           question_type: str = "choice", record_id: str = "x/0") -> dict:
    """A corpus record in the shape ``kev.data.load_records`` hands the trainer."""
    question = {"type": question_type, "label": label, "criteria": criteria}
    if question_type == "noul":
        question.pop("label")
        question.pop("criteria")
        question["instructions"] = {"candidate": "CD3D", "reference": "the reference"}
    return {
        "state": state,
        "questions": {"q": question},
        "_meta": {"id": record_id, "family": family, "cl_id": cl_id, "dataset_id": "unit", "variant": "permuted"},
    }


# ------------------------------------------------------------------ the parse


def test_parse_definition_round_trips_the_whole_register(register):
    """The rule reads markers out of option text, so the parse must be exact, not close."""
    for entry in register.values():
        parsed = armb.parse_definition(option(entry))
        assert list(parsed[0]) == list(entry["pos_markers"]), entry["name"]
        assert list(parsed[1]) == list(entry["neg_markers"]), entry["name"]


def test_parse_definition_tolerates_a_definition_with_no_marker_sentence():
    assert armb.parse_definition("A cell with no marker sentence.") == ((), ())


def test_register_names_and_ids_are_bijections(index):
    """The panel clause resolves a name to a type through this table; it has to be exact."""
    assert len(index.by_name) == len(index.by_cl) == len(set(index.by_name))
    assert all(index.cl_of(name) == cl for name, cl in index.by_name.items())
    assert all(index.name_of(cl) == name for cl, name in index.by_cl.items())


# ------------------------------------------------------------------ the rule


def test_panel_choice_is_supported(subject, index):
    """The true option is on the panel and the state has not been deprived of anything."""
    criteria = {"other type": "prose", subject["name"]: option(subject)}
    answer = armb.evidence_answer(
        record("panel_choice", criteria, subject["name"], state_of(tuple(subject["usable_pos"])), subject["cl_id"]),
        index)
    assert answer["side"] == "supported"
    assert answer["target"] == subject["name"]


def test_panel_miss_withholds(subject, index):
    """The state carries the evidence, but no option on this panel names the cell's type."""
    criteria = {"some other type": "prose", **abstain_at()}
    answer = armb.evidence_answer(
        record("panel_miss", criteria, NONE_OPTIONS[0][0], state_of(tuple(subject["usable_pos"])), subject["cl_id"]),
        index)
    assert answer["side"] == "withheld"
    assert answer["target"] == NONE_OPTIONS[0][0]


def test_out_of_register_withholds_without_a_name_to_look_for(subject, index):
    """A type the register cannot name has no option name to find, so the panel clause fails."""
    criteria = {subject["name"]: option(subject), **abstain_at()}
    answer = armb.evidence_answer(
        record("out_of_register", criteria, NONE_OPTIONS[0][0], state_of(tuple(subject["usable_pos"])), "CL:9999999"),
        index)
    assert answer["side"] == "withheld"
    assert "no register panel can name" in answer["reason"]


def test_evidence_removed_withholds_by_construction(subject, index):
    """The corpus removed the deciding markers, so the state clause fails before the panel's."""
    criteria = {subject["name"]: option(subject), **abstain_at()}
    answer = armb.evidence_answer(
        record("evidence_removed", criteria, NONE_OPTIONS[0][0], state_of(("CD3D",)), subject["cl_id"]), index)
    assert answer["side"] == "withheld"
    assert "deciding markers were removed" in answer["reason"]


def test_evidence_diluted_is_decided_by_the_state(subject, index):
    """Masking random genes is only a reason to abstain once the definition stopped holding."""
    criteria = {subject["name"]: option(subject), **abstain_at()}
    cl_id = subject["cl_id"]

    holds = armb.evidence_answer(
        record("evidence_diluted", criteria, subject["name"], state_of(tuple(subject["usable_pos"])), cl_id), index)
    assert (holds["side"], holds["target"]) == ("supported", subject["name"])

    broke = armb.evidence_answer(
        record("evidence_diluted", criteria, NONE_OPTIONS[0][0], state_of(("CD3D", "CD8A")), cl_id), index)
    assert (broke["side"], broke["target"]) == ("withheld", NONE_OPTIONS[0][0])


def test_a_noul_record_has_no_target(subject, index):
    answer = armb.evidence_answer(
        record("marker_noul", {}, None, state_of(("CD3D",)), subject["cl_id"], question_type="noul"), index)
    assert (answer["side"], answer["target"]) == ("n/a", None)


# ------------------------------------------------------------------ B1


def test_insert_abstain_is_deterministic_and_does_not_touch_the_label(subject):
    source = record("panel_choice", {subject["name"]: option(subject)}, subject["name"],
                    state_of(tuple(subject["usable_pos"])), subject["cl_id"])
    first, note = armb.insert_abstain(source, seed=7)
    again, _ = armb.insert_abstain(source, seed=7)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert note["inserted"] and note["abstain_key"] in armb.ABSTAIN_KEYS
    question = first["questions"]["q"]
    assert question["label"] == subject["name"]
    assert set(question["criteria"]) == {subject["name"], note["abstain_key"]}
    assert set(source["questions"]["q"]["criteria"]) == {subject["name"]}  # the source is not mutated


def test_insert_abstain_leaves_an_existing_abstain_option_alone(subject):
    criteria = {subject["name"]: option(subject), NONE_OPTIONS[0][0]: NONE_OPTIONS[0][1]}
    out, note = armb.insert_abstain(
        record("panel_choice", criteria, subject["name"], state_of(("CD3D",)), subject["cl_id"]), seed=1)
    assert note["inserted"] is False and note["abstain_key"] == NONE_OPTIONS[0][0]
    assert list(out["questions"]["q"]["criteria"]) == list(criteria)


def test_the_inserted_key_is_one_the_list_does_not_already_carry():
    """The draw is over the wordings this option list does not carry, not over all of them."""
    import random

    for taken in (NONE_OPTIONS[0][0], NONE_OPTIONS[len(NONE_OPTIONS) // 2][0]):
        criteria = {"some type": "prose", taken: "already here"}
        for seed in ("a", "b", "c", "d"):
            key, wording = armb.choose_abstain_option(criteria, random.Random(seed))
            assert key != taken and key in armb.ABSTAIN_KEYS
            assert wording == dict(NONE_OPTIONS)[key]


def test_every_abstain_key_is_available_when_the_list_carries_none():
    """A list with no abstain option can be given any of the wordings."""
    keys = {armb.choose_abstain_option({}, rng).__getitem__(0)
            for rng in (__import__("random").Random(str(i)) for i in range(50))}
    assert keys == armb.ABSTAIN_KEYS


def test_arm_b_record_stamps_the_rule_it_checked(subject, other, index):
    out, note = armb.arm_b_record(
        record("panel_miss", {other["name"]: option(other), **abstain_at()}, NONE_OPTIONS[0][0],
               state_of(tuple(subject["usable_pos"])), subject["cl_id"]),
        index, seed=0)
    assert note["label_checked"] is True and note["side"] == "withheld"
    assert out["_meta"][armb.META_KEY]["target"] == NONE_OPTIONS[0][0]


def test_arm_b_record_refuses_a_label_the_rule_does_not_answer(subject, other, index):
    """Arm B does not re-label: a row the rule disagrees with is a build failure, named."""
    criteria = {subject["name"]: option(subject), other["name"]: option(other)}
    liar = record("panel_choice", criteria, other["name"],
                  state_of(tuple(subject["usable_pos"])), subject["cl_id"], record_id="unit/liar")
    with pytest.raises(ValueError) as raised:
        armb.arm_b_record(liar, index, seed=0)
    assert "unit/liar" in str(raised.value)
    assert "does not re-label" in str(raised.value)


def test_choice_question_refuses_two_questions(subject):
    with pytest.raises(ValueError):
        armb.choice_question({"questions": {"a": {"type": "choice"}, "b": {"type": "choice"}}})


def test_abstain_key_refuses_two_abstain_options():
    """Wordings repeat across keys (13 wordings, 6 keys), so the check is on keys.

    Two abstain keys on one list is a *training* state, not a corpus one: the frozen corpus
    holds at most one on every choice question (4,434 train and 70,010 test questions, counted
    in ``scratch/armB/render-shape.py``), while ``kev.data.augment`` puts a second none wording
    on 526 of the 4,434 augmented rows (11.86%) — its ``p_none`` and ``p_none_distract``
    branches both add a key the list does not already carry, and neither removes an existing
    one. So the singular reader refuses rather than choosing, and names the plural one.
    """
    distinct = [option for option in NONE_OPTIONS if option[0] != NONE_OPTIONS[0][0]]
    criteria = {NONE_OPTIONS[0][0]: NONE_OPTIONS[0][1], distinct[0][0]: distinct[0][1]}
    assert len(distinct) and len(criteria) == 2
    assert armb.abstain_keys(criteria) == (NONE_OPTIONS[0][0], distinct[0][0])  # both, in option order
    with pytest.raises(ValueError) as raised:
        armb.abstain_key(criteria)
    assert "abstain_keys" in str(raised.value) and "augmented training row" in str(raised.value)


def test_abstain_keys_reads_every_abstain_option_on_a_rendered_row():
    """The plural reader is what the trainer's stream needs: it never refuses, it reports."""
    criteria = {"a type": "prose", NONE_OPTIONS[0][0]: NONE_OPTIONS[0][1],
                NONE_OPTIONS[-1][0]: NONE_OPTIONS[-1][1]}
    assert armb.abstain_keys(criteria) == (NONE_OPTIONS[0][0], NONE_OPTIONS[-1][0])
    assert armb.abstain_keys({"a type": "prose"}) == ()
    assert armb.abstain_key({"a type": "prose"}) is None


# ------------------------------------------------------------------ the build


def mini_corpus(root: Path, subject: dict, other: dict) -> Path:
    """A two-partition corpus whose labels are the evidence rule's, written into ``root``."""
    supported = record("panel_choice", {subject["name"]: option(subject), other["name"]: option(other)},
                       subject["name"], state_of(tuple(subject["usable_pos"])), subject["cl_id"],
                       record_id="train/0")
    withheld = record("panel_miss", {other["name"]: option(other), **abstain_at()}, NONE_OPTIONS[0][0],
                      state_of(tuple(subject["usable_pos"])), subject["cl_id"], record_id="train/1")
    scored = record("panel_choice", {subject["name"]: option(subject), other["name"]: option(other)},
                    subject["name"], state_of(tuple(subject["usable_pos"])), subject["cl_id"],
                    record_id="test/0")

    files: dict[str, dict] = {}
    root.mkdir(parents=True, exist_ok=True)
    for name, records in (("train.jsonl", [supported, withheld]), ("test.jsonl", [scored])):
        body = "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in records)
        (root / name).write_text(body, encoding="utf-8")
        files[name] = {"sha256": text_digest(body), "bytes": len(body.encode()), "kind": "data",
                       "records": len(records), "questions": len(records)}
    (root / "manifest.json").write_text(json.dumps({
        "name": "unit", "version": "v2", "seed": 11, "config_digest": "unit",
        "artifacts": {"version": "v2"}, "files": files, "splits": ["train", "test"],
    }, ensure_ascii=False), encoding="utf-8")
    return root


def test_build_rewrites_only_the_named_partition(tmp_path, subject, other):
    source = mini_corpus(tmp_path / "corpus", subject, other)
    manifest = armb.build(source, tmp_path / "armB", splits=("train",))

    assert manifest["armB"]["transformed_splits"] == ["train"]
    assert manifest["armB"]["unchanged_splits"] == ["test"]
    assert (tmp_path / "armB" / "test.jsonl").read_bytes() == (source / "test.jsonl").read_bytes()
    assert (tmp_path / "armB" / "train.jsonl").read_bytes() != (source / "train.jsonl").read_bytes()
    assert manifest["armB"]["labels_changed"] == 0
    assert manifest["counts"]["choice_rows_checked"] == 2
    assert manifest["counts"]["abstain_options_inserted"] == 1  # the panel_miss row already offered one
    assert manifest["armB"]["per_stratum"]["train"]["unit/panel_choice/permuted"] == {
        "rows": 1, "withheld": 0, "supported": 1, "abstain_offered": 1}
    assert manifest["armB"]["per_stratum"]["train"]["unit/panel_miss/permuted"] == {
        "rows": 1, "withheld": 1, "supported": 0, "abstain_offered": 1}


def test_build_is_reproducible(tmp_path, subject, other):
    source = mini_corpus(tmp_path / "corpus", subject, other)
    first = armb.build(source, tmp_path / "a", splits=("train",))
    second = armb.build(source, tmp_path / "b", splits=("train",))
    for name in ("train.jsonl", "test.jsonl"):
        assert ((tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()), name
    assert first["files"]["train.jsonl"]["sha256"] == second["files"]["train.jsonl"]["sha256"]


def test_build_refuses_a_corpus_whose_labels_are_not_the_rule(tmp_path, subject, other):
    source = mini_corpus(tmp_path / "corpus", subject, other)
    body = (source / "train.jsonl").read_text(encoding="utf-8").replace(
        f'"label": {json.dumps(subject["name"])}', f'"label": {json.dumps(other["name"])}', 1)
    (source / "train.jsonl").write_text(body, encoding="utf-8")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["train.jsonl"]["sha256"] = text_digest(body)
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        armb.build(source, tmp_path / "armB", splits=("train",))
    assert "different corpus" in str(raised.value)


def test_build_refuses_to_overwrite_a_partition_whose_bytes_changed(tmp_path, subject, other):
    source = mini_corpus(tmp_path / "corpus", subject, other)
    armb.build(source, tmp_path / "armB", splits=("train",))
    (tmp_path / "armB" / "test.jsonl").write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError):
        armb.build(source, tmp_path / "armB", splits=("train",))
    armb.build(source, tmp_path / "armB", splits=("train",), force=True)


def test_build_verifies_the_copied_partition_against_the_source_digest(tmp_path, subject, other):
    """A copy that is not the source's bytes would make an arm-B score incomparable."""
    source = mini_corpus(tmp_path / "corpus", subject, other)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["test.jsonl"]["sha256"] = hashlib.sha256(b"not the corpus").hexdigest()
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        armb.build(source, tmp_path / "armB", splits=("train",))
    assert "does not match the source digest" in str(raised.value)


# ------------------------------------------------------------------ the real corpus, if present


@pytest.mark.skipif(not (ROOT / "runs" / "real-v2" / "manifest.json").exists(),
                    reason="the real corpus is not in this checkout")
def test_the_real_corpus_labels_are_the_evidence_rules(tmp_path):
    """The premise arm B rests on, checked against the frozen partition itself.

    This is the slow one — it reads and rule-checks all 5,592 training records — and it is
    the one that must not be silently skipped if it fails, because a corpus whose labels are
    not the evidence rule's would make arm B a different experiment.

    It also holds the *producer-side* invariant the guard in :func:`armb.abstain_key` exists
    for: a corpus choice question carries at most one abstain option. Two would mean the
    builder double-added one, and the strict reader is what would catch it, so the count is
    asserted here where it can see the real partition.
    """
    index = armb.NameIndex.load("v2")
    source = ROOT / "runs" / "real-v2"
    counts: dict[str, int] = {}
    offered: dict[int, int] = {}
    with (source / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            answer = armb.evidence_answer(payload, index)
            if answer["side"] == "n/a":
                continue
            _, question = armb.choice_question(payload)
            assert question["label"] == answer["target"], payload["_meta"]["id"]
            counts[answer["side"]] = counts.get(answer["side"], 0) + 1
            present = armb.abstain_keys(question["criteria"])
            offered[len(present)] = offered.get(len(present), 0) + 1
    assert counts == {"withheld": 2923, "supported": 1511}
    assert offered == {0: 872, 1: 3562}  # never two: the builder adds at most one
