"""The suite builder: kev must be able to load what this writes, byte for byte.

The contract is kev's, so the strongest test here is not an assertion about the
files this module writes — it is ``kev.suite.load_split`` reading them back and
verifying the digests and counts itself, exactly as a training run would.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from sckevtype.contracts.freeze import file_digest, read_json
from sckevtype.train.suite import build_suite

PARTITIONS = ("train", "calibration", "development", "test")


@pytest.fixture
def suite(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> Path:
    """A freshly built suite over the frozen smoke corpus."""
    out = tmp_path / "suite"
    build_suite(corpus_dir, out, tokenizer=stand_in_tokenizer)
    return out


# --------------------------------------------------------------------------- kev round-trip


def test_kev_loads_every_partition_it_finds(corpus_dir: Path, suite: Path, kev_installed: None) -> None:
    import kev.suite

    manifest = read_json(suite / "manifest.json")
    for split in PARTITIONS:
        if f"{split}.jsonl" not in manifest["files"]:
            continue
        records = kev.suite.load_split(suite, split, allow_test=True)
        assert len(records) == manifest["files"][f"{split}.jsonl"]["records"]
        assert all(record["questions"] for record in records)


def test_kev_validates_the_training_partition_against_the_suite_policy(suite: Path, kev_installed: None) -> None:
    """``validate_training`` enforces the trainable/eval-only split this manifest declares."""
    import kev.suite

    manifest = read_json(suite / "manifest.json")
    kev.suite.validate_training(kev.suite.load_split(suite, "train"), manifest)


def test_every_record_is_a_request_kev_can_materialize(suite: Path, kev_installed: None) -> None:
    import kev.suite
    from kev.data import materialize

    for record in kev.suite.load_split(suite, "train"):
        materialized = materialize(record)
        assert materialized["questions"] and materialized["state"]


def test_split_metadata_survives_the_round_trip(suite: Path, kev_installed: None) -> None:
    """Training keys its augmentation seeds off ``id``, which must be unique across partitions."""
    import kev.suite

    seen = [record["_meta"]["id"] for split in PARTITIONS
            for record in kev.suite.load_split(suite, split, allow_test=True)]
    assert len(seen) == len(set(seen))


# --------------------------------------------------------------------------- the manifest


def test_every_manifest_digest_matches_the_file_it_describes(corpus_dir: Path, suite: Path) -> None:
    manifest = read_json(suite / "manifest.json")
    for name, entry in manifest["files"].items():
        path = suite / name
        assert file_digest(path) == entry["sha256"], name
        assert path.stat().st_size == entry["bytes"], name
        assert len(path.read_text(encoding="utf-8").splitlines()) == entry["records"], name


def test_the_manifest_records_the_corpus_it_came_from(corpus_dir: Path, suite: Path) -> None:
    manifest = read_json(suite / "manifest.json")
    assert manifest["corpus"]["manifest_sha256"] == file_digest(corpus_dir / "manifest.json")
    assert manifest["corpus"]["name"]
    assert manifest["corpus"]["version"]


def test_counts_add_up(suite: Path) -> None:
    manifest = read_json(suite / "manifest.json")
    assert sum(manifest["counts"]["per_split"].values()) == manifest["counts"]["records"]
    assert sum(manifest["counts"]["per_family"].values()) == manifest["counts"]["records"]
    assert manifest["counts"]["questions"] == sum(manifest["counts"]["per_question_type"].values())
    assert sum(manifest["counts"]["per_question_type"].values()) == sum(
        entry["questions"] for entry in manifest["files"].values()
    )


def test_the_training_partition_declares_its_sources(suite: Path) -> None:
    """A suite whose trainable sources were guessed would train on the wrong data."""
    manifest = read_json(suite / "manifest.json")
    sources = {json.loads(line)["_meta"]["source"]
               for line in (suite / "train.jsonl").read_text(encoding="utf-8").splitlines() if line}
    assert manifest["trainable_sources"] == sorted(sources)
    assert not set(manifest["trainable_sources"]) & set(manifest["eval_only_sources"])


# --------------------------------------------------------------------------- the state budget


def test_the_recorded_budget_is_measured_from_all_three_of_kevs_limits(
    suite: Path, kev_installed: None, stand_in_tokenizer
) -> None:
    """The budget is the smallest one that covers the state, the row *and* the packed record.

    kev derives its row and packed limits from ``--max_state`` alone
    (``kev.model.training_context``), so a suite recording only the state axis records
    a number its own records cannot be trained under. Recomputed here from the
    manifest's observations, so a manifest that hard-coded a value disagrees with
    itself; one rung down must fall short, or it is not the smallest.
    """
    from kev.model import MAX_STATE, training_context

    from sckevtype.contracts import budget

    manifest = read_json(suite / "manifest.json")
    context = manifest["context"]
    lengths = [budget.state_tokens(stand_in_tokenizer, json.loads(line)["state"])
               for line in (suite / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert context["observed"]["per_split"]["train"]["max"] == max(lengths)

    observed = budget.observed_limits(context["observed"])
    assert observed is not None
    assert observed["max_state"] == max(context["observed"]["per_split"][split]["max"] for split in manifest["splits"])
    limits = training_context(context["max_state"])
    assert context["max_state"] >= observed["max_state"]
    assert limits["max_branch"] >= observed["max_row"]
    assert limits["max_packed"] >= observed["max_packed"]
    assert context["max_state"] % budget.BUDGET_UNIT == 0
    assert context["source"] == "measured"
    assert context["observed"]["over_budget"] == 0
    assert context["observed"]["truncated"] == 0
    assert context["max_state"] == budget.required_budget(**observed)
    if context["max_state"] > MAX_STATE:
        smaller = training_context(context["max_state"] - budget.BUDGET_UNIT)
        assert (smaller["max_state"] < observed["max_state"] or smaller["max_branch"] < observed["max_row"]
                or smaller["max_packed"] < observed["max_packed"])


def test_the_recorded_context_is_kevs_own_for_that_budget(suite: Path, kev_installed: None) -> None:
    """Whatever is recorded, kev must agree: these are the limits a run will hand the encoder.

    One number, not two. kev derives all three of its limits from ``--max_state``, so
    the manifest records *that* derivation rather than a second, unenforced set of
    numbers beside it.
    """
    from kev.model import training_context

    context = read_json(suite / "manifest.json")["context"]
    assert training_context(context["max_state"]) == {
        "max_state": context["max_state"],
        "max_branch": context["max_branch"],
        "max_packed": context["max_packed"],
    }


def test_the_row_and_packed_axes_are_measured_over_the_training_partition(
    suite: Path, kev_installed: None, stand_in_tokenizer
) -> None:
    """The other two limits are measured from the records kev's trainer will encode.

    Recomputed here through kev's own encoder from the suite's bytes, so a manifest
    that recorded a guess instead of a measurement disagrees with itself. The
    partition is the training one because that is what the trainer encodes — a frozen
    suite's records are not filtered — so a measurement over every partition would
    record a budget for records no run ever hands the encoder.

    Encoding at the *recorded* budget rather than at the one the states were written
    to fit is the same measurement: ``kev.model.encode`` truncates a state to
    ``max_state - 1`` tokens and every state is already within both, so nothing is cut.
    """
    from kev.suite import load_split

    from sckevtype.contracts import budget

    context = read_json(suite / "manifest.json")["context"]
    rows, packed = budget.encoded_lengths(load_split(suite, "train"), stand_in_tokenizer,
                                          max_state=context["max_state"])
    assert context["observed"]["max_row"] == max(rows)
    assert context["observed"]["max_packed"] == max(packed)


def test_every_observed_split_is_recorded(suite: Path) -> None:
    manifest = read_json(suite / "manifest.json")
    assert set(manifest["context"]["observed"]["per_split"]) == set(manifest["splits"])
    for observed in manifest["context"]["observed"]["per_split"].values():
        assert observed["min"] <= observed["median"] <= observed["p95"] <= observed["max"]
        assert observed["n"] > 0


def test_a_budget_that_cannot_hold_the_records_fails_the_build(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> None:
    """The failure the cluster paid for, moved to build time and named in full.

    The message has to carry what a rebuild needs: how many records are over,
    how long the longest one is, the budget that was recorded, and the budget the
    data actually requires.
    """
    from sckevtype.contracts import budget

    corpus = _long_state_corpus(corpus_dir, tmp_path, "long", genes=600)
    longest = 3 * 600 - 1 + 1  # "GENEi i" twice per gene, "|" between them, plus kev's delimiter
    with pytest.raises(ValueError) as raised:
        build_suite(corpus, tmp_path / "out", tokenizer=stand_in_tokenizer, max_state=384)
    message = str(raised.value)
    assert "1 of 1 records exceed the recorded max_state 384" in message
    assert f"longest state {longest} tokens" in message
    assert f"required budget {budget.budget_for(longest)}" in message
    assert not (tmp_path / "out" / "train.jsonl").exists()


def test_truncate_cuts_the_states_and_records_the_count(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> None:
    """``--truncate`` performs the cut it reports: the bytes are what kev would encode."""
    from sckevtype.contracts import budget

    corpus = _long_state_corpus(corpus_dir, tmp_path, "long", genes=600)
    manifest = build_suite(corpus, tmp_path / "out", tokenizer=stand_in_tokenizer, max_state=384, truncate=True)
    context = manifest["context"]
    assert context["truncate"] is True
    assert context["source"] == "pinned"
    assert context["observed"]["truncated"] == 1
    assert context["observed"]["over_budget"] == 0
    # `observed` describes the suite as written, so the longest state is the budget the
    # cut brought it to and nothing is over it.
    assert context["observed"]["max_state"] <= context["max_state"]
    for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines():
        assert budget.state_tokens(stand_in_tokenizer, json.loads(line)["state"]) <= context["max_state"]


def test_a_pinned_budget_outside_kevs_range_is_refused(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> None:
    """kev refuses anything outside [384, 7552]; so does the builder, before writing."""
    from kev.model import MAX_STATE, MAX_TRAIN_STATE

    corpus = _long_state_corpus(corpus_dir, tmp_path, "short", genes=20)
    for pinned in (MAX_STATE - 1, MAX_TRAIN_STATE + 1):
        with pytest.raises(ValueError, match=r"outside kev's range"):
            build_suite(corpus, tmp_path / str(pinned), tokenizer=stand_in_tokenizer, max_state=pinned)
    assert build_suite(corpus, tmp_path / "floor", tokenizer=stand_in_tokenizer,
                       max_state=MAX_STATE)["context"]["max_state"] == MAX_STATE


def test_a_required_budget_above_kevs_ceiling_names_it(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> None:
    """A state too long to serve cannot be trained on either, whatever the suite records."""
    from kev.model import MAX_TRAIN_STATE

    corpus = _long_state_corpus(corpus_dir, tmp_path, "huge", genes=4000)
    with pytest.raises(ValueError, match=f"exceeds kev's MAX_TRAIN_STATE \\({MAX_TRAIN_STATE}\\)"):
        build_suite(corpus, tmp_path / "out", tokenizer=stand_in_tokenizer)


def test_the_budget_is_recomputable_from_the_manifest_alone(suite: Path) -> None:
    """A reader re-derives the recorded budget from ``context.observed``, without re-tokenizing."""
    from sckevtype.contracts import budget

    context = read_json(suite / "manifest.json")["context"]
    observed = context["observed"]
    assert set(observed) >= {"max_state", "max_row", "max_packed"}
    assert budget.required_budget(observed["max_state"], observed["max_row"],
                                  observed["max_packed"]) == context["max_state"]


def test_a_record_too_long_to_pack_fails_the_build(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> None:
    """A budget that cannot hold the packed axis is a suite that cannot be trained, so it is not written.

    The state axis is not where this one dies: the states fit the pin comfortably and it
    is the questions — which kev packs into the same limit as the state — that the
    budget cannot afford. Recording it anyway is what the third live failure paid for.
    """
    corpus = _long_packed_corpus(corpus_dir, tmp_path, "packed", questions=20)
    with pytest.raises(ValueError) as raised:
        build_suite(corpus, tmp_path / "out", tokenizer=stand_in_tokenizer, max_state=384)
    assert "cannot afford this data" in str(raised.value)
    assert not (tmp_path / "out" / "train.jsonl").exists()


def test_a_packed_record_over_the_ceiling_names_kevs_ceiling_and_the_count(
    corpus_dir: Path, tmp_path: Path, stand_in_tokenizer
) -> None:
    """Requirement: name the budget the data needs, the ceiling, and the records that forced it."""
    from kev.model import MAX_TRAIN_STATE

    corpus = _long_packed_corpus(corpus_dir, tmp_path, "packed", questions=400)
    with pytest.raises(ValueError) as raised:
        build_suite(corpus, tmp_path / "out", tokenizer=stand_in_tokenizer)
    message = str(raised.value)
    assert f"exceeds kev's MAX_TRAIN_STATE ({MAX_TRAIN_STATE})" in message
    assert "1 of 1 training records cannot be trained on at any budget kev serves" in message
    assert not (tmp_path / "out" / "train.jsonl").exists()


def test_a_real_suite_is_measured_with_the_pinned_tokenizer(corpus_dir: Path, tmp_path: Path, pinned_tokenizer) -> None:
    """The end-to-end claim: a suite built with no tokenizer named is measured with the registry's."""
    from sckevtype.contracts import budget

    manifest = build_suite(corpus_dir, tmp_path / "pinned")
    context = manifest["context"]
    assert context["tokenizer"]["id"] == "Qwen/Qwen3.5-0.8B-Base"
    assert context["tokenizer"]["revision"]  # a suite records the commit, never a moving target
    assert context["max_state"] % budget.BUDGET_UNIT == 0
    assert context["max_state"] >= context["observed"]["max_state"]
    assert context["observed"]["over_budget"] == 0
    assert context["observed"]["truncated"] == 0


# --------------------------------------------------------------------------- determinism


def test_rebuilding_into_another_directory_is_byte_identical(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    build_suite(corpus_dir, first, tokenizer=stand_in_tokenizer)
    build_suite(corpus_dir, second, tokenizer=stand_in_tokenizer)
    for path in sorted(first.iterdir()):
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name


def test_rebuilding_over_itself_is_not_an_error(corpus_dir: Path, suite: Path, stand_in_tokenizer) -> None:
    """An identical rebuild is a no-op; only *changed* bytes are refused."""
    manifest = build_suite(corpus_dir, suite, tokenizer=stand_in_tokenizer)
    assert manifest["counts"]["records"]


def test_a_corpus_whose_bytes_changed_is_refused(corpus_dir: Path, tmp_path: Path) -> None:
    """A suite over a corpus that no longer matches its manifest would pin the wrong thing."""
    copy = tmp_path / "corpus"
    shutil.copytree(corpus_dir, copy)
    (copy / "train.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        build_suite(copy, tmp_path / "out")


def test_changed_suite_bytes_are_refused_without_force(corpus_dir: Path, suite: Path, stand_in_tokenizer) -> None:
    (suite / "train.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different content"):
        build_suite(corpus_dir, suite, tokenizer=stand_in_tokenizer)
    build_suite(corpus_dir, suite, tokenizer=stand_in_tokenizer, force=True)
    assert (suite / "train.jsonl").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- the test cap


#: The manifest an *uncapped* build of the smoke corpus produces, over every field that does not
#: depend on where the corpus sits on this box. ``artifacts/suites/real-v2`` — the suite the model
#: was trained on — was built by that path, and the acceptance for the cap is that its flags default
#: off: the same command without them still writes this. Re-pin only if the default build was
#: *meant* to change, and say why in the commit that does it.
_UNCAPPED_MANIFEST_DIGEST = "c0eb7e0af6faf40a3280889d377bc4eeaa5e921fdaa88e1f6b4fef67e2e50260"

#: A cap small enough that the smoke corpus's thin strata need the floor, and a corpus small enough
#: to build several times in a test.
_CAP, _CAP_FLOOR = 100, 20


def _manifest_digest(manifest: dict) -> str:
    """The manifest's digest, with the corpus's absolute path dropped (it is this box's, not ours)."""
    stable = {**manifest, "corpus": {k: v for k, v in manifest["corpus"].items() if k != "path"}}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def test_the_uncapped_build_is_the_manifest_real_v2_was_built_from(suite: Path) -> None:
    """The flags default off, and off means the bytes this module has always written.

    ``real-v2`` was built by this path with no flags; a cap that leaked into an uncapped build would
    rewrite the manifest of the suite the model was trained on.
    """
    manifest = read_json(suite / "manifest.json")
    assert "cap" not in manifest
    assert _manifest_digest(manifest) == _UNCAPPED_MANIFEST_DIGEST, (
        "the default build changed; the cap is meant to be additive. If this change is deliberate, "
        "re-pin _UNCAPPED_MANIFEST_DIGEST and say why"
    )


def test_the_cap_touches_only_the_test_partition(corpus_dir: Path, tmp_path: Path, suite: Path,
                                                 stand_in_tokenizer) -> None:
    """Calibration is what every fit reads: shrinking it would change *what* is reported, not how well."""
    capped = tmp_path / "capped"
    manifest = build_suite(corpus_dir, capped, tokenizer=stand_in_tokenizer, cap_test=_CAP, cap_floor=_CAP_FLOOR)
    whole = read_json(suite / "manifest.json")
    assert manifest["counts"]["per_split"]["test"] < whole["counts"]["per_split"]["test"]
    assert manifest["cap"]["partition"] == "test"
    for name in whole["files"]:
        if name != "test.jsonl":
            assert (capped / name).read_bytes() == (suite / name).read_bytes(), name


def test_the_cap_records_its_draw_and_rebuilds_byte_identically(corpus_dir: Path, tmp_path: Path,
                                                               stand_in_tokenizer) -> None:
    """What was scored, what was not, and the same draw every time — the manifest is the disclosure.

    The histograms are checked against the file on disk rather than against themselves: a manifest
    that describes bytes it did not write is the failure this whole module is arranged to prevent.
    """
    from sckevtype.contracts.schema import size_bin

    first, second = tmp_path / "first", tmp_path / "second"
    manifest = build_suite(corpus_dir, first, tokenizer=stand_in_tokenizer, cap_test=_CAP, cap_floor=_CAP_FLOOR)
    assert build_suite(corpus_dir, second, tokenizer=stand_in_tokenizer,
                       cap_test=_CAP, cap_floor=_CAP_FLOOR) == manifest
    for name in manifest["files"]:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name

    cap = manifest["cap"]
    corpus_seed = read_json(corpus_dir / "manifest.json").get("seed")
    assert cap["requested"] == _CAP and cap["floor"] == _CAP_FLOOR and cap["seed"] == corpus_seed
    assert cap["available"] == len((corpus_dir / "test.jsonl").read_text(encoding="utf-8").splitlines())
    assert cap["achieved"] == manifest["counts"]["per_split"]["test"] == manifest["files"]["test.jsonl"]["records"]
    assert cap["achieved"] == sum(stratum["taken"] for stratum in cap["per_stratum"].values())
    assert cap["strata_used"] == len(cap["per_stratum"])
    # Every stratum is allocated its share, and a thin one is raised to the floor rather than left
    # too small to report — the two halves of the rule, both visible in the block.
    for stratum, counts in cap["per_stratum"].items():
        assert 0 < counts["taken"] <= counts["available"], stratum
        assert counts["taken"] >= min(counts["available"], _CAP_FLOOR), stratum
    assert any(counts["taken"] > round(counts["available"] * _CAP / cap["available"])
               for counts in cap["per_stratum"].values()), "no stratum needed the floor; the cap is too small to test it"

    written = Counter(size_bin(len(question["criteria"]))
                      for line in (first / "test.jsonl").read_text(encoding="utf-8").splitlines()
                      for question in json.loads(line)["questions"].values())
    assert cap["panel_size_bins"]
    for name, rows in cap["panel_size_bins"].items():
        assert rows["capped"] == written[name], name
        assert rows["available"] >= rows["capped"]
        assert rows["capped"] >= _CAP_FLOOR, f"{name} carries less than the floor and the report would summarise it anyway"


def test_a_cap_that_would_starve_a_panel_size_bin_is_refused(corpus_dir: Path, tmp_path: Path,
                                                            stand_in_tokenizer) -> None:
    """A bin whose ECE rests on a handful of rows is a number, not an estimate — so no suite is written."""
    out = tmp_path / "starved"
    with pytest.raises(ValueError) as raised:
        build_suite(corpus_dir, out, tokenizer=stand_in_tokenizer, cap_test=_CAP, cap_floor=1000)
    message = str(raised.value)
    assert "below the 1000-row floor" in message
    # The message names the bins that would be starved, and the levers that would fix it.
    assert "2-4" in message and "5-9" in message and "--cap-test" in message and "--cap-floor" in message
    assert not out.exists()


def test_a_cap_with_no_floor_is_refused(corpus_dir: Path, tmp_path: Path, stand_in_tokenizer) -> None:
    """Without a floor a stratum can be thinned to nothing, which is the cap this one exists to stop."""
    with pytest.raises(ValueError, match="floor must be at least 1"):
        build_suite(corpus_dir, tmp_path / "floorless", tokenizer=stand_in_tokenizer, cap_test=_CAP, cap_floor=0)


def test_the_cli_flags_reach_the_builder(corpus_dir: Path, tmp_path: Path, pinned_tokenizer) -> None:
    """The cluster scripts call the CLI, so the flag names are part of the contract with them."""
    from sckevtype.train.suite import main

    out = tmp_path / "cli"
    assert main(["--corpus", str(corpus_dir), "--out", str(out), "--cap-test", "100", "--cap-floor", "20"]) == 0
    assert read_json(out / "manifest.json")["cap"]["floor"] == 20
    with pytest.raises(SystemExit):
        main(["--corpus", str(corpus_dir), "--out", str(tmp_path / "bad"), "--cap-test", "100", "--cap-floor", "0"])


def _corpus(tmp_path: Path, name: str, lines: dict[str, str]) -> Path:
    """A corpus of one-line partitions, so a malformed case can be built in isolation."""
    corpus = tmp_path / name
    corpus.mkdir()
    files = {}
    for partition, line in lines.items():
        path = corpus / partition
        path.write_text(line, encoding="utf-8")
        files[partition] = {"sha256": file_digest(path), "records": 1}
    (corpus / "manifest.json").write_text(json.dumps({"name": name, "files": files}), encoding="utf-8")
    return corpus


def _long_packed_corpus(corpus_dir: Path, tmp_path: Path, name: str, questions: int) -> Path:
    """A one-record corpus whose *packed* record is long while its state stays short.

    The record is the smoke corpus's own first record with its question repeated under
    fresh keys. kev packs a record as its state plus every question, so this grows the
    packed axis without growing the state one — which is how a budget derived from the
    state alone comes to look affordable.
    """
    record = json.loads((corpus_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    question = next(iter(record["questions"].values()))
    record["questions"] = {f"q{index}": question for index in range(questions)}
    return _corpus(tmp_path, name, {"train.jsonl": json.dumps(record, ensure_ascii=False) + "\n"})


def _long_state_corpus(corpus_dir: Path, tmp_path: Path, name: str, genes: int) -> Path:
    """A one-record corpus whose state is ``genes`` symbols long, so a budget can be made to fail.

    The record is the smoke corpus's own first record with its state replaced:
    everything else — a real question with real criteria — is what makes it a
    request kev accepts, and the builder validates before it measures.
    """
    record = json.loads((corpus_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    record["state"] = " | ".join(f"GENE{index} {index}" for index in range(genes))
    return _corpus(tmp_path, name, {"train.jsonl": json.dumps(record, ensure_ascii=False) + "\n"})


def test_a_corpus_with_no_training_partition_is_refused(tmp_path: Path) -> None:
    """A suite with nothing to train on is not a suite."""
    corpus = _corpus(tmp_path, "empty", {})
    with pytest.raises(ValueError, match="no train.jsonl"):
        build_suite(corpus, tmp_path / "out")


def test_a_record_kev_cannot_read_is_refused_with_the_file_named(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, "broken", {"train.jsonl": '{"state": "A 1", "questions": {"label": 5}}\n'})
    with pytest.raises(ValueError, match="kev cannot read this partition"):
        build_suite(corpus, tmp_path / "out")


def test_a_record_kev_cannot_materialize_is_refused(tmp_path: Path) -> None:
    """The validator runs before anything is written, so a bad suite leaves no half-file behind."""
    record = '{"state": "A 1", "questions": {"label": {"type": "not_a_type", "label": 0}}}\n'
    corpus = _corpus(tmp_path, "unusable", {"train.jsonl": record})
    with pytest.raises(ValueError, match="not a kev request"):
        build_suite(corpus, tmp_path / "out")
    assert not (tmp_path / "out" / "train.jsonl").exists()
