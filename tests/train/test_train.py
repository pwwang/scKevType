"""The training CLI: the flags it mirrors, the tokens it counts, the record it leaves.

The run itself needs a card, so what is tested here is everything that decides
what a run *means*: that the CLI hands the flags back to kev's own parser, that
the token count is the one the backbone actually pays for, and that the
provenance record carries what reproducing a run needs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sckevtype.train.model import BACKBONE, Trainables
from sckevtype.train.train import forward_tokens, main, parse_args, provenance_record

REQUIRED_KEYS = {
    "tool", "repos", "suite", "base", "trainable", "hyperparameters", "environment",
    "determinism", "epochs", "optimizer_steps", "peak_device_bytes", "stopped_early", "peak_rss_bytes",
}


# --------------------------------------------------------------------------- the tokens


class StubModel:
    """Only the two things ``forward_tokens`` asks a model: which form, and how to split."""

    def __init__(self, rows: bool) -> None:
        self._rows = rows

    def rows_form(self, encs) -> bool:
        return self._rows


def test_a_packed_encoding_counts_its_packed_length() -> None:
    assert forward_tokens(StubModel(rows=False), {"ids": [0] * 7}) == 7


def test_a_row_form_encoding_counts_the_state_once_per_question() -> None:
    """The hybrid backbone repeats the state for every question; not counting that is undercounting."""
    pytest.importorskip("kev.model", reason="rows_of is kev's splitter")
    enc = {
        "ids": list(range(10)),
        "pos": list(range(10)),
        "seg": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
        "decide_idx": [5, 9],
        "opt_idx": [[3, 4], [6, 7]],
    }
    state, rows = 3, 3 + 4
    assert len(enc["ids"]) == 10
    assert forward_tokens(StubModel(rows=True), enc) == state + state + rows


# --------------------------------------------------------------------------- the flags


@pytest.fixture
def kev_train():
    return pytest.importorskip("kev.train", reason="the flags are kev's; requires the train extra")


@pytest.fixture
def budgeted_suite(tmp_path: Path) -> Path:
    """A suite whose manifest records a budget — the only thing ``kev_namespace`` reads from one."""
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text(json.dumps({"name": "smoke", "context": {"max_state": 1664}}), encoding="utf-8")
    return suite


def test_the_cli_accepts_the_flags_the_spec_names(kev_train) -> None:
    args = parse_args([
        "--suite", "s", "--base", "b", "--base_revision", "r", "--epochs", "2", "--lr", "1e-4",
        "--batch", "3", "--accum", "4", "--dtype", "fp32", "--checkpointing", "1", "--out", "o",
        "--steps", "30", "--seed", "7",
    ])
    assert (args.steps, args.seed, args.batch, args.accum) == (30, 7, 3, 4)


def test_bf16_without_cuda_is_refused(kev_train) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--suite", "s", "--base", "b", "--dtype", "bf16", "--device", "cpu", "--out", "o"])


def test_a_negative_step_budget_is_refused(kev_train) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--suite", "s", "--base", "b", "--steps", "-1", "--out", "o"])


def test_the_kev_namespace_carries_kevs_own_defaults(kev_train, budgeted_suite: Path) -> None:
    """The flags this CLI does not expose must still arrive at the trainer with kev's value."""
    from sckevtype.train.train import kev_namespace

    ns = kev_namespace(parse_args(["--suite", str(budgeted_suite), "--base", "b", "--out", "o"]), "o")
    assert ns.base == "b" and ns.suite == str(budgeted_suite)
    assert ns.lora > 0 and ns.weight_decay >= 0 and ns.label_smoothing == 0
    assert ns.p_none > 0 and ns.max_state > 0        # the augmentation and context knobs exist with kev's values


def test_the_pair_rate_reaches_kev_and_defaults_to_kevs_own(kev_train, budgeted_suite: Path) -> None:
    """The rejected alternative has to stay reachable, or its measurement cannot be repeated.

    Arm B trains at 0.0 — kev's own default — because the pair was measured on this corpus and
    rejected: it duplicates the source row on every record that already answers the abstain
    option, so the counter-example share falls 30.3% -> 21.5% while the epoch grows 2.6x
    (``scratch/armB/measure-stream.py``). The flag exists so that number can be re-derived.
    """
    from sckevtype.train.train import kev_namespace

    off = kev_namespace(parse_args(["--suite", str(budgeted_suite), "--base", "b", "--out", "o"]), "o")
    assert off.p_none_pair == 0.0          # kev's default, and arm B's value
    on = kev_namespace(
        parse_args(["--suite", str(budgeted_suite), "--base", "b", "--p_none_pair", "1.0", "--out", "o"]), "o")
    assert on.p_none_pair == 1.0           # the measured-and-rejected alternative, still runnable


def test_a_pair_rate_outside_kevs_bound_is_refused(kev_train) -> None:
    """kev's own search bounds this knob to [0, 1]; a rate outside it is a typo, not a run."""
    for value in ("-0.1", "1.5"):
        with pytest.raises(SystemExit):
            parse_args(["--suite", "s", "--base", "b", "--p_none_pair", value, "--out", "o"])


def test_the_recorded_hyperparameters_are_json_safe(kev_train, budgeted_suite: Path) -> None:
    """``training_config.json`` is ``vars(namespace)``; a non-serializable field would fail the run at the last step."""
    from sckevtype.train.train import kev_namespace

    ns = kev_namespace(parse_args(["--suite", str(budgeted_suite), "--base", "b", "--out", "o"]), "o")
    assert json.loads(json.dumps(vars(ns), allow_nan=False)) == vars(ns)


# --------------------------------------------------------------- the state budget, propagated


def test_kev_is_handed_the_budget_the_suite_records(kev_train, budgeted_suite: Path) -> None:
    """The failure this step exists for: kev's 384-token default is never what a run trains against."""
    from sckevtype.train.train import kev_namespace

    ns = kev_namespace(parse_args(["--suite", str(budgeted_suite), "--base", "b", "--out", "o"]), "o")
    assert ns.max_state == 1664


def test_the_number_kev_is_handed_covers_the_branch_axis(kev_train, tmp_path: Path) -> None:
    """A suite whose questions are long records a trainer budget above its state budget.

    kev derives the row and packed limits from ``--max_state`` alone, so handing it
    ``context.max_state`` would train a suite whose states fit and whose branches do
    not. ``train_max_state`` is the number that covers both, and it is what every
    caller — the trainer and ``compute`` — reads through one function.
    """
    from sckevtype.train.train import recorded_max_state

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text(
        json.dumps({"context": {"max_state": 2048, "train_max_state": 6144}}), encoding="utf-8")
    assert recorded_max_state(json.loads((suite / "manifest.json").read_text())["context"]) == 6144


def test_a_context_without_a_trainer_budget_falls_back_to_its_state_budget(kev_train) -> None:
    """An older suite records only ``max_state``; that is still a recorded number, not kev's default."""
    from sckevtype.train.train import recorded_max_state

    assert recorded_max_state({"max_state": 1664}) == 1664


def test_a_stale_suite_is_refused_by_its_own_observations(kev_train) -> None:
    """The budget a manifest records must satisfy the three limits that manifest measured.

    A suite built under the state-only rule records 2,048 beside a 7,714-token packed
    observation, which is a run that dies on its first batch after a GPU boot. Read
    from ``observed``, it dies here instead, naming the axis that is short. The nested
    shape is the one those suites actually wrote.
    """
    from sckevtype.train.train import verify_recorded_budget

    flat = {"max_state": 2048, "observed": {"max_state": 2020, "max_row": 4891, "max_packed": 7771}}
    for context in (flat, {"max_state": 2048, "observed": {"max": 2020, "row": {"max": 4891}, "packed": {"max": 7771}}}):
        with pytest.raises(ValueError, match=r"cannot hold the records it measured: the longest packed"):
            verify_recorded_budget(context, "artifacts/suites/real-v2")
    # The same observations under the budget they need are accepted, and that budget is returned.
    assert verify_recorded_budget({**flat, "max_state": 6144}, "suite") == 6144
    # A suite that records no observations at all cannot be re-derived; the measurement
    # that follows is what catches it, so the check passes it through rather than guessing.
    assert verify_recorded_budget({"max_state": 1664, "observed": {}}, "suite") == 1664


def test_a_pinned_budget_is_what_kev_is_handed(kev_train, budgeted_suite: Path) -> None:
    from sckevtype.train.train import kev_namespace

    ns = kev_namespace(parse_args(["--suite", str(budgeted_suite), "--base", "b", "--max_state", "2048", "--out", "o"]), "o")
    assert ns.max_state == 2048


def test_a_suite_that_records_no_budget_is_refused(kev_train, tmp_path: Path) -> None:
    """No path here falls back to kev's ``MAX_STATE``; a suite that cannot say what it was measured to stops the run."""
    from sckevtype.train.train import kev_namespace

    suite = tmp_path / "bare"
    suite.mkdir()
    (suite / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"records no context\.max_state"):
        kev_namespace(parse_args(["--suite", str(suite), "--base", "b", "--out", "o"]), "o")


def test_a_suite_with_no_manifest_is_refused(kev_train, tmp_path: Path) -> None:
    from sckevtype.train.train import kev_namespace

    with pytest.raises(FileNotFoundError, match="no suite manifest"):
        kev_namespace(parse_args(["--suite", str(tmp_path / "gone"), "--base", "b", "--out", "o"]), "o")


def test_the_run_stops_on_a_record_the_budget_cannot_hold(kev_train, stand_in_tokenizer) -> None:
    """kev's trainer does not filter a frozen suite's records, so this is where an over-long one is caught."""
    from kev.model import training_context

    from sckevtype.train.train import verify_state_budget

    records = [{"state": " ".join(["G"] * 500)}]
    with pytest.raises(ValueError) as raised:
        verify_state_budget(stand_in_tokenizer, records, training_context(384), "suites/real-v2")
    message = str(raised.value)
    assert "1 of 1 training records exceed the 384-token budget suites/real-v2 records" in message
    assert "longest state 501 tokens" in message
    assert "required budget 512" in message


def test_the_run_stops_on_a_record_whose_branch_the_budget_cannot_hold(kev_train, corpus_dir, stand_in_tokenizer) -> None:
    """The second axis: the state fits, the questions do not.

    kev refuses a branch longer than ``max_branch - len(state)``, and derives that
    row limit from ``--max_state`` alone — so a budget large enough for every state
    can still be too small for the questions asked of one, which is the failure
    ``branch too long: 2197 tokens with a 1585-token state`` was.
    """
    from kev.model import training_context

    from sckevtype.train.train import verify_state_budget

    record = json.loads((corpus_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    limits = training_context(384)
    limits["max_branch"] = 1  # no branch can fit beside a state
    with pytest.raises(ValueError) as raised:
        verify_state_budget(stand_in_tokenizer, [record], limits, "suites/real-v2")
    assert "1 of 1 training records exceed the 1-token row limit suites/real-v2 records" in str(raised.value)


def test_a_run_against_a_budget_that_holds_every_record_measures_them(kev_train, corpus_dir, stand_in_tokenizer) -> None:
    """The distribution it reports is the tokenizer's own count of the records' states."""
    from kev.model import training_context

    from sckevtype.contracts.budget import state_tokens
    from sckevtype.train.train import verify_state_budget

    record = json.loads((corpus_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    limits = training_context(384)
    limits["max_state"] = 100_000  # nothing is over budget, so the measurement completes
    limits["max_branch"] = limits["max_packed"] = 1_000_000
    summary = verify_state_budget(stand_in_tokenizer, [record], limits, "suites/real-v2")
    length = state_tokens(stand_in_tokenizer, record["state"])
    assert summary == {"n": 1, "min": length, "median": length, "p95": length, "max": length}


def test_an_existing_run_directory_is_never_overwritten(kev_train, tmp_path, capsys) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "keep").write_text("partial artifacts from an interrupted run", encoding="utf-8")
    assert main(["--suite", str(tmp_path), "--base", "b", "--out", str(out)]) == 1
    assert "refusing" in capsys.readouterr().err
    assert (out / "keep").exists()


def test_kev_refuses_an_existing_run_too(kev_train, budgeted_suite: Path, tmp_path: Path) -> None:
    """The refusal is kev's as well as ours: a run must not land on top of another."""
    from sckevtype.train.train import kev_namespace

    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(SystemExit):
        kev_namespace(parse_args(["--suite", str(budgeted_suite), "--base", "b", "--out", str(out)]), str(out))


# --------------------------------------------------------------------------- provenance


def test_the_provenance_record_carries_what_reproducing_a_run_needs(tmp_path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text('{"name": "smoke"}\n', encoding="utf-8")
    namespace = argparse.Namespace(suite=str(suite), seed=3, base="Qwen/Qwen3.5-0.8B-Base", lr=2e-4, batch=2)
    record = provenance_record(
        namespace=namespace, revision="dc7cdfe2ee41", manifest={
            "name": "smoke", "corpus": {"manifest_sha256": "ab" * 32},
            "files": {"train.jsonl": {"sha256": "cd" * 32}},
        },
        suite_dir=suite, trainables=Trainables(11_347_456, ("head.proj.weight",), {"adapter": 10_822_656, "head": 524_800}),
        base={"id": namespace.base, "revision": "dc7cdfe2ee41", "text_only": True},
        environment={"device": "cuda", "gpu": "NVIDIA RTX A5000", "torch": "2.8.0", "transformers": "5.17.0"},
        epochs=[{"epoch": 0, "tokens_per_second": 1234.5, "peak_device_bytes": 2**33}],
        optimizer_steps=30, peak_device_bytes=2**33, stopped_early=True,
    )
    assert REQUIRED_KEYS <= set(record)
    assert {"sckevtype", "kev"} <= set(record["repos"])
    assert record["suite"]["manifest_sha256"] and record["suite"]["partitions"]["train.jsonl"]
    assert record["base"]["revision"] == "dc7cdfe2ee41"
    assert record["environment"]["gpu"] == "NVIDIA RTX A5000"
    assert record["determinism"]["seed"] == 3
    assert record["optimizer_steps"] == 30 and record["stopped_early"] is True
    assert {"tokens_per_second", "peak_device_bytes"} <= set(record["epochs"][0])
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_the_provenance_record_pins_the_code_that_trained(tmp_path) -> None:
    """Commit ids come from the checkouts; the module digests cover the case where there is none."""
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text("{}\n", encoding="utf-8")
    record = provenance_record(
        namespace=argparse.Namespace(suite=str(suite), seed=0), revision=None, manifest={}, suite_dir=suite,
        trainables=Trainables(0, (), {BACKBONE: 0}),
        base={}, environment={}, epochs=[], optimizer_steps=0, peak_device_bytes=0, stopped_early=False,
    )
    assert set(record["repos"]["sckevtype_source"]) == {"model.py", "suite.py", "train.py", "compute.py"}
    assert all(len(digest) == 64 for digest in record["repos"]["sckevtype_source"].values())


def test_the_environment_block_records_the_libraries_a_number_is_attributed_to() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from sckevtype.train.train import _environment

    environment = _environment("cpu")
    assert environment["device"] == "cpu"
    assert {"python", "torch", "transformers", "peft"} <= set(environment)


# --------------------------------------------------------------------------- compute


def test_compute_refuses_a_model_that_pins_no_revision(tmp_path) -> None:
    from sckevtype.train.compute import load_registry

    config = tmp_path / "models.yaml"
    config.write_text("models:\n  loose:\n    id: Qwen/Qwen3.5-0.8B-Base\n    revision: null\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pins no revision"):
        load_registry(config)


def test_the_registry_has_the_base_the_smoke_run_uses() -> None:
    from sckevtype.train.compute import load_registry

    entry = load_registry()["0.8b"]
    assert entry["id"] == "Qwen/Qwen3.5-0.8B-Base"
    assert entry["revision"]


def test_the_batch_of_longest_rows_nests() -> None:
    """A larger batch must contain the smaller one's records, or the two peaks are not comparable."""
    from sckevtype.train.compute import take_batch

    class Shape:
        """``forward_tokens`` asks a model only whether it is in the row form."""

        def rows_form(self, encs):
            return False

    variants = [SimpleNamespace(enc={"ids": [0] * length}, request_id=f"r{length}") for length in (5, 9, 2, 7)]
    for size in (1, 2, 3):
        assert take_batch(Shape(), variants, size) == take_batch(Shape(), variants, size + 1)[:size]
    assert [variant.request_id for variant in take_batch(Shape(), variants, 3)] == ["r9", "r7", "r5"]
