"""The resume path: a periodic checkpoint, and a continuation that is the same run.

What this proves, in the order a lost epoch hurts: that a checkpoint exists *before* the end of a
run; that a resumed attempt picks the run up at the step the checkpoint recorded rather than at step
1; that the steps it then trains are the steps the killed attempt would have trained — the same
records, the same LR, the same loss, bit for bit; and that a run which is *not* resumed is unchanged
by any of it. The last one is what the 4B/9B/27B comparison rests on: the interrupted attempts here
write checkpoints, and their logs and final weights still come out identical to a run that write
none. A resume that restarted the data, or reseeded the dropout, would "continue" one run into
another.

The corpus is the smoke suite's own records (a six-record prefix, with a manifest rewritten to
match, because ``kev.suite.load_split`` checks both the partition's sha256 and its record count) and
the tokenizer is the one those records were measured with, so the encodings, the delimiter ids and
the state budget are a real run's. The backbone is the only thing shrunk — two 32-wide layers with
random weights, built here rather than downloaded — because what has to be identical is a sequence,
not an accuracy.

The interruption is raised from ``torch.optim.AdamW.step``, which is where a node dies in practice:
after N optimizer steps and before the (N+1)th. Everything the trainer had written by then — the
rows of ``run.log.jsonl`` and the checkpoint — is what the resume finds, and the row the resume
re-runs is the row it has to drop.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sckevtype.train import train as T

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "artifacts" / "suites" / "smoke"

#: Where the tiny backbone's *tokenizer* comes from. The model is built from a config below; the
#: tokenizer cannot be, because the suite's records and kev's delimiter tokens are ids in its
#: vocabulary, and a run encodes with the tokenizer the suite was measured with.
BASE = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "tiny"        # a local directory ignores it; the suite pins no base, so one is required

#: The backbone this file builds: Qwen3 (attention-only, so the packed path and no CUDA kernels),
#: two layers wide enough to have all seven of kev's LoRA targets.
BACKBONE = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 2048}

#: Records the copied suite trains on: six, at ``--batch 1 --accum 2``, are three steps per epoch.
#: Small on purpose — this is a CPU test, and what it checks is the *sequence* of steps.
RECORDS = 6

#: The rows a continuation has to reproduce. ``elapsed_s`` and ``peak_device_bytes`` are measurements
#: of this attempt's machine rather than of the run, and are not among them.
STEP_FIELDS = ("step", "epoch", "microbatch", "records_seen", "loss", "kl", "anchor", "window_tokens",
               "step_tokens", "passes", "slots_max", "epoch_tokens", "lr")


class NodeReboot(RuntimeError):
    """Stands in for the node going away: raised where a reboot interrupts a run."""


@contextlib.contextmanager
def dies_after(steps: int) -> Iterator[None]:
    """Make the (``steps`` + 1)-th optimizer step raise, as a node that reboots mid-step would.

    Patching the class's ``step`` rather than counting forward passes keeps the interruption where it
    belongs — between two optimizer steps — whatever the batch and accumulation sizes are.
    """
    torch = pytest.importorskip("torch", reason="the run is torch's")
    original = torch.optim.AdamW.step
    calls = {"n": 0}

    def step(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > steps:
            raise NodeReboot(f"the node went away before optimizer step {calls['n']}")
        return original(self, *args, **kwargs)

    torch.optim.AdamW.step = step
    try:
        yield
    finally:
        torch.optim.AdamW.step = original


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A backbone small enough for a CPU test, with the tokenizer a real suite is measured with."""
    torch = pytest.importorskip("torch", reason="the model is torch's")
    transformers = pytest.importorskip("transformers", reason="the backbone is built by transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    except Exception as error:  # noqa: BLE001 - any load failure means "not on this box"
        pytest.skip(f"no tokenizer for {BASE} on this box ({error}); it is what the suite was measured with")

    directory = tmp_path_factory.mktemp("tiny-base")
    tokenizer.save_pretrained(directory)
    config = transformers.Qwen3Config(vocab_size=len(tokenizer), **BACKBONE)
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(directory)
    del torch
    return directory


@pytest.fixture(scope="module")
def suite_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The smoke suite's first ``RECORDS`` training records, with a manifest that matches them.

    Rewritten rather than copied: ``load_split`` verifies the partition's sha256 *and* its record
    count against the manifest, so a subset with the original manifest is refused before training —
    and a suite whose manifest described records it does not hold would be a fixture that lies. The
    records themselves are the real ones, so kev augments and encodes what a real run encodes.
    """
    if not (SMOKE / "train.jsonl").exists():
        pytest.skip(f"no smoke suite at {SMOKE}; build one with `python -m sckevtype.train.suite`")
    manifest = json.loads((SMOKE / "manifest.json").read_text(encoding="utf-8"))
    lines = [line for line in (SMOKE / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:RECORDS]

    directory = tmp_path_factory.mktemp("suite") / "smoke-prefix"
    directory.mkdir()
    text = "".join(f"{line}\n" for line in lines)
    (directory / "train.jsonl").write_text(text, encoding="utf-8")
    # per_family/per_question_type describe the *whole* partition, so they are not carried over
    manifest["files"] = {"train.jsonl": {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                                         "bytes": len(text.encode()), "kind": "data", "records": len(lines)}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return directory


def cli(suite: Path, base: Path, out: Path, *extra: str) -> list[str]:
    """A trainer invocation, made through the CLI rather than through ``run`` — the way the lane does it."""
    return ["--suite", str(suite), "--base", str(base), "--base_revision", REVISION, "--device", "cpu",
            "--out", str(out), *extra]


@pytest.fixture(scope="module")
def reference(tiny_base: Path, suite_copy: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The run every interrupted attempt is compared against: six steps over two epochs, no checkpoints.

    Module-scoped because it is the same six steps for every test here and a CPU run of them is the
    expensive part of this file. ``--checkpoint_every 0`` on purpose: the interrupted attempts *do*
    write checkpoints and are compared to this run, so matching it is also the statement that writing
    one does not perturb the training it is taken from.
    """
    out = tmp_path_factory.mktemp("reference") / "run"
    assert T.main(cli(suite_copy, tiny_base, out, "--epochs", "2", "--accum", "2", "--checkpoint_every", "0")) == 0
    return out


def rows(out: Path) -> list[dict[str, Any]]:
    """``run.log.jsonl``, as rows."""
    text = (out / "run.log.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def sequence(out: Path) -> list[tuple[Any, ...]]:
    """The steps a run trained, in order, as the fields a continuation has to reproduce."""
    return [tuple(row[field] for field in STEP_FIELDS) for row in rows(out)]


def weights(out: Path) -> dict[str, Any]:
    """A run's trained tensors: the adapter and the pointer head, as the run saved them."""
    torch = pytest.importorskip("torch", reason="the adapter is torch's format")
    from safetensors.torch import load_file

    head = torch.load(out / "head.pt", map_location="cpu", weights_only=False)["head"]
    return {**load_file(str(out / "adapter_model.safetensors")), **{f"head.{key}": value for key, value in head.items()}}


# --------------------------------------------------------------------------- the continuation


def test_a_resume_continues_the_step_and_loss_sequence_instead_of_restarting(
    tiny_base: Path, suite_copy: Path, reference: Path, tmp_path: Path,
) -> None:
    """The failure this exists for, in miniature: die after five of six steps, resume, and read the log.

    The attempt dies *during* the sixth step's update, so the run's last completed step is five, the
    newest checkpoint is step four (every second step) and the log holds rows 1..5 — one of which the
    resume is about to write again. The resumed attempt must continue at step five, and the log it
    leaves must be the log the uninterrupted run has: steps 1..6, one row each, no step twice.
    """
    out = tmp_path / "interrupted"
    with dies_after(5), pytest.raises(NodeReboot):
        T.main(cli(suite_copy, tiny_base, out, "--epochs", "2", "--accum", "2", "--checkpoint_every", "2"))

    # what the kill left behind: two checkpoints, and a log one row past the newest of them
    assert [path.name for path in T.checkpoint_files(out)] == ["checkpoint-step-2.pt", "checkpoint-step-4.pt"]
    killed = rows(out)
    assert [row["step"] for row in killed] == [1, 2, 3, 4, 5]
    assert not list(out.glob("*.tmp")), "the interrupted write left a scratch file behind"

    assert T.main(cli(suite_copy, tiny_base, out, "--epochs", "2", "--accum", "2", "--checkpoint_every", "2",
                      "--resume")) == 0

    resumed = rows(out)
    assert [row["step"] for row in resumed] == [1, 2, 3, 4, 5, 6], "the resumed log is not one row per step"
    assert sequence(out) == sequence(reference), "the resumed steps are not the steps of the uninterrupted run"
    their, ours = weights(reference), weights(out)
    assert set(their) == set(ours)
    assert all(their[name].equal(ours[name]) for name in their), "the resumed run trained different weights"

    record = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert record["optimizer_steps"] == 6                       # the run's total, not this attempt's
    assert record["resume"]["from_step"] == 4 and record["resume"]["attempt"] == 2
    assert record["resume"]["steps_this_attempt"] == 2
    assert record["epochs"][-1]["attempt"] == 2                 # which attempt measured that epoch


def test_a_resume_at_an_epoch_boundary_takes_the_next_epochs_shuffle(
    tiny_base: Path, suite_copy: Path, reference: Path, tmp_path: Path,
) -> None:
    """The other half of the cursor: a checkpoint taken on an epoch's last step.

    Nothing of that epoch is left to run, so the resume continues into the next one — and the next
    epoch's order is a draw from the RNG stream, not from the seed, so it is exactly the draw the
    checkpoint recorded. Get this wrong and every resumed multi-epoch run trains a different order.
    """
    out = tmp_path / "boundary"
    with dies_after(4), pytest.raises(NodeReboot):
        T.main(cli(suite_copy, tiny_base, out, "--epochs", "2", "--accum", "2", "--checkpoint_every", "3"))

    # the newest checkpoint is the last step of epoch 0, and the log holds one row past it
    assert [path.name for path in T.checkpoint_files(out)] == ["checkpoint-step-3.pt"]
    assert [row["step"] for row in rows(out)] == [1, 2, 3, 4]
    assert [row["epoch"] for row in rows(out)] == [0, 0, 0, 1]

    assert T.main(cli(suite_copy, tiny_base, out, "--epochs", "2", "--accum", "2", "--checkpoint_every", "3",
                      "--resume")) == 0

    assert [row["step"] for row in rows(out)] == [1, 2, 3, 4, 5, 6]
    assert sequence(out) == sequence(reference), "the resumed epoch is not the epoch the checkpoint recorded"


# --------------------------------------------------------------------------- the checkpoint itself


def payload(step: int) -> dict[str, Any]:
    """A checkpoint payload with every key a resume reads — the file, without a run behind it."""
    torch = pytest.importorskip("torch", reason="a checkpoint is torch's format")
    return {"format": T.CHECKPOINT_FORMAT, "step": step, "epoch": 0, "step_in_epoch": step, "records_seen": step,
            "attempt": 1, "peak_device_bytes": 0, "epoch_records": 0, "epoch_tokens": 0, "passes": 0,
            "order": [0], "recipe": {}, "adapter": {}, "head": {}, "optimizer": {}, "scheduler": {},
            "rng": {"python": random.Random(0).getstate(), "random_module": random.getstate(),
                    "torch": torch.get_rng_state()}}


def test_only_the_latest_two_checkpoints_are_kept(tmp_path: Path) -> None:
    """A 9B checkpoint is ~0.5 GiB (its adapter is 165 MiB fp32 and AdamW keeps two moments of it).

    Two are kept: the newest is the file the crash itself may have truncated, and the write being
    atomic (a temp file renamed over the name) means the older one is cheap insurance rather than a
    second copy of a run in progress.
    """
    pytest.importorskip("torch", reason="a checkpoint is torch's format")
    for step in (2, 4, 6):
        T.save_checkpoint(tmp_path, payload(step))
    assert [path.name for path in T.checkpoint_files(tmp_path)] == ["checkpoint-step-4.pt", "checkpoint-step-6.pt"]
    assert not list(tmp_path.glob("*.tmp"))


def test_a_checkpoint_the_crash_truncated_falls_back_to_the_one_before_it(tmp_path: Path) -> None:
    """The reason for keeping two, exercised: the newest file does not load, so the older one is used."""
    pytest.importorskip("torch", reason="a checkpoint is torch's format")
    for step in (2, 4):
        T.save_checkpoint(tmp_path, payload(step))
    newest = tmp_path / "checkpoint-step-4.pt"
    newest.write_bytes(newest.read_bytes()[: newest.stat().st_size // 3])       # as a reboot mid-write leaves it

    path, loaded = T.load_resume_checkpoint(tmp_path)
    assert path.name == "checkpoint-step-2.pt" and loaded["step"] == 2


def test_resume_without_a_checkpoint_is_refused_by_naming_the_flag(
    tiny_base: Path, suite_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A populated directory is still refused — unless it holds a checkpoint, which is what --resume needs."""
    out = tmp_path / "artifacts-only"
    out.mkdir()
    (out / "run.log.jsonl").write_text("", encoding="utf-8")      # the remains of a run that never checkpointed

    assert T.main(cli(suite_copy, tiny_base, out, "--resume")) == 1
    err = capsys.readouterr().err
    assert "--resume" in err and "no checkpoint in" in err
    assert T.checkpoint_files(out) == []

    # ... and without the flag, the same directory is refused for the reason it always was
    capsys.readouterr()
    assert T.main(cli(suite_copy, tiny_base, out)) == 1
    assert "refusing to overwrite an existing run" in capsys.readouterr().err


def test_a_resume_whose_recipe_moved_is_refused_naming_the_field(
    tiny_base: Path, suite_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A different learning rate is a different run, and the refusal names the field that differs.

    One step long (``--accum 6`` folds the six records into a single optimizer step), so this costs a
    checkpoint and nothing else. It is also the case a run *ends* on: the last step of the last epoch
    is checkpointed, because the epoch bound is not a stop the trainer knows about, and the refusal
    below fires before the resume would have to say there is nothing left to train.
    """
    out = tmp_path / "recipe"
    assert T.main(cli(suite_copy, tiny_base, out, "--epochs", "1", "--accum", "6", "--checkpoint_every", "1",
                      "--lr", "2e-4")) == 0
    assert [path.name for path in T.checkpoint_files(out)] == ["checkpoint-step-1.pt"]

    assert T.main(cli(suite_copy, tiny_base, out, "--epochs", "1", "--accum", "6", "--checkpoint_every", "1",
                      "--lr", "1e-3", "--resume")) == 1
    err = capsys.readouterr().err
    assert "was written by a different run" in err and "lr" in err


def test_a_run_that_is_not_resumed_records_no_resume_key(tmp_path: Path) -> None:
    """The key is absent, not null: an ordinary run's provenance stays the file it was."""
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text('{"name": "smoke"}\n', encoding="utf-8")
    record = T.provenance_record(
        namespace=argparse.Namespace(suite=str(suite), seed=0), revision=REVISION, manifest={"name": "smoke"},
        suite_dir=suite, trainables=argparse.Namespace(count=0, by_kind={}, names=()), base={}, environment={},
        epochs=[], optimizer_steps=0, peak_device_bytes=0, stopped_early=False,
    )
    assert "resume" not in record
    resumed = T.provenance_record(
        namespace=argparse.Namespace(suite=str(suite), seed=0), revision=REVISION, manifest={"name": "smoke"},
        suite_dir=suite, trainables=argparse.Namespace(count=0, by_kind={}, names=()), base={}, environment={},
        epochs=[], optimizer_steps=6, peak_device_bytes=0, stopped_early=False,
        resume={"checkpoint": "checkpoint-step-4.pt", "attempt": 2, "from_step": 4},
    )
    assert resumed["resume"]["from_step"] == 4
    assert json.loads(json.dumps(resumed, allow_nan=False)) == resumed


# --------------------------------------------------------------------------- the flags


def test_the_27b_lane_forwards_the_recovery_pair_only_when_asked() -> None:
    """The other lane that calls ``train.run``: its driver must hand the trainer today's argv, exactly.

    ``scripts/kev/kev_fullft_train.py`` is the 27B's entry point (a full-weight checkpoint, a fresh
    LoRA, the trainer's own loop). Its --resume/--checkpoint_every are the only way the 27B's 18-hour
    epoch can be continued, and a driver that started forwarding them unconditionally would change
    every existing command line's argv -- so both are asserted here: absent by default, present when
    asked for.
    """
    import importlib.util

    driver = ROOT / "scripts" / "kev" / "kev_fullft_train.py"
    if not driver.exists():
        pytest.skip(f"no 27B driver at {driver}")
    pytest.importorskip("torch", reason="the driver loads torch to shard a full-weight checkpoint")
    pytest.importorskip("accelerate", reason="the driver's placement policy imports accelerate")
    spec = importlib.util.spec_from_file_location("_kev_fullft_train", driver)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = argparse.Namespace(path="/hub/snapshots/tiny")

    def argv(*extra: str) -> list[str]:
        return module.train_argv(module.parse_args(["--full_weight", "x", "--suite", "s", "--out", "o", *extra]),
                                 checkpoint, "rev0")

    assert argv() == [                                          # exactly the flags it passed before this
        "--suite", "s", "--base", "/hub/snapshots/tiny", "--base_revision", "rev0",
        "--dtype", "bf16", "--weights_dtype", "bf16", "--checkpointing", "1", "--device", "cuda",
        "--gpus", "4", "--token_budget", "0", "--seed", "1", "--steps", "20", "--out", "o",
    ]
    assert argv("--resume")[-1] == "--resume"
    assert argv("--checkpoint_every", "50")[-2:] == ["--checkpoint_every", "50"]


def test_the_interval_is_a_flag_with_an_environment_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is 200 steps, an env var moves it without touching the command line, and a typo stops."""
    base = ["--suite", "s", "--base", "b", "--out", "o"]
    assert T.DEFAULT_CHECKPOINT_EVERY == 200
    assert T.parse_args(base).checkpoint_every == 200
    assert T.parse_args(base).resume is False

    monkeypatch.setenv(T.ENV_CHECKPOINT_EVERY, "25")
    assert T.parse_args(base).checkpoint_every == 25
    assert T.parse_args([*base, "--checkpoint_every", "7"]).checkpoint_every == 7       # the flag wins
    monkeypatch.setenv(T.ENV_CHECKPOINT_EVERY, "often")
    with pytest.raises(SystemExit):
        T.parse_args(base)
    monkeypatch.setenv(T.ENV_CHECKPOINT_EVERY, "-3")
    with pytest.raises(SystemExit):
        T.parse_args(base)


def test_a_negative_interval_is_refused() -> None:
    with pytest.raises(SystemExit):
        T.parse_args(["--suite", "s", "--base", "b", "--out", "o", "--checkpoint_every", "-1"])


def test_the_kev_namespace_still_carries_the_runs_own_directory_on_a_resume(tmp_path: Path) -> None:
    """kev refuses an ``--out`` that exists, and a resume's has to — so the check is bypassed for that
    one value, and only for it: the namespace this run records still names the run's directory, and
    every other check kev's parser makes still ran to produce it.
    """
    pytest.importorskip("kev.train", reason="the namespace is kev's; requires the train extra")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text('{"context": {"max_state": 1664}}\n', encoding="utf-8")
    out = tmp_path / "run"
    out.mkdir()

    with pytest.raises(SystemExit):
        T.kev_namespace(T.parse_args(["--suite", str(suite), "--base", "b", "--out", str(out)]), str(out))

    args = T.parse_args(["--suite", str(suite), "--base", "b", "--resume", "--out", str(out)])
    namespace = T.kev_namespace(args, str(out))
    assert namespace.out == str(out)                 # what training_config.json records
    assert namespace.max_state == 1664               # kev's own budget resolution, still kev's
    json.dumps(vars(namespace), allow_nan=False)     # and still JSON-safe for that file
