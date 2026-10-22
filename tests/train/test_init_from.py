"""``--init_from``: delta fine-tuning, warm-started from a released checkpoint.

The load and its compatibility checks are ``kev.checkpoint.Checkpoint.warm_start``'s and are
exercised here through the one seam this trainer adds (:func:`sckevtype.train.train.warm_start`)
rather than through a second implementation. What is tested: the CLI accepts the flag, the flag
never leaks into kev's namespace (which is what the recorded hyperparameters are read from, so
an ordinary run's ``training_config.json`` and ``provenance.json`` keep today's bytes), a
mismatched source is refused *before* a tensor is loaded, and — with ``--run-slow --run-network``
— that a warm-started model scores identically to the checkpoint it came from and that a short
delta run completes and names its source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sckevtype.train.model import Trainables
from sckevtype.train.train import (
    kev_namespace,
    main,
    parse_args,
    provenance_record,
    warm_start,
)

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "artifacts" / "suites" / "smoke"

#: The base and commit the released kev checkpoints carry in their own ``head.pt``, and the
#: commit ``configs/models.yaml`` pins — a warm start refuses two different revisions, so a
#: real delta run is launched with this pair.
BASE = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SOURCE = "jaredpalmer/kev-0.8b"

#: How many test-partition records the parity check scores. Small: it is a load check, and one
#: disagreeing question is as damning with eight records as with eight hundred.
SLICE = 12


def cli(suite: Path, out: Path, *extra: str) -> list[str]:
    """A CLI invocation that parses; the flag tests stop long before anything is loaded."""
    return ["--suite", str(suite), "--base", BASE, "--base_revision", REVISION, "--out", str(out), *extra]


# --------------------------------------------------------------------------- the flag


def test_the_flag_defaults_to_off_and_is_accepted(tmp_path: Path) -> None:
    assert parse_args(cli(tmp_path, tmp_path / "run")).init_from == ""
    assert parse_args(cli(tmp_path, tmp_path / "run", "--init_from", SOURCE)).init_from == SOURCE


def test_the_flag_never_reaches_kevs_namespace(tmp_path: Path) -> None:
    """``vars(ns)`` is what ``training_config.json``'s args and the provenance hyperparameters
    record, so a flag that landed in kev's namespace would change both files on every run —
    including the flag-absent one. ``--init_from`` is this CLI's own; kev keeps its default."""
    pytest.importorskip("kev.train", reason="the namespace is kev's; requires the train extra")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text('{"context": {"max_state": 384}}\n', encoding="utf-8")
    args = parse_args(cli(suite, tmp_path / "run", "--init_from", SOURCE))
    namespace = kev_namespace(args, str(tmp_path / "run"))
    assert namespace.init_from == ""            # kev's own default, not ours
    assert namespace.base_revision == REVISION  # the flags this CLI does mirror still arrive


# --------------------------------------------------------------------------- provenance


def provenance(tmp_path: Path, **extra) -> dict:
    """A provenance record for a minimal suite — the same call shape the tests in
    ``test_train.py`` use, so only the ``init_from`` argument under test varies."""
    suite = tmp_path / "suite"
    suite.mkdir(exist_ok=True)
    (suite / "manifest.json").write_text('{"name": "smoke"}\n', encoding="utf-8")
    return provenance_record(
        namespace=argparse.Namespace(suite=str(suite), seed=0), revision=REVISION,
        manifest={"name": "smoke"}, suite_dir=suite, trainables=Trainables(0, (), {}),
        base={}, environment={}, epochs=[], optimizer_steps=0, peak_device_bytes=0,
        stopped_early=False, **extra,
    )


def test_an_ordinary_run_records_no_init_from_key(tmp_path: Path) -> None:
    """No flag = today's record: the key must be absent, not present and null, so the file
    stays byte-identical to a run made before the flag existed."""
    assert "init_from" not in provenance(tmp_path)


def test_a_delta_run_records_the_source_it_warm_started_from(tmp_path: Path) -> None:
    source = {"init_from": SOURCE, "resolved": "/hub/snapshots/54f4f877", "adapter_sha256": "a" * 64,
              "head_sha256": "b" * 64, "adapter_tensors": 372, "base": BASE, "base_revision": REVISION}
    record = provenance(tmp_path, init_from=source)
    assert record["init_from"] == source
    assert json.loads(json.dumps(record, allow_nan=False)) == record


# --------------------------------------------------------------------------- refusals


def fake_checkpoint(tmp_path: Path, **meta: object) -> Path:
    """A checkpoint directory holding only what a refusal reads: ``head.pt``.

    The architecture comparison runs before the adapter is opened, so the directory needs
    nothing else — and the model argument to :func:`warm_start` can be an object that fails
    the moment it is touched, which is exactly what these tests pass.
    """
    torch = pytest.importorskip("torch", reason="head.pt is torch's format")
    from kev.checkpoint import Meta

    directory = tmp_path / "checkpoint"
    directory.mkdir()
    fields = {"base": BASE, "base_revision": REVISION, "lora": 16, "head_dim": 256,
              "option_isolation": False, "special_embeddings": False}
    fields.update(meta)
    torch.save(Meta(**fields).to_dict(), directory / "head.pt")
    return directory


@pytest.mark.parametrize("field, value", [("head_dim", 512), ("lora", 32), ("base", "Qwen/Qwen3-0.6B-Base")])
def test_a_mismatched_source_is_refused_before_anything_is_loaded(
    tmp_path: Path, field: str, value: object
) -> None:
    """A refusal that cannot fail is worse than no check: this one fires on kev's own
    comparison, names the offending field, and does so before any tensor is read."""
    pytest.importorskip("kev.checkpoint", reason="the comparison is kev's; requires the train extra")
    from kev.checkpoint import Meta

    source = fake_checkpoint(tmp_path, **{field: value})
    ours = Meta(base=BASE, base_revision=REVISION, lora=16, head_dim=256,
                option_isolation=False, special_embeddings=False)
    with pytest.raises(ValueError, match=field) as refusal:
        warm_start(str(source), object(), ours)   # object(): raises before the model is touched
    assert repr(value) in str(refusal.value)      # ... and names what it found


# --------------------------------------------------------------------------- the load itself


def _device(torch) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _warm_predictor(model, tokenizer, device: str, context: dict):
    """kev's own scoring path (:class:`kev.predictors.LocalPredictor`) over an in-memory model.

    The source checkpoint is scored by a normally constructed ``LocalPredictor``; the warm-started
    model has no run directory yet, so the loader is skipped and the two models are compared
    through the *same* ``__call__`` — the code that encodes, forwards and softmaxes.
    """
    from kev.predictors import LocalPredictor

    predictor = LocalPredictor.__new__(LocalPredictor)
    predictor.run, predictor.tok, predictor.model = "<warm-start>", tokenizer, model
    predictor.temperature, predictor.device, predictor.context = 1.0, device, context
    return predictor


def _compare(theirs, ours, records) -> tuple[float, list, int]:
    """Worst |dp| and any argmax disagreement between two predictors over ``records``."""
    def top(probabilities: dict) -> str:
        return max(probabilities, key=probabilities.__getitem__)

    worst, flips, questions = 0.0, [], 0
    for record in records:
        left, right = theirs(record)["probabilities"], ours(record)["probabilities"]
        for qid, probabilities in left.items():
            questions += 1
            worst = max(worst, *(abs(p - right[qid][key]) for key, p in probabilities.items()))
            if top(probabilities) != top(right[qid]):
                flips.append((record["_meta"]["id"], qid))
    return worst, flips, questions


@pytest.mark.slow
@pytest.mark.network
def test_a_warm_started_model_scores_identically_to_its_source() -> None:
    """Before any optimizer step, the warm-started model and the source must agree on a slice of
    the suite. Training would hide a wrong load in the loss; this is the check that cannot be
    trained past.

    Two legs, because the source has two scoring paths. Against the source loaded **unmerged**
    (``KEV_MERGE=0``, the same computation the warm-started model runs): exact. Against the
    source **as served** (``Checkpoint.load``'s default, the adapter folded into the base):
    argmax identical and probabilities to the merge's own fp tolerance — folding rounds the
    weights at ~1e-7 relative, and this hybrid backbone's recurrence amplifies that to ~4e-4
    probability (a 1e-7 weight nudge moves it the same amount), so the merged path *cannot*
    be expected to agree more closely than that.
    """
    torch = pytest.importorskip("torch", reason="the load is torch's")
    pytest.importorskip("kev.predictors", reason="the scoring path is kev's")
    from kev.checkpoint import LoadOptions, Meta
    from kev.predictors import LocalPredictor
    from kev.suite import load_split, read_manifest

    from sckevtype.train.model import build_model

    manifest = read_manifest(str(SMOKE))
    records = load_split(str(SMOKE), "test", allow_test=True)[:SLICE]
    device = _device(torch)
    # T = 1: the raw-logit path both sides share. The checkpoint's shipped temperature (2.35) is a
    # serve-time calibration kev's warm start does not load — it is not in head.pt's state dict — and
    # this project's eval scores at T = 1 too, so parity is a statement about the weights alone.
    unmerged = LocalPredictor(SOURCE, device, LoadOptions(temperature=1.0, merge=False), context=manifest["context"])
    built = build_model(BASE, REVISION, {"lora": 16, "head_dim": 256}, device=device, tokenizer=unmerged.tok)
    built.eval()
    warm_start(SOURCE, built, Meta(base=BASE, base_revision=REVISION, lora=16, head_dim=256))
    warm = _warm_predictor(built, unmerged.tok, device, manifest["context"])

    exact, flips_exact, questions = _compare(unmerged, warm, records)
    del unmerged
    if device == "cuda":
        torch.cuda.empty_cache()   # the served copy is a second full model
    served = LocalPredictor(SOURCE, device, LoadOptions(temperature=1.0), context=manifest["context"])
    tolerance, flips, _ = _compare(served, warm, records)

    # the measurement the acceptance report quotes, printed rather than only asserted
    print(f"parity: {len(records)} records, {questions} questions; vs unmerged max |dp| {exact:.3e} "
          f"(flips {len(flips_exact)}); vs served max |dp| {tolerance:.3e} (flips {len(flips)})")
    assert questions, "the slice scored no questions at all"
    assert not flips_exact and exact < 1e-6, f"the load is not exact: {exact:.2e} against the unmerged source"
    assert not flips, f"argmax disagrees on {flips[:3]}"
    assert tolerance < 1e-3, f"the served probabilities differ by {tolerance:.2e}"


@pytest.mark.slow
@pytest.mark.network
def test_a_source_whose_adapter_does_not_match_the_model_is_refused_naming_the_tensor(tmp_path: Path) -> None:
    """The set of adapter tensors must match exactly, in both directions — an adapter that is
    short a target module (or long one) is refused with the tensor's name, because peft loads
    matching keys silently and a half-loaded adapter still trains and still reports a loss."""
    torch = pytest.importorskip("torch", reason="the model is torch's")
    pytest.importorskip("peft", reason="the adapter is peft's")
    from kev.checkpoint import Meta, resolve_run

    from sckevtype.train.model import build_model

    if not SMOKE.exists():
        pytest.skip(f"no suite at {SMOKE}; the smoke recipe builds it")
    device = _device(torch)
    built = build_model(BASE, REVISION, {"lora": 16, "head_dim": 256}, device=device)
    ours = Meta(base=BASE, base_revision=REVISION, lora=16, head_dim=256)

    from safetensors.torch import load_file, save_file

    released = Path(resolve_run(SOURCE))
    weights = load_file(str(released / "adapter_model.safetensors"))
    dropped = min(weights)
    bogus = "base_model.model.layers.0.mlp.gate_proj.lora_A.not_a_tensor"
    doctored = {"short": {k: v for k, v in weights.items() if k != dropped},
                "long": {**weights, bogus: weights[dropped].clone()}}
    for case, tensors in doctored.items():
        directory = tmp_path / case
        directory.mkdir()
        for name in ("head.pt", "adapter_config.json"):
            (directory / name).write_bytes((released / name).read_bytes())
        save_file(tensors, str(directory / "adapter_model.safetensors"))
        with pytest.raises(ValueError, match="adapter tensors") as refusal:
            warm_start(str(directory), built, ours)
        named = dropped if case == "short" else bogus
        assert named in str(refusal.value), f"{case} refusal does not name {named}"


@pytest.mark.slow
@pytest.mark.network
def test_a_delta_run_completes_and_names_its_source(tmp_path: Path) -> None:
    """A short run warm-started from the released 0.8b: it completes, its provenance names the
    source, and the directory it wrote reads back as a checkpoint through kev's own reader."""
    torch = pytest.importorskip("torch", reason="the run is torch's")
    pytest.importorskip("kev.train", reason="the run is kev's; requires the train extra")
    if not SMOKE.exists():
        pytest.skip(f"no suite at {SMOKE}; the smoke recipe builds it")
    if not torch.cuda.is_available():
        pytest.skip("the smoke recipe trains bf16 on cuda")

    out = tmp_path / "delta"
    code = main(["--suite", str(SMOKE), "--base", BASE, "--base_revision", REVISION,
                 "--steps", "2", "--batch", "2", "--dtype", "bf16", "--checkpointing", "1",
                 "--device", "cuda", "--init_from", SOURCE, "--out", str(out)])
    assert code == 0, "the delta run did not complete"
    record = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    source = record["init_from"]
    assert source["init_from"] == SOURCE and source["base"] == BASE
    assert len(source["adapter_sha256"]) == 64 and len(source["head_sha256"]) == 64

    from kev.checkpoint import Checkpoint

    written, released = Checkpoint(str(out)), Checkpoint(SOURCE)
    assert written.meta.base == BASE and written.meta.head_dim == released.meta.head_dim
    # the head the run saved is the source's head, trained on — same tensors, same shapes
    assert {name: tuple(t.shape) for name, t in written.meta.head.items()} == \
           {name: tuple(t.shape) for name, t in released.meta.head.items()}
