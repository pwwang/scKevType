"""A run's own ``training_config.json`` must name its true initialization.

The bug these tests pin, and why it mattered: ``training_config.json`` recorded kev's *defaults*
where the run's own initialization belonged.

* ``args.init_from`` was kev's own empty default on every run — including a delta run — because
  the flag is deliberately kept out of ``vars(ns)``. A delta run's config therefore said
  "not a delta run".
* ``model_settings.base`` was kev's template default ``Qwen/Qwen3-0.6B-Base`` on every run,
  including runs whose base was ``Qwen/Qwen3.5-0.8B-Base``. The one field a reader would check
  to ask "which base?" answered with a model that was never loaded.

Together those two made five runs warm-started from ``jaredpalmer/kev-0.8b`` read as
base-initialized — their ``provenance.json`` said otherwise, but the config is the file a reader
opens first. The fix is not a new file: it is the two fields telling the truth, plus the resolved
source (with its digests and its commit) written where the base is written.

The fast tests cover the revision derivation. The two slow ones run the trainer for real, because
``training_config.json`` is written by ``run()`` and nothing cheaper exercises it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sckevtype.train.train import main, source_revision

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "artifacts" / "suites" / "smoke"

#: The base and commit the released kev-0.8b checkpoint carries in its own ``head.pt``.
BASE = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SOURCE = "jaredpalmer/kev-0.8b"

#: kev's template default, which the config used to report as the base of every run.
KEVS_DEFAULT_BASE = "Qwen/Qwen3-0.6B-Base"


# --------------------------------------------------------------------------- the revision


def test_a_hub_snapshot_path_names_the_commit_it_holds() -> None:
    """``Checkpoint.path`` is a snapshot directory, and huggingface_hub names it after the commit.

    That is the only place kev's own record carries the source *repo's* revision, so it is read
    back out rather than left for a reader to know the convention.
    """
    resolved = "/hub/models--jaredpalmer--kev-0.8b/snapshots/9a45d25eb2ab761841196625383fa1dff0e56c1e"
    assert source_revision(resolved) == "9a45d25eb2ab761841196625383fa1dff0e56c1e"


@pytest.mark.parametrize("resolved", ["/runs/delta-kev-0.8b-s1", "/tmp/initfrom/delta", ""])
def test_a_local_run_directory_has_no_commit_to_name(resolved: str) -> None:
    """A local source is not a commit; the field says so instead of inventing one."""
    assert source_revision(resolved) is None


# --------------------------------------------------------------------------- the run itself


def _device(torch) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _train(out: Path, *extra: str) -> dict:
    """Run the trainer for real and return the ``training_config.json`` it wrote."""
    code = main(["--suite", str(SMOKE), "--base", BASE, "--base_revision", REVISION,
                 "--steps", "2", "--batch", "2", "--dtype", "bf16", "--checkpointing", "1",
                 "--device", "cuda", "--out", str(out), *extra])
    assert code == 0, "the run did not complete"
    return json.loads((out / "training_config.json").read_text(encoding="utf-8"))


@pytest.fixture
def smoke_run():
    """The slow legs need torch, a card and the smoke suite; skip by name when any is absent."""
    torch = pytest.importorskip("torch", reason="the run is torch's")
    pytest.importorskip("kev.train", reason="the run is kev's; requires the train extra")
    if not SMOKE.exists():
        pytest.skip(f"no suite at {SMOKE}; the smoke recipe builds it")
    if not torch.cuda.is_available():
        pytest.skip("the smoke recipe trains bf16 on cuda")
    return torch


@pytest.mark.slow
@pytest.mark.network
def test_a_delta_runs_config_names_the_source_and_the_base_it_used(smoke_run, tmp_path: Path) -> None:
    """The whole point: open a delta run's config and it says where the run started.

    Three things must be true of it and none of them were: ``args.init_from`` is the flag that
    was passed, ``model_settings.base`` is the base that was built, and the resolved source — its
    commit and the digests of the adapter and head that were loaded — is in the file, so the
    config alone is enough to say which released checkpoint this run continued.
    """
    config = _train(tmp_path / "delta", "--init_from", SOURCE)

    assert config["args"]["init_from"] == SOURCE, "the config does not name the source it warm-started from"
    assert config["model_settings"]["base"] == BASE, "model_settings.base is still kev's template default"
    assert config["model_settings"]["base"] != KEVS_DEFAULT_BASE
    assert config["base"] == {"id": BASE, "revision": REVISION}

    source = config["init_from"]
    assert source["init_from"] == SOURCE and source["base"] == BASE
    assert source["base_revision"] == REVISION
    assert len(source["adapter_sha256"]) == 64 and len(source["head_sha256"]) == 64
    assert source["adapter_tensors"] == 372
    # The commit is whatever the cache resolved the id to — not a constant, because two caches
    # can hold different snapshots of the same repo (this box's and the cluster's do). What must
    # hold is that the field names the snapshot `resolved` points at.
    assert source["source_revision"] == Path(source["resolved"]).name
    assert len(source["source_revision"]) == 40
    assert json.loads(json.dumps(config, allow_nan=False)) == config


@pytest.mark.slow
@pytest.mark.network
def test_an_ordinary_runs_config_names_its_base_and_carries_no_init_from(smoke_run, tmp_path: Path) -> None:
    """The other direction, and the one the fix must not break: a run with no flag must not claim
    a source, and must still name its true base rather than kev's template default.

    ``init_from`` is absent rather than null — the absence is the statement "this run started from
    the base model alone", and a key that is present-and-null would be indistinguishable from a
    delta run whose source failed to resolve.
    """
    config = _train(tmp_path / "plain")

    assert config["args"]["init_from"] == ""
    assert "init_from" not in config
    assert config["model_settings"]["base"] == BASE
    assert config["base"] == {"id": BASE, "revision": REVISION}
