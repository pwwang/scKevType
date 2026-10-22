"""The builder: what it writes, what it refuses, and that it writes the same bytes twice."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sckevtype.contracts.render import parse_rank_list
from sckevtype.contracts.schema import RecordFamily
from sckevtype.data import build as build_module
from sckevtype.data import fetch as fetch_module
from sckevtype.data.build import (
    CorpusConfig,
    DatasetConfig,
    build,
    load_config,
    main,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "corpus" / "smoke.yaml"
#: A minimal valid corpus: the tests below vary one dataset against it, not the whole config.
BARE = CorpusConfig(
    name="x",
    seed=1,
    datasets=[DatasetConfig(id="placeholder", registered=False, format="csv_expr", tier="T3", path="/tmp/x", cap=1)],
    state={"max_tokens": 384},
)


def test_the_config_loads_and_forbids_what_it_should():
    config = load_config(CONFIG)
    assert isinstance(config, CorpusConfig)
    assert config.state.max_tokens == 384
    assert config.datasets and all(dataset.cap for dataset in config.datasets)
    with pytest.raises(ValueError):
        DatasetConfig.model_validate({"id": "x", "surprise": 1})  # extra keys are errors, not ignored
    with pytest.raises(ValueError, match="label_map"):
        build_module.Artifacts(Path(__file__).resolve().parents[2] / "artifacts" / "v1").label_map(DatasetConfig(id="x"))
    assert config.families == tuple(RecordFamily)


def test_a_dataset_the_registry_does_not_know_must_say_everything():
    with pytest.raises(ValueError, match="must be set"):
        build_module.resolve_dataset(DatasetConfig(id="mystery", registered=False), BARE)
    with pytest.raises(KeyError):
        build_module.resolve_dataset(DatasetConfig(id="not-a-dataset", registered=True), BARE)


def test_build_writes_the_four_kev_splits_and_a_manifest(smoke_build):
    manifest = smoke_build.manifest
    assert set(smoke_build.lines) == {"train", "calibration", "development", "test"}
    assert manifest["counts"]["records"] == len(smoke_build.records) > 0
    assert manifest["counts"]["per_split"]["test"] > 0
    assert set(manifest["counts"]["per_family"]) == {family.value for family in RecordFamily}
    assert (smoke_build.out / "qc.json").exists()
    assert (smoke_build.out / "manifest.json").exists()


def test_the_manifest_is_what_kev_load_split_verifies(smoke_build):
    """``kev.suite.load_split`` reads ``files[name]["sha256"]`` and ``["records"]``."""
    for name, entry in smoke_build.manifest["files"].items():
        path = smoke_build.out / name
        written = path.read_bytes()
        assert entry["sha256"] == hashlib.sha256(written).hexdigest()
        assert entry["bytes"] == len(written)
        if name.endswith(".jsonl"):
            assert entry["records"] == len(smoke_build.lines[name.replace(".jsonl", "")])
            assert entry["questions"] == sum(len(payload["questions"]) for payload in smoke_build.lines[name.replace(".jsonl", "")])


def test_records_round_trip_through_the_kev_shape(smoke_build):
    for split, payloads in smoke_build.lines.items():
        for payload in payloads:
            assert set(payload) == {"state", "questions", "_meta"}
            assert payload["_meta"]["family"] in {family.value for family in RecordFamily}
            assert payload["state"].strip()
            for question in payload["questions"].values():
                assert question["label"] is not None
            assert split in ("train", "calibration", "development", "test")


def test_train_is_permuted_only_while_the_held_out_splits_carry_both_variants(smoke_build):
    variants = {split: {payload["_meta"]["variant"] for payload in payloads} for split, payloads in smoke_build.lines.items()}
    assert variants["train"] == {"permuted"}
    assert variants["calibration"] == {"clean", "permuted"}
    assert variants["development"] == {"clean", "permuted"}
    assert variants["test"] == {"clean", "permuted"}
    # Both variants of a record are present, and they are the same record otherwise.
    clean = [payload for payload in smoke_build.lines["test"] if payload["_meta"]["variant"] == "clean"]
    assert clean and all(payload["state"] for payload in clean)


def test_rebuild_is_byte_identical_into_a_different_directory(smoke_build, tmp_path):
    other = tmp_path / "elsewhere"
    manifest = build(CONFIG, other)
    assert manifest["files"] == smoke_build.manifest["files"]
    for name in smoke_build.manifest["files"]:
        assert (other / name).read_bytes() == (smoke_build.out / name).read_bytes()
    assert manifest == smoke_build.manifest


def test_nothing_written_carries_a_timestamp_or_an_absolute_path(smoke_build):
    for name in smoke_build.manifest["files"]:
        if name.endswith(".jsonl"):
            body = (smoke_build.out / name).read_text(encoding="utf-8")
            assert str(smoke_build.out) not in body
    assert "created_at" not in smoke_build.manifest
    for entry in smoke_build.manifest["datasets"].values():
        assert not Path(entry["path"]).is_absolute()
    assert set(smoke_build.manifest["artifacts"]["digests"]) == {"render_config.json", "definitions.json", "tissue_map.json", "cl_terms.json"}


def test_a_changed_file_is_refused_without_force(smoke_build, tmp_path):
    out = tmp_path / "guarded"
    build(CONFIG, out)
    (out / "train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists with different content"):
        build(CONFIG, out)
    manifest = build(CONFIG, out, force=True)
    assert manifest["counts"]["records"] == smoke_build.manifest["counts"]["records"]
    assert (out / "train.jsonl").read_bytes() == (smoke_build.out / "train.jsonl").read_bytes()


def test_qc_json_holds_every_probe(smoke_build):
    payload = json.loads((smoke_build.out / "qc.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "counts",
        "checks",
        "state_lengths",
        "identifiability",
        "datasets",
        "labels",
        "skipped",
        "tissue_resolution",
    }
    assert payload["checks"]["pairing"]["unpaired"] == 0
    assert payload["checks"]["pairing"]["paired"] > 0
    assert payload["state_lengths"]["over_budget"] == 0
    assert payload["skipped"]  # the two evidence families are counted when they do not apply
    assert set(payload["datasets"]) == {"smoke_h5ad", "smoke_10x"}
    assert all("path" not in entry for entry in payload["datasets"].values())
    assert payload["labels"]["smoke_10x"]["dropped"] >= 1  # the "Doublet" label


def test_states_are_rendered_by_the_canonical_renderer(smoke_build):
    for payload in smoke_build.lines["test"][:50]:
        symbols = [symbol for symbol, _ in parse_rank_list(payload["state"], smoke_build.cfg)]
        assert payload["state"] == payload["state"].strip()
        assert all(" " in symbol or symbol.isprintable() for symbol in symbols)


def test_the_cli_reports_failure_instead_of_tracebacking(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.yaml"), "--out", str(tmp_path / "out")]) == 1
    assert "build failed" in capsys.readouterr().err
    assert main(["--config", str(CONFIG), "--out", str(tmp_path / "ok")]) == 0
    assert "smoke:" in capsys.readouterr().out


def test_committed_fixtures_match_the_generator(fixtures_dir, fixture_generator, tmp_path):
    """The committed fixture bytes are the generator's bytes — no drift, no hand edits."""
    copy = tmp_path / "data"
    shutil.copytree(fixtures_dir, copy)
    assert fixture_generator.main(["--out", str(copy), "--check"]) == 0
    (copy / "tenx_blood" / "labels.tsv").write_text("tampered\n", encoding="utf-8")
    assert fixture_generator.main(["--out", str(copy), "--check"]) == 1


def test_the_build_needs_no_network(smoke_build, monkeypatch):
    """A corpus build must never reach for the network: `fetch` is opt-in per dataset."""

    def explode(*_args, **_kwargs):
        raise AssertionError("the builder attempted a network call")

    monkeypatch.setattr(fetch_module, "_http_stream", explode)
    monkeypatch.setattr(fetch_module, "_http_json", explode)
    manifest = build(CONFIG, smoke_build.out.parent / "offline", force=True)
    assert manifest["counts"]["records"] > 0
