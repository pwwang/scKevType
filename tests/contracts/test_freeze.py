"""Digests and manifests: write, verify, detect tampering, branch a version."""

from __future__ import annotations

import json

import pytest

from sckevtype.contracts.freeze import (
    artifact_bytes,
    canonical_json,
    file_digest,
    new_version,
    payload_digest,
    read_manifest,
    update_manifest,
    verify_manifest,
    write_artifact,
)
from sckevtype.contracts.schema import SuiteManifest


def test_digests_are_key_order_independent_and_stable():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1}) == '{"a":2,"b":1}'
    assert payload_digest({"a": [1, 2]}) == payload_digest({"a": [1, 2]})
    assert payload_digest({"a": [1, 2]}) != payload_digest({"a": [2, 1]})


def test_artifact_bytes_matches_what_is_written(tmp_path):
    path = tmp_path / "payload.json"
    digest = write_artifact(tmp_path, "payload.json", {"a": 1})
    assert path.read_bytes() == artifact_bytes({"a": 1})
    assert digest == file_digest(path)


def test_write_then_verify_is_clean(tmp_path):
    write_artifact(tmp_path, "a.json", {"a": 1}, kind="vocabulary", counts=3)
    write_artifact(tmp_path, "b.txt", "raw text", kind="note")
    update_manifest(tmp_path / "manifest.json", name="t", version="v1")
    assert verify_manifest(tmp_path) == []

    manifest = SuiteManifest.model_validate(read_manifest(tmp_path))
    assert manifest.files["a.json"].counts == 3 and manifest.files["a.json"].kind == "vocabulary"
    assert manifest.files["b.txt"].sha256 == file_digest(tmp_path / "b.txt")
    assert manifest.counts == {}, "counts belong to the builder, not the writer"


def test_a_single_flipped_byte_is_reported(tmp_path):
    write_artifact(tmp_path, "definitions.json", {"definitions": [{"cl_id": "CL:0000624"}]})
    write_artifact(tmp_path, "vocabulary.json", {"symbols": ["CD3D"]})
    assert verify_manifest(tmp_path) == []

    target = tmp_path / "vocabulary.json"
    target.write_text(target.read_text(encoding="utf-8").replace("CD3D", "CD3E"), encoding="utf-8")
    problems = verify_manifest(tmp_path)
    assert len(problems) == 1 and "vocabulary.json" in problems[0] and "digest mismatch" in problems[0]


def test_a_missing_file_and_a_missing_manifest_are_reported(tmp_path):
    write_artifact(tmp_path, "vocabulary.json", {"symbols": []})
    (tmp_path / "vocabulary.json").unlink()
    assert "missing file: vocabulary.json" in verify_manifest(tmp_path)
    assert verify_manifest(tmp_path / "nowhere") == [f"missing manifest: {tmp_path / 'nowhere' / 'manifest.json'}"]


def test_manifest_merge_is_order_independent(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_artifact(first, "a.json", {"a": 1})
    write_artifact(first, "b.json", {"b": 2})
    write_artifact(second, "b.json", {"b": 2})
    write_artifact(second, "a.json", {"a": 1})
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_created_at_survives_a_rebuild(tmp_path):
    write_artifact(tmp_path, "a.json", {"a": 1})
    created = read_manifest(tmp_path)["created_at"]
    write_artifact(tmp_path, "b.json", {"b": 2})
    assert read_manifest(tmp_path)["created_at"] == created, "a rebuild must not re-stamp the manifest"


def test_update_manifest_sets_header_fields_and_keeps_files(tmp_path):
    write_artifact(tmp_path, "a.json", {"a": 1})
    manifest = update_manifest(tmp_path / "manifest.json", name="t", version="v1", seed=7, git_commit=None)
    assert manifest["name"] == "t" and manifest["version"] == "v1" and manifest["seed"] == 7
    assert "git_commit" not in manifest, "None means unknown, not present"
    assert set(manifest["files"]) == {"a.json"}
    assert SuiteManifest.model_validate(json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))).seed == 7


def test_new_version_copies_forward_and_still_verifies(tmp_path):
    write_artifact(tmp_path / "v1", "a.json", {"a": 1})
    update_manifest(tmp_path / "v1" / "manifest.json", name="t", version="v1", seed=7)

    target = new_version(tmp_path, "v1")
    assert target == tmp_path / "v2"
    manifest = read_manifest(target)
    assert manifest["version"] == "v2" and manifest["previous_version"] == "v1" and manifest["seed"] == 7
    assert verify_manifest(target) == []
    assert verify_manifest(tmp_path / "v1") == [], "the source version is untouched"

    with pytest.raises(FileExistsError):
        new_version(tmp_path, "v1")
    with pytest.raises(FileNotFoundError):
        new_version(tmp_path, "v9")
