"""The gene vocabulary: coverage reporting, digest-pinned save/load, tokenizer probe."""

from __future__ import annotations

import pytest

from sckevtype.contracts.freeze import read_json
from sckevtype.contracts.vocab import VocabConfig, Vocabulary, build_vocabulary


def test_build_dedupes_and_sorts():
    vocabulary = build_vocabulary(["CD3D", "LYZ", "CD3D", "  ", "MS4A1"])
    assert vocabulary.symbols == ("CD3D", "LYZ", "MS4A1")
    assert "LYZ" in vocabulary and "GAPDH" not in vocabulary
    assert len(vocabulary) == 3


def test_source_digest_tracks_the_input_list_not_its_order():
    assert build_vocabulary(["CD3D", "LYZ"]).source_digest == build_vocabulary(["LYZ", "CD3D"]).source_digest
    assert build_vocabulary(["CD3D"]).source_digest != build_vocabulary(["CD3D", "LYZ"]).source_digest


def test_coverage_reports_the_out_of_vocabulary_rate():
    vocabulary = build_vocabulary(["CD3D", "CD3E", "LYZ"])
    report = vocabulary.check_coverage(["CD3D", "CD3E", "LYZ", "GAPDH"])
    assert (report.total, report.mapped, report.unmapped) == (4, 3, 1)
    assert report.rate == pytest.approx(0.75)
    assert report.unmapped_items == ("GAPDH",)
    assert report.ok(0.75) and not report.ok(0.9)
    assert vocabulary.check_coverage([]).rate == 1.0


def test_save_load_round_trips_with_a_stable_digest(tmp_path):
    vocabulary = build_vocabulary(["CD3D", "LYZ"], VocabConfig(name="t", version="v9"))
    path = tmp_path / "vocabulary.json"
    digest = vocabulary.save(str(path))
    assert digest == vocabulary.digest()
    assert read_json(path)["digest"] == digest
    loaded = Vocabulary.load(str(path))
    assert loaded == vocabulary and loaded.digest() == digest and loaded.version == "v9"


def test_load_refuses_a_tampered_vocabulary(tmp_path):
    path = tmp_path / "vocabulary.json"
    build_vocabulary(["CD3D", "LYZ"]).save(str(path))
    path.write_text(path.read_text(encoding="utf-8").replace("CD3D", "GAPDH"), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        Vocabulary.load(str(path))


def test_tokenizer_probe_hook_is_optional_and_filters_by_cost():
    def tokens(symbol: str) -> int:
        return len(symbol)  # stand-in for a real tokenizer

    probed = build_vocabulary(["A", "AB", "ABCDEF"], VocabConfig(max_tokens=2), probe=tokens)
    assert probed.symbols == ("A", "AB")
    assert probed.probe == "tokens"
    assert probed.digest() != build_vocabulary(["A", "AB", "ABCDEF"], VocabConfig(max_tokens=2)).digest()

    unprobed = build_vocabulary(["A", "AB", "ABCDEF"])
    assert unprobed.symbols == ("A", "AB", "ABCDEF") and unprobed.probe is None


def test_config_digest_tracks_every_field():
    assert VocabConfig().digest() == VocabConfig().digest()
    assert VocabConfig().digest() != VocabConfig(max_tokens=3).digest()
    assert VocabConfig().digest() != VocabConfig(version="seed-2").digest()
