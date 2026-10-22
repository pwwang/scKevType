"""The QC probes: they must run, and they must measure the thing they claim to."""

from __future__ import annotations

from sckevtype.contracts.schema import RecordFamily
from sckevtype.data import qc, seed_int


def test_identifiability_separates_two_datasets_and_reports_a_null(smoke_build):
    records = smoke_build.records
    report = qc.dataset_identifiability(records, smoke_build.cfg, seed=seed_int(1, "qc"), n_permutations=10)
    assert report["groups"] == len({record.meta.group_id for record in records})
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["baseline_mean"] is not None
    assert 0 <= report["permutations_at_or_above"] <= 10
    assert 0.0 < report["permutation_p"] <= 1.0
    # A second call with the same seed is the same number: the probe is not a coin toss.
    assert qc.dataset_identifiability(records, smoke_build.cfg, seed=seed_int(1, "qc"), n_permutations=10) == report


def test_identifiability_with_one_dataset_has_nothing_to_predict(smoke_build):
    records = [record for record in smoke_build.records if record.meta.dataset_id == "smoke_h5ad"]
    report = qc.dataset_identifiability(records, smoke_build.cfg, seed=1)
    assert report["datasets"] == ["smoke_h5ad"]
    assert report["accuracy"] is not None  # a one-class problem is trivially solvable


def test_label_coverage_counts_cells_per_dataset_and_type(smoke_build):
    coverage = qc.label_coverage(smoke_build.records)
    assert set(coverage) == {"smoke_10x", "smoke_h5ad"}
    for entry in coverage.values():
        assert entry["cells"] == sum(entry["by_type"].values())
        assert entry["types"] == len(entry["by_type"])


def test_family_counts_are_complete_and_balanced_enough_to_learn_from(smoke_build):
    counts = qc.family_counts(smoke_build.records)
    assert set(counts) == {family.value for family in RecordFamily}
    assert all(entry["records"] > 0 for entry in counts.values())
    nool = counts[RecordFamily.MARKER_NOUL.value]["targets"]
    assert set(nool) == {"true", "false"}
    assert nool["true"] > 0


def test_panel_miss_pairing_finds_every_sibling(smoke_build):
    report = qc.panel_miss_pairing(smoke_build.records)
    assert report["miss_records"] > 0
    assert report["paired"] == report["miss_records"]
    assert report["unpaired"] == 0 and report["unpaired_groups"] == []
    assert report["siblings"] >= report["paired"]


def test_panel_miss_pairing_notices_a_missing_sibling(smoke_build):
    """Drop a miss record's correct-panel sibling: the audit must report it unpaired."""
    records = smoke_build.records
    miss = next(record for record in records if record.meta.family is RecordFamily.PANEL_MISS)
    orphaned = [
        record
        for record in records
        if not (record.meta.group_id == miss.meta.group_id and record.meta.family is RecordFamily.PANEL_CHOICE)
    ]
    report = qc.panel_miss_pairing(orphaned)
    assert report["unpaired"] >= 1
    assert miss.meta.group_id in report["unpaired_groups"]


def test_state_lengths_are_exact_and_never_truncate(smoke_build):
    records = smoke_build.records
    report = qc.state_lengths(records, smoke_build.cfg, max_tokens=384, tokens_per_symbol=3.0)
    assert sum(report["histogram"].values()) == len(records)
    assert report["min"] <= report["mean"] <= report["max"]
    assert report["over_budget"] + sum(n for length, n in report["histogram"].items() if int(length) * 3.0 <= 384) == len(records)
    # The histogram counts the symbols actually rendered: the cap is 200 by contract.
    assert report["max"] <= smoke_build.cfg.max_genes


def test_the_token_block_is_measured_only_when_a_tokenizer_is_given(smoke_build):
    """``tokens`` is a measurement, so it is absent when nothing measured it.

    The ``data`` extra installs neither kev nor transformers, so the counter arrives
    as a callable — the corpus build passes the pinned tokenizer's
    (:func:`sckevtype.contracts.budget.state_tokens`). Without one, the report keeps
    the symbol estimate and says ``tokens: null`` rather than guessing.
    """
    records = smoke_build.records
    assert qc.state_lengths(records, smoke_build.cfg, max_tokens=384)["tokens"] is None
    measured = qc.state_lengths(records, smoke_build.cfg, max_tokens=384, tokenize=lambda state: 1000)
    assert measured["tokens"]["n"] == len(records)
    assert measured["tokens"]["over_budget"] == len(records)
    assert measured["tokens"]["max_tokens"] == 384


def test_report_has_every_probe(smoke_build):
    records = smoke_build.records
    payload = qc.report(records, smoke_build.cfg, seed=seed_int(2, "qc"), max_tokens=384, tokens_per_symbol=3.0, n_permutations=0, max_features=50)
    assert set(payload) == {"counts", "checks", "state_lengths", "identifiability"}
    assert payload["counts"]["records"] == len(records)
    assert set(payload["checks"]) == {"families", "coverage", "pairing"}
    assert payload["identifiability"]["permutations"] == 0
