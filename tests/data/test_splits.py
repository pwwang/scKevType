"""Split registry: donor disjointness, tier constraints, caps, and realised counts."""

from __future__ import annotations

import pytest

from sckevtype.data import seed_int
from sckevtype.data.splits import (
    SPLIT_NAMES,
    TIER_SPLITS,
    SplitCaps,
    SplitRegistry,
    reads_test_frac,
)


def registry(**overrides) -> SplitRegistry:
    """Ten donors of ten cells each, spread over three tiers."""
    fields = {"seed": seed_int(4, "splits"), "tiers": {"a": "train", "b": "T2", "c": "T3"}, "caps": SplitCaps()}
    fields |= overrides
    splits = SplitRegistry(**fields)
    for dataset, donors in (("a", 6), ("b", 2), ("c", 2)):
        splits.register(dataset, [(f"{dataset}-cell-{donor}-{index}", f"donor-{donor}", "CL:0000624") for donor in range(donors) for index in range(10)])
    return splits.build()


def all_tier_registry(**overrides) -> SplitRegistry:
    """The three old tiers plus one dataset on ``all``, the tier that trains *and* tests.

    ``d`` carries twenty donors so that all four of its buckets are drawn from — the
    property the tier exists for, and one a four-donor atlas could not demonstrate. The
    older tiers are here so that "the new knob left them alone" is a comparison rather
    than a claim.
    """
    fields = {
        "seed": seed_int(11, "v4"),
        "tiers": {"a": "train", "b": "T2", "c": "T3", "d": "all"},
        "caps": SplitCaps(),
    }
    fields |= overrides
    splits = SplitRegistry(**fields)
    for dataset, donors in (("a", 6), ("b", 2), ("c", 2), ("d", 20)):
        splits.register(dataset, [(f"{dataset}-cell-{donor}-{index}", f"donor-{donor}", "CL:0000624") for donor in range(donors) for index in range(10)])
    return splits.build()


def assignments_of(splits: SplitRegistry, dataset: str) -> dict[str, str]:
    """``cell id -> split`` for one dataset: the whole assignment, comparable between builds."""
    return {cell_id: splits.split_of(dataset, cell_id) for cell_id, _, _ in splits.cells[dataset]}


def cells_in(splits: SplitRegistry, split: str, dataset: str) -> int:
    return sum(splits.counts().get(split, {}).get(dataset, {}).values())


def test_split_names_are_kevs_and_the_tiers_are_the_papers():
    assert SPLIT_NAMES == ("train", "calibration", "development", "test")
    assert TIER_SPLITS["train"] == ("train", "calibration", "development")
    assert TIER_SPLITS["T2"] == ("development",)
    assert TIER_SPLITS["T3"] == ("test",)
    # The fourth tier: every split, so a dataset can train the model and be tested on
    # donors of its own.
    assert TIER_SPLITS["all"] == SPLIT_NAMES
    assert reads_test_frac("all")
    assert not any(reads_test_frac(tier) for tier in ("train", "T2", "T3"))
    assert not reads_test_frac("nope") and not reads_test_frac(None)


def test_no_donor_is_in_two_splits():
    splits = registry()
    seen: dict[str, str] = {}
    for dataset, cells in splits.cells.items():
        for cell_id, donor_id, _cl_id in cells:
            key = f"{dataset}/{donor_id}"
            split = splits.split_of(dataset, cell_id)
            assert seen.setdefault(key, split) == split, f"{key} straddles {split} and {seen[key]}"


def test_every_cell_is_assigned_exactly_once():
    splits = registry()
    assigned = set(splits.assignments)
    expected = {f"{dataset}/{cell_id}" for dataset, cells in splits.cells.items() for cell_id, _, _ in cells}
    assert assigned == expected


def test_tier_decides_the_splits_a_dataset_may_feed():
    splits = registry()
    assert {splits.split_of("b", cell_id) for cell_id, _, _ in splits.cells["b"]} == {"development"}
    assert {splits.split_of("c", cell_id) for cell_id, _, _ in splits.cells["c"]} == {"test"}
    assert {splits.split_of("a", cell_id) for cell_id, _, _ in splits.cells["a"]} <= set(TIER_SPLITS["train"])


def test_the_train_tier_reaches_all_three_of_its_splits():
    """With 6 donors and fracs of 0.34, every train-tier split must be populated."""
    splits = registry(calibration_frac=0.34, development_frac=0.34)
    reached = {splits.split_of("a", cell_id) for cell_id, _, _ in splits.cells["a"]}
    assert reached == {"train", "calibration", "development"}


def test_caps_are_never_exceeded():
    """Every dataset here is a single type, so the per-type cap bites first."""
    splits = registry(caps=SplitCaps(per_dataset={"a": 25, "c": 5}, per_type=15))
    assert len(splits.cells["a"]) == 60  # the cap bit
    assert len(splits.kept("a")) == 15  # per-type 15, then the dataset cap of 25 does not bite
    assert len(splits.kept("b")) == 15  # unlisted in per_dataset: the per-type cap still applies
    assert len(splits.kept("c")) == 5  # the tighter of per-type 15 and per-dataset 5
    # With no per-type cap, the dataset cap is the only one.
    typed = registry(caps=SplitCaps(per_dataset={"a": 25}, per_type=9))
    assert len(typed.kept("a")) == 9
    assert len(registry(caps=SplitCaps(per_dataset={"a": 25})).kept("a")) == 25


def test_per_type_caps_apply_before_the_donor_draw():
    """Two types, capped at two cells each: the draw sees only what the caps kept."""
    splits = SplitRegistry(seed=seed_int(1), tiers={"a": "train"}, caps=SplitCaps(per_type=2))
    splits.register(
        "a",
        [(f"c-{cl_id[-1]}-{donor}", f"donor-{donor}", cl_id) for cl_id in ("CL:0000624", "CL:0000236") for donor in range(4)],
    )
    splits.build()
    kept = splits.kept("a")
    assert len(kept) == 4
    for cl_id in ("CL:0000624", "CL:0000236"):
        assert sum(1 for cell_id, _, cl in splits.cells["a"] if cell_id in kept and cl == cl_id) == 2
    # Capped away before the draw, so a donor whose cells were all dropped never
    # appears: the donor table holds exactly the donors that still have a cell.
    assert set(splits.donors) == {f"a/{donor}" for cell_id, donor, _ in splits.cells["a"] if cell_id in kept}


def test_sampling_is_deterministic_and_capping_is_seeded():
    first, second = registry(caps=SplitCaps(per_dataset={"a": 25})), registry(caps=SplitCaps(per_dataset={"a": 25}))
    assert first.assignments == second.assignments
    assert first.kept("a") == second.kept("a")
    other = registry(seed=seed_int(5), caps=SplitCaps(per_dataset={"a": 25}))
    assert other.assignments != first.assignments or other.kept("a") != first.kept("a")


def test_unknown_dataset_and_unregistered_cells_are_errors():
    splits = registry()
    with pytest.raises(KeyError):
        splits.split_of("nope", "cell")
    with pytest.raises(KeyError):
        splits.split_of("a", "cell-that-does-not-exist")


def test_payload_reports_realised_counts_per_dataset_and_type():
    payload = registry().payload()
    assert payload["seed"] == seed_int(4, "splits")
    assert set(payload["splits"]) == set(SPLIT_NAMES)
    assert payload["splits"]["test"]["datasets"] == {"c": {"CL:0000624": 20}}
    assert payload["splits"]["test"]["donors"] == 2
    # Dataset `a` is the train tier: all 60 of its cells land in its three splits, and
    # `c` (T3) appears in `test` alone.
    train_tier = [payload["splits"][split]["datasets"].get("a", {}).get("CL:0000624", 0) for split in TIER_SPLITS["train"]]
    assert sum(train_tier) == 60
    assert set(payload["splits"]["test"]["datasets"]) == {"c"}
    assert all(set(payload["splits"][split]["datasets"]) <= {"a", "b"} for split in TIER_SPLITS["train"])
    assert all(split in TIER_SPLITS["train"] + TIER_SPLITS["T2"] + TIER_SPLITS["T3"] for split in payload["donors"].values())


def test_a_registry_must_be_built_before_it_is_read():
    splits = SplitRegistry(seed=1, tiers={"a": "train"})
    splits.register("a", [("c", "d", "CL:0000624")])
    with pytest.raises(RuntimeError, match="build"):
        splits.split_of("a", "c")


# --- the `all` tier: one dataset trains and is tested on its own donors ---------


def test_a_dataset_on_the_all_tier_feeds_all_four_splits():
    """The property the tier exists for: the atlas trains the model *and* tests it."""
    splits = all_tier_registry()
    assert set(assignments_of(splits, "d").values()) == set(SPLIT_NAMES)
    for split in SPLIT_NAMES:
        assert cells_in(splits, split, "d") > 0
        assert splits.donor_counts()[split]["d"] > 0


def test_no_donor_on_the_all_tier_straddles_a_split():
    """Four splits, still one donor table: a donor is in exactly one of them."""
    splits = all_tier_registry()
    by_donor: dict[str, str] = {}
    for dataset, cells in splits.cells.items():
        for cell_id, donor_id, _cl_id in cells:
            key = f"{dataset}/{donor_id}"
            split = splits.split_of(dataset, cell_id)
            assert by_donor.setdefault(key, split) == split, f"{key} straddles {split} and {by_donor[key]}"
    # ...and it is the same partition the registry reports: no donor assigned a split
    # that no cell of it is in, no donor missing from the table.
    assert by_donor == splits.donors
    assert set(splits.donors.values()) == set(SPLIT_NAMES)


def test_the_test_fraction_moves_the_test_count():
    """The knob, on one seed and one donor set: a bigger bucket is more test cells."""
    small, large = all_tier_registry(test_frac=0.05), all_tier_registry(test_frac=0.5)
    assert 0 < cells_in(small, "test", "d") < cells_in(large, "test", "d")
    # Only the boundary moved: every cell is in one of the four splits either way, and
    # what test gained, train gave up.
    assert sum(cells_in(small, split, "d") for split in SPLIT_NAMES) == sum(cells_in(large, split, "d") for split in SPLIT_NAMES)
    assert cells_in(small, "train", "d") + cells_in(small, "test", "d") == cells_in(large, "train", "d") + cells_in(large, "test", "d")
    # The test bucket is the interval [calibration + development, + test) of one uniform
    # draw per donor, so widening it can only add donors, never swap them.
    def test_donors(fraction: float) -> set[str]:
        return {key for key, split in all_tier_registry(test_frac=fraction).donors.items() if split == "test" and key.startswith("d/")}

    assert test_donors(0.05) <= test_donors(0.2) <= test_donors(0.5)


def test_the_old_tiers_are_unchanged_by_the_test_fraction():
    """A dataset on `train` gets no test cells, `T2` only development, `T3` only test —
    and `test_frac` cannot move any of them, at either end of its range.

    This is what keeps the corpus versions before v4 reproducible: their tiers never
    advance the donor draw past development, so the draw they make is the one they made
    before the test bucket existed.
    """
    off, saturated = all_tier_registry(test_frac=0.0), all_tier_registry(test_frac=0.9)
    for dataset in ("a", "b", "c"):
        assert assignments_of(off, dataset) == assignments_of(saturated, dataset)
    reached = set(assignments_of(off, "a").values())
    assert "test" not in reached and reached <= set(TIER_SPLITS["train"])
    assert set(assignments_of(off, "b").values()) == set(TIER_SPLITS["T2"]) == {"development"}
    assert set(assignments_of(off, "c").values()) == set(TIER_SPLITS["T3"]) == {"test"}


def test_the_payload_records_the_test_fraction():
    """The manifest's split block carries the knob, so a corpus says how it was drawn."""
    payload = all_tier_registry(test_frac=0.3).payload()
    assert payload["test_frac"] == 0.3
    assert payload["calibration_frac"] == 0.1 and payload["development_frac"] == 0.1
    assert payload["splits"]["test"]["datasets"]["d"], "the all-tier dataset is in the test table"
    assert set(payload["splits"]["test"]["datasets"]) <= {"c", "d"}, "only T3 and the all tier reach test"
