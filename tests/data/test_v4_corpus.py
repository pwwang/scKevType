"""The v4 corpus design: every atlas trains, and every atlas keeps donors back as test.

`sckevtype/data/splits.py`'s donor draw is pinned in `test_splits.py`. This file checks
the three things a *corpus* can get wrong: that `configs/corpus/real-v4.yaml` is real-v3
with the tiers moved and nothing else, that the corpora already on disk are neither
edited nor re-split by the change, and that the new tier survives a whole build — on the
smoke fixtures, since no atlas is read here and an atlas build is an hour of I/O.

The claim the tier is for is *per dataset*: each of the six atlases ends up with donors
of its own in `test`, so "does the model know cell types" is no longer the same question
as "does it carry to another lab's protocol". It is not a claim that a four-donor atlas
fills all four of its buckets — with four donors and buckets of 0.1 / 0.1 / 0.2, an empty
calibration or development is a fact of donor-level splitting, and the tests below say
which of the two claims is being made where.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sckevtype.data import registry as registry_module
from sckevtype.data.build import CorpusConfig, SplitConfig, build, load_config
from sckevtype.data.splits import SPLIT_NAMES, SplitRegistry

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "configs" / "corpus" / "real-v4.yaml"
V3 = ROOT / "configs" / "corpus" / "real-v3.yaml"
SMOKE = ROOT / "configs" / "corpus" / "smoke.yaml"
#: The frozen v3 corpus: its manifest's donor table is the record of what the old tiers
#: did, and every test below that talks about "unchanged" is measured against it.
FROZEN = ROOT / "artifacts" / "corpus" / "real-v3" / "manifest.json"
#: The tiers the v1–v3 design put the six atlases on, i.e. the registry's own.
V3_TIERS = {
    "muraro": "T3",
    "hao_cite": "train",
    "dominguez_conde": "T2",
    "tabula_muris_senis": "T3",
    "siletti": "T3",
    "tabula_sapiens": "T3",
}


@pytest.fixture(scope="module")
def v4() -> CorpusConfig:
    return load_config(V4)


@pytest.fixture(scope="module")
def v3() -> CorpusConfig:
    return load_config(V3)


def _as_the_old_design(config: CorpusConfig) -> dict:
    """The payload a config unpacks to with the three v4 differences normalised away.

    `test_frac` is a field of *every* config once the schema has it — that is what a
    default means — so it comes out here beside the corpus's name and each dataset's
    tier: the things real-v4 is allowed to differ in. What is left is every other value
    the two configs share, compared as one block rather than field by field.
    """
    payload = config.model_dump(mode="json")
    payload["name"] = "real-v3"
    payload["splits"].pop("test_frac", None)
    for dataset in payload["datasets"]:
        dataset["tier"] = V3_TIERS[dataset["id"]]
    return payload


def test_the_v4_config_is_real_v3_with_the_tiers_moved_and_nothing_else(v4, v3):
    """Same seed, register, atlases, caps, label maps, panel draw and state budget."""
    assert v4.name == "real-v4"
    assert v4.version == v3.version and v4.artifacts == v3.artifacts and v4.cache == v3.cache
    assert v4.seed == v3.seed and v4.variant_policy == v3.variant_policy
    assert v4.panel == v3.panel and v4.state == v3.state
    assert v4.families == v3.families and v4.holds == v3.holds and v4.axis == v3.axis
    assert [dataset.id for dataset in v4.datasets] == [dataset.id for dataset in v3.datasets]
    assert _as_the_old_design(v4) == _as_the_old_design(v3)
    assert {dataset.id: dataset.tier for dataset in v4.datasets} == {dataset.id: "all" for dataset in v3.datasets}


def test_the_option_surface_is_the_marker_free_one(v4):
    """v4 inherits v3's marker strip (the shortcut `panel.option_markers: false` closes)."""
    assert v4.panel.option_markers is False
    assert v4.variant_policy == "single"


def test_real_v3_still_describes_the_old_three_way_design(v3):
    """The file the v3 corpus was built from is not edited, and names no v4 tier."""
    assert {dataset.id: dataset.tier for dataset in v3.datasets} == V3_TIERS
    assert (v3.splits.calibration_frac, v3.splits.development_frac) == (0.1, 0.1)


def test_the_shared_registry_still_describes_the_v1_to_v3_design():
    """`configs/datasets.yaml` is not edited: the new tier lives in the corpus config.

    The registry has no `all` entry and is not asked to grow one — a tier is a corpus's
    decision about a dataset (train it here, test it there), while the registry records
    what an atlas *is*. `configs/datasets-v4.yaml` does not exist for the same reason:
    there is nothing for the corpus to point at.
    """
    tiers = {entry.id: entry.tier for entry in registry_module.load().datasets}
    assert set(tiers.values()) <= {"train", "T2", "T3"}
    assert tiers["hao_cite"] == "train" and tiers["muraro"] == tiers["tabula_sapiens"] == "T3"
    assert not (ROOT / "configs" / "datasets-v4.yaml").exists()


def test_the_papers_default_test_fraction_is_point_two(v4):
    assert SplitConfig().test_frac == 0.2
    assert SplitRegistry(seed=1, tiers={"x": "all"}).test_frac == 0.2
    assert (v4.splits.calibration_frac, v4.splits.development_frac, v4.splits.test_frac) == (0.1, 0.1, 0.2)


def test_a_config_that_reads_the_test_fraction_must_leave_room_for_train():
    """The check is tier-aware: it bites where the fraction is read, and only there."""
    dataset = {"id": "x", "registered": False, "format": "csv_expr", "tier": "all", "path": "/tmp/x", "cap": 1,
               "label_map": {"CL:0000624": "CL:0000624"}}
    with pytest.raises(ValueError, match="leave something for train"):
        CorpusConfig(name="x", seed=1, datasets=[dataset], state={"max_tokens": 384},
                     splits={"calibration_frac": 0.5, "development_frac": 0.4, "test_frac": 0.2})
    # The same three numbers on a tier that cannot reach test: nothing is drawn from the
    # sum, so a corpus that names them is not refused.
    old = CorpusConfig(name="x", seed=1, datasets=[{**dataset, "tier": "train"}], state={"max_tokens": 384},
                       splits={"calibration_frac": 0.5, "development_frac": 0.4, "test_frac": 0.2})
    assert old.splits.test_frac == 0.2
    # ...which is what keeps the smoke corpus (0.34 / 0.34) a valid config.
    assert load_config(SMOKE).splits.test_frac == 0.2


def test_the_frozen_v3_donors_are_not_re_split_by_the_new_code():
    """Every donor of the frozen v3 corpus lands in the split its manifest recorded.

    The new code is handed the same seed, the same tiers (from the frozen manifest's own
    dataset table) and the same number of donors, one cell each: if the added test bucket
    moved a single donor, the assignment the v3 corpus was drawn with would not come back.
    """
    manifest = json.loads(FROZEN.read_text(encoding="utf-8"))
    tiers = {dataset_id: entry["tier"] for dataset_id, entry in manifest["datasets"].items()}
    splits = SplitRegistry(
        seed=manifest["splits"]["seed"],
        tiers=tiers,
        calibration_frac=manifest["splits"]["calibration_frac"],
        development_frac=manifest["splits"]["development_frac"],
        test_frac=manifest["splits"].get("test_frac", 0.2),
    )
    frozen_donors: dict[str, str] = manifest["splits"]["donors"]
    assert frozen_donors, "the frozen manifest has no donor table to check"
    for donor_key in frozen_donors:
        dataset_id, donor_id = donor_key.split("/", 1)
        splits.register(dataset_id, [(f"cell-{donor_id}", donor_id, "CL:0000624")])
    splits.build()
    assert splits.donors == frozen_donors
    assert set(splits.donors.values()) <= {"train", "calibration", "development", "test"}


def test_the_v4_draw_gives_every_atlas_test_donors():
    """The design claim, checked on the donor sets the v3 corpus recorded.

    v4 keeps v3's seed, caps and atlases, so the donors it draws are the ones the frozen
    manifest lists; what changes is that they are drawn four ways. Every atlas must reach
    `test` (that is the point), and the corpus's four splits must all be populated. The
    four-donor atlases (muraro, siletti) are *not* required to fill calibration and
    development as well: three buckets over four donors cannot be a guarantee, and the
    three-way buckets are shared by every atlas in the pooled corpus anyway.
    """
    manifest = json.loads(FROZEN.read_text(encoding="utf-8"))
    frozen_donors: dict[str, str] = manifest["splits"]["donors"]
    splits = SplitRegistry(seed=manifest["splits"]["seed"], tiers={dataset_id: "all" for dataset_id in manifest["datasets"]},
                           calibration_frac=0.1, development_frac=0.1, test_frac=0.2)
    for donor_key in frozen_donors:
        dataset_id, donor_id = donor_key.split("/", 1)
        splits.register(dataset_id, [(f"cell-{donor_id}", donor_id, "CL:0000624")])
    splits.build()
    assert set(splits.donors) == set(frozen_donors), "the donor set is v3's — same seed, same caps, same atlases"
    donors = splits.donor_counts()
    for dataset_id in sorted(manifest["datasets"]):
        assert donors.get("test", {}).get(dataset_id, 0) > 0, f"{dataset_id} has no test donors"
    assert all(sum(donors.get(split, {}).values()) > 0 for split in SPLIT_NAMES)
    # If the real build has landed, its donor table is the check on this prediction.
    built = FROZEN.parent / "real-v4" / "manifest.json"
    if built.exists():
        real = json.loads(built.read_text(encoding="utf-8"))
        assert real["splits"]["donors"] == splits.donors, "the built v4 corpus drew other donors than the prediction"
        assert real["splits"]["test_frac"] == 0.2


def test_a_whole_build_on_the_new_tier_reaches_all_four_splits(tmp_path):
    """The knob is wired end to end: config -> registry -> the four written partitions.

    The smoke fixtures are the only datasets here, and they are read from the committed
    fixture directory (`configs/corpus/fixtures/smoke/data`), which the fixture
    generator's own `--check` test pins against a fresh generation.
    """
    text = SMOKE.read_text(encoding="utf-8")
    text = text.replace("tier: train", "tier: all").replace("tier: T3", "tier: all")
    text = text.replace("calibration_frac: 0.34", "calibration_frac: 0.15")
    text = text.replace("development_frac: 0.34", f"development_frac: 0.15\n  test_frac: {'0.5'}")
    config = tmp_path / "smoke-v4.yaml"
    config.write_text(text, encoding="utf-8")
    out = tmp_path / "corpus"
    manifest = build(config, out)

    assert manifest["splits"]["test_frac"] == 0.5
    for split in SPLIT_NAMES:
        assert manifest["counts"]["per_split"][split] > 0, f"{split} came out empty"
        assert manifest["splits"]["splits"][split]["cells"] > 0
        assert (out / f"{split}.jsonl").stat().st_size > 0
    assert set(manifest["splits"]["splits"]["test"]["datasets"]) & {"smoke_h5ad", "smoke_10x"}
    donors = manifest["splits"]["donors"]
    assert all(split in SPLIT_NAMES for split in donors.values())
    assert len(set(donors)) == len(donors), "a donor key appears twice"
    assert {donor_key.split("/", 1)[0] for donor_key in donors} == {"smoke_h5ad", "smoke_10x"}


def test_the_test_fraction_leaves_a_train_or_T3_corpus_byte_identical(tmp_path):
    """`test_frac` at either end of its range, on the tiers v1–v3 used: same bytes.

    The smoke fixtures are the datasets here, and they are on the old tiers — a `train`
    one and a `T3` one, the two v1–v3 corpora were built from (a `T2` dataset is the same
    code path as `T3`: a single split, drawn without a fraction). Nothing a v3 config
    reads moves: not the four partitions, not the donor table, not the per-split counts.
    The manifest itself does carry the fraction, because the manifest says how the corpus
    was drawn — which is why a v3 corpus rebuilt with this code differs in that one field
    and in nothing else.
    """
    base = SMOKE.read_text(encoding="utf-8")
    assert "tier: train" in base and "tier: T3" in base and "tier: all" not in base
    builds = {}
    for fraction in ("0.0", "0.9"):
        text = base.replace("development_frac: 0.34", f"development_frac: 0.34\n  test_frac: {fraction}")
        config = tmp_path / f"smoke-test-frac-{fraction}.yaml"
        config.write_text(text, encoding="utf-8")
        out = tmp_path / f"out-{fraction}"
        builds[fraction] = (build(config, out), out)

    (off_manifest, off_out), (up_manifest, up_out) = builds["0.0"], builds["0.9"]
    assert off_manifest["splits"]["test_frac"] == 0.0 and up_manifest["splits"]["test_frac"] == 0.9
    assert off_manifest["splits"]["donors"] == up_manifest["splits"]["donors"]
    assert off_manifest["splits"]["splits"] == up_manifest["splits"]["splits"]
    for split in SPLIT_NAMES:
        name = f"{split}.jsonl"
        assert off_manifest["files"][name]["sha256"] == up_manifest["files"][name]["sha256"], f"{name} moved"
        assert (off_out / name).read_bytes() == (up_out / name).read_bytes()
    assert off_manifest["counts"]["per_split"] == up_manifest["counts"]["per_split"]
