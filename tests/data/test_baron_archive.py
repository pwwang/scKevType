"""The Baron archive adapter, and the corpus config that wires it.

Baron (``GSE84133``) is the one atlas the factory reads out of a *tar of six
per-donor tables* rather than one table per dataset, so the two things worth
testing are the ones a hand-rolled reader gets wrong:

* the donor is the **member name**, and it is the only thing that separates two
  files' barcodes — 308 barcode memberships are shared between files, so an
  un-namespaced id would merge two donors' cells into one split-registry key;
* 104 barcodes repeat **inside** one file (Baron's inDrop barcodes are not unique),
  so an id that is not disambiguated silently loses a cell: the registry keys cells
  by id, so two rows with one id are one cell to the corpus.

Both are exercised on a two-member synthetic archive, built here so the test needs
no 29 MB download. A final test reads the real archive when it is staged and skips
when it is not, so the numbers this corpus was built on are checked on a box that
has the bytes instead of being restated in a comment.
"""

from __future__ import annotations

import csv
import gzip
import io
import tarfile
from pathlib import Path

import pytest

from sckevtype.contracts.ontology import LabelMap
from sckevtype.data import adapters
from sckevtype.data.adapters import LabelStats, baron_archive
from sckevtype.data.build import Artifacts, load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "corpus" / "baron-external.yaml"
ARCHIVE = ROOT / "data" / "raw" / "baron" / "GSE84133_RAW.tar"

#: The mapping the corpus config carries, as a LabelMap: nine labels in, six dropped.
MAPPING = {
    "beta": "CL:0000169",
    "alpha": "CL:0000171",
    "delta": "CL:0000173",
    "gamma": "CL:0002275",
    "ductal": "CL:0002079",
    "endothelial": "CL:0000115",
    "macrophage": "CL:0000235",
    "mast": "CL:0000097",
    "t_cell": "CL:0000084",
    "T_cell": "CL:0000084",
    "B_cell": "CL:0000236",
}


def _member(name: str, header: list[str], rows: list[list[str]]) -> tuple[str, bytes]:
    """One ``*_umifm_counts.csv.gz`` member, gzipped exactly as the archive holds it."""
    body = io.StringIO()
    writer = csv.writer(body, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return name, gzip.compress(body.getvalue().encode("utf-8"))


@pytest.fixture
def archive(tmp_path) -> Path:
    """A two-member archive: one human file with a repeated barcode, one mouse file.

    The repeated ``bc2`` carries a *mappable* label (``delta``), so it is a real cell
    the corpus keeps — a repeat that also happened to be unmappable would be dropped
    before it was ever keyed. ``acinar``, a label the register cannot name (its term,
    CL:0002064, is not in scope), sits on its own row, and the mouse file re-uses
    ``bc1``: the two failure modes above, on separate rows.
    """
    path = tmp_path / "GSE84133_RAW.tar"
    columns = ["", "barcode", "assigned_cluster", "ACTB", "INS", "GCG"]
    mouse_columns = ["", "barcode", "assigned_cluster", "Actb", "Ins1"]
    human = _member(
        "GSM0000001_humanX_umifm_counts.csv.gz",
        columns,
        [
            ["0", "bc1", "beta", "5", "9", "0"],
            ["1", "bc2", "alpha", "3", "0", "7"],
            ["2", "bc2", "delta", "4", "0", "1"],
            ["3", "bc3", "acinar", "0", "2", "0"],
        ],
    )
    mouse = _member(
        "GSM0000002_mouseY_umifm_counts.csv.gz",
        mouse_columns,
        [
            ["0", "bc1", "beta", "1", "2"],
            ["1", "bc9", "T_cell", "1", "0"],
        ],
    )
    with tarfile.open(path, "w") as tar:
        for name, payload in (human, mouse):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


def _label_map() -> LabelMap:
    return LabelMap(dataset_id="baron", mapping=dict(MAPPING), source="test")


def test_donor_of_reads_the_member_name():
    assert baron_archive.donor_of("GSM2230757_human1_umifm_counts.csv.gz") == "human1"
    assert baron_archive.donor_of("GSM2230761_mouse1_umifm_counts.csv.gz") == "mouse1"
    # An unexpected name keeps its stem: unique and visible beats invented.
    assert baron_archive.donor_of("sample7_counts.csv.gz") == "sample7_counts"


def test_the_format_is_registered_alongside_the_other_adapters():
    assert adapters.ADAPTERS["baron_archive"] == "baron_archive"
    assert adapters.adapter_for("baron_archive") is baron_archive


def test_probe_reports_all_members_not_just_the_first(archive):
    info = baron_archive.probe(archive)
    assert info.format == "baron_archive"
    assert info.n_cells == 6                      # 4 human rows + 2 mouse rows
    assert info.n_genes == 3                     # the long member's gene count, not the short one's
    assert info.obs_keys == ("assigned_cluster", "barcode")
    # The probe reports the file, never the corpus's map: `acinar` is visible here
    # even though the corpus drops it.
    assert info.labels == ("T_cell", "acinar", "alpha", "beta", "delta")
    assert info.label_counts["beta"] == 2


def test_probe_refuses_a_tar_that_is_not_this_archive(tmp_path):
    other = tmp_path / "other.tar"
    with tarfile.open(other, "w") as tar:
        payload = gzip.compress(b"a,b\n1,2\n")
        info = tarfile.TarInfo("notes.txt.gz")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="not the Baron archive"):
        baron_archive.probe(other)


def test_cells_are_namespaced_by_donor_and_repeats_are_disambiguated(archive):
    cells = list(baron_archive.to_cells(archive, _label_map(), default_tissue="pancreas"))
    ids = [cell.cell_id for cell in cells]
    assert ids == ["humanX:bc1", "humanX:bc2", "humanX:bc2#2", "mouseY:bc1", "mouseY:bc9"]
    # Every id is unique, which is what the split registry keys on; the two `bc1`
    # rows are two donors' cells and must not collapse into one.
    assert len(set(ids)) == len(ids)
    assert [cell.donor_id for cell in cells] == ["humanX"] * 3 + ["mouseY"] * 2
    assert {cell.tissue for cell in cells} == {"pancreas"}
    assert {cell.dataset_id for cell in cells} == {"baron"}


def test_unmappable_labels_are_dropped_and_counted(archive):
    stats = LabelStats()
    cells = list(baron_archive.to_cells(archive, _label_map(), default_tissue="pancreas", stats=stats))
    assert stats.total == 6 and stats.mapped == len(cells) == 5
    assert stats.dropped == 1
    assert stats.dropped_labels["acinar"] == 1
    # Dropped, never coerced: no cell carries a CL id the map does not name.
    assert {cell.cl_id for cell in cells} == {"CL:0000169", "CL:0000171", "CL:0000173", "CL:0000084"}


def test_expression_and_only_behave_like_the_other_adapters(archive):
    labelled = list(baron_archive.to_cells(archive, _label_map(), default_tissue="pancreas", expression=False))
    assert labelled and all(cell.genes == () and cell.values == () for cell in labelled)

    full = list(baron_archive.to_cells(archive, _label_map(), default_tissue="pancreas"))
    beta = next(cell for cell in full if cell.cell_id == "humanX:bc1")
    # `barcode`/`assigned_cluster`/the unnamed row key are metadata, not genes: a
    # metadata column read as a gene is a crash on the first non-numeric value.
    assert beta.genes == ("ACTB", "INS") and beta.values == (5.0, 9.0)

    only = list(baron_archive.to_cells(archive, _label_map(), default_tissue="pancreas", only={"humanX:bc2"}))
    assert [cell.cell_id for cell in only] == ["humanX:bc2"]
    # The suffix is assigned before `only` is applied, so the second `bc2` keeps its
    # own id and is filtered on it rather than on the stem.
    both = list(baron_archive.to_cells(archive, _label_map(), default_tissue="pancreas", only={"humanX:bc2", "humanX:bc2#2"}))
    assert [cell.cell_id for cell in both] == ["humanX:bc2", "humanX:bc2#2"]


# --------------------------------------------------------------------------- the config


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def register() -> dict[str, str]:
    """The v2 register's in-scope ``cl_id -> name``, the set the label map is a scope over."""
    artifacts = Artifacts(ROOT / "artifacts" / "v2-at-build")
    return {cl_id: term.name for cl_id, term in artifacts.definitions.by_cl_id.items()}


def test_the_external_atlas_is_held_out_and_fully_scored(config):
    baron = next(dataset for dataset in config.datasets if dataset.id == "baron")
    # `T3`: every donor goes to `test`, so the scored population is the whole atlas
    # rather than a donor-level quarter of it. That is the property of this corpus.
    assert baron.tier == "T3"
    assert baron.format == "baron_archive"
    assert baron.cap == 2600
    assert baron.label_column == "assigned_cluster"
    assert baron.tissue == "pancreas"
    # The v4 scoring surface and nothing else.
    assert [family.value for family in config.families] == ["panel_choice", "panel_miss", "marker_noul", "evidence_diluted"]
    # The cells the register cannot name are counted, not built.
    assert config.splits.unregistered == {"baron": 0, "muraro": 0}


def test_every_target_of_the_label_map_is_an_in_scope_register_term(config, register):
    baron = next(dataset for dataset in config.datasets if dataset.id == "baron")
    mapping = baron.label_map
    assert isinstance(mapping, dict)
    assert set(mapping) == set(MAPPING)
    assert "beta" in mapping and mapping["beta"] == "CL:0000169"       # type B pancreatic cell
    assert mapping["alpha"] == "CL:0000171"                            # pancreatic A cell — not beta
    for label, cl_id in mapping.items():
        assert cl_id in register, f"{label} -> {cl_id} is not in the register's in-scope set"
    # The labels whose CL term is out of scope are absent from the map: they are
    # dropped and counted by the build, never coerced to a near miss. In particular
    # `acinar` must not resolve to CL:0002623 ("acinar cell of salivary gland").
    for label in ("acinar", "activated_stellate", "quiescent_stellate", "schwann", "epsilon", "immune_other"):
        assert label not in mapping
    assert "CL:0002623" not in set(mapping.values())


@pytest.mark.skipif(not ARCHIVE.exists(), reason=f"{ARCHIVE.relative_to(ROOT)} is not staged on this box")
def test_the_staged_archive_is_the_one_the_corpus_was_built_on(config):
    """The archive's own numbers: the cells, the genes, and the in-scope rate."""
    info = baron_archive.probe(ARCHIVE)
    assert info.n_cells == 10455
    assert info.n_genes == 20125
    assert len(info.labels) == 17

    baron = next(dataset for dataset in config.datasets if dataset.id == "baron")
    mapping = baron.label_map
    assert isinstance(mapping, dict)
    stats = LabelStats()
    cells = list(baron_archive.to_cells(ARCHIVE, LabelMap(dataset_id="baron", mapping=mapping, source="test"),
                                        default_tissue="pancreas", stats=stats, expression=False))
    assert stats.total == 10455
    assert len(cells) == stats.mapped == 8934
    assert stats.dropped == 1521
    assert stats.dropped_labels == {
        "acinar": 958,
        "activated_stellate": 298,
        "epsilon": 18,
        "immune_other": 8,
        "quiescent_stellate": 220,
        "schwann": 19,
    }
    # Six donors, and every one of them survives the corpus's cap: mouse1 is the
    # smallest at 782 in-scope cells, which is why a 29% draw cannot empty it.
    assert {cell.donor_id for cell in cells} == {"human1", "human2", "human3", "human4", "mouse1", "mouse2"}
    assert len({cell.cell_id for cell in cells}) == len(cells)
