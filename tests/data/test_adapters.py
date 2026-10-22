"""The three adapters: what they find in a file, and the cells they hand back."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from sckevtype.contracts.ontology import LabelMap
from sckevtype.data import adapters
from sckevtype.data.adapters import LabelStats, csv_expr, probe, tenx_mtx, to_cells
from sckevtype.data.adapters import cellxgene_h5ad as h5
from sckevtype.data.registry import ObsColumns

H5AD_COLUMNS = ObsColumns(label="cell_type_ontology_term_id", donor="donor_id", tissue="tissue_ontology_term_id")


def _label_map(dataset_id: str, *cl_ids: str) -> LabelMap:
    return LabelMap(dataset_id=dataset_id, mapping={cl_id: cl_id for cl_id in cl_ids}, source="test")


def test_probe_reports_columns_and_label_vocabulary(smoke_data):
    info = probe("cellxgene_h5ad", smoke_data / "cellxgene_lung.h5ad", columns=H5AD_COLUMNS)
    assert info.format == "cellxgene_h5ad"
    assert sum(info.label_counts.values()) == info.n_cells
    assert info.obs_keys == ("_index", "cell_type_ontology_term_id", "donor_id", "tissue_ontology_term_id")
    assert info.label_column == "cell_type_ontology_term_id"
    assert info.donor_column == "donor_id"
    assert info.tissue_column == "tissue_ontology_term_id"
    assert "CL:0000084" in info.labels
    # The index holds Ensembl ids, the symbols are in feature_name: the probe must have
    # preferred the symbol column, or every state would be rendered from accession numbers.
    assert all(not gene.startswith("ENSG") for gene in info.genes_head)
    assert info.genes_head


def test_tenx_probe_finds_the_silver_standard_sidecar(smoke_data):
    info = probe("tenx_mtx", smoke_data / "tenx_blood")
    assert info.format == "tenx_mtx"
    assert info.n_cells == 15
    assert info.n_genes == 41
    assert "CL:0000624" in info.labels
    # The unmappable label is still in the vocabulary: dropping is the label map's job,
    # and the probe reports the file, not the filter.
    assert "Doublet" in info.labels


def test_cellxgene_to_cells_maps_drops_and_counts(smoke_data):
    stats = LabelStats()
    cells = list(to_cells("cellxgene_h5ad", smoke_data / "cellxgene_lung.h5ad", _label_map("smoke", "CL:0000084", "CL:0000623"), columns=H5AD_COLUMNS, default_tissue="lung", stats=stats))
    assert cells and {cell.tissue for cell in cells} == {"lung"}
    assert {cell.cl_id for cell in cells} == {"CL:0000084", "CL:0000623"}
    assert {cell.dataset_id for cell in cells} == {"smoke"}
    assert all(cell.donor_id for cell in cells)
    # Dropped, and *counted* — the number the paper reports as label-map coverage.
    # Everything outside the map is dropped, not silently mixed into a fixed label space.
    assert stats.dropped == sum(stats.dropped_labels.values()) == stats.total - len(cells)
    assert stats.dropped_labels["CL:9999999"] > 0
    assert stats.mapped == len(cells)
    assert stats.payload()["mapped_rate"] == pytest.approx(stats.rate)


def test_tenx_to_cells_needs_a_label_sidecar(smoke_data, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("matrix.mtx", "barcodes.tsv", "features.tsv"):
        (bundle / name).write_bytes((smoke_data / "tenx_blood" / name).read_bytes())
    with pytest.raises(ValueError, match="no label sidecar"):
        list(to_cells("tenx_mtx", bundle, _label_map("smoke", "CL:0000624")))
    # And with one, the barcodes the sidecar does not name are dropped, not defaulted.
    labels = tmp_path / "labels.tsv"
    labels.write_text("d1_cell0\tCL:0000624\nd2_cell0\tnot_a_type\n", encoding="utf-8")
    stats = LabelStats()
    cells = list(
        to_cells("tenx_mtx", bundle, _label_map("smoke", "CL:0000624"), label_file=str(labels), default_tissue="blood", stats=stats)
    )
    assert [cell.cell_id for cell in cells] == ["d1_cell0"]
    assert stats.dropped_labels["not_a_type"] == 1


def test_tenx_labels_are_positional_without_a_header(smoke_data):
    labels = tenx_mtx.read_labels(smoke_data / "tenx_blood" / "labels.tsv")
    assert labels["d1_cell0"] == "CL:0000624"


def test_matrixmarket_reader_is_one_based_and_buckets_per_cell(smoke_data):
    (genes, cells, nonzeros), rows = tenx_mtx.read_matrix(smoke_data / "tenx_blood" / "matrix.mtx")
    assert (genes, cells) == (41, 15)
    assert len(rows) == cells
    assert sum(len(row) for row in rows) == nonzeros
    assert all(0 <= gene < genes and value > 0 for row in rows for gene, value in row)


def test_csv_adapter_reads_a_genes_by_cells_table(tmp_path):
    table = tmp_path / "sample_a.csv"
    table.write_text(
        "cell,barcode,assigned_cluster,CD3D,MS4A1,ACTB\nc1,b1,alpha,4,0,9\nc2,b2,beta,0,3,8\n",
        encoding="utf-8",
    )
    columns = ObsColumns(label="assigned_cluster")
    cells = list(
        to_cells(
            "csv_expr",
            table,
            LabelMap(dataset_id="baron", mapping={"alpha": "CL:0000624", "beta": "CL:0000236"}, source="test"),
            columns=columns,
            donor_from_stem=True,
            default_tissue="pancreas",
        )
    )
    assert [cell.cl_id for cell in cells] == ["CL:0000624", "CL:0000236"]
    # Only non-zero genes, and the metadata columns are not mistaken for genes.
    assert cells[0].genes == ("CD3D", "ACTB")
    assert cells[0].values == (4.0, 9.0)
    assert cells[0].donor_id == "sample_a"
    assert csv_expr.header(table)[:3] == ["cell", "barcode", "assigned_cluster"]


def test_csv_adapter_rejects_a_non_numeric_gene_column(tmp_path):
    table = tmp_path / "broken.csv"
    table.write_text("cell,assigned_cluster,CD3D\nc1,alpha,unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        list(
            to_cells(
                "csv_expr",
                table,
                LabelMap(dataset_id="x", mapping={"alpha": "CL:0000624"}, source="test"),
                columns=ObsColumns(label="assigned_cluster"),
                default_tissue="pancreas",
            )
        )


def test_csc_encoded_matrix_is_transposed_not_misread_as_csr(tmp_path):
    """A CSC group read as CSR would transpose the matrix — silently, per cell."""
    path = tmp_path / "csc.h5ad"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset("_index", data=np.array(["c0", "c1"], dtype=object), dtype=h5py.string_dtype())
        obs.create_dataset("cell_type_ontology_term_id", data=np.array(["CL:0000624"] * 2, dtype=object), dtype=h5py.string_dtype())
        var = handle.create_group("var")
        var.create_dataset("_index", data=np.array(["A", "B", "C"], dtype=object), dtype=h5py.string_dtype())
        matrix = handle.create_group("X")
        matrix.attrs["encoding-type"] = "csc_matrix"
        matrix.attrs["shape"] = np.array([2, 3], dtype=np.int64)
        # Dense [[1, 0, 2], [0, 3, 0]] in CSC: column 0 -> row 0; column 1 -> row 1; column 2 -> row 0.
        matrix.create_dataset("data", data=np.array([1.0, 3.0, 2.0], dtype=np.float32))
        matrix.create_dataset("indices", data=np.array([0, 1, 0], dtype=np.int32))
        matrix.create_dataset("indptr", data=np.array([0, 1, 2, 3], dtype=np.int64))
    info = probe("cellxgene_h5ad", path, columns=ObsColumns(label="cell_type_ontology_term_id"))
    assert (info.n_cells, info.n_genes) == (2, 3)
    cells = list(
        to_cells(
            "cellxgene_h5ad",
            path,
            _label_map("x", "CL:0000624"),
            columns=ObsColumns(label="cell_type_ontology_term_id"),
            default_tissue="lung",
        )
    )
    assert [sorted(zip(cell.genes, cell.values)) for cell in cells] == [[("A", 1.0), ("C", 2.0)], [("B", 3.0)]]


def test_dense_matrix_and_categorical_obs(smoke_data):
    """The fixture's donor column is categorical (codes + categories); its X is CSR."""
    cells = list(
        to_cells("cellxgene_h5ad", smoke_data / "cellxgene_lung.h5ad", _label_map("smoke", "CL:0000235"), columns=H5AD_COLUMNS, default_tissue="lung")
    )
    assert cells and {cell.cl_id for cell in cells} == {"CL:0000235"}
    assert all(cell.donor_id in {"a", "b", "c", "d", "e", "f"} for cell in cells)


def test_unknown_format_and_missing_columns_are_errors(smoke_data):
    with pytest.raises(ValueError):
        adapters.adapter_for("parquet")
    with pytest.raises(ValueError, match="is not in"):
        h5.probe(smoke_data / "cellxgene_lung.h5ad", columns=ObsColumns(label="nope"))
    with pytest.raises(ValueError):
        csv_expr.frames(smoke_data / "tenx_blood" / "features.tsv", ObsColumns(label="assigned_cluster"))


def test_resolve_tissue_maps_or_drops_rather_than_mislabelling():
    """A tissue the map does not know is dropped, never filed under the default."""
    assert adapters.resolve_tissue("UBERON:0002048", {"UBERON:0002048": "lung"}, None) == "lung"
    assert adapters.resolve_tissue("UBERON:9999999", {"UBERON:0002048": "lung"}, "lung") is None
    # A single-tissue dataset (no map) uses its default, including for a missing value.
    assert adapters.resolve_tissue("UBERON:0002048", {}, "lung") == "lung"
    assert adapters.resolve_tissue(None, {}, "lung") == "lung"
    assert adapters.resolve_tissue(None, {}, None) is None
