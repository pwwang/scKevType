"""CELLxGENE h5ad: the adapter that covers most of the corpus.

One reader covers Domínguez Conde, Muraro, Tabula Sapiens, Tabula Muris Senis and
Siletti, because the CELLxGENE schema mandates the columns the factory keys on:
``obs["cell_type_ontology_term_id"]`` (already a CL id — the curator did the
annotation), ``obs["donor_id"]`` (the grouping key that must never straddle a
split) and ``obs["tissue_ontology_term_id"]``.

Read with ``h5py``, not ``anndata``: anndata would materialise the whole matrix,
and the factory only needs one cell's non-zero entries at a time. ``obs`` is
decoded eagerly (it is small); ``X`` stays on disk and is read row by row.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any

import h5py

from ...contracts.ontology import LabelMap
from . import (
    CellRecord,
    CellTable,
    DatasetInfo,
    DenseMatrix,
    LabelStats,
    SparseMatrix,
    resolve_column,
    resolve_tissue,
)

CELLXGENE_COLUMNS = ("cell_type_ontology_term_id", "donor_id", "tissue_ontology_term_id")
#: Where the gene symbol lives when ``var/_index`` holds Ensembl ids (the CELLxGENE
#: convention). The registry can override it per dataset (``var_genes``).
GENE_SYMBOL_COLUMNS = ("feature_name", "gene_symbol", "symbol")
INDEX_KEYS = ("_index", "index")


def _decode(value: Any) -> str:
    """h5py returns bytes for strings; everything downstream wants str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "decode") and not isinstance(value, str):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_group(node: Any) -> bool:
    return isinstance(node, h5py.Group)


def _column(handle: h5py.File, group: str, name: str) -> list:
    """Read one ``obs``/``var`` column, whether it is a dataset or a categorical group."""
    node = handle[group][name]
    if _is_group(node):
        # anndata categorical: categories + integer codes (-1 = missing)
        categories = [_decode(value) for value in node["categories"][:]]
        return [categories[code] if code >= 0 else None for code in node["codes"][:]]
    values = node[:]
    if getattr(values, "ndim", 1) == 0:
        return [values.item()]
    return [None if value is None else _decode(value) for value in values]


def _obs_keys(handle: h5py.File) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in handle["obs"]))


def _var_keys(handle: h5py.File) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in handle["var"]))


def _index(handle: h5py.File, group: str, cell_column: str | None = None) -> list[str]:
    """The index column, or the configured cell-id column when there is none.

    Not every CELLxGENE file keeps its cell ids in ``obs/_index``: Siletti's ABC
    atlas names the column ``CellID`` and has no index at all, so a reader that
    insists on ``_index`` cannot build it.
    """
    for key in ((cell_column,) if cell_column else ()) + INDEX_KEYS:
        if key in handle[group]:
            return [text for text in _column(handle, group, key)]
    raise ValueError(f"{handle.filename}: {group!r} has no index column {INDEX_KEYS}")


def var_names(handle: h5py.File, genes_column: str | None = None) -> tuple[str, ...]:
    """Gene symbols: the configured column, else the CELLxGENE symbol column, else the index."""
    keys = _var_keys(handle)
    for candidate in ((genes_column,) if genes_column else ()) + GENE_SYMBOL_COLUMNS:
        if candidate in keys:
            return tuple(_column(handle, "var", candidate))
    return tuple(_index(handle, "var"))


def _matrix(handle: h5py.File, key: str = "X", *, materialise: bool = True):
    """The expression matrix as a row-addressable object (dense, CSR or CSC).

    ``materialise=False`` keeps a CSR matrix's ``data``/``indices`` on disk and
    reads one row per :meth:`SparseMatrix.row` call: a build that keeps 2,000 of
    an atlas's 330,000 cells should not read 3 GB of non-zeros to use 0.6% of
    them. A caller that walks the whole file wants the default (one sequential
    read, then cheap rows); a caller that knows its rows wants the lazy reader.
    """
    node = handle[key]
    if not _is_group(node):
        return DenseMatrix(node[:])
    # h5py reads a string attr back as bytes, so decode it: an undecoded
    # ``b"csc_matrix"`` would compare unequal and be read as CSR, silently
    # transposing every cell's genes.
    encoding = _decode(node.attrs.get("encoding-type", "csr_matrix"))
    shape = tuple(int(size) for size in node.attrs.get("shape", ()))
    if encoding == "csc_matrix":
        return SparseMatrix(node["data"][:], node["indices"][:], node["indptr"][:], shape, axis="csc")
    if not materialise:
        return SparseMatrix(node["data"], node["indices"], node["indptr"], shape)
    return SparseMatrix(node["data"][:], node["indices"][:], node["indptr"][:], shape)


def _shape(handle: h5py.File, key: str = "X") -> tuple[int, int]:
    """The matrix's shape without reading it: a group states it, a dataset is its own."""
    node = handle[key]
    if _is_group(node):
        return tuple(int(size) for size in node.attrs.get("shape", ()))  # type: ignore[return-value]
    return tuple(int(size) for size in node.shape)  # type: ignore[return-value]


def probe(path: Path, *, columns=None, genes_column: str | None = None) -> DatasetInfo:
    """Read the columns, size and label vocabulary of a CELLxGENE h5ad.

    A probe reads metadata, never expression: it wants the shape, and asking the
    matrix for its shape used to mean reading every non-zero in the file (42 GB
    for tabula_sapiens) before a single record was built.
    """
    with h5py.File(path, "r") as handle:
        keys = _obs_keys(handle)
        label_column = resolve_column(keys, columns.label if columns else None, "label")
        donor_column = resolve_column(keys, columns.donor if columns else None, "donor")
        tissue_column = resolve_column(keys, columns.tissue if columns else None, "tissue")
        names = var_names(handle, genes_column)
        matrix = _shape(handle)
        labels: list[str] = []
        counts: Counter = Counter()
        if label_column:
            labels = [label for label in _column(handle, "obs", label_column) if label is not None]
            counts = Counter(labels)
        return DatasetInfo(
            path=str(path),
            format="cellxgene_h5ad",
            n_cells=matrix[0],
            n_genes=matrix[1] if len(matrix) > 1 else len(names),
            obs_keys=keys,
            label_column=label_column,
            donor_column=donor_column,
            tissue_column=tissue_column,
            labels=tuple(sorted(counts)),
            label_counts=dict(counts),
            genes_head=tuple(names[:8]),
        )


def to_cells(
    path: Path,
    label_map: LabelMap,
    *,
    columns=None,
    genes_column: str | None = None,
    cell_column: str | None = None,
    tissue_labels: dict[str, str] | None = None,
    default_tissue: str | None = None,
    stats: LabelStats | None = None,
    expression: bool = True,
    only: Collection[str] | None = None,
    key: str = "X",
) -> Iterator[CellRecord]:
    """Yield labelled cells of a CELLxGENE h5ad file, dropping (and counting) the rest.

    ``label_map`` is the frozen ``dataset_label -> CL id`` table; for CELLxGENE
    files the "label" is already a CL id, so the map's job here is to decide which
    terms are in scope (an unmapped one is dropped rather than silently mixed into
    a corpus whose label space is fixed — the map's coverage report is the record
    of that decision).

    ``expression=False`` yields the cells without touching ``X`` — the label and
    donor columns are what a split assignment needs, and reading a 40 GB matrix to
    find out which cells are called what is a build's largest avoidable cost.
    ``only`` restricts the yielded cells to a set of cell ids (the cells a split
    kept), so a capped atlas reads the expression of the cells it will use.
    """
    stats = stats if stats is not None else LabelStats()
    with h5py.File(path, "r") as handle:
        keys = _obs_keys(handle)
        label_column = resolve_column(keys, columns.label if columns else None, "label", required=True)
        donor_column = resolve_column(keys, columns.donor if columns else None, "donor")
        tissue_column = resolve_column(keys, columns.tissue if columns else None, "tissue")
        cell_ids = _index(handle, "obs", cell_column)
        labels = _column(handle, "obs", label_column)
        donors = _column(handle, "obs", donor_column) if donor_column else [None] * len(cell_ids)
        tissues = _column(handle, "obs", tissue_column) if tissue_column else [None] * len(cell_ids)
        # `only` means the caller knows which rows it will use: keep the non-zeros on
        # disk and read those rows. Without it, one sequential read is cheaper.
        matrix = _matrix(handle, key, materialise=only is None) if expression else None
        names = var_names(handle, genes_column) if matrix is not None else ()
        for index, cell_id in enumerate(cell_ids):
            label = labels[index]
            cl_id = label_map.resolve(label) if label is not None else None
            stats.record(str(label), cl_id)
            if cl_id is None:
                continue
            # An unmapped tissue is *not* a reason to drop the cell: the register
            # has a list for a tissue it maps and the whole register for one it
            # does not, and the factory resolves between them per cell (Step 8 §1).
            # A cell that resolves to neither is only ever a labelling matter.
            tissue = resolve_tissue(tissues[index], tissue_labels, default_tissue)
            if tissue is None:
                stats.record_tissue(tissues[index])
            if only is not None and str(cell_id) not in only:
                continue
            row, values = matrix.row(index) if matrix is not None else ((), ())
            yield CellRecord(
                cell_id=str(cell_id),
                dataset_id=label_map.dataset_id,
                donor_id=str(donors[index]) if donors[index] not in (None, "") else None,
                tissue=tissue,
                cl_id=cl_id,
                genes=tuple(names[column] for column in row),
                values=values,
            )


def table(path: Path, *, columns=None, genes_column: str | None = None) -> CellTable:
    """The whole file as a :class:`CellTable` (used by tests and by small fixtures)."""
    with h5py.File(path, "r") as handle:
        keys = _obs_keys(handle)
        obs = {"cell_id": _index(handle, "obs")}
        for name in (*CELLXGENE_COLUMNS, *(key for key in keys if key not in CELLXGENE_COLUMNS)):
            if name in keys:
                obs[name] = _column(handle, "obs", name)
        names = var_names(handle, genes_column)
        matrix = _matrix(handle)
        if isinstance(matrix, DenseMatrix):
            matrix = DenseMatrix(matrix.array.copy())
        return CellTable(obs=obs, var_names=names, matrix=matrix)
