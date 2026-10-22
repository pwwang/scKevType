"""Dataset adapters: four file layouts in, one ``CellRecord`` stream out.

Every adapter exposes the same two functions::

    probe(path, *, columns=None)            -> DatasetInfo      # what is in this file
    to_cells(path, label_map, *, ...)       -> Iterator[CellRecord]

and no adapter reads a label vocabulary of its own — labels become CL ids only
through the frozen :class:`~sckevtype.contracts.ontology.LabelMap`, and a label
the map cannot resolve is **dropped and counted** (``LabelStats``), never coerced
to a nearby term. That count is what the build reports, so "we dropped 12% of
dataset X" is visible in the corpus manifest rather than invisible in the model.

``CellTable`` is deliberately duck-typed — ``obs`` (column -> list), ``var_names``
and a ``matrix`` with a ``row(i)`` accessor — so nothing here imports ``anndata``:
the machines that run the tests are CPU-only, and the factory reads only what it
needs (a rank list built from non-zero entries).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ...contracts.ontology import LabelMap


@dataclass(frozen=True)
class CellRecord:
    """One cell, reduced to what the factory needs to build supervision for it.

    Spec §1's tuple is ``(cell_id, dataset_id, donor_id, tissue, cl_id, genes,
    values)`` (see :meth:`as_tuple`). ``genes``/``values`` hold the cell's
    non-zero entries only: a rank list does not care about zeros, and a
    2.5M-cell atlas does not fit in memory.
    """

    cell_id: str
    dataset_id: str
    donor_id: str | None
    tissue: str | None
    cl_id: str
    genes: tuple[str, ...]
    values: tuple[float, ...]

    def as_tuple(self) -> tuple:
        """The spec's tuple form, for tests and for anything that wants it flat."""
        return (self.cell_id, self.dataset_id, self.donor_id, self.tissue, self.cl_id, self.genes, self.values)


@dataclass
class LabelStats:
    """What the adapter kept and what it dropped, with the labels that caused it.

    ``registered`` is the set of CL ids the register can name. Passing it lets a
    reader run with a *permissive* label map (one that resolves every label, so
    the out-of-register cells can be built rather than skipped) and still report
    exactly the numbers a strict map would: an id outside the register counts as
    dropped, under its raw label. ``None`` means "every resolved id is in scope".
    """

    total: int = 0
    mapped: int = 0
    dropped: int = 0
    dropped_labels: Counter = field(default_factory=Counter)
    dropped_tissues: Counter = field(default_factory=Counter)
    registered: frozenset[str] | None = None

    def knows(self, cl_id: str | None) -> bool:
        """Is this resolved id one the register can name?"""
        return cl_id is not None and (self.registered is None or cl_id in self.registered)

    def record(self, label: str, cl_id: str | None) -> None:
        self.total += 1
        if self.knows(cl_id):
            self.mapped += 1
        else:
            self.dropped += 1
            self.dropped_labels[label] += 1

    def record_tissue(self, raw: str | None) -> None:
        self.dropped_tissues[str(raw)] += 1

    @property
    def rate(self) -> float:
        return self.mapped / self.total if self.total else 0.0

    def payload(self) -> dict:
        """Deterministic (sorted) form for the corpus manifest."""
        return {
            "total": self.total,
            "mapped": self.mapped,
            "dropped": self.dropped,
            "mapped_rate": round(self.rate, 6),
            "dropped_labels": {label: self.dropped_labels[label] for label in sorted(self.dropped_labels)},
            "dropped_tissues": {tissue: self.dropped_tissues[tissue] for tissue in sorted(self.dropped_tissues)},
        }


@dataclass
class DatasetInfo:
    """What ``probe`` found: which columns exist, how big it is, what labels it has."""

    path: str
    format: str
    n_cells: int
    n_genes: int
    obs_keys: tuple[str, ...]
    label_column: str | None
    donor_column: str | None
    tissue_column: str | None
    labels: tuple[str, ...] = ()
    label_counts: dict[str, int] = field(default_factory=dict)
    genes_head: tuple[str, ...] = ()

    def payload(self) -> dict:
        return {
            "path": str(self.path),
            "format": self.format,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "obs_keys": list(self.obs_keys),
            "label_column": self.label_column,
            "donor_column": self.donor_column,
            "tissue_column": self.tissue_column,
            "n_labels": len(self.labels),
            "labels": list(self.labels),
            "label_counts": {label: self.label_counts[label] for label in sorted(self.label_counts)},
            "genes_head": list(self.genes_head),
        }


class Matrix(Protocol):
    """Row access into an expression matrix: non-zero entries of one cell."""

    shape: tuple[int, int]

    def row(self, index: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """``(column indices, values)`` for row ``index`` — non-zero entries only."""
        ...


@dataclass
class CellTable:
    """A minimal cell table: ``obs`` columns, gene names, and a matrix accessor."""

    obs: dict[str, list]
    var_names: tuple[str, ...]
    matrix: Matrix

    def __len__(self) -> int:
        column = next(iter(self.obs.values()), [])
        return len(column)

    def cell_ids(self) -> list[str]:
        for key in ("cell_id", "_index", "index"):
            if key in self.obs:
                return [str(value) for value in self.obs[key]]
        raise ValueError("the cell table has no cell-id column (_index)")


class DenseMatrix:
    """A dense ``cells x genes`` array (numpy or any nested sequence)."""

    def __init__(self, array: Any) -> None:
        self.array = array
        self.shape = (len(array), len(array[0]) if len(array) else 0)

    def row(self, index: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        values = self.array[index]
        indices = [column for column, value in enumerate(values) if value]
        return tuple(indices), tuple(float(values[column]) for column in indices)


class SparseMatrix:
    """A sparse matrix in CSR (or CSC, transposed once on construction).

    Hand-rolled over the raw ``data``/``indices``/``indptr`` arrays an h5ad or
    MatrixMarket file stores, so no ``scipy`` import is needed on the CPU-only
    boxes the tests run on. ``axis="csc"`` costs one bucketing pass over the
    non-zeros and is recorded in ``materialised`` — an h5ad is normally CSR, so
    this is the uncommon path.
    """

    def __init__(self, data, indices, indptr, shape: tuple[int, int], *, axis: str = "csr") -> None:
        self.shape = shape
        if axis == "csc":
            data, indices, indptr = _csc_to_csr(data, indices, indptr, shape)
            self.materialised = True
        else:
            self.materialised = False
        self.data, self.indices, self.indptr = data, indices, indptr

    def row(self, index: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        start, stop = int(self.indptr[index]), int(self.indptr[index + 1])
        columns = [int(column) for column in self.indices[start:stop]]
        values = [float(value) for value in self.data[start:stop]]
        entries = [(column, value) for column, value in zip(columns, values) if value]
        return tuple(column for column, _ in entries), tuple(value for _, value in entries)


def _csc_to_csr(data, indices, indptr, shape: tuple[int, int]):
    """Transpose a CSC structure into CSR (no arithmetic — structure only)."""
    rows, _ = shape
    counts = [0] * rows
    for row in indices:
        counts[int(row)] += 1
    offsets = [0] * (rows + 1)
    for row, count in enumerate(counts):
        offsets[row + 1] = offsets[row] + count
    cursor = offsets[:-1].copy()
    out_indices = [0] * len(indices)
    out_data = [0.0] * len(data)
    for column in range(len(indptr) - 1):
        for position in range(int(indptr[column]), int(indptr[column + 1])):
            row = int(indices[position])
            target = cursor[row]
            out_indices[target] = column
            out_data[target] = float(data[position])
            cursor[row] += 1
    return out_data, out_indices, offsets


def resolve_tissue(raw: object, tissue_labels: dict[str, str] | None, default: str | None) -> str | None:
    """Raw ``obs`` tissue value -> canonical tissue name (``None`` when unmapped).

    A dataset that is single-tissue names it in the registry (``default``); a
    multi-tissue one maps its own vocabulary through ``tissue_labels``. A value
    that resolves to neither is returned as ``None`` and counted by the caller —
    the middleware has no panel for a tissue it does not know.
    """
    text = str(raw) if raw is not None else ""
    if tissue_labels and text in tissue_labels:
        return tissue_labels[text]
    if default and (not text or not tissue_labels):
        return default
    return None


def resolve_column(obs_keys: Sequence[str], name: str | None, what: str, *, required: bool = False) -> str | None:
    """The column to read: the configured name (validated) or None."""
    if name is None:
        if required:
            raise ValueError(f"no {what} column configured (available: {sorted(obs_keys)})")
        return None
    if name not in obs_keys:
        raise ValueError(f"{what} column {name!r} is not in the file (available: {sorted(obs_keys)})")
    return name


#: format -> adapter module name. Imported lazily so a missing extra (say ``h5py``
#: on a box that only builds the CSV fixture) fails at use, not at import.
ADAPTERS: dict[str, str] = {
    "cellxgene_h5ad": "cellxgene_h5ad",
    "tenx_mtx": "tenx_mtx",
    "csv_expr": "csv_expr",
    # One dataset that is an *archive* of per-donor tables rather than one table:
    # Baron's six `GSM*_umifm_counts.csv.gz` members, donor in the member name.
    "baron_archive": "baron_archive",
}


def adapter_for(fmt: str):
    """The adapter module for a registry ``format`` value."""
    import importlib

    try:
        module_name = ADAPTERS[fmt]
    except KeyError:
        raise ValueError(f"unknown dataset format {fmt!r} (known: {sorted(ADAPTERS)})") from None
    return importlib.import_module(f"{__name__}.{module_name}")


def probe(fmt: str, path: str | Path, **kwargs) -> DatasetInfo:
    """Probe ``path`` with the adapter for ``fmt``."""
    return adapter_for(fmt).probe(Path(path), **kwargs)


def to_cells(fmt: str, path: str | Path, label_map: LabelMap, **kwargs) -> Iterator[CellRecord]:
    """Iterate cells of ``path`` with the adapter for ``fmt``, labelled via ``label_map``."""
    return adapter_for(fmt).to_cells(Path(path), label_map, **kwargs)


__all__ = [
    "ADAPTERS",
    "CellRecord",
    "CellTable",
    "DatasetInfo",
    "DenseMatrix",
    "LabelStats",
    "Matrix",
    "SparseMatrix",
    "adapter_for",
    "probe",
    "resolve_column",
    "resolve_tissue",
    "to_cells",
]
