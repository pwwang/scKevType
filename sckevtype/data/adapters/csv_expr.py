"""CSV/TSV expression: a cells-by-genes table with the labels in its metadata columns.

Covers Baron (``GSE84133``), whose ``*_umifm_counts.csv`` files carry
``cell, barcode, assigned_cluster, <gene>...``: the first column is the cell id,
the columns named by the registry's ``obs`` config are metadata, and everything
else is a gene. Baron's four human samples are four files and four donors, so the
file stem can name the donor (``donor_from_stem``) — the important part is that
those cells are grouped, not smeared across splits.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import TextIO

from ...contracts.ontology import LabelMap
from . import CellRecord, DatasetInfo, LabelStats, resolve_tissue

#: Values a cell id column may be called (Baron writes ``cell``).
CELL_COLUMNS = ("cell", "cell_id", "barcode", "index")


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _dialect(path: Path) -> str:
    return "," if path.name.lower().removesuffix(".gz").endswith(".csv") else "\t"


def header(path: Path) -> list[str]:
    """The column names of the expression table."""
    with _open_text(path) as stream:
        reader = csv.reader(stream, delimiter=_dialect(path))
        for row in reader:
            return [field.strip() for field in row]
    raise ValueError(f"{path}: empty expression table")


def frames(path: Path, columns=None, *, cell_column: str | None = None):
    """``(cell column, metadata columns, gene columns)`` from the header.

    Metadata is whatever the registry's ``obs`` config names; the rest are genes.
    The cell id column is the configured one, else the first column if it looks
    like one (:data:`CELL_COLUMNS`), else the first column regardless — a
    cells-by-genes table always has a row key.
    """
    names = header(path)
    if not names:
        raise ValueError(f"{path}: empty expression table")
    labelled = {
        name
        for name in (getattr(columns, "label", None), getattr(columns, "donor", None), getattr(columns, "tissue", None))
        if name
    }
    for name in labelled:
        if name not in names:
            raise ValueError(f"{path}: configured column {name!r} is not in the table (have: {names[:8]}…)")
    if cell_column and cell_column not in names:
        raise ValueError(f"{path}: cell column {cell_column!r} is not in the table")
    identifiers = {name for name in names if name.strip().casefold() in CELL_COLUMNS}
    key = cell_column or next((name for name in names if name in identifiers), names[0])
    # Every identifier-looking column is metadata, not a gene: Baron's tables carry a
    # `barcode` column beside `cell`, and a metadata column read as a gene is a crash
    # on the first non-numeric value (or, worse, a silently numeric one).
    metadata = set(labelled) | identifiers | {key}
    genes = [name for name in names if name not in metadata]
    if not genes:
        raise ValueError(f"{path}: no gene columns (metadata: {sorted(metadata)})")
    return key, metadata, genes


def probe(path: Path, *, columns=None, cell_column: str | None = None) -> DatasetInfo:
    """Read column layout, size and label vocabulary of an expression table."""
    _key, metadata, genes = frames(path, columns, cell_column=cell_column)
    labels: Counter = Counter()
    count = 0
    label_column = getattr(columns, "label", None) if columns else None
    with _open_text(path) as stream:
        reader = csv.DictReader(stream, delimiter=_dialect(path))
        for row in reader:
            count += 1
            if label_column:
                label = (row.get(label_column) or "").strip()
                if label:
                    labels[label] += 1
    return DatasetInfo(
        path=str(path),
        format="csv_expr",
        n_cells=count,
        n_genes=len(genes),
        obs_keys=tuple(sorted(metadata)),
        label_column=label_column,
        donor_column=getattr(columns, "donor", None) if columns else None,
        tissue_column=getattr(columns, "tissue", None) if columns else None,
        labels=tuple(sorted(labels)),
        label_counts=dict(labels),
        genes_head=tuple(genes[:8]),
    )


def to_cells(
    path: Path,
    label_map: LabelMap,
    *,
    columns=None,
    cell_column: str | None = None,
    donor_id: str | None = None,
    donor_from_stem: bool = False,
    tissue_labels: dict[str, str] | None = None,
    default_tissue: str | None = None,
    stats: LabelStats | None = None,
    expression: bool = True,
    only: Collection[str] | None = None,
) -> Iterator[CellRecord]:
    """Yield labelled cells of an expression table, dropping (and counting) the rest.

    ``expression=False`` / ``only`` behave as in the h5ad adapter: a caller that
    only needs the labels, or only some cells, says so.
    """
    stats = stats if stats is not None else LabelStats()
    key, _, genes = frames(path, columns, cell_column=cell_column)
    label_column = getattr(columns, "label", None) if columns else None
    donor_column = getattr(columns, "donor", None) if columns else None
    tissue_column = getattr(columns, "tissue", None) if columns else None
    if label_column is None:
        raise ValueError(f"{path}: no label column configured — a CSV corpus needs `obs.label`")
    fallback_donor = donor_id or (path.name.removesuffix(".gz").split(".")[0] if donor_from_stem else None)
    with _open_text(path) as stream:
        reader = csv.DictReader(stream, delimiter=_dialect(path))
        for row in reader:
            label = (row.get(label_column) or "").strip()
            cl_id = label_map.resolve(label) if label else None
            stats.record(label, cl_id)
            if cl_id is None:
                continue
            tissue = resolve_tissue(row.get(tissue_column) if tissue_column else None, tissue_labels, default_tissue)
            if tissue is None:
                stats.record_tissue(row.get(tissue_column) if tissue_column else None)
                continue
            cell_id = (row.get(key) or "").strip()
            if only is not None and cell_id not in only:
                continue
            symbols: list[str] = []
            values: list[float] = []
            for gene in genes if expression else ():
                field = (row.get(gene) or "").strip()
                if not field:
                    continue
                try:
                    value = float(field)
                except ValueError:
                    raise ValueError(f"{path}: gene column {gene!r} holds a non-numeric value {field!r}") from None
                if value:
                    symbols.append(gene)
                    values.append(value)
            donor = (row.get(donor_column) or "").strip() if donor_column else ""
            yield CellRecord(
                cell_id=cell_id,
                dataset_id=label_map.dataset_id,
                donor_id=donor or fallback_donor,
                tissue=tissue,
                cl_id=cl_id,
                genes=tuple(symbols),
                values=tuple(values),
            )
