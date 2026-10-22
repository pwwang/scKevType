"""10x Genomics Matrix Market: ``matrix.mtx`` + ``barcodes`` + ``features``.

Covers pbmc3k and Zheng 68k. Both ship **no author labels** (``research/lit/08``):
the 10x download is an expression matrix, and a cell type can only come from a
community silver standard. So the label is read from a sidecar file next to the
matrix (``labels.tsv``: barcode, label) and the registry records where that
sidecar came from — a silver-standard label must never be mistaken for a curated
one. A 10x directory with no label sidecar yields a clear error, not an empty
corpus.

Reading is streaming and dependency-free: ``.gz`` is handled with the stdlib, the
MatrixMarket header is parsed by hand, and non-zero entries are bucketed per cell
in one pass (``matrix.mtx`` is column-ordered by construction, so one pass is
enough).
"""

from __future__ import annotations

import csv
import gzip
from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import TextIO

from ...contracts.ontology import LabelMap
from . import CellRecord, DatasetInfo, LabelStats, resolve_tissue

#: File stems, in preference order, for the three parts of a 10x bundle.
MATRIX_STEMS = ("matrix.mtx",)
BARCODE_STEMS = ("barcodes.tsv", "barcodes.txt")
FEATURE_STEMS = ("features.tsv", "features.txt", "genes.tsv", "genes.txt")
LABEL_STEMS = ("labels.tsv", "labels.csv", "labels.txt")


def _open_text(path: Path) -> TextIO:
    """Open a possibly-gzipped text file (the 10x download is often ``*.gz``)."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _find(directory: Path, stems: tuple[str, ...], *, required: bool = True) -> Path | None:
    """The first of ``stems`` (plain or ``.gz``) that exists in ``directory``."""
    for stem in stems:
        for name in (stem, f"{stem}.gz"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    if required:
        raise FileNotFoundError(f"{directory}: none of {list(stems)} (or their .gz) is present")
    return None


def _read_lines(path: Path) -> Iterator[str]:
    with _open_text(path) as stream:
        for line in stream:
            yield line.rstrip("\n")


def read_labels(path: Path, *, label_column: str | None = None, barcode_column: str | None = None) -> dict[str, str]:
    """Read a ``barcode -> label`` sidecar (2 columns; a header row is optional).

    ``label_column``/``barcode_column`` name the header fields when the sidecar has
    a header. Without them the first two columns are used positionally, which is
    how the community silver standards for pbmc3k and Zheng 68k are distributed.
    """
    delimiter = "," if path.suffix.lstrip(".gz").lower().endswith((".csv", "csv")) else "\t"
    with _open_text(path) as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        rows = [row for row in reader if row and any(field.strip() for field in row)]
    if not rows:
        return {}
    labels: dict[str, str] = {}
    header = [field.strip().casefold() for field in rows[0]]
    has_header = barcode_column is not None or any(field in ("barcode", "cell", "cell_id", "index") for field in header)
    start = 0
    first, second = 0, 1
    if has_header:
        start = 1
        first = header.index(barcode_column.casefold()) if barcode_column and barcode_column.casefold() in header else 0
        second = header.index(label_column.casefold()) if label_column and label_column.casefold() in header else 1
    for row in rows[start:]:
        if len(row) <= max(first, second):
            continue
        barcode, label = row[first].strip(), row[second].strip()
        if barcode and label:
            labels[barcode] = label
    return labels


def read_features(path: Path) -> list[tuple[str, str]]:
    """``features.tsv`` rows as ``(gene_id, symbol)`` (v2 files have two columns too)."""
    features = []
    for line in _read_lines(path):
        if not line.strip():
            continue
        fields = line.split("\t")
        gene_id = fields[0].strip()
        symbol = fields[1].strip() if len(fields) > 1 and fields[1].strip() else gene_id
        features.append((gene_id, symbol))
    return features


def read_matrix(path: Path) -> tuple[tuple[int, int, int], list[list[tuple[int, float]]]]:
    """Parse ``matrix.mtx`` into per-cell ``(gene index, value)`` lists (0-based).

    Returns ``((genes, cells, nonzeros), rows)`` where ``rows[cell]`` holds that
    cell's non-zero entries. MatrixMarket is 1-based; the conversion happens here
    so nothing downstream has to remember it.
    """
    genes = cells = nonzeros = 0
    seen_header = False
    rows: list[list[tuple[int, float]]] = []
    with _open_text(path) as stream:
        for line in stream:
            if line.startswith("%"):
                continue
            fields = line.split()
            if not seen_header:
                if len(fields) < 3:
                    raise ValueError(f"{path}: malformed MatrixMarket header: {line!r}")
                genes, cells, nonzeros = (int(fields[0]), int(fields[1]), int(fields[2]))
                rows = [[] for _ in range(cells)]
                seen_header = True
                continue
            if len(fields) < 3:
                continue
            row, cell, value = int(fields[0]) - 1, int(fields[1]) - 1, float(fields[2])
            if value == 0.0:
                continue
            rows[cell].append((row, value))
    if not seen_header:
        raise ValueError(f"{path}: empty MatrixMarket file")
    return (genes, cells, nonzeros), rows


def _bundle(path: Path) -> tuple[Path, Path, Path, Path | None]:
    """Locate the four parts of a 10x bundle inside ``path`` (a directory)."""
    if not path.is_dir():
        raise NotADirectoryError(f"{path}: a 10x bundle is a directory (matrix.mtx, barcodes, features)")
    return (
        _find(path, MATRIX_STEMS),
        _find(path, BARCODE_STEMS),
        _find(path, FEATURE_STEMS),
        _find(path, LABEL_STEMS, required=False),
    )


def probe(path: Path, *, columns=None) -> DatasetInfo:
    """Read sizes and the label vocabulary of a 10x bundle."""
    _matrix_path, barcodes_path, features_path, label_path = _bundle(path)
    barcodes = list(_read_lines(barcodes_path))
    features = read_features(features_path)
    labels: dict[str, str] = {}
    label_column = getattr(columns, "label", None) if columns else None
    if label_path:
        labels = read_labels(label_path, label_column=label_column)
    counts = Counter(labels.values()) if labels else Counter()
    return DatasetInfo(
        path=str(path),
        format="tenx_mtx",
        n_cells=len(barcodes),
        n_genes=len(features),
        obs_keys=("barcode",) + (("label",) if labels else ()),
        label_column="label" if labels else (label_column or None),
        donor_column=getattr(columns, "donor", None) if columns else None,
        tissue_column=None,
        labels=tuple(sorted(counts)),
        label_counts=dict(counts),
        genes_head=tuple(symbol for _, symbol in features[:8]),
    )


def to_cells(
    path: Path,
    label_map: LabelMap,
    *,
    columns=None,
    label_file: str | None = None,
    label_column: str | None = None,
    barcode_column: str | None = None,
    donor_id: str | None = None,
    tissue_labels: dict[str, str] | None = None,
    default_tissue: str | None = None,
    stats: LabelStats | None = None,
    expression: bool = True,
    only: Collection[str] | None = None,
) -> Iterator[CellRecord]:
    """Yield labelled cells of a 10x bundle (labelled via the silver-standard sidecar).

    ``expression=False`` / ``only`` behave as in the h5ad adapter: a caller that
    only needs the labels, or only some cells, says so.
    """
    stats = stats if stats is not None else LabelStats()
    _matrix_path, barcodes_path, features_path, found_labels = _bundle(path)
    if label_file and Path(label_file).is_absolute():
        label_path: Path | None = Path(label_file)
    elif label_file:
        label_path = _find(path, (label_file,), required=False) or (path / label_file)
    else:
        label_path = found_labels
    if label_path is None or not label_path.exists():
        raise ValueError(
            f"{path}: no label sidecar (looked for {list(LABEL_STEMS)}). A 10x download ships no author "
            "labels; supply a community silver standard as `<barcode>\\t<label>` and record its provenance."
        )
    labels = read_labels(label_path, label_column=label_column, barcode_column=barcode_column)
    barcodes = list(_read_lines(barcodes_path))
    symbols = [symbol for _, symbol in read_features(features_path)]
    _, rows = read_matrix(_matrix_path)
    if len(rows) != len(barcodes):
        raise ValueError(f"{path}: matrix has {len(rows)} cells but {len(barcodes)} barcodes")
    for index, barcode in enumerate(barcodes):
        label = labels.get(barcode)
        cl_id = label_map.resolve(label) if label is not None else None
        stats.record(str(label), cl_id)
        if cl_id is None:
            continue
        tissue = resolve_tissue(None, tissue_labels, default_tissue)
        if tissue is None:
            stats.record_tissue(None)
            continue
        if only is not None and barcode not in only:
            continue
        entries = rows[index] if expression else ()
        yield CellRecord(
            cell_id=barcode,
            dataset_id=label_map.dataset_id,
            donor_id=donor_id,
            tissue=tissue,
            cl_id=cl_id,
            genes=tuple(symbols[row] for row, _ in entries),
            values=tuple(value for _, value in entries),
        )
