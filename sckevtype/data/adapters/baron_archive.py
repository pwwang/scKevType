"""Baron 2016 pancreas (GEO ``GSE84133``): six per-donor cells-by-genes CSV members in one tar.

Why this is not ``csv_expr``. Every other CSV layout the factory reads is *one table
for one dataset*. Baron ships **six** tables inside a single ``.tar`` — four human
samples and two mouse ones — and the donor is not a column anywhere in them: it is
the *member name* (``GSM2230757_human1_umifm_counts.csv.gz``). ``csv_expr`` reads a
path as a file, so handing it a tar fails one level down, and handing it one member
would silently make a six-donor atlas into a one-donor one. So the unpack lives
here: open the tar once, stream each member, and label every cell with the donor its
file names.

What the members hold, measured on the archive's own bytes (29,122,560 B, sha256
``aed2d208…``): a header ``,barcode,assigned_cluster,<20125 genes>`` (the mouse
files carry 14,878 genes), one row per cell, and 17 distinct ``assigned_cluster``
values across the six files — the 14 the human files carry plus ``B_cell``,
``T_cell`` (the capitalised spelling of the same label) and ``immune_other``, which
the mouse files alone add.

Two facts about the ids, both measured rather than assumed:

* the first column has **no name** and its values are the row index; the cell key is
  ``barcode`` (which is also why ``barcode`` is never a gene column — a metadata
  column read as a gene is a crash on the first non-numeric value), and
* barcodes are **not unique**: 308 barcode memberships are shared between files (one
  barcode can be in two of them), and 104 repeat *within* a single file (106 extra
  rows). So a cell id is ``<donor>:<barcode>``, and a
  repeat inside one file becomes ``<donor>:<barcode>#2``, ``#3``, …. A bare barcode
  would merge two donors' cells into one registry key, and an unsuffixed repeat would
  silently lose a cell — the split registry keys cells by id, so a duplicate id is one
  cell to the corpus while the file holds two.

Labels become CL ids only through the frozen :class:`~sckevtype.contracts.ontology.LabelMap`;
a label the map cannot resolve is dropped and counted by ``LabelStats``, exactly as
in the other adapters. Nothing here decides which free-text label is which term:
that is the corpus config's ``label_map``, where a wrong entry is reviewable.
"""

from __future__ import annotations

import csv
import gzip
import re
import tarfile
from collections import Counter
from collections.abc import Collection, Iterator
from pathlib import Path

from ...contracts.ontology import LabelMap
from . import CellRecord, DatasetInfo, LabelStats, resolve_tissue

#: The member suffix every expression table in the archive carries.
MEMBER_SUFFIX = "_umifm_counts.csv.gz"

#: The column the free-text type is in, when the corpus config names none.
LABEL_COLUMN = "assigned_cluster"

#: Columns that are identifiers or metadata, never genes.
IDENTIFIER_COLUMNS = ("", "barcode", "cell", "cell_id", "index", "assigned_cluster")

#: ``GSM2230757_human1_umifm_counts.csv.gz`` -> ``human1``.
_ACCESSION = re.compile(r"^(?:GSM\d+[_-])?(?P<donor>.+?)(?:_umifm|$)")


def donor_of(member: str) -> str:
    """The donor a member names: the part between the GEO accession and the suffix.

    ``GSM2230757_human1_umifm_counts.csv.gz`` is donor ``human1``. A member that does
    not match the archive's naming keeps its whole stem, which is at least unique and
    visible, rather than being dropped or given a made-up donor.
    """
    stem = Path(member).name
    for suffix in (".gz", ".csv"):
        stem = stem.removesuffix(suffix)
    match = _ACCESSION.match(stem)
    donor = (match.group("donor") if match else stem).strip("_-")
    return donor or stem


def members(path: Path) -> list[tarfile.TarInfo]:
    """The expression members of the archive, in file-name order (a fixed order)."""
    with tarfile.open(path) as archive:
        found = [
            member for member in archive.getmembers() if member.isfile() and member.name.endswith(MEMBER_SUFFIX)
        ]
    if not found:
        raise ValueError(f"{path}: no *{MEMBER_SUFFIX} member — this is not the Baron archive")
    return sorted(found, key=lambda member: member.name)


def _stream(archive: tarfile.TarFile, member: tarfile.TarInfo):
    """One member as text: the tar's bytes, decompressed, decoded, newline-translated."""
    extracted = archive.extractfile(member)
    if extracted is None:  # a member with no body; tarfile's own edge
        raise ValueError(f"{member.name}: not a regular file")
    return gzip.open(extracted, "rt", encoding="utf-8", newline="")


def column_indices(header: list[str], label_column: str) -> tuple[int, int, list[int]]:
    """``(cell key, label, gene columns)`` for one member's header.

    The key is ``barcode`` when the table has one (Baron does), else the first column:
    a cells-by-genes table always has a row key, and Baron's first column is unnamed.
    """
    names = [name.strip() for name in header]
    if label_column not in names:
        raise ValueError(f"column {label_column!r} is not in the table (have: {names[:8]}…)")
    key = next((index for index, name in enumerate(names) if name.casefold() == "barcode"), 0)
    label = names.index(label_column)
    metadata = {key, label} | {index for index, name in enumerate(names) if name.casefold() in IDENTIFIER_COLUMNS}
    genes = [index for index in range(len(names)) if index not in metadata]
    if not genes:
        raise ValueError(f"no gene columns (metadata: {sorted(names[index] for index in metadata)})")
    return key, label, genes


def probe(path: Path, *, columns: object = None, cell_column: str | None = None) -> DatasetInfo:
    """What the archive holds: cells, genes, and the label vocabulary of all six files.

    ``n_genes`` is the longest member's gene count (the human files carry 20,125 and
    the mouse ones 14,878) and ``labels`` is the union over members — the probe
    reports the *file*, never the corpus's label map, so an unmappable label stays
    visible instead of vanishing into a coverage rate.
    """
    label_name = str(getattr(columns, "label", None) or LABEL_COLUMN)
    labels: Counter = Counter()
    count = genes_seen = 0
    head: tuple[str, ...] = ()
    with tarfile.open(path) as archive:
        for member in members(path):
            with _stream(archive, member) as stream:
                reader = csv.reader(stream)
                header = next(reader, None)
                if header is None:
                    raise ValueError(f"{path}: member {member.name} is empty")
                _key, label, genes = column_indices(header, label_name)
                names = [name.strip() for name in header]
                if len(genes) > genes_seen:
                    genes_seen, head = len(genes), tuple(names[index] for index in genes[:8])
                for row in reader:
                    count += 1
                    value = row[label].strip() if label < len(row) else ""
                    if value:
                        labels[value] += 1
    return DatasetInfo(
        path=str(path),
        format="baron_archive",
        n_cells=count,
        n_genes=genes_seen,
        obs_keys=("assigned_cluster", "barcode"),
        label_column=label_name,
        donor_column=None,
        tissue_column=None,
        labels=tuple(sorted(labels)),
        label_counts=dict(labels),
        genes_head=head,
    )


def to_cells(
    path: Path,
    label_map: LabelMap,
    *,
    columns: object = None,
    label_column: str | None = None,
    cell_column: str | None = None,
    donor_id: str | None = None,
    donor_from_stem: bool = False,
    tissue_labels: dict[str, str] | None = None,
    default_tissue: str | None = None,
    stats: LabelStats | None = None,
    expression: bool = True,
    only: Collection[str] | None = None,
) -> Iterator[CellRecord]:
    """Yield every labelled cell of the archive, dropping (and counting) the rest.

    ``donor_id``/``donor_from_stem``/``cell_column`` are accepted and ignored: the donor
    *is* the member (there is no column to name) and the key is the ``barcode``. They
    are in the signature because the factory's ``adapter_kwargs`` is written for the
    CSV layouts, and a strict signature would turn a config field into a crash rather
    than into a no-op a reader can see.

    ``expression=False`` / ``only`` behave as in the other adapters: the registration
    pass reads no expression, and the record pass reads it only for the cells a split
    kept. ``only`` is compared against the **namespaced** id (``<donor>:<barcode>``),
    which is what the split registry keys on.
    """
    label_name = str(label_column or getattr(columns, "label", None) or LABEL_COLUMN)
    stats = stats if stats is not None else LabelStats()
    tissue = resolve_tissue(None, tissue_labels, default_tissue)
    seen: Counter = Counter()
    with tarfile.open(path) as archive:
        for member in members(path):
            donor = donor_of(member.name)
            with _stream(archive, member) as stream:
                reader = csv.reader(stream)
                header = next(reader, None)
                if header is None:
                    raise ValueError(f"{path}: member {member.name} is empty")
                key, label_at, genes = column_indices(header, label_name)
                names = [name.strip() for name in header]
                for number, row in enumerate(reader):
                    label = row[label_at].strip() if label_at < len(row) else ""
                    cl_id = label_map.resolve(label) if label else None
                    stats.record(label, cl_id)
                    if cl_id is None:
                        continue
                    if tissue is None:
                        stats.record_tissue(None)
                        continue
                    barcode = row[key].strip() if key < len(row) else ""
                    stem_id = f"{donor}:{barcode or f'row{number}'}"
                    seen[stem_id] += 1
                    cell_id = stem_id if seen[stem_id] == 1 else f"{stem_id}#{seen[stem_id]}"
                    if only is not None and cell_id not in only:
                        continue
                    symbols: list[str] = []
                    values: list[float] = []
                    for index in genes if expression else ():
                        field = row[index].strip() if index < len(row) else ""
                        if not field:
                            continue
                        try:
                            value = float(field)
                        except ValueError:
                            raise ValueError(
                                f"{path}: member {member.name} column {names[index]!r} holds a "
                                f"non-numeric value {field!r}"
                            ) from None
                        if value:
                            symbols.append(names[index])
                            values.append(value)
                    yield CellRecord(
                        cell_id=cell_id,
                        dataset_id=label_map.dataset_id,
                        donor_id=donor,
                        tissue=tissue,
                        cl_id=cl_id,
                        genes=tuple(symbols),
                        values=tuple(values),
                    )
