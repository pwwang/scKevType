"""Step 2 -- the matrices the baselines see.

For each dataset: read ``raw/X`` (raw integer counts) for exactly the cells the
corpus uses, and persist csr arrays plus the full gene vocabulary.

Read restriction: the cell ids come from ``out/cells_{split}.json``, which step 1
built off the corpus. If a corpus cell id is not found in the atlas's own id column
that is a hard error here, not a silent drop -- a baseline scored on a different
cell set than the corpus is not a baseline for this corpus.

Memory stays bounded although the atlases are large: contiguous runs of the
requested (sorted) rows are read as single block slices of ``raw/X``'s
``data``/``indices``. Measured cost of the largest read (tabula_sapiens, 42 GB on
disk, 5,000 of 1,136,218 rows) is in the run log.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    DATASETS,
    EXPR,
    QCOUT,
    ROOT,
    cell_ids,
    count_matrix_key,
    expr_paths,
    gene_names,
    ids_sha256,
    matrix_rows,
    write_json,
    write_order_sidecar,
)

#: Rows re-read straight from the atlas and compared, by content, with what was persisted.
#: Cheap (a handful of rows) and independent of the reader the extraction used.
SPOT_CHECKS = 5


def assert_ascending_rows(rows: np.ndarray, where: str) -> np.ndarray:
    """Refuse to re-order rows: the caller's array order IS the output row order.

    Both readers below used to call ``np.sort`` on the requested rows, which is exactly how
    the cell-order misalignment happened: the matrix came back in ascending source-row order
    while the cell ids saved beside it stayed in request order, and nothing failed. A
    non-ascending array is a caller bug; it must raise, not be quietly repaired -- the repair
    is the step that loses the row-to-id correspondence.
    """
    if rows.size and np.any(rows[1:] <= rows[:-1]):
        raise SystemExit(
            f"{where}: row indices must be unique and strictly ascending, got {len(rows)} indices "
            f"starting {rows[:5].tolist()}. The reader emits rows in the order it is given, so "
            f"re-ordering here would silently desynchronise the matrix rows from the cell ids "
            f"saved beside them -- pass the cell ids sorted by source position (see extract())."
        )
    return rows


def read_rows(handle: h5py.File, key: str, rows: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """The requested rows of a CSR matrix stored under ``key``, as a csr matrix."""
    grp = handle[key]
    indptr = grp["indptr"][:]
    rows = assert_ascending_rows(np.asarray(rows, dtype=np.int64), "read_rows")
    n = len(rows)
    nnz_per_row = (indptr[rows + 1] - indptr[rows]).astype(np.int64)
    out_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nnz_per_row, out=out_indptr[1:])
    total = int(out_indptr[-1])
    data = np.empty(total, dtype=grp["data"].dtype)
    indices = np.empty(total, dtype=np.int32)
    # Merge consecutive row indices into runs: one read per run instead of per row.
    start = 0
    while start < n:
        end = start
        while end + 1 < n and rows[end + 1] == rows[end] + 1:
            end += 1
        lo, hi = int(rows[start]), int(rows[end])
        s, e = int(indptr[lo]), int(indptr[hi + 1])
        off = int(out_indptr[start])
        data[off : off + (e - s)] = grp["data"][s:e]
        indices[off : off + (e - s)] = grp["indices"][s:e]
        start = end + 1
    return sp.csr_matrix((data, indices, out_indptr), shape=(n, n_genes))


def read_rows_streaming(
    handle: h5py.File, key: str, rows: np.ndarray, n_genes: int, block_elements: int = 20_000_000
) -> sp.csr_matrix:
    """The requested rows, read by streaming ``data``/``indices`` sequentially.

    Reading a few thousand scattered rows out of a 30-40 GB CSR means one HDF5 read
    per row, and each of those reads decompresses whatever chunk the row falls in --
    measured at well under 1 MB/s on these files, i.e. hours. Walking ``data`` and
    ``indices`` front to back instead costs one sequential pass over the non-zeros
    (single-digit GB) and keeps memory bounded to one block, which is the difference
    between minutes and hours for siletti and tabula_sapiens.
    """
    grp = handle[key]
    indptr = grp["indptr"][:]
    rows = assert_ascending_rows(np.asarray(rows, dtype=np.int64), "read_rows_streaming")
    # ``wanted`` must span every row of the matrix, not just up to rows.max(): the block
    # scan below visits all rows, and one index past the end is an IndexError 20 minutes
    # into a pass over a 30 GB file. The guard makes that a first-second failure instead.
    n_rows = len(indptr) - 1
    if rows.size and int(rows.max()) >= n_rows:
        raise SystemExit(
            f"read_rows_streaming: row {int(rows.max())} is outside {key} ({n_rows} rows) "
            f"-- the cell list and the matrix disagree"
        )
    wanted = np.zeros(n_rows, dtype=bool)
    wanted[rows] = True
    total_nnz = int(indptr[-1])
    nnz_per_wanted = (indptr[rows + 1] - indptr[rows]).astype(np.int64)
    out_indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(nnz_per_wanted, out=out_indptr[1:])
    data = np.empty(int(out_indptr[-1]), dtype=grp["data"].dtype)
    indices_dtype = grp["indices"].dtype if grp["indices"].dtype in (np.int32, np.int64) else np.int64
    indices = np.empty(int(out_indptr[-1]), dtype=indices_dtype)
    filled = 0
    start = 0
    print(f"    streaming {key}: rows_in_matrix={n_rows} rows_wanted={len(rows)} "
          f"max_row={int(rows.max()) if rows.size else None} nnz={total_nnz} "
          f"nnz_wanted={int(out_indptr[-1])}", flush=True)
    while start < total_nnz:
        stop = min(start + block_elements, total_nnz)
        offsets = np.arange(start, stop, dtype=np.int64)
        block_rows = np.searchsorted(indptr, offsets, side="right") - 1
        if block_rows.size and int(block_rows.max()) >= n_rows:
            raise SystemExit(f"read_rows_streaming: block row index {int(block_rows.max())} >= {n_rows}")
        pick = wanted[block_rows]
        if pick.any():
            count = int(pick.sum())
            data[filled : filled + count] = grp["data"][start:stop][pick]
            indices[filled : filled + count] = grp["indices"][start:stop][pick]
            filled += count
        start = stop
    if filled != int(out_indptr[-1]):
        raise SystemExit(f"read_rows_streaming: filled {filled} slots but expected {int(out_indptr[-1])}")
    # Elements stream in ascending (row, column) order, so slots fill exactly in the
    # order out_indptr (built from the sorted wanted rows) expects.
    return sp.csr_matrix((data, indices, out_indptr), shape=(len(rows), n_genes))


def fingerprint(indices: np.ndarray, data: np.ndarray) -> str:
    digest = hashlib.md5()
    digest.update(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
    digest.update(np.ascontiguousarray(data, dtype=np.float32).tobytes())
    return digest.hexdigest()


def verify_alignment(
    dataset: str,
    split: str,
    source_path: Path,
    key: str,
    row_order: list[str],
    positions: np.ndarray,
    ids: list[str],
    genes: list[str],
    matrix: sp.csr_matrix,
) -> dict:
    """THE GUARD: prove the id at row i is the id of the cell whose expression is row i.

    Three independent checks, all cheap enough to run on every extraction:

    1. by construction -- ``ids[positions[i]] == row_order[i]`` for every i, over the ids the
       source itself reports at those positions;
    2. from disk -- the persisted id array is exactly ``row_order`` in that order, and the
       persisted matrix has one row per id;
    3. by content -- ``SPOT_CHECKS`` persisted rows are re-read from the atlas at the source
       row *their own id* names, with ``lib_common.matrix_rows`` (a different reader from the
       one the extraction used) and compared fingerprint for fingerprint.

    Check 3 is the one that would have caught the original bug: it asks the consumers'
    question of the file on disk, and the answer does not depend on how the file was built.
    """
    paths = expr_paths(dataset, split)
    rebuilt_at_position = [ids[int(p)] for p in positions]
    if rebuilt_at_position != list(row_order):
        raise SystemExit(
            f"{dataset}/{split}: ALIGNMENT FAILED at construction -- obs[{key}] does not name the "
            f"cells the row order claims; refusing to persist a matrix whose rows and ids disagree"
        )
    stored_matrix = sp.load_npz(paths["matrix"]).tocsr()
    stored_ids = [str(c) for c in np.load(paths["cells"], allow_pickle=True)]
    if stored_ids != list(row_order):
        raise SystemExit(
            f"{dataset}/{split}: ALIGNMENT FAILED on disk -- {paths['cells'].name} is not the "
            f"row-ordered id sequence ({sum(1 for a, b in zip(stored_ids, row_order) if a != b)} "
            f"positions differ); refusing to leave it there"
        )
    if stored_matrix.shape[0] != len(stored_ids):
        raise SystemExit(
            f"{dataset}/{split}: {paths['matrix'].name} has {stored_matrix.shape[0]} rows but "
            f"{len(stored_ids)} cell ids -- they cannot describe the same cells"
        )
    checks = {
        "row_positions_strictly_ascending": True,
        "ids_at_their_source_position_match_row_order": len(row_order),
        "persisted_ids_equal_row_order": len(row_order),
        "matrix_rows_equal_persisted_ids": int(stored_matrix.shape[0]),
        "spot_checks": 0,
        "spot_checks_matched": 0,
        "spot_check_rows": [],
    }
    picks = sorted(
        np.random.default_rng(0)
        .choice(len(row_order), size=min(SPOT_CHECKS, len(row_order)), replace=False)
        .tolist()
    )
    want = np.array([positions[i] for i in picks], dtype=np.int64)
    with h5py.File(source_path, "r") as handle:
        parts = matrix_rows(handle, key, want, len(genes))
    if len(parts) != 5:  # the source node is dense, not a CSR group: nothing to rebuild
        checks["spot_check_note"] = f"{key} is not stored as a CSR group; content re-read skipped"
        return checks
    data, indices, indptr, n_genes, read_rows = parts
    counts = indptr[read_rows + 1] - indptr[read_rows]
    source_rows = sp.csr_matrix(
        (data, indices, np.concatenate([[0], np.cumsum(counts)])), shape=(len(read_rows), n_genes)
    )
    mismatches = []
    for local, row_index in enumerate(picks):
        stored_row = stored_matrix.getrow(row_index)
        source_row = source_rows.getrow(local)
        if fingerprint(stored_row.indices, stored_row.data) != fingerprint(source_row.indices, source_row.data):
            mismatches.append(
                {"row": int(row_index), "source_row": int(positions[row_index]), "cell_id": row_order[row_index]}
            )
        del stored_row, source_row
    checks["spot_checks"] = len(picks)
    checks["spot_checks_matched"] = len(picks) - len(mismatches)
    checks["spot_check_rows"] = [
        {"row": int(i), "source_row": int(positions[i]), "cell_id": row_order[i]} for i in picks
    ]
    if mismatches:
        raise SystemExit(
            f"{dataset}/{split}: ALIGNMENT FAILED on content -- persisted row(s) {mismatches} do not "
            f"reproduce the source row their own cell id names. The ids and the rows disagree; "
            f"refusing to leave this extraction on disk."
        )
    return checks


def extract(dataset: str, cell_list: list[str], split: str, key: str | None = None, stream: bool = False) -> dict:
    path, id_column, gene_column = DATASETS[dataset]
    path = ROOT / path
    t0 = time.time()
    requested = sorted(set(cell_list))
    with h5py.File(path, "r") as handle:
        key = key or count_matrix_key(handle)
        ids = cell_ids(handle, id_column)
        position = {cid: i for i, cid in enumerate(ids)}
        missing = [c for c in requested if c not in position]
        if missing:
            raise SystemExit(
                f"{dataset}: {len(missing)} of {len(requested)} corpus cell ids are not in obs[{id_column}] "
                f"(e.g. {missing[:3]}) -- refusing to score a different cell set than the corpus"
            )
        genes = gene_names(handle, gene_column)
        # ------------------------------------------------------- ROW ORDER (the invariant)
        # The readers emit rows in the order they are handed, and that order *is* the source
        # matrix's own row index. So the ids must be saved sorted by source position -- not in
        # request order, not alphabetically: matrix row i is row_order[i], whose source row is
        # positions[i]. Saving request order here is precisely the misalignment that put 1,979
        # of muraro's 2,126 ids at the wrong index for every positional consumer.
        row_order = sorted(requested, key=lambda cid: position[cid])
        positions = np.array([position[cid] for cid in row_order], dtype=np.int64)
        assert_ascending_rows(positions, f"{dataset}: cell-id row positions")
        reader = read_rows_streaming if stream else read_rows
        matrix = reader(handle, key, positions, len(genes))
    if matrix.nnz and not np.all(matrix.data == np.round(matrix.data)):
        raise SystemExit(
            f"{dataset}: {key} is not integer-valued -- refusing to treat a normalised "
            f"matrix as counts (found {matrix.data[:5]})"
        )

    EXPR.mkdir(parents=True, exist_ok=True)
    paths = expr_paths(dataset, split)
    sp.save_npz(paths["matrix"], matrix.astype(np.float32), compressed=True)
    np.save(paths["genes"], np.array(genes, dtype=object), allow_pickle=True)
    np.save(paths["cells"], np.array(row_order, dtype=object), allow_pickle=True)
    alignment = verify_alignment(dataset, split, path, key, row_order, positions, ids, genes, matrix)
    write_order_sidecar(
        dataset,
        split,
        row_order,
        positions,
        verified_by="step2_expr.extract -- ids written in source-row order and the persisted rows "
        f"re-read from the atlas by content ({alignment['spot_checks_matched']}/"
        f"{alignment['spot_checks']} matched)",
        checks=alignment,
        extra={
            "ids_sha256_if_saved_in_request_order": ids_sha256(requested),
            "positions_where_request_order_would_differ": sum(
                1 for a, b in zip(requested, row_order) if a != b
            ),
            "request_order_note": "what the request-sorted id array would have hashed to; retained so a "
            "future regression that saves request order again can be recognised by digest instead of by "
            "an afternoon of probing",
        },
    )
    detected = np.diff(matrix.indptr)
    stats = {
        "dataset": dataset,
        "split": split,
        "path": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
        "matrix_key": key,
        "cells": len(cell_list),
        "genes": len(genes),
        "nnz": int(matrix.nnz),
        "median_detected_genes_per_cell": int(np.median(detected)),
        "min_detected_genes_per_cell": int(detected.min()),
        "max_detected_genes_per_cell": int(detected.max()),
        "max_value": float(matrix.data.max()) if matrix.nnz else 0.0,
        "all_integer_counts": bool(np.all(matrix.data == np.round(matrix.data))) if matrix.nnz else True,
        "rows_in_source_row_order": True,
        "alignment_checks": alignment,
        "order_certificate": str(paths["order"].relative_to(ROOT)),
        "ids_sha256": ids_sha256(row_order),
        "seconds": round(time.time() - t0, 1),
        "out": str(paths["matrix"].relative_to(ROOT)),
    }
    print(
        f"[{dataset}/{split}] cells={stats['cells']} genes={stats['genes']} nnz={stats['nnz']} "
        f"median_detected={stats['median_detected_genes_per_cell']} int_counts={stats['all_integer_counts']} "
        f"row_order=ids_in_source_order({alignment['spot_checks_matched']}/{alignment['spot_checks']} rows "
        f"reproduced from the source) ({stats['seconds']}s)",
        flush=True,
    )
    return stats


def main() -> None:
    """Usage: step2_expr.py [split[:dataset[:stream]] ...]   (default: test, train)

    ``stream`` forces the sequential reader -- the right choice for the 30-42 GB
    atlases, where scattered single-row reads decompress a chunk per row.
    """
    import json

    specs = sys.argv[1:] or ["test", "train"]
    stats = {}
    for spec in specs:
        parts = spec.split(":")
        split, only = parts[0], (parts[1] if len(parts) > 1 else None)
        stream = len(parts) > 2 and parts[2] == "stream"
        cell_file = QCOUT / f"cells_{split}.json"
        if not cell_file.exists():
            print(f"[{split}] no cell inventory at {cell_file}; skipping")
            continue
        cells = json.loads(cell_file.read_text(encoding="utf-8"))
        per_dataset: dict[str, list[str]] = {}
        for record in cells.values():
            per_dataset.setdefault(record["dataset_id"], []).append(record["cell_id"])
        for dataset in sorted(per_dataset):
            if only and dataset != only:
                continue
            ids = sorted(set(per_dataset[dataset]))
            stats[f"{dataset}/{split}"] = extract(dataset, ids, split, stream=stream)
    from lib_common import write_json

    existing = QCOUT / "step2_expr_summary.json"
    if existing.exists():
        stats = {**json.loads(existing.read_text(encoding="utf-8")), **stats}
    write_json(existing, stats)


if __name__ == "__main__":
    main()
