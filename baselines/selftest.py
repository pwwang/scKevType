"""Self-tests for the baselines' own plumbing -- the parts that can fail silently.

Three invariants, each of which has already bitten this work once:

1. ``collapse_duplicate_genes`` returns a matrix whose columns match the gene list it
   returns, sums the duplicates rather than dropping them, and multiplies on the right.
   (A left-side (n_unique x n_genes) aggregation raised a matmul dimension error on
   tabula_muris_senis: the operands were (n, k=17943) and (k=3500, m), i.e. the whole
   gene axis against the whole cell axis.)
2. ``load()`` pairs the matrix and the gene names from the same file, so a matrix from
   one dataset can never be read against another dataset's genes.
3. The streaming row reader returns exactly what the per-run reader returns, both now refusing
   a row array that is not already ascending -- the silent `np.sort` that let the cell ids and
   the matrix rows drift apart (test_row_order_guards).

Run: uv run --no-sync python baselines/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    DATASETS,
    EXPR,
    ROOT,
    cell_ids,
    count_matrix_key,
    gene_names,
    load_aligned_cells,
    write_order_sidecar,
)
from step2_expr import read_rows, read_rows_streaming  # noqa: E402
from step3_celltypist import collapse_duplicate_genes, load  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def test_collapse() -> None:
    # 5 columns, 3 unique symbols: A twice (1+2 -> 3), B once, C twice (0+4 -> 4).
    matrix = sp.csr_matrix(np.array([[1.0, 5.0, 2.0, 0.0, 4.0], [2.0, 0.0, 0.0, 1.0, 3.0]], dtype=np.float32))
    genes = ["A", "B", "A", "C", "C"]
    check("collapse: input column count equals gene-name count", matrix.shape[1] == len(genes),
          f"{matrix.shape[1]} columns vs {len(genes)} names")
    collapsed, out_genes, n_dropped = collapse_duplicate_genes(matrix, genes)
    check("collapse: returns one column per unique gene", collapsed.shape[1] == len(out_genes),
          f"{collapsed.shape[1]} columns vs {len(out_genes)} names")
    check("collapse: unique names in first-seen order", out_genes == ["A", "B", "C"], str(out_genes))
    check("collapse: counts the symbols it merged", n_dropped == 2, f"n_dropped={n_dropped}")
    dense = collapsed.toarray()
    check("collapse: sums duplicates rather than dropping them",
          np.allclose(dense, np.array([[3.0, 5.0, 4.0], [2.0, 0.0, 4.0]])), str(dense.tolist()))
    check("collapse: conserves total counts", np.isclose(collapsed.sum(), matrix.sum()))

    # A matrix from a different gene space must be refused, not multiplied positionally.
    try:
        collapse_duplicate_genes(sp.csr_matrix(np.ones((2, 3), dtype=np.float32)), ["A", "B"])
        check("collapse: refuses a mismatched gene space", False, "no exception raised")
    except ValueError as error:
        check("collapse: refuses a mismatched gene space", "different gene spaces" in str(error))

    # Double collapse, the upper-casing path step3 takes for the mouse atlas.
    matrix = sp.csr_matrix(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    genes = ["Cd3e", "CD3E", "Cd3e"]
    first, first_genes, _ = collapse_duplicate_genes(matrix, genes)
    upper, upper_genes, _ = collapse_duplicate_genes(first, [g.upper() for g in first_genes])
    check("collapse: upper-cased second pass still aligns", upper.shape[1] == len(upper_genes),
          f"{upper.shape[1]} columns vs {len(upper_genes)} names")
    check("collapse: upper-cased second pass sums what it merged", np.isclose(upper.toarray().sum(), 6.0))


def test_load_pairs_matrix_and_genes() -> None:
    for dataset, split in (("muraro", "test"), ("hao_cite", "train")):
        matrix, genes, cells = load(dataset, split)
        check(f"load({dataset}/{split}): matrix columns equal gene names", matrix.shape[1] == len(genes),
              f"{matrix.shape[1]} columns vs {len(genes)} names")
        check(f"load({dataset}/{split}): matrix rows equal cell ids", matrix.shape[0] == len(cells),
              f"{matrix.shape[0]} rows vs {len(cells)} cells")
        collapsed, unique_genes, dropped = collapse_duplicate_genes(matrix, genes)
        check(f"load({dataset}/{split}): gene names unique after collapse",
              len(set(unique_genes)) == len(unique_genes) and collapsed.shape[1] == len(unique_genes),
              f"raw file carries {dropped} duplicate symbols")
        check(f"load({dataset}/{split}): collapse conserves counts",
              np.isclose(collapsed.sum(), matrix.sum()))


def test_row_order_guards() -> None:
    """The guards that keep matrix row i paired with cell id i.

    All of this has failed once: `extract` saved the ids in request order while the readers
    emitted rows in ascending source-row order, which mislabelled 1,979 of muraro's 2,126 rows
    and produced a sub-floor "baseline" that was really a permutation artifact.
    """
    path, id_column, gene_column = DATASETS["muraro"]
    with h5py.File(ROOT / path, "r") as handle:
        genes = gene_names(handle, gene_column)
        key = count_matrix_key(handle)
        for reader, name in ((read_rows, "read_rows"), (read_rows_streaming, "read_rows_streaming")):
            try:
                reader(handle, key, np.array([5, 3, 1], dtype=np.int64), len(genes))
                check(f"{name}: refuses a non-ascending row array", False, "no exception raised")
            except SystemExit as error:
                check(f"{name}: refuses a non-ascending row array", "strictly ascending" in str(error))
        # A re-ordered id array must be refused even when it holds exactly the right ids.
        scratch = EXPR / "zz_selftest_test_cells.npy"
        scratch_order = EXPR / "zz_selftest_test_order.json"
        np.save(scratch, np.array(["a", "b", "c"], dtype=object))
        write_order_sidecar("zz_selftest", "test", ["a", "b", "c"], [7, 8, 9],
                            verified_by="selftest", checks={})
        try:
            check("certificate: a certified id array loads", load_aligned_cells("zz_selftest") == ["a", "b", "c"])
            np.save(scratch, np.array(["b", "a", "c"], dtype=object))
            try:
                load_aligned_cells("zz_selftest")
                check("certificate: refuses a re-ordered id array", False, "no exception raised")
            except SystemExit as error:
                check("certificate: refuses a re-ordered id array", "ids_sha256 differs" in str(error))
        finally:
            scratch.unlink(missing_ok=True)
            scratch_order.unlink(missing_ok=True)

        # And the arrays actually on disk must be ascending in the source's own row order.
        ids = load_aligned_cells("muraro", "test")
        position = {c: i for i, c in enumerate(cell_ids(handle, id_column))}
        positions = [position[c] for c in ids]
        check("muraro/test: certified ids ascend in source-row order",
              all(a < b for a, b in zip(positions, positions[1:])),
              f"first positions {positions[:4]}, last {positions[-1]}")


def test_streaming_matches_run_reader() -> None:
    dataset = "muraro"
    path, id_column, gene_column = DATASETS[dataset]
    with h5py.File(ROOT / path, "r") as handle:
        ids = cell_ids(handle, id_column)
        position = {c: i for i, c in enumerate(ids)}
        genes = gene_names(handle, gene_column)
        n_rows = len(ids)
        ordered = sorted(position)
        # The last row of the matrix is the case that overran: the block scan visits every
        # row, so a wanted-set that stops short of the end used to index past it. Include
        # it, include the first row, and force many blocks (including a short final one).
        wanted = sorted({ordered[0], ordered[1], ordered[-1], ordered[-2]} | set(ordered[::7][:300]))
        rows = np.array([position[c] for c in wanted], dtype=np.int64)
        # The rows must be handed to the readers in ascending order -- that IS the invariant the
        # readers now enforce (a request order is not a row order).
        rows = np.sort(rows)
        key = count_matrix_key(handle)
        a = read_rows(handle, key, rows, len(genes))
        for block in (10_000, 100_000, 10_000_000):
            b = read_rows_streaming(handle, key, rows, len(genes), block_elements=block)
            check(
                f"streaming reader matches the per-run reader (block={block}, last row included)",
                np.array_equal(a.data, b.data)
                and np.array_equal(a.indices, b.indices)
                and np.array_equal(a.indptr, b.indptr),
            )
        # A row number past the end of the matrix must fail loudly and immediately.
        try:
            read_rows_streaming(handle, key, np.array([n_rows], dtype=np.int64), len(genes), block_elements=10_000)
            check("streaming reader rejects an out-of-range row", False, "no exception raised")
        except SystemExit as error:
            check("streaming reader rejects an out-of-range row", "outside" in str(error), str(error)[:60])


if __name__ == "__main__":
    test_collapse()
    test_load_pairs_matrix_and_genes()
    test_row_order_guards()
    test_streaming_matches_run_reader()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        raise SystemExit(1)
    print("all checks passed")
