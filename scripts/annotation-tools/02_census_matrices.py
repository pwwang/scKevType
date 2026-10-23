"""Step 2 of the standard-annotator baseline: the census cells' count matrices.

One matrix per dataset, holding exactly the cells the population's `panel_choice` questions
name, in a certified row order, so that a tool's label for row i is the label of ``cells[i]``.

ROW ORDER: the source atlas's own order, selected by a boolean mask
-------------------------------------------------------------------
A certified extraction already fixes what row i of it holds. Subsetting it by a boolean mask
keeps that order: row i of the subset is the i-th selected row of the source, and
``cells[i]`` is that row's own cell id. The certificate records the id sequence's digest and
the source positions read, and ``load_aligned_cells`` refuses an array whose order is not
certified -- the refusal is the point, not an obstacle.

WHERE EACH MATRIX COMES FROM
----------------------------
* ``muraro``, ``siletti``, ``tabula_muris_senis``, ``hao_cite`` -- a certified extraction the
  project's own baselines already read (``baselines/out/expr/<dataset>_<split>.npz``; hao_cite's
  is the ``train`` split, which is its matrix). Its row order is certified by
  ``<dataset>_<split>_order.json`` and the loader here is ``lib_common.load_aligned_cells``,
  which refuses an uncertified array.
* ``tabula_sapiens`` -- no extraction exists for this atlas in the project's baselines, so it
  is read here from the atlas itself (``data/raw.9p/tabula_sapiens/tabula_sapiens.h5ad``,
  45,013,733,785 bytes, sha256 sidecar quoted in the summary) with ``lib_common.matrix_rows``,
  the same contiguous-run reader ``step2_expr.py`` uses.
* ``dominguez_conde`` -- likewise read from ``data/raw.9p/dominguez_conde_cxg/global.h5ad``
  (3,036,698,635 bytes) through the same ``matrix_rows`` path.

Both atlas reads are spot-checked: five of the persisted rows are re-read one at a time
straight from the atlas (by that cell's own source row) and their gene content is compared with
what was persisted; a mismatch refuses to keep the matrix.

The count matrix is ``raw/X`` in every file (integer counts; ``X`` is normalised in some of
them), which is what CellTypist's own tutorial expects to normalise itself.

RESUMABLE
---------
Each dataset's matrix, gene list, cell list and certificate are written as soon as that dataset
completes, and ``matrices.summary.json`` is rewritten after every dataset, so a job that dies
part way through can be continued with ``--resume`` (which skips a dataset whose four files
already exist) or built one dataset at a time with ``--only``. Elapsed seconds are reported per
dataset either way.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/02_census_matrices.py

    # the v4 population, into its own working directory
    uv run --no-sync python scripts/annotation-tools/02_census_matrices.py \
        --population reports/census-suite-ids-v4.jsonl \
        --dir reports/annotation-tools-v4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

ROOT = Path(os.environ.get("SCKEVTYPE_ROOT", "/mnt/f/E/hermes-workspace/scKevType"))
sys.path.insert(0, str(ROOT / "baselines"))
from lib_common import (
    cell_ids,
    count_matrix_key,
    gene_names,
    ids_sha256,
    load_aligned_cells,
    matrix_rows,
)

DEFAULT_POPULATION = ROOT / "reports" / "census-suite-ids.jsonl"
DEFAULT_DIR = ROOT / "reports" / "annotation-tools"
BASE_EXPR = ROOT / "baselines" / "out" / "expr"

#: dataset -> (split, why) for the extractions the project's baselines already certified
FROM_BASELINES: dict[str, tuple[str, str]] = {
    "muraro": ("test", "certified extraction, in-baselines"),
    "siletti": ("test", "certified extraction, in-baselines"),
    "tabula_muris_senis": ("test", "certified extraction, in-baselines"),
    "hao_cite": ("train", "certified extraction, in-baselines (its own matrix)"),
}
#: dataset -> (atlas path, cell-id column, gene-symbol column, why) for the atlases read here
FROM_ATLAS: dict[str, tuple[Path, str, str, str]] = {
    "tabula_sapiens": (
        ROOT / "data" / "raw.9p" / "tabula_sapiens" / "tabula_sapiens.h5ad",
        "_index",
        "feature_name",
        "no certified extraction exists for this atlas in the project's baselines",
    ),
    "dominguez_conde": (
        ROOT / "data" / "raw.9p" / "dominguez_conde_cxg" / "global.h5ad",
        "_index",
        "feature_name",
        "no certified extraction exists for this atlas in the project's baselines",
    ),
}

SPOT_CHECKS = 5


def show(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def census_cells(questions: Path) -> dict[str, list[str]]:
    """dataset -> the population's cells for it, in the order the question table names them."""
    per: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    with open(questions, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ds, cell = row["dataset"], row["cell"]
            seen.setdefault(ds, set())
            if cell not in seen[ds]:
                seen[ds].add(cell)
                per.setdefault(ds, []).append(cell)
    return per


def population_datasets(population: Path) -> dict[str, int]:
    """dataset -> panel_choice rows, straight off the population file."""
    per: dict[str, int] = {}
    with open(population, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["family"] == "panel_choice":
                per[row["dataset"]] = per.get(row["dataset"], 0) + 1
    if not per:
        raise SystemExit(f"{population}: no panel_choice rows -- refusing to build nothing")
    return per


def write_certificate(dataset: str, cells: list[str], source_positions: list[int] | None, extra: dict) -> dict:
    payload = {
        "dataset": dataset,
        "cells": len(cells),
        "ids_sha256": ids_sha256(cells),
        "ids_in_source_row_order": True,
        "row_positions_strictly_ascending": True,
        "verified_by": "scripts/annotation-tools/02_census_matrices.py",
        "source_positions_sha256": (
            hashlib.sha256(np.asarray(source_positions, dtype=np.int64).tobytes()).hexdigest()
            if source_positions is not None
            else None
        ),
        "what_this_proves": "matrix row i holds the counts of `cells[i]`; a tool that labels "
        "row i labels that cell",
        **extra,
    }
    (OUT / f"{dataset}.certificate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def save(dataset: str, matrix, genes: list[str], cells: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sp.save_npz(OUT / f"{dataset}.npz", matrix.tocsr())
    np.save(OUT / f"{dataset}_genes.npy", np.array(genes, dtype=object))
    np.save(OUT / f"{dataset}_cells.npy", np.array(cells, dtype=object))


def stats_of(matrix, cells: list[str]) -> dict:
    return {
        "cells": len(cells),
        "median_detected_genes_per_cell": float(np.median(np.diff(matrix.indptr))),
        "all_integer_counts": bool(np.all(matrix.data == np.round(matrix.data))),
        "row_order": "source order (a boolean mask over the certified/source rows)",
    }


def from_baseline(dataset: str, cells: list[str]) -> tuple[dict, list[str], list[int]]:
    t0 = time.time()
    split, why = FROM_BASELINES[dataset]
    matrix = sp.load_npz(BASE_EXPR / f"{dataset}_{split}.npz").tocsr()
    genes = [str(g) for g in np.load(BASE_EXPR / f"{dataset}_{split}_genes.npy", allow_pickle=True)]
    certified = load_aligned_cells(dataset, split)  # raises unless the row order is certified
    if matrix.shape[0] != len(certified) or matrix.shape[1] != len(genes):
        raise SystemExit(f"{dataset}: matrix shape {matrix.shape} does not match the certified ids/genes")
    index = {c: i for i, c in enumerate(certified)}
    missing = [c for c in cells if c not in index]
    if missing:
        raise SystemExit(f"{dataset}: {len(missing)} population cells are not in the certified matrix, e.g. {missing[:5]}")
    # the boolean mask: keeps the certified array's row order, so the persisted ids are the
    # source order and the certificate's claim about source rows is literal
    mask = np.zeros(len(certified), dtype=bool)
    mask[[index[c] for c in cells]] = True
    if int(mask.sum()) != len(set(cells)):
        raise SystemExit(f"{dataset}: the mask selects {int(mask.sum())} rows for {len(set(cells))} distinct cells")
    subset = matrix[mask]
    kept = [c for c, m in zip(certified, mask) if m]
    positions = np.nonzero(mask)[0]
    if not np.all(np.diff(positions) > 0):
        raise SystemExit(f"{dataset}: the selected source positions are not strictly ascending")
    # the cheap positional sanity check: a mask must not change the number of stored values
    if subset.nnz != sum(matrix.getrow(r).nnz for r in positions):
        raise SystemExit(f"{dataset}: subset nnz {subset.nnz} != sum of the selected rows' nnz")
    save(dataset, subset, genes, kept)
    out = {
        "source": f"baselines/out/expr/{dataset}_{split}.npz (certified extraction)",
        "source_split": split,
        "source_is_certified_extraction": True,
        "source_note": why,
        "genes": len(genes),
        "nnz": int(subset.nnz),
        **stats_of(subset, kept),
        "seconds": round(time.time() - t0, 1),
    }
    out["source_positions_ascending"] = True
    return out, kept, [int(p) for p in positions]


def from_atlas(dataset: str, cells: list[str]) -> tuple[dict, list[str], list[int]]:
    t0 = time.time()
    atlas, cell_column, gene_column, why = FROM_ATLAS[dataset]
    if not atlas.exists():
        raise SystemExit(f"{dataset}: atlas not found at {atlas}")
    with h5py.File(atlas, "r") as fh:
        # lib_common's readers decode anndata categoricals stored as groups, which these
        # atlases' var/feature_name and obs columns are.
        ids = cell_ids(fh, cell_column)
        genes = gene_names(fh, gene_column)
        key = count_matrix_key(fh)
        index = {c: i for i, c in enumerate(ids)}
        missing = [c for c in cells if c not in index]
        if missing:
            raise SystemExit(f"{dataset}: {len(missing)} population cells not in the atlas, e.g. {missing[:5]}")
        mask_positions = np.array(sorted(index[c] for c in cells), dtype=np.int64)
        data, indices, indptr, n_genes, sorted_positions = matrix_rows(fh, key, mask_positions, len(genes))
        # matrix_rows reads contiguous runs and returns the rows it read, ascending; rebuild the
        # csr in that order -- which IS the source order the mask would have selected.
        nnz = (indptr[sorted_positions + 1] - indptr[sorted_positions]).astype(np.int64)
        out_indptr = np.zeros(len(sorted_positions) + 1, dtype=np.int64)
        np.cumsum(nnz, out=out_indptr[1:])
        subset = sp.csr_matrix((data, indices, out_indptr), shape=(len(sorted_positions), n_genes))
        kept = [ids[int(p)] for p in sorted_positions]
        if set(kept) != set(cells):
            raise SystemExit(f"{dataset}: the rows read are not the cells asked for")
        row_of = {c: i for i, c in enumerate(kept)}
        # spot checks: re-read the named rows one at a time, straight from the atlas, and compare
        picks = [kept[i] for i in range(0, len(kept), max(1, len(kept) // SPOT_CHECKS))][:SPOT_CHECKS]
        spot = []
        for c in picks:
            r = index[c]
            s, e = int(fh[key]["indptr"][r]), int(fh[key]["indptr"][r + 1])
            gidx = np.asarray(fh[key]["indices"][s:e])
            gval = np.asarray(fh[key]["data"][s:e])
            mine = subset.getrow(row_of[c])
            ok = (
                mine.nnz == len(gidx)
                and np.array_equal(np.asarray(mine.indices), gidx)
                and np.allclose(np.asarray(mine.data), gval)
            )
            spot.append({"cell_id": c, "source_row": r, "persisted_row": row_of[c], "matched": bool(ok),
                         "nnz": len(gidx)})
        if not all(x["matched"] for x in spot):
            raise SystemExit(f"{dataset}: spot check failed -- refusing to keep a misaligned matrix: {spot}")
        sidecar = atlas.with_suffix(".h5ad.sha256")
        atlas_sha = sidecar.read_text().strip() if sidecar.exists() else None
        save(dataset, subset, genes, kept)
        out = {
            "source": show(atlas),
            "atlas_bytes": atlas.stat().st_size,
            "atlas_sha256_sidecar": atlas_sha,
            "matrix_key": key,
            "source_is_certified_extraction": False,
            "source_note": why,
            "genes": len(genes),
            "nnz": int(subset.nnz),
            **stats_of(subset, kept),
            "spot_checks": spot,
            "seconds": round(time.time() - t0, 1),
        }
        return out, kept, [int(p) for p in sorted_positions]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the population's per-dataset count matrices.")
    parser.add_argument("--population", default=str(DEFAULT_POPULATION),
                        help="the census id list the question table was built from (checked against it)")
    parser.add_argument("--dir", default=str(DEFAULT_DIR),
                        help="the working directory: questions.jsonl is read from it, expr/ written to it")
    parser.add_argument("--only", default=None,
                        help="comma-separated datasets to build (default: every dataset the table names)")
    parser.add_argument("--resume", action="store_true",
                        help="skip a dataset whose matrix, genes, cells and certificate all exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    global OUT
    out_dir = Path(args.dir).resolve()
    OUT = out_dir / "expr"
    OUT.mkdir(parents=True, exist_ok=True)
    questions = out_dir / "questions.jsonl"
    if not questions.exists():
        raise SystemExit(f"{questions}: not found -- run 01_census_questions.py --dir {args.dir} first")
    population = Path(args.population)
    if not population.exists():
        raise SystemExit(f"{population}: not found")

    per_dataset_rows = population_datasets(population)
    cells_by_dataset = census_cells(questions)
    if set(cells_by_dataset) != set(per_dataset_rows):
        raise SystemExit(
            f"the question table's datasets {sorted(cells_by_dataset)} differ from the population's "
            f"{sorted(per_dataset_rows)} -- 01 was run against a different population"
        )
    summary_path = out_dir / "matrices.summary.json"
    summary: dict[str, dict] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    order = [d for d in list(FROM_BASELINES) + list(FROM_ATLAS) if d in cells_by_dataset]
    unknown = [d for d in cells_by_dataset if d not in FROM_BASELINES and d not in FROM_ATLAS]
    if unknown:
        raise SystemExit(f"no source is declared for {unknown} -- refusing to guess where its matrix lives")
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        unknown = sorted(wanted - set(order))
        if unknown:
            raise SystemExit(f"--only={sorted(unknown)} name no dataset of this population ({sorted(order)})")
        order = [d for d in order if d in wanted]

    for dataset in order:
        cells = cells_by_dataset[dataset]
        print(f"[{dataset}] {len(cells)} population cells ({per_dataset_rows[dataset]} questions)", flush=True)
        files = [OUT / f"{dataset}.npz", OUT / f"{dataset}_genes.npy", OUT / f"{dataset}_cells.npy",
                 OUT / f"{dataset}.certificate.json"]
        if args.resume and all(p.exists() for p in files):
            cert = json.loads(files[3].read_text(encoding="utf-8"))
            stats = {k: v for k, v in cert.items()
                     if k not in ("what_this_proves", "verified_by")}
            stats["resumed"] = True
            summary[dataset] = stats
            print(f"    resumed from {files[3].name}: {cert['cells']} cells, ids_sha256 {cert['ids_sha256'][:12]}", flush=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            continue
        if dataset in FROM_ATLAS:
            stats, kept, positions = from_atlas(dataset, cells)
        elif dataset in FROM_BASELINES:
            stats, kept, positions = from_baseline(dataset, cells)
        else:
            raise SystemExit(f"no source is declared for {dataset}")
        if len(kept) != len(cells):
            raise SystemExit(f"{dataset}: kept {len(kept)} rows for {len(cells)} cells")
        cert = write_certificate(dataset, kept, positions, stats)
        stats["certificate_ids_sha256"] = cert["ids_sha256"]
        summary[dataset] = stats
        # written after EVERY dataset, so the run is resumable and each dataset's elapsed time
        # and digest are on disk as soon as that dataset completes
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"    {stats['seconds']}s nnz={stats['nnz']:,} integer={stats['all_integer_counts']} "
              f"median_genes={stats['median_detected_genes_per_cell']:.0f} "
              f"ids_sha256={cert['ids_sha256'][:12]} -> {OUT.name}/{dataset}.npz", flush=True)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
