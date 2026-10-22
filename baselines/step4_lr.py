"""Step 4 -- a plain logistic regression on highly variable genes.

Three arms, all on the corpus's own cells and labels:

* ``splitexact`` -- the literal request: HVGs from the train split, fit on the train
  split (``runs/real-v2/train.jsonl``, the hao_cite cells it holds), predict the test
  split. Its class space is whatever the train split contains, so it can only ever
  answer with those labels -- a fact about the transfer, reported rather than patched.
* ``atlas`` -- the generous reading of the same instruction: same HVGs, fit on every
  in-scope hao_cite cell of the corpus's *train tier*. hao_cite is the train atlas; its
  cells in this corpus land in the train, calibration and development partitions, and
  no test atlas is touched. Same atlas as our model with ~70x the cells, so nobody can
  say the baseline was starved of data.
* ``cv_indomain`` -- NOT a baseline, and labelled as such wherever it appears: 5-fold
  cross-validation inside each individual test dataset, with the HVGs re-selected
  inside every fold. It uses the test labels, which no honest baseline may do. It
  exists so the transfer numbers can be read against the ceiling of this same method
  with in-domain supervision. Its folds are **stratified over cells**, so cells from
  one donor sit on both sides of every fold; ``cv_grouped`` is the same measurement
  with the folds grouped by donor, and that is the one to quote.
* ``cv_grouped`` -- the same in-domain measurement with **leave-one-donor-out** folds:
  every fold scores one donor's cells with a model fitted only on the other donors, and
  the HVGs are re-selected inside each of those training folds. On an atlas with four
  donors this is four fits; with 23 donors it is 23. Reference only, on the same grounds
  as ``cv_indomain`` -- and the twenty-fold difference those two arms can show is a
  statement about donor structure, not about the method.

Leakage checks, recorded in the summary as ``leakage_checks`` because a number this
high invites the question: the transfer arms fit on nothing but the corpus's hao_cite
train tier (asserted disjoint, by dataset and cell id, from every scored dataset), the
in-domain arms re-select HVGs inside each training fold, and the in-domain arms never
score a cell their fit saw.

Pipeline: counts -> CP10K -> log1p -> the HVG columns. Gene symbols are matched
exactly; for the mouse atlas a case-insensitive match is added and reported, because
the human and mouse symbols for one ortholog differ mostly in case (Cd3e/CD3E) and the
corpus's own definitions list both spellings.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    ARM_TAGS,
    ASYMMETRY,
    DATASETS,
    QCOUT,
    ROOT,
    cell_ids,
    count_matrix_key,
    definitions,
    gene_names,
    obs_column,
    tag_string,
    write_json,
)
from step2_expr import read_rows  # noqa: E402
from step3_celltypist import collapse_duplicate_genes, load  # noqa: E402

TEST_DATASETS = ["muraro", "siletti", "tabula_sapiens", "tabula_muris_senis"]
N_HVG = 2000
LR_PATH = QCOUT / "lr"
CASE_INSENSITIVE_DATASETS = {"tabula_muris_senis"}
#: The two in-domain fold plans, named so a summary line says which one produced a number.
CELL_STRATIFIED = "5-fold StratifiedKFold over cells (donor structure can cross folds)"
GROUPED = "leave-one-donor-out (no cell is scored by a fit that saw its own donor)"


def lognorm(counts: sp.csr_matrix) -> sp.csr_matrix:
    """Counts -> CP10K -> log1p, staying sparse. Every caller passes counts."""
    counts = counts.tocsr().astype(np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    out = (sp.diags(1e4 / totals) @ counts).tocsr()
    out.data = np.log1p(out.data)
    return out


def select_hvg(counts: sp.csr_matrix, genes: list[str], n_top: int = N_HVG) -> tuple[list[int], dict]:
    """Highly variable genes by normalised dispersion, from counts.

    This is the classic ``flavor='seurat'`` definition -- dispersion = variance/mean of
    the log-normalised values, averaged in bins of mean expression and z-scored within
    bin -- computed on sparse arrays so the baseline needs no optional scanpy extra and
    never densifies the matrix.
    """
    log = lognorm(counts)
    mean = np.asarray(log.mean(axis=0)).ravel()
    mean_sq = np.asarray(log.multiply(log).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.where(mean > 0, variance / np.where(mean > 0, mean, 1.0), 0.0)
    detected = np.asarray((counts > 0).sum(axis=0)).ravel()
    usable = detected >= 3
    bins = int(min(20, max(1, usable.sum() // 50)))
    edges = np.quantile(mean[usable], np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    which = np.digitize(mean, edges) - 1
    score = np.full(len(genes), -np.inf, dtype=np.float64)
    for b in range(bins):
        in_bin = usable & (which == b)
        if in_bin.sum() < 2:
            continue
        sub = dispersion[in_bin]
        spread = sub.std()
        score[in_bin] = (sub - sub.mean()) / spread if spread > 0 else 0.0
    order = np.argsort(-score)
    chosen = [int(i) for i in order[:n_top] if np.isfinite(score[i])]
    info = {
        "method": "normalised dispersion (scanpy flavor='seurat' definition), mean-binned and z-scored, "
        "genes detected in >=3 cells",
        "n_train_cells": int(counts.shape[0]),
        "n_genes_input": len(genes),
        "n_genes_detected_in_3_cells": int(usable.sum()),
        "n_selected": len(chosen),
        "bins": bins,
    }
    return chosen, info


def take_columns(
    counts: sp.csr_matrix, genes: list[str], chosen: list[str], case_insensitive: bool
) -> tuple[sp.csr_matrix, dict]:
    """Log-normalised counts restricted to ``chosen`` genes, in that column order.

    A gene the dataset does not carry becomes an all-zero column -- the standard
    treatment -- which is why the coverage numbers ride along with every score. All
    absent genes point at one appended zero column, so this is a single vectorised
    column take rather than a Python loop over 2,000 columns.
    """
    log = lognorm(counts)
    genes = list(genes)
    if case_insensitive:
        log, genes, _ = collapse_duplicate_genes(log, [g.upper() for g in genes])
    log = sp.hstack([log, sp.csr_matrix((log.shape[0], 1), dtype=np.float32)], format="csr")
    position = {g: i for i, g in enumerate(genes)}
    absent_column = log.shape[1] - 1
    columns = np.array([position.get(g, absent_column) for g in chosen], dtype=np.int64)
    out = log[:, columns].tocsr()
    present = int(sum(1 for g in chosen if g in position))
    return out, {
        "hvg_present": present,
        "hvg_absent": len(chosen) - present,
        "hvg_coverage": round(present / len(chosen), 4),
        "case_insensitive_symbols": case_insensitive,
    }


def fit_lr(x, y, seed: int = 0) -> LogisticRegression:
    """L2 logistic regression on standardised features -- the ordinary strong linear baseline.

    Scaling is not a detail here: log-normalised expression columns differ in scale by
    orders of magnitude, and an unscaled fit spends most of its L2 budget on the
    high-variance columns. CellTypist's own pipeline scales its input before predicting
    ("Scaling input data" in its output), so scaling is also what makes the two baselines
    comparable. ``with_mean=False`` keeps the sparse matrix sparse. The scaler lives inside
    the fitted pipeline so that fitting it on the training fold only is automatic.
    """
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
    )
    model.fit(x, y)
    return model


def sanity_accuracy(dataset: str, cells, predicted) -> dict:
    """Top-1 agreement with the corpus's own label, over that dataset's registered cells.

    Printed and stored next to every arm so a run that produced nothing usable cannot be
    mistaken for a run that produced a number.
    """
    inventory = json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
    truth = {
        c["cell_id"]: c["cl_id"]
        for c in inventory.values()
        if c["dataset_id"] == dataset and c["registered"]
    }
    hit = total = 0
    for cell, value in zip(cells, predicted):
        if cell not in truth:
            continue
        total += 1
        if str(value) == truth[cell]:
            hit += 1
    coverage = sum(1 for c in cells if c in truth)
    return {
        "registered_cells_in_dataset": len(truth),
        "cells_overlapping_truth": coverage,
        "exact_top1_vs_corpus_label": round(hit / total, 4) if total else None,
    }


def predict_and_save(path: Path, cells, model, x, extra: dict | None = None) -> dict:
    probabilities = model.predict_proba(x)
    classes = np.array([str(c) for c in model.classes_], dtype=object)
    predicted = classes[probabilities.argmax(axis=1)]
    np.savez_compressed(
        path,
        cells=np.array(cells, dtype=object),
        predicted=predicted,
        classes=classes,
        probabilities=probabilities.astype(np.float32),
        meta=np.array(json.dumps(extra or {}), dtype=object),
    )
    return {
        "top_predicted": dict(Counter(predicted).most_common(8)),
        "mean_top_probability": round(float(probabilities.max(axis=1).mean()), 4),
    }


def read_atlas_design(chosen: list[str]) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """Every in-scope hao_cite cell of the corpus's train tier, restricted to the HVG columns.

    Row blocks are read sequentially and log-normalised with their own totals before the
    column take, so peak memory is one block's non-zeros rather than the atlas's. A cell
    is in scope when its own label is one the register names -- the corpus's rule ("a
    cell whose label is not a key is dropped and counted", configs/corpus/real-v2.yaml).
    """
    register = set(definitions())
    path, id_column, gene_column = DATASETS["hao_cite"]
    path = ROOT / path
    blocks_x, blocks_y = [], []
    stats = {"atlas_rows": 0, "atlas_rows_in_scope": 0, "atlas_rows_out_of_scope": 0}
    with h5py.File(path, "r") as handle:
        ids = cell_ids(handle, id_column)
        labels = obs_column(handle, "cell_type_ontology_term_id")
        genes = gene_names(handle, gene_column)
        key = count_matrix_key(handle)
        chunk = 20000
        for start in range(0, len(ids), chunk):
            stop = min(start + chunk, len(ids))
            block = np.arange(start, stop, dtype=np.int64)
            counts = read_rows(handle, key, block, len(genes))
            stats["atlas_rows"] += len(block)
            keep = np.array([labels[int(r)] in register for r in block])
            stats["atlas_rows_in_scope"] += int(keep.sum())
            stats["atlas_rows_out_of_scope"] += int((~keep).sum())
            if not keep.any():
                continue
            x_block, _ = take_columns(counts[keep], genes, chosen, case_insensitive=False)
            blocks_x.append(x_block)
            blocks_y.extend(labels[int(r)] for r in block[keep])
            print(f"    hao_cite rows {start}-{stop}: in scope {int(keep.sum())}", flush=True)
    x = sp.vstack(blocks_x).tocsr()
    y = np.array(blocks_y, dtype=object)
    stats["matrix_key"] = key
    stats["atlas_genes"] = len(genes)
    return x, y, stats


def main() -> None:
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print("Usage: step4_lr.py [--datasets=a,b]   (default: every dataset in TEST_DATASETS)")
        return
    only = None
    for arg in argv:
        if arg.startswith("--datasets="):
            only = {name.strip() for name in arg.split("=", 1)[1].split(",") if name.strip()}
    datasets = [d for d in TEST_DATASETS if not only or d in only]
    if not datasets:
        raise SystemExit(f"--datasets={sorted(only) if only else None} matched none of {TEST_DATASETS}")
    LR_PATH.mkdir(parents=True, exist_ok=True)
    summary_path = QCOUT / "step4_lr_summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    provenance = previous.get("_provenance", [])
    summary: dict = {k: v for k, v in previous.items() if k not in ("_provenance", "splitexact", "atlas")}
    # Per-dataset test blocks accumulate across invocations: running this with --datasets=muraro
    # must not drop the blocks an earlier run wrote for siletti. Only the datasets in this run
    # are replaced; the rest are carried over unchanged.
    test_blocks: dict = dict(previous.get("splitexact", {}).get("tests", {}))
    summary["n_hvg_requested"] = N_HVG
    print(f"parsed args: argv={argv} datasets={datasets}")

    # ---------- training material: HVGs from the train split ----------
    train_counts, train_genes, train_cells = load("hao_cite", "train")
    train_counts, train_genes, collapsed = collapse_duplicate_genes(train_counts, train_genes)
    inventory = json.loads((QCOUT / "cells_train.json").read_text(encoding="utf-8"))
    label_of = {c["cell_id"]: c["cl_id"] for c in inventory.values() if c["dataset_id"] == "hao_cite"}
    y_all = np.array([label_of[c] for c in train_cells], dtype=object)
    registered = np.array([not str(v).startswith("local:") for v in y_all])
    hvg_index, hvg_info = select_hvg(train_counts[registered], train_genes)
    hvg = [train_genes[i] for i in hvg_index]
    hvg_info.update(
        {
            "duplicate_symbols_collapsed": collapsed,
            "n_classes": int(len(set(y_all[registered]))),
            "n_cells_used": int(registered.sum()),
            "n_cells_local_label_dropped": int((~registered).sum()),
            "class_counts": dict(Counter(str(v) for v in y_all[registered]).most_common()),
        }
    )
    summary["hvg"] = hvg_info
    (LR_PATH / "hvg.json").write_text(json.dumps({"hvg": hvg, **hvg_info}, indent=2) + "\n", encoding="utf-8")
    print(f"HVG: {len(hvg)} of {len(train_genes)} genes from {int(registered.sum())} registered train cells")
    print(f"    train classes: {hvg_info['n_classes']}; local-label cells dropped: {hvg_info['n_cells_local_label_dropped']}")

    x_train, train_cov = take_columns(train_counts, train_genes, hvg, case_insensitive=False)

    # ---------- arm 1: split-exact ----------
    t0 = time.time()
    model_split = fit_lr(x_train[registered], y_all[registered])
    summary["splitexact"] = {
        "train_cells": int(registered.sum()),
        "n_train_classes": int(len(model_split.classes_)),
        "train_classes": [str(c) for c in model_split.classes_],
        "train_design": train_cov,
        "fit_seconds": round(time.time() - t0, 1),
        "tests": {},
    }
    print(f"split-exact LR fitted: {int(registered.sum())} cells, {len(model_split.classes_)} classes "
          f"({time.time() - t0:.1f}s)")

    # ---------- arm 2: the whole train tier ----------
    t0 = time.time()
    x_atlas, y_atlas, atlas_stats = read_atlas_design(hvg)
    model_atlas = fit_lr(x_atlas, y_atlas)
    summary["atlas"] = {
        "train_cells": int(x_atlas.shape[0]),
        "n_classes": int(len(model_atlas.classes_)),
        "train_classes": [str(c) for c in model_atlas.classes_],
        "class_counts": dict(Counter(str(v) for v in y_atlas).most_common(12)),
        "read_and_fit_seconds": round(time.time() - t0, 1),
        **atlas_stats,
    }
    print(f"atlas LR fitted: {x_atlas.shape[0]} cells, {len(model_atlas.classes_)} classes ({time.time() - t0:.1f}s)")
    del x_atlas, y_atlas

    # ---------- test predictions ----------
    for dataset in datasets:
        counts, genes, cells = load(dataset, "test")
        counts, genes, dropped = collapse_duplicate_genes(counts, genes)
        case_insensitive = dataset in CASE_INSENSITIVE_DATASETS
        x_test, cov = take_columns(counts, genes, hvg, case_insensitive=case_insensitive)
        cov["duplicate_symbols_collapsed"] = dropped
        per_arm = {}
        for name, model in (("splitexact", model_split), ("atlas", model_atlas)):
            t1 = time.time()
            per_arm[name] = predict_and_save(LR_PATH / f"{name}_{dataset}.npz", cells, model, x_test)
            per_arm[name]["seconds"] = round(time.time() - t1, 1)
            per_arm[name]["sanity"] = sanity_accuracy(dataset, cells, model.predict(x_test))
            print(f"  [{dataset}/{name}] classes={len(model.classes_)} "
                  f"exact_top1={per_arm[name]['sanity']['exact_top1_vs_corpus_label']} "
                  f"top={list(per_arm[name]['top_predicted'])[:3]}", flush=True)
        per_arm["cv_indomain"] = cross_validate(counts, genes, cells, dataset, case_insensitive)
        per_arm["cv_grouped"] = cross_validate_grouped(counts, genes, cells, dataset, case_insensitive)
        test_blocks[dataset] = {**cov, **per_arm}
        print(
            f"[{dataset}] hvg_coverage={cov['hvg_coverage']} "
            f"splitexact_top1={per_arm['splitexact']['sanity']['exact_top1_vs_corpus_label']} "
            f"atlas_top1={per_arm['atlas']['sanity']['exact_top1_vs_corpus_label']}",
            flush=True,
        )
        for arm in ("cv_indomain", "cv_grouped"):
            block = per_arm[arm]
            if block.get("skipped"):
                print(f"  [{dataset}/{arm}] skipped: {block['skipped']}", flush=True)
                continue
            thin = block["classes_with_fewer_cells_than_folds"]
            print(
                f"  [{dataset}/{arm}] tags={tag_string(arm)} acc={block['accuracy']} "
                f"floor={block['majority_class_rate']} ({block['n_classes']} classes, "
                f"{block['folds']} folds) grouping={block['grouping']}"
                + (f" | THIN CLASSES (fewer cells than folds): {thin}" if thin else ""),
                flush=True,
            )
    summary["leakage_checks"] = leakage_checks()
    summary["splitexact"]["tests"] = dict(sorted(test_blocks.items()))
    summary["arm_tags"] = {
        name: ARM_TAGS[name] for name in ("splitexact", "atlas", "cv_indomain", "cv_grouped")
    }
    summary["asymmetry"] = ASYMMETRY
    provenance.append(
        {
            "argv": argv,
            "datasets_filter": sorted(only) if only else "ALL",
            "datasets_run": datasets,
            "hvg_source": "train split (hao_cite), registered cells only",
            "arms": ["splitexact", "atlas", "cv_indomain(reference only)"],
        }
    )
    write_json(summary_path, {"_provenance": provenance, **summary})
    print(f"wrote {summary_path}")


def cv_truth(dataset: str, cells) -> tuple[np.ndarray, np.ndarray, dict]:
    """(labels, registered mask, cell_id -> donor_id) for one dataset's test cells."""
    inventory = json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
    rows = [c for c in inventory.values() if c["dataset_id"] == dataset]
    truth = {c["cell_id"]: c["cl_id"] for c in rows}
    donors = {c["cell_id"]: c["donor_id"] for c in rows}
    y = np.array([truth[c] for c in cells], dtype=object)
    keep = np.array([not str(v).startswith("local:") for v in y])
    return y, keep, donors


def score_folds(
    counts_kept,
    genes,
    y_kept,
    cells,
    positions,
    folds,
    fold_labels,
    grouping: str,
    case_insensitive: bool,
    out_path: Path,
) -> dict:
    """Run one fold plan: HVGs re-selected inside every training fold, score the held-out cells.

    ``folds`` are (train, test) index arrays into the registered cells. A fold's fit never
    touches the cells that fold scores -- asserted here rather than assumed, because "the fit
    saw the cells it is scored on" is the one failure mode that would silently inflate these
    numbers instead of raising.
    """
    predicted = np.array([""] * len(cells), dtype=object)
    fold_coverage, per_fold = [], []
    for label, (tr, te) in zip(fold_labels, folds):
        if set(tr.tolist()) & set(te.tolist()):
            raise SystemExit(f"{grouping}: fold {label} would fit on cells it scores")
        fold_hvg_index, _ = select_hvg(counts_kept[tr], genes)
        # The fold's HVGs are selected from the dataset's own (untransformed) symbols, so
        # they must go through the same symbol transform that take_columns applies to the
        # matrix -- otherwise every column misses and the "model" is a majority-class guess.
        fold_hvg = [genes[i].upper() if case_insensitive else genes[i] for i in fold_hvg_index]
        x_tr, cov_tr = take_columns(counts_kept[tr], genes, fold_hvg, case_insensitive)
        x_te, _ = take_columns(counts_kept[te], genes, fold_hvg, case_insensitive)
        fold_coverage.append(cov_tr["hvg_coverage"])
        model = fit_lr(x_tr, y_kept[tr])
        fold_predicted = model.predict(x_te)
        predicted[positions[te]] = fold_predicted
        per_fold.append(
            {
                "fold": str(label),
                "cells_scored": int(len(te)),
                "accuracy": round(float(np.mean(np.array([str(p) for p in fold_predicted]) == y_kept[te])), 4),
                "hvg_coverage": cov_tr["hvg_coverage"],
            }
        )
    correct = np.array([str(p) for p in predicted[positions]]) == y_kept
    accuracy = float(np.mean(correct))
    np.savez_compressed(
        out_path,
        cells=np.array(cells, dtype=object),
        predicted=predicted,
        classes=np.array(sorted(set(str(v) for v in y_kept)), dtype=object),
    )
    counts_true = Counter(str(v) for v in y_kept)
    total = sum(counts_true.values())
    majority_floor = max(counts_true.values()) / total
    # A class with fewer cells than there are folds cannot appear in every training fold of a
    # stratified split; sklearn only warns about it, so it is recorded here and named in the
    # summary rather than left in a log.
    thin = {cl: n for cl, n in counts_true.items() if n < len(folds)}
    return {
        "folds": len(folds),
        "grouping": grouping,
        "cells_scored": int(len(positions)),
        "accuracy": accuracy,
        # The floor travels with the accuracy: on a 56-class dataset an accuracy and a 7-class
        # one are not comparable reads, and a below-floor accuracy is a bug in the arm rather
        # than a finding about the data (exactly how the row-order misalignment showed up).
        "majority_class_rate": round(majority_floor, 4),
        "stratified_chance_rate": round(sum((v / total) ** 2 for v in counts_true.values()), 4),
        "accuracy_minus_floor": round(accuracy - majority_floor, 4),
        "n_classes": int(len(counts_true)),
        "class_counts": dict(counts_true.most_common(8)),
        "classes_with_fewer_cells_than_folds": thin,
        "mean_fold_hvg_coverage": round(float(np.mean(fold_coverage)), 4),
        "per_fold": per_fold,
        "note": "reference only: fits on this dataset's own labels; not a transfer baseline. "
        "Read the accuracy next to majority_class_rate: below it, the arm is broken, not beaten."
        + (
            " Folds are grouped by donor, so no cell is scored by a fit that saw a cell from its "
            "own donor."
            if grouping == GROUPED
            else " Folds are stratified over CELLS, so cells from one donor sit on both sides of "
            "every fold; prefer the donor-grouped arm as the in-domain number."
        ),
    }


def leakage_checks() -> dict:
    """The three questions a number this high invites, answered and asserted.

    (1) Did a scored atlas contribute a cell to any fit? The transfer arms fit on the corpus's
        hao_cite train tier only -- asserted disjoint from every scored dataset by
        ``(dataset_id, cell_id)`` pairs, not by bare barcode (barcodes collide across atlases).
    (2) Were the HVGs chosen on the whole matrix? No: ``select_hvg`` is called inside each fold
        on that fold's training rows, and the resulting column list is used for both sides.
    (3) Does a fold score a cell its fit saw? No -- ``score_folds`` raises on that, per fold.

    The check covers every dataset ``cells_test.json`` knows about, not only the ones this
    invocation scored, so a partial re-run cannot quietly narrow the claim.
    """
    fit_ids: set[tuple[str, str]] = set()
    for name in ("cells_train.json", "cells_calibration.json", "cells_development.json"):
        path = QCOUT / name
        if not path.exists():
            continue
        inventory = json.loads(path.read_text(encoding="utf-8"))
        fit_ids |= {(c["dataset_id"], c["cell_id"]) for c in inventory.values()}
    inventory = json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
    per_dataset: dict[str, set] = {}
    for c in inventory.values():
        per_dataset.setdefault(c["dataset_id"], set()).add((c["dataset_id"], c["cell_id"]))
    overlap = {ds: len(fit_ids & ids) for ds, ids in sorted(per_dataset.items())}
    if any(overlap.values()):
        raise SystemExit(
            f"a scored dataset contributed cells to the fit material: {overlap} -- the transfer "
            f"arms would be scoring cells they were fitted on"
        )
    return {
        "scored_datasets_disjoint_from_fit": overlap,
        "fit_material": "the corpus's hao_cite train tier (train + calibration + development "
        "partitions), asserted disjoint by (dataset_id, cell_id) from every scored dataset",
        "hvg_selection": "inside each training fold only: select_hvg is called on the fold's own "
        "training rows and that fold's column list is then used for both of its sides",
        "fold_fit_disjoint_from_fold_score": "asserted per fold in score_folds -- a fold that "
        "would fit on a cell it scores raises instead of producing a number",
        "donor_grouping": "cv_grouped fits on all other donors and scores one donor's cells; "
        "cv_indomain is stratified over cells and therefore leaks donor structure -- recorded "
        "here rather than quietly reported as an in-domain ceiling",
    }


def cross_validate(counts, genes, cells, dataset: str, case_insensitive: bool) -> dict:
    """5-fold CV inside one test dataset, with the HVGs re-selected inside each fold.

    Reference only: it fits on this dataset's own labels. Reported next to the transfer
    arms, never as one of them. The folds are stratified over cells, which leaks donor
    structure -- the donor-grouped arm is the one to quote.
    """
    y, keep, _donors = cv_truth(dataset, cells)
    if keep.sum() < 100:
        return {"skipped": f"only {int(keep.sum())} registered cells"}
    positions = np.where(keep)[0]
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(int(keep.sum())), y[keep])
    )
    return score_folds(
        counts[keep],
        genes,
        y[keep],
        cells,
        positions,
        folds,
        [str(i) for i in range(len(folds))],
        CELL_STRATIFIED,
        case_insensitive,
        LR_PATH / f"cv_indomain_{dataset}.npz",
    )


def cross_validate_grouped(counts, genes, cells, dataset: str, case_insensitive: bool) -> dict:
    """Leave-one-donor-out CV inside one test dataset: the honest in-domain number.

    Every fold fits on all the *other* donors and scores one donor's cells, with the HVGs
    re-selected inside that fold's training part. This costs one fit per donor, and it is
    the measurement that answers "would this method still look like this on a donor it has
    not seen", which a per-cell split cannot answer.
    """
    y, keep, donors = cv_truth(dataset, cells)
    if keep.sum() < 100:
        return {"skipped": f"only {int(keep.sum())} registered cells"}
    positions = np.where(keep)[0]
    groups = np.array([str(donors.get(c)) for c in np.array(cells, dtype=object)[keep]], dtype=object)
    unique = sorted(set(groups.tolist()))
    if len(unique) < 2:
        return {"skipped": f"only {len(unique)} donor(s) in this dataset"}
    folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in unique]
    metrics = score_folds(
        counts[keep],
        genes,
        y[keep],
        cells,
        positions,
        folds,
        unique,
        GROUPED,
        case_insensitive,
        LR_PATH / f"cv_grouped_{dataset}.npz",
    )
    metrics["donors"] = len(unique)
    metrics["donor_sizes"] = {g: int((groups == g).sum()) for g in unique}
    metrics["cells_scored_per_donor_min"] = min(m["cells_scored"] for m in metrics["per_fold"])
    return metrics


if __name__ == "__main__":
    main()
