"""Step 3 -- CellTypist, the standard logistic-regression annotator, on the corpus cells.

What runs, and why:

* input  -- the count matrix from step 2, ``normalize_total(1e4)`` + ``log1p`` (what
  CellTypist's own tutorial feeds its models).
* model  -- Immune_All_Low for every human dataset. The CellTypist repertoire
  reachable from this install is {Immune_All_Low, Immune_All_High, Human_Lung_Atlas,
  Adult_Mouse_Gut} (out/celltypist_models.json); there is no pancreas and no brain
  model, so the human immune model is the only defensible choice for muraro and
  siletti and is also the model CellTypist's own documentation calls the default.
  Human_Lung_Atlas is run as a second arm on tabula_sapiens (the only tissue-matched
  human model that exists here), and Adult_Mouse_Gut as the species-matched arm on
  tabula_muris_senis. Arms that do not exist are recorded as null, not guessed.
* mouse  -- the human model is also run on tabula_muris_senis with uppercased gene
  symbols (``--upper-symbols``): the human and mouse symbols for one ortholog differ
  mostly in case (Cd3e/CD3E) and the corpus's own definitions carry both spellings
  (e.g. erythrocyte: HBB, GYPA, HBA1, TFRC, Hba-a1, Alas2), so a case-insensitive
  match follows the corpus's own cross-species stance. Recorded as its own arm.

Output per (dataset, arm): predictions, mapped CL ids with their basis, and the full
label-probability matrix, so the per-family step can ask a constrained question
without re-running the model.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_common import (  # noqa: E402
    ARM_TAGS,
    EXPR,
    QCOUT,
    ROOT,
    TAG_LEGEND,
    load_aligned_cells,
    write_json,
)

#: dataset -> [(arm name, celltypist model, symbol transform, why)]
#:
#: Two kinds of arm per human dataset: the tissue-matched model (the fair version of this
#: baseline) and Immune_All_Low (the model CellTypist's own quickstart puts on anything).
#: The matched models only exist because the model index was force-refreshed -- see
#: scripts/baselines/extend_label_map.py for why the first pass missed them.
ARMS = {
    "muraro": [
        ("pancreatic_islet", "Adult_Human_PancreaticIslet.pkl", "none",
         "tissue-matched: the corpus's muraro labels are alpha/beta/delta/PP/ductal/mesenchymal"),
        ("immune_low", "Immune_All_Low.pkl", "none",
         "CellTypist's default model; no pancreas tissue in its reference"),
        ("fetal_pancreas", "Fetal_Human_Pancreas.pkl", "none",
         "developmental pancreas, secondary matched arm"),
    ],
    "siletti": [
        ("mtg", "Adult_Human_MTG.pkl", "none",
         "tissue-matched: adult human middle temporal gyrus; labels are per-cluster"),
        ("immune_low", "Immune_All_Low.pkl", "none",
         "CellTypist's default model; the corpus register's brain coverage is neuron/medium spiny neuron"),
    ],
    "tabula_sapiens": [
        ("immune_low", "Immune_All_Low.pkl", "none", "default model, all tissues"),
        ("lung_atlas", "Human_Lung_Atlas.pkl", "none", "tissue-matched for the lung cells in this multi-tissue dataset"),
    ],
    "tabula_muris_senis": [
        ("mouse_gut", "Adult_Mouse_Gut.pkl", "none", "species-matched (mouse), the only mouse tissue model applicable here"),
        ("immune_low_upper", "Immune_All_Low.pkl", "upper", "human model, mouse symbols case-matched"),
    ],
}


def load(dataset: str, split: str = "test"):
    """The persisted matrix, genes and cell ids -- with the row order asserted, not assumed.

    The ids come from ``load_aligned_cells``, which refuses an array whose row order is not
    certified (see lib_common): these baselines label matrix row i with the label of
    ``cells[i]``, so an id array in request order rather than row order silently mislabels
    every row. That is exactly the failure this loader now makes impossible to reach.
    """
    matrix = sp.load_npz(EXPR / f"{dataset}_{split}.npz").tocsr()
    genes = [str(g) for g in np.load(EXPR / f"{dataset}_{split}_genes.npy", allow_pickle=True)]
    cells = load_aligned_cells(dataset, split)
    if matrix.shape[0] != len(cells):
        raise SystemExit(
            f"{dataset}/{split}: the matrix has {matrix.shape[0]} rows but the certified id array "
            f"holds {len(cells)} ids -- they cannot describe the same cells"
        )
    if matrix.shape[1] != len(genes):
        raise SystemExit(
            f"{dataset}/{split}: the matrix has {matrix.shape[1]} columns but {len(genes)} gene names"
        )
    return matrix, genes, cells


def collapse_duplicate_genes(matrix, genes):
    """Sum columns that share a symbol -- ``var_names`` must be unique for CellTypist.

    Duplicate symbols are real in these files (the CELLxGENE var table can carry the
    same feature_name twice). Summing is the standard aggregation and is recorded in
    the arm's stats, never silent.

    Alignment is by gene NAME: the aggregation matrix is built from ``genes`` itself, and
    the shape of ``matrix`` is asserted against it first, so a matrix from another gene
    space can never be multiplied positionally by mistake.
    """
    matrix = matrix.tocsr()
    if matrix.shape[1] != len(genes):
        raise ValueError(
            f"collapse_duplicate_genes: matrix has {matrix.shape[1]} columns but {len(genes)} gene names "
            "-- refusing a positional multiply across two different gene spaces"
        )
    seen: dict[str, int] = {}
    keep: list[int] = []
    for index, gene in enumerate(genes):
        if gene in seen:
            continue
        seen[gene] = len(keep)
        keep.append(index)
    if len(keep) == len(genes):
        return matrix, list(genes), 0
    groups = np.array([seen[gene] for gene in genes], dtype=np.int64)
    # (n_genes x n_unique) aggregation applied on the right: (cells x genes) @ (genes x unique).
    aggregation = sp.csr_matrix(
        (np.ones(len(genes), dtype=np.float32), (np.arange(len(genes)), groups)),
        shape=(len(genes), len(keep)),
    )
    collapsed = (matrix.astype(np.float32) @ aggregation).tocsr()
    if collapsed.shape != (matrix.shape[0], len(keep)):
        raise ValueError(f"collapse_duplicate_genes: unexpected result shape {collapsed.shape}")
    return collapsed, [genes[i] for i in keep], len(genes) - len(keep)


def run_arm(dataset: str, arm: str, model: str, transform: str, why: str) -> dict:
    import anndata as ad
    import scanpy as sc
    from celltypist import models

    t0 = time.time()
    matrix, genes, cells = load(dataset)
    matrix, genes, n_collapsed = collapse_duplicate_genes(matrix, genes)
    if transform == "upper":
        genes = [g.upper() for g in genes]
        matrix, genes, extra = collapse_duplicate_genes(matrix, genes)
        n_collapsed += extra
    adata = ad.AnnData(X=matrix.astype(np.float32), obs={"cell_id": cells}, var={"gene": genes})
    adata.var_names = genes
    adata.obs_names = cells
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    instance = models.Model.load(str(Path(models.models_path) / model))
    model_features = set(instance.features)
    overlap = [g for g in genes if g in model_features]
    result = _annotate(adata, instance)
    predicted = np.asarray(result.predicted_labels.iloc[:, 0], dtype=object)
    mapping = _mapping(model)
    labels = list(result.probability_matrix.columns)
    probabilities = np.asarray(result.probability_matrix.to_numpy(), dtype=np.float32)

    mapped_cl = [mapping.get(p, {}).get("cl_id") for p in predicted]
    basis = [mapping.get(p, {}).get("basis") for p in predicted]
    out_stem = EXPR.parent / "celltypist" / f"{dataset}_{arm}"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_stem.with_suffix(".npz"),
        cells=np.array(cells, dtype=object),
        predicted=np.array(predicted, dtype=object),
        mapped_cl=np.array(mapped_cl, dtype=object),
        basis=np.array(basis, dtype=object),
        labels=np.array(labels, dtype=object),
        probabilities=probabilities,
    )
    digest = hashlib.sha256((Path(models.models_path) / model).read_bytes()).hexdigest()
    resolved = sum(1 for c in mapped_cl if c is not None)
    if len(predicted) == 0:
        raise SystemExit(f"{dataset}/{arm}: the model returned no prediction for any cell -- refusing to report")
    # A worked-example sanity number, printed so a silent no-op cannot look like a result:
    # how often the model's own top-1 (mapped to a CL id) equals the corpus's label for
    # this cell, restricted to cells whose label is a register term. The per-family, per-
    # dataset figures live in step5_score.py.
    exact_n = exact_hit = unmapped = 0
    for cell, truth in _corpus_labels(dataset).items():
        if cell not in set(cells):
            continue
        exact_n += 1
        value = dict(zip(cells, mapped_cl)).get(cell)
        if value is None:
            unmapped += 1
        elif value == truth:
            exact_hit += 1
    stats = {
        "dataset": dataset,
        "arm": arm,
        "model": model,
        "model_sha256": digest,
        "model_why": why,
        "symbol_transform": transform,
        "cells": len(cells),
        "genes_input": len(genes),
        "duplicate_symbols_collapsed": n_collapsed,
        "model_features": len(model_features),
        "model_features_present": len(overlap),
        "model_feature_coverage": round(len(overlap) / len(model_features), 4),
        "labels_in_model": len(labels),
        "labels_mapped_to_cl": len(mapping),
        "predictions": len(predicted),
        "predictions_mapped_to_cl": resolved,
        "predictions_mapped_rate": round(resolved / len(predicted), 4),
        "distinct_predicted_labels": len(set(predicted)),
        "top_predicted_labels": _top(predicted),
        "mean_max_probability": float(np.mean(probabilities.max(axis=1))) if probabilities.size else None,
        "sanity_registered_cells": exact_n,
        "sanity_exact_top1_vs_corpus_label": round(exact_hit / exact_n, 4) if exact_n else None,
        "sanity_unmapped_label_rate": round(unmapped / exact_n, 4) if exact_n else None,
        "majority_voting": None,
        "seconds": round(time.time() - t0, 1),
        "out": str(out_stem.with_suffix(".npz").relative_to(ROOT)),
    }
    print(
        f"[{dataset}/{arm}] model={model} transform={transform} cells={stats['cells']} "
        f"feature_coverage={stats['model_feature_coverage']} mapped={stats['predictions_mapped_rate']} "
        f"({stats['seconds']}s)",
        flush=True,
    )
    print(f"    top labels: {stats['top_predicted_labels'][:6]}", flush=True)
    return stats


def _annotate(adata, instance):
    import celltypist

    return celltypist.annotate(adata, model=instance, majority_voting=False)


def _mapping(model: str) -> dict:
    from lib_common import resolve_celltypist_map

    return resolve_celltypist_map(model)


_CORPUS_LABELS: dict[str, dict[str, str]] | None = None


def _corpus_labels(dataset: str) -> dict[str, str]:
    """cell_id -> the corpus's own CL id for that cell, for the registered cells of one dataset."""
    global _CORPUS_LABELS
    if _CORPUS_LABELS is None:
        payload = json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
        per_dataset: dict[str, dict[str, str]] = {}
        for record in payload.values():
            if record["registered"]:
                per_dataset.setdefault(record["dataset_id"], {})[record["cell_id"]] = record["cl_id"]
        _CORPUS_LABELS = per_dataset
    return _CORPUS_LABELS.get(dataset, {})


def _top(predicted, k: int = 8) -> list:
    from collections import Counter

    return [f"{name} x{n}" for name, n in Counter(predicted).most_common(k)]


def main() -> None:
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        print("Usage: step3_celltypist.py [--datasets=a,b]   (default: every dataset in ARMS)")
        print("Note: this script takes a --datasets flag, unlike step2_expr.py, which takes positional")
        print("split:dataset:stream arguments. The two conventions differ; the parsed values are printed")
        print("below on every run so a mismatch is visible immediately instead of scoring nothing.")
        return
    only = None
    for arg in argv:
        if arg.startswith("--datasets="):
            only = {name.strip() for name in arg.split("=", 1)[1].split(",") if name.strip()}
    if only:
        unknown = only - set(ARMS)
        if unknown:
            raise SystemExit(f"--datasets: {sorted(unknown)} not in ARMS {sorted(ARMS)} -- nothing would be scored")
    print(f"parsed args: argv={argv} datasets={sorted(only) if only else 'ALL'}")
    import json as _json

    cells_test = _json.loads((QCOUT / "cells_test.json").read_text(encoding="utf-8"))
    per_dataset_counts: dict[str, int] = {}
    for record in cells_test.values():
        per_dataset_counts[record["dataset_id"]] = per_dataset_counts.get(record["dataset_id"], 0) + 1
    selected = [d for d in ARMS if not only or d in only]
    if not selected:
        raise SystemExit(
            f"--datasets={sorted(only) if only else None} matched none of ARMS {sorted(ARMS)} -- "
            "refusing to write an empty summary that looks like a completed run"
        )
    stats = {}
    for dataset in selected:
        arms = ARMS[dataset]
        print(f"plan: {dataset} has {per_dataset_counts.get(dataset, 0)} corpus cells; arms: "
              f"{[a[0] for a in arms]}", flush=True)
        for arm, model, transform, why in arms:
            stats[f"{dataset}/{arm}"] = run_arm(dataset, arm, model, transform, why)
    for key, value in stats.items():
        value["tags"] = ARM_TAGS["celltypist"]
        value["tag_legend"] = TAG_LEGEND
    out = QCOUT / "step3_celltypist_summary.json"
    provenance = json.loads(out.read_text(encoding="utf-8")).get("_provenance", []) if out.exists() else []
    previous = {k: v for k, v in json.loads(out.read_text(encoding="utf-8")).items() if k != "_provenance"} if out.exists() else {}
    merged = {**previous, **stats}
    provenance.append(
        {
            "argv": argv,
            "datasets_filter": sorted(only) if only else "ALL",
            "datasets_scored": selected,
            "arms_written": sorted(stats),
            "models": sorted({model for d in selected for _, model, _, _ in ARMS[d]}),
            "cells_scored_by_dataset": {
                dataset: sum(1 for c in cells_test.values() if c["dataset_id"] == dataset) for dataset in selected
            },
        }
    )
    write_json(out, {"_provenance": provenance, **merged})
    print(f"wrote {out} with {len(merged)} arms (provenance entries: {len(provenance)})")
    for key, value in sorted(stats.items()):
        print(f"  {key}: cells={value['cells']} labels={value['distinct_predicted_labels']} "
              f"mapped={value['predictions_mapped_rate']} "
              f"sanity_exact_top1={value['sanity_exact_top1_vs_corpus_label']}")


if __name__ == "__main__":
    main()
