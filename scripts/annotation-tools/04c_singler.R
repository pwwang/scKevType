#!/usr/bin/env Rscript
# Step 4c: SingleR on the census cells.
#
# THE CONVENTIONAL RUN, stated so it can be checked:
#   * reference  -- celldex::HumanPrimaryCellAtlasData() (HPCA, 713 reference cells, 36 main
#                   labels) for the three human datasets; celldex::MouseRNAseqData()
#                   (358 reference cells, 18 main labels) for tabula_muris_senis. These are
#                   the two references SingleR's own documentation uses.
#   * labels     -- `label.main` is SingleR's documented default and is scored as the primary
#                   arm; `label.fine` is scored as a second arm, because a coarse reference
#                   vocabulary cannot express a fine register and reporting only the coarse
#                   one would understate the tool. Both are written out; the scoring step
#                   harmonises each to the corpus's register and takes the better of the two
#                   as the tool's best reading.
#   * input      -- raw counts, normalised here exactly as CellTypist's arm is: CP10K + log1p.
#                   SingleR is run on the intersection of genes between test and reference.
#   * parameters -- SingleR's defaults (`de.method = "classic"`, `fine.tune = TRUE`).
#
# SingleR does not abstain: `pruned.labels` marks low-confidence calls with NA, and those are
# reported as a separate column rather than folded into the accuracy, because a standard
# annotator that declines to call a cell is still answering the question by not answering it.
#
# Usage:  Rscript scripts/annotation-tools/04c_singler.R

suppressPackageStartupMessages({
  library(celldex)
  library(SingleR)
  library(Matrix)
  library(SummarizedExperiment)
})

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
ROOT <- "/mnt/f/E/hermes-workspace/scKevType"
IN <- file.path(ROOT, "reports", "annotation-tools", "r-input")
OUT <- file.path(ROOT, "reports", "annotation-tools", "singler")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

refs <- list(human = HumanPrimaryCellAtlasData(), mouse = MouseRNAseqData())
which_ref <- c(muraro = "human", siletti = "human", tabula_sapiens = "human",
               tabula_muris_senis = "mouse")

meta <- list()
for (dataset in names(which_ref)) {
  d <- file.path(IN, dataset)
  m <- readMM(file.path(d, "matrix.mtx"))
  genes <- readLines(file.path(d, "genes.txt"))
  cells <- readLines(file.path(d, "cells.txt"))
  rownames(m) <- genes
  colnames(m) <- cells
  stopifnot(nrow(m) == length(genes), ncol(m) == length(cells))

  # CP10K + log1p, the same normalisation the CellTypist arm gets
  m <- as(m, "CsparseMatrix")
  lib <- Matrix::colSums(m)
  m <- t(t(m) / lib) * 1e4
  m <- log1p(m)

  ref <- refs[[which_ref[[dataset]]]]
  common <- intersect(rownames(m), rownames(ref))
  cat(sprintf("[%s] ref=%s common genes=%d of %d test / %d ref\n",
              dataset, which_ref[[dataset]], length(common), nrow(m), nrow(ref)))
  if (length(common) < 100) stop("too few common genes")

  sub <- m[common, , drop = FALSE]
  rsub <- ref[common, ]

  res <- list()
  for (labs in c("main", "fine")) {
    labels <- as.character(SummarizedExperiment::colData(rsub)[[paste0("label.", labs)]])
    pred <- SingleR(test = sub, ref = rsub, labels = labels)
    res[[labs]] <- data.frame(
      cell = colnames(sub),
      label = as.character(pred$labels),
      pruned = as.character(pred$pruned.labels),
      stringsAsFactors = FALSE
    )
    cat(sprintf("    label.%s: %d distinct predicted labels, %d pruned (NA)\n",
                labs, length(unique(res[[labs]]$label)),
                sum(is.na(res[[labs]]$pruned))))
  }

  out <- data.frame(
    cell = res$main$cell,
    label_main = res$main$label,
    pruned_main = res$main$pruned,
    label_fine = res$fine$label,
    pruned_fine = res$fine$pruned,
    stringsAsFactors = FALSE
  )
  write.csv(out, file.path(OUT, paste0(dataset, ".csv")), row.names = FALSE)
  meta[[dataset]] <- list(
    dataset = dataset,
    reference = which_ref[[dataset]],
    reference_name = ifelse(which_ref[[dataset]] == "human",
                            "celldex::HumanPrimaryCellAtlasData()",
                            "celldex::MouseRNAseqData()"),
    reference_cells = ncol(ref), reference_genes = nrow(ref),
    test_genes = nrow(m), common_genes = length(common),
    cells = ncol(m),
    label_main_distinct = length(unique(res$main$label)),
    label_fine_distinct = length(unique(res$fine$label)),
    label_main_pruned_na = sum(is.na(res$main$pruned)),
    label_fine_pruned_na = sum(is.na(res$fine$pruned)),
    top_main = as.list(head(sort(as.integer(table(res$main$label)), decreasing = TRUE), 8))
  )
  cat(sprintf("  wrote %s/%s.csv\n", OUT, dataset))
}

meta$versions <- list(
  R = as.character(getRversion()),
  SingleR = as.character(packageVersion("SingleR")),
  celldex = as.character(packageVersion("celldex")),
  Matrix = as.character(packageVersion("Matrix")),
  SummarizedExperiment = as.character(packageVersion("SummarizedExperiment"))
)
meta$parameters <- list(de.method = "classic", fine.tune = TRUE,
                        normalisation = "CP10K then log1p", labels = "label.main primary, label.fine second arm")
writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
           file.path(ROOT, "reports", "annotation-tools", "singler.summary.json"))
cat("wrote singler.summary.json\n")
