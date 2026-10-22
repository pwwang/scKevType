#!/usr/bin/env Rscript
# Bounded Azimuth attempt: load the human MTG reference and annotate the census siletti cells.
#
# WHY ONLY siletti: Azimuth ships a fixed menu of tissue-specific references. Of the four
# atlases in this corpus, only two tissues are covered at all -- pancreas (muraro) and brain
# (siletti) -- and the pancreas references are built from the same public islet studies as the
# muraro atlas, so a pancreas number would be in-domain by construction rather than a baseline.
# The brain reference is the Allen human middle temporal gyrus, which is the same region the
# corpus's siletti atlas covers; that overlap is recorded as a caveat beside the number.
#
# Everything is capped: this is one bounded attempt, not an installation fight.
set.seed(1)
suppressPackageStartupMessages({ library(Azimuth); library(Seurat); library(Matrix) })
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
ROOT <- "/mnt/f/E/hermes-workspace/scKevType"
IN <- file.path(ROOT, "reports", "annotation-tools", "r-input", "siletti")
OUT <- file.path(ROOT, "reports", "annotation-tools", "azimuth")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

cat("Azimuth", as.character(packageVersion("Azimuth")), "| Seurat", as.character(packageVersion("Seurat")), "\n")
cat("available references:\n")
print(tryCatch(Azimuth:::ReferenceVersion(), error = function(e) conditionMessage(e)))

cat("=== loading reference humanMTCortex ===\n")
t0 <- Sys.time()
ref <- tryCatch(Azimuth::LoadReference("humanMTCortex"),
                error = function(e) { cat("LOAD_REFERENCE_FAILED:", conditionMessage(e), "\n"); NULL })
if (is.null(ref)) { cat("AZIMUTH_SCORE=not_attempted\n"); quit(status = 0) }
cat("reference loaded in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")

m <- readMM(file.path(IN, "matrix.mtx"))
genes <- readLines(file.path(IN, "genes.txt"))
cells <- readLines(file.path(IN, "cells.txt"))
rownames(m) <- genes; colnames(m) <- cells
obj <- CreateSeuratObject(counts = as(m, "CsparseMatrix"), project = "siletti-census")
obj <- SCTransform(obj, verbose = FALSE)
cat("=== RunAzimuth ===\n")
t1 <- Sys.time()
obj <- tryCatch(Azimuth::RunAzimuth(obj, reference = "humanMTCortex", verbose = TRUE),
                error = function(e) { cat("RUN_AZIMUTH_FAILED:", conditionMessage(e), "\n"); NULL })
if (is.null(obj)) { cat("AZIMUTH_SCORE=not_attempted\n"); quit(status = 0) }
cat("RunAzimuth done in", round(as.numeric(difftime(Sys.time(), t1, units = "mins")), 1), "min\n")

labs <- obj$predicted.id
write.csv(data.frame(cell = colnames(obj), azimuth_label = labs,
                     mapped_label = obj$predicted.id),
          file.path(OUT, "siletti.csv"), row.names = FALSE)
cat("wrote", file.path(OUT, "siletti.csv"), "\n")
cat("distinct labels:", length(unique(labs)), "\n")
print(head(sort(table(labs), decreasing = TRUE), 12))
cat("AZIMUTH_SCORE=ran\n")
