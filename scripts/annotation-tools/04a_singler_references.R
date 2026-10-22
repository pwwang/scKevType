#!/usr/bin/env Rscript
# Step 4a: fetch the celldex references SingleR will use, and dump the label vocabularies.
#
# WHY THESE REFERENCES: SingleR's own documentation and tutorial use
# `celldex::HumanPrimaryCellAtlasData()` (HPCA) for human and `celldex::MouseRNAseqData()`
# for mouse. They are the field-standard pan-tissue references and neither is derived from
# the four atlases this corpus scores on (see the report's leakage section).
#
# The fetch goes through ExperimentHub on first use, so this is the slow, network-bound step;
# it caches under ~/.cache/R/ExperimentHub and is a no-op on re-run.

suppressPackageStartupMessages({
  library(celldex)
  library(SingleR)
  library(SummarizedExperiment)
})

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
OUT <- "/mnt/f/E/hermes-workspace/scKevType/reports/annotation-tools"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

refs <- list(human = HumanPrimaryCellAtlasData(), mouse = MouseRNAseqData())

vocab <- list()
for (nm in names(refs)) {
  ref <- refs[[nm]]
  vocab[[nm]] <- list(
    name = nm,
    n_genes = nrow(ref),
    n_cells = ncol(ref),
    label_main = sort(unique(as.character(ref$label.main))),
    label_fine = sort(unique(as.character(ref$label.fine)))
  )
  cat(sprintf("[%s] %d genes x %d cells; %d main labels, %d fine labels\n",
              nm, nrow(ref), ncol(ref),
              length(vocab[[nm]]$label_main), length(vocab[[nm]]$label_fine)))
  cat("  main:", paste(vocab[[nm]]$label_main, collapse = ", "), "\n")
}

# the package versions and the reference's own provenance, so the report can pin them
vocab$versions <- list(
  R = as.character(getRversion()),
  SingleR = as.character(packageVersion("SingleR")),
  celldex = as.character(packageVersion("celldex")),
  SummarizedExperiment = as.character(packageVersion("SummarizedExperiment")),
  ExperimentHub = tryCatch(as.character(packageVersion("ExperimentHub")), error = function(e) NA)
)
vocab$source <- "celldex::HumanPrimaryCellAtlasData() / celldex::MouseRNAseqData()"
writeLines(jsonlite::toJSON(vocab, auto_unbox = TRUE, pretty = TRUE),
           file.path(OUT, "singler.references.json"))
cat("wrote", file.path(OUT, "singler.references.json"), "\n")
