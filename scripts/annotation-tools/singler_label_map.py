"""Label harmonisation: a tool's own vocabulary -> the corpus's register (CL ids).

THE PROBLEM THIS SOLVES
-----------------------
CellTypist and SingleR both emit labels in their own vocabulary -- ``"Alpha cell"``,
``"T_cells"``, ``"Fibroblasts:foreskin"`` -- while a `panel_choice` question's answer is one of
the corpus's register names (``"pancreatic A cell"``, ``"T cell"``, ``"fibroblast"``). Scoring
by string equality would measure vocabulary mismatch, not annotation. Every number in this
report is therefore computed on **CL ids**:

    tool label  --(this map)-->  CL id  ==  the target option's CL id ?

CellTypist's map is the project's own curated table (``baselines/celltypist_label_map.json``,
resolved through ``lib_common.resolve_celltypist_map``) so the CellTypist numbers here rest on
exactly the mapping the project's published baseline tables rest on. SingleR's is built here,
because no such table existed, and is documented below.

SINGLER'S MAP, AND HOW IT WAS BUILT
-----------------------------------
Two references are used (see ``04c_singler.R``): ``celldex::HumanPrimaryCellAtlasData()``
(HPCA) for the human datasets and ``celldex::MouseRNAseqData()`` for the mouse one. Their
label vocabularies are small and coarse -- 36 and 18 ``label.main`` values, 157 and 28
``label.fine`` values -- so the map is an explicit table, not a heuristic:

* **main labels** -- an explicit ``label.main -> CL id`` table below. Every entry carries a
  ``basis``: ``name`` (the label names that CL term, modulo case and plural), ``synonym`` (a
  standard abbreviation, e.g. ``NK_cell`` = natural killer cell, ``MSC`` = mesenchymal stem
  cell), ``coarsened`` (the label denotes a state or subtype the register has no term for and
  is mapped to the nearest term it *is* a subtype of -- a coarsened prediction is counted as
  wrong unless that coarser term is itself the answer), or ``unmapped`` (the label denotes
  something that is not a cell type in this register at all, e.g. ``BM`` = bone marrow,
  ``Gametocytes``, ``iPS_cells``).
* **fine labels** -- ``label.fine`` values are mostly ``main:free_annotation``
  (``"Macrophage:monocyte-derived:M-CSF"``), so the default rule is: strip everything from the
  first ``:`` and look the base up in the main table, recording the basis as
  ``coarsened-from-main``. Where the suffix carries real identity rather than an experimental
  condition, an explicit override recovers it (``"T_cell:CD4+"`` -> CL:0000624
  ``CD4-positive, alpha-beta T cell``, ``"B_cell:Memory"`` -> CL:0000787 ``memory B cell``,
  ``"Monocyte:CD16+"`` -> CL:0000875 ``non-classical monocyte``, and so on). The override list
  is exhaustive over the fine labels that actually occur in these four runs -- the script
  prints every label it could not place, and the count of those is reported rather than
  absorbed.

A label that maps to a CL id **outside** the corpus's register can never equal an option and
is scored as wrong; that is a property of the tool's vocabulary, not of the scoring, and the
report quantifies it (``off_register`` in the per-arm tables).
"""

from __future__ import annotations

#: HPCA (celldex::HumanPrimaryCellAtlasData) label.main -> (cl_id, basis)
HPCA_MAIN: dict[str, tuple[str | None, str]] = {
    "Astrocyte": ("CL:0000127", "name"),
    "B_cell": ("CL:0000236", "synonym"),
    "BM": (None, "unmapped"),
    "BM & Prog.": (None, "unmapped"),
    "Chondrocytes": ("CL:0000138", "name"),
    "CMP": ("CL:0000049", "synonym"),
    "DC": ("CL:0000451", "synonym"),
    "Embryonic_stem_cells": ("CL:0002322", "name"),
    "Endothelial_cells": ("CL:0000115", "name"),
    "Epithelial_cells": ("CL:0000066", "name"),
    "Erythroblast": ("CL:0000765", "name"),
    "Fibroblasts": ("CL:0000057", "name"),
    "Gametocytes": (None, "unmapped"),
    "GMP": ("CL:0000557", "synonym"),
    "HSC_-G-CSF": ("CL:0000037", "coarsened"),
    "HSC_CD34+": ("CL:0000037", "coarsened"),
    "Hepatocytes": ("CL:0000182", "name"),
    "iPS_cells": (None, "unmapped"),
    "Keratinocytes": ("CL:0000312", "name"),
    "Macrophage": ("CL:0000235", "name"),
    "MEP": ("CL:0000050", "synonym"),
    "MSC": ("CL:0000134", "synonym"),
    "Monocyte": ("CL:0000576", "name"),
    "Myelocyte": ("CL:0000580", "coarsened"),
    "NK_cell": ("CL:0000623", "synonym"),
    "Neuroepithelial_cell": ("CL:0002259", "coarsened"),
    "Neurons": ("CL:0000540", "name"),
    "Neutrophil": ("CL:0000775", "name"),
    "Neutrophils": ("CL:0000775", "name"),
    "Osteoblasts": ("CL:0000062", "name"),
    "Platelets": ("CL:0000233", "name"),
    "Pre-B_cell_CD34-": ("CL:0000816", "coarsened"),
    "Pro-B_cell_CD34+": ("CL:0000826", "coarsened"),
    "Pro-Myelocyte": ("CL:0000836", "coarsened"),
    "Smooth_muscle_cells": ("CL:0000192", "name"),
    "T_cells": ("CL:0000084", "name"),
    "Tissue_stem_cells": ("CL:0000048", "coarsened"),
}

#: celldex::MouseRNAseqData label.main -> (cl_id, basis)
MOUSE_MAIN: dict[str, tuple[str | None, str]] = {
    "Adipocytes": ("CL:0000136", "synonym"),
    "Astrocytes": ("CL:0000127", "name"),
    "B cells": ("CL:0000236", "name"),
    "Cardiomyocytes": ("CL:0000746", "synonym"),
    "Dendritic cells": ("CL:0000451", "name"),
    "Endothelial cells": ("CL:0000115", "name"),
    "Epithelial cells": ("CL:0000066", "name"),
    "Erythrocytes": ("CL:0000232", "name"),
    "Fibroblasts": ("CL:0000057", "name"),
    "Granulocytes": ("CL:0000094", "name"),
    "Hepatocytes": ("CL:0000182", "name"),
    "Macrophages": ("CL:0000235", "name"),
    "Microglia": ("CL:0000129", "synonym"),
    "Monocytes": ("CL:0000576", "name"),
    "Neurons": ("CL:0000540", "name"),
    "NK cells": ("CL:0000623", "synonym"),
    "Oligodendrocytes": ("CL:0000128", "name"),
    "T cells": ("CL:0000084", "name"),
}

#: label.fine values whose suffix carries identity rather than an experimental condition.
#: Everything not listed here falls back to the main table via the part before the first ":".
HPCA_FINE_OVERRIDE: dict[str, tuple[str | None, str]] = {
    "B_cell:Memory": ("CL:0000787", "name"),
    "B_cell:Naive": ("CL:0000788", "name"),
    "B_cell:Plasma_cell": ("CL:0000786", "name"),
    "B_cell:immature": ("CL:0000816", "name"),
    "B_cell:CXCR4-_centrocyte": ("CL:0000844", "coarsened"),
    "B_cell:CXCR4+_centroblast": ("CL:0000843", "coarsened"),
    "B_cell:Germinal_center": ("CL:0000844", "coarsened"),
    "Chondrocytes:MSC-derived": ("CL:0000138", "coarsened"),
    "DC:monocyte-derived": ("CL:0000990", "coarsened"),
    "Endothelial_cells:lymphatic": ("CL:0002138", "name"),
    "Endothelial_cells:blood_vessel": ("CL:0002543", "coarsened"),
    "Epithelial_cells:bladder": ("CL:0000066", "coarsened"),
    "Epithelial_cells:bronchial": ("CL:0000066", "coarsened"),
    "Fibroblasts:breast": ("CL:0000057", "coarsened"),
    "Fibroblasts:foreskin": ("CL:0000057", "coarsened"),
    "Macrophage:Alveolar": ("CL:0000583", "name"),
    "Monocyte:CD14+": ("CL:0001054", "name"),
    "Monocyte:CD16+": ("CL:0000875", "synonym"),
    "Monocyte:CD16-": ("CL:0000860", "synonym"),
    "Neuroepithelial_cell:ESC-derived": ("CL:0002259", "coarsened"),
    "Neurons:Schwann_cell": ("CL:0002573", "coarsened"),
    "Smooth_muscle_cells:bronchial": ("CL:0000192", "coarsened"),
    "Smooth_muscle_cells:umbilical_vein": ("CL:0002588", "coarsened"),
    "Smooth_muscle_cells:vascular": ("CL:0000359", "name"),
    "T_cell:CD4+": ("CL:0000624", "synonym"),
    "T_cell:CD8+": ("CL:0000625", "synonym"),
    "T_cell:CD4+_Naive": ("CL:0000895", "name"),
    "T_cell:CD8+_naive": ("CL:0000900", "name"),
    "T_cell:CD4+_central_memory": ("CL:0000897", "name"),
    "T_cell:CD4+_effector_memory": ("CL:0000905", "name"),
    "T_cell:CD8+_Central_memory": ("CL:0000907", "name"),
    "T_cell:CD8+_effector_memory": ("CL:0000913", "name"),
    "T_cell:CD8+_effector_memory_RA": ("CL:0000913", "coarsened"),
    "T_cell:gamma-delta": ("CL:0000798", "name"),
    "T_cell:Treg:Naive": ("CL:0000815", "coarsened"),
    "Tissue_stem_cells:BM_MSC": ("CL:0000134", "coarsened"),
    "Tissue_stem_cells:adipose-derived_MSC_AM3": ("CL:0000134", "coarsened"),
    "Tissue_stem_cells:iliac_MSC": ("CL:0000134", "coarsened"),
    "Tissue_stem_cells:lipoma-derived_MSC": ("CL:0000134", "coarsened"),
    "Tissue_stem_cells:BM_MSC:BMP2": ("CL:0000134", "coarsened"),
    "Tissue_stem_cells:BM_MSC:osteogenic": ("CL:0000134", "coarsened"),
    "Tissue_stem_cells:BM_MSC:TGFb3": ("CL:0000134", "coarsened"),
}

MOUSE_FINE_OVERRIDE: dict[str, tuple[str | None, str]] = {
    # the mouse fine vocabulary carries state words ("activated", "senescent") and a few
    # stem/progenitor abbreviations; neither is a cell *type* in the register
    "aNSCs": ("CL:0000047", "coarsened"),
    "qNSCs": ("CL:0000047", "coarsened"),
    "NPCs": ("CL:0000047", "coarsened"),
    "OPCs": ("CL:0002453", "name"),
    "Ependymal": ("CL:0000065", "coarsened"),
}

#: state words that a mouse fine label may carry as a trailing token
MOUSE_STATE_SUFFIXES = {"activated", "senescent"}


def singler_label_to_cl(label: str, reference: str, level: str) -> tuple[str | None, str]:
    """(cl_id, basis) for one SingleR label.

    ``reference`` is ``"human"`` or ``"mouse"``; ``level`` is ``"main"`` or ``"fine"``.
    Returns ``(None, "unmapped")`` for a label with no CL term in this register's snapshot.
    """
    table = HPCA_MAIN if reference == "human" else MOUSE_MAIN
    override = HPCA_FINE_OVERRIDE if reference == "human" else MOUSE_FINE_OVERRIDE
    if label is None:
        return None, "missing"
    label = str(label)
    if level == "main":
        return table.get(label, (None, "unmapped"))
    if label in override:
        return override[label]
    # human fine labels are "main:free_annotation"; mouse ones may instead be
    # "main <state word>"
    base = label.split(":", 1)[0]
    if base in table:
        cl_id, basis = table[base]
        return cl_id, ("coarsened-from-main" if cl_id is not None else basis)
    parts = label.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].lower() in MOUSE_STATE_SUFFIXES and parts[0] in table:
        cl_id, basis = table[parts[0]]
        return cl_id, ("coarsened-from-main" if cl_id is not None else basis)
    return None, "unmapped"


def unknown_singler_labels(labels: set[str], reference: str, level: str) -> list[str]:
    """The labels this map cannot place, so a run can report them instead of absorbing them."""
    out = []
    for label in sorted(labels):
        cl_id, basis = singler_label_to_cl(label, reference, level)
        if cl_id is None:
            out.append(label)
    return out
