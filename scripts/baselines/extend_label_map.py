"""Merge the extra CellTypist label -> CL entries into baselines/celltypist_label_map.json.

Kept as a script rather than hand-edited JSON so the additions are reviewable, stay
attributable, and can be re-applied. It is idempotent: running it twice changes nothing.

Why these models at all: the model index CellTypist ships locally is stale --
``models.get_all_models()`` reported a single model and ``download_models()`` rejected
plausible names, so the first pass concluded "no pancreas model exists". After
``download_models(force_update=True)`` the repertoire is 61 models, including
``Adult_Human_PancreaticIslet`` (muraro is pancreas) and ``Adult_Human_MTG`` (siletti is
human brain). A tissue-matched model is the fair version of this baseline, so those arms
exist and the labels they produce had to be mapped.

Two conventions, both enforced in lib_common.resolve_celltypist_map:
  * basis='synonym'   the label is the standard alternative name of that CL term
                      (alpha cell = pancreatic A cell, 'Duct' = pancreatic ductal cell).
  * basis='coarsened' the label denotes something narrower than any register term; it is
                      mapped to the register term it is a subtype of, and step5 counts a
                      coarsened prediction as hierarchy-consistent only, never as an
                      exact hit.
Labels deliberately left unmapped are listed in "_model_notes" so the gap is visible in
the data rather than buried in prose.

Run: uv run --no-sync python scripts/baselines/extend_label_map.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
MAP = ROOT / "baselines" / "celltypist_label_map.json"
CL_TERMS = ROOT / "artifacts" / "v2" / "cl_terms.json"


def entry(cl_id: str, basis: str, reason: str) -> dict:
    return {"cl_id": cl_id, "basis": basis, "reason": reason}


#: Adult_Human_PancreaticIslet -- the tool's own islet model; muraro's registered cells are
#: almost entirely alpha/beta/delta/PP/ductal/mesenchymal/endothelial cells.
PANCREATIC_ISLET = {
    "alpha_mature": entry("CL:0000171", "synonym", "alpha cell = pancreatic A cell, a register term"),
    "alpha_immature": entry("CL:0000171", "coarsened", "developmental stage of the A cell"),
    "alpha_stress II": entry("CL:0000171", "coarsened", "stress state of the A cell"),
    "alpha_MHC II": entry("CL:0000171", "coarsened", "MHC-II-high state of the A cell"),
    "beta_mature": entry("CL:0000169", "synonym", "beta cell = type B pancreatic cell, a register term"),
    "beta_immature": entry("CL:0000169", "coarsened", "developmental stage of the B cell"),
    "beta_stress I": entry("CL:0000169", "coarsened", "stress state of the B cell"),
    "beta_stress II": entry("CL:0000169", "coarsened", "stress state of the B cell"),
    "beta_MHC/autoantigen": entry("CL:0000169", "coarsened", "antigen-presenting state of the B cell"),
    "beta_mtDNA deficient": entry("CL:0000169", "coarsened", "mitochondrial state of the B cell"),
    "delta": entry("CL:0000173", "synonym", "delta cell = pancreatic D cell, a register term"),
    "PP": entry("CL:0002275", "synonym", "PP = pancreatic polypeptide cell, a register term"),
}

#: Fetal_Human_Pancreas -- developmental pancreas model, kept as a secondary arm for muraro.
FETAL_PANCREAS = {
    "Duct": entry("CL:0002079", "synonym", "pancreatic ductal cell, a register term"),
    "Endothelium": entry("CL:0000115", "synonym", "Endothelium = endothelial cell"),
    "Mesenchyme": entry("CL:0008019", "synonym", "Mesenchyme = mesenchymal cell"),
    "Immune cell": entry("CL:0000738", "coarsened", "register has leukocyte and its subtypes, not a generic immune cell"),
    "PP": entry("CL:0002275", "synonym", "PP = pancreatic polypeptide cell"),
    "α": entry("CL:0000171", "synonym", "alpha cell = pancreatic A cell"),
    "β": entry("CL:0000169", "synonym", "beta cell = type B pancreatic cell"),
    "δ": entry("CL:0000173", "synonym", "delta cell = pancreatic D cell"),
    "EP1": entry("CL:0000066", "coarsened", "pancreatic epithelial progenitor; register term is the parent epithelial cell"),
    "EP2": entry("CL:0000066", "coarsened", "pancreatic epithelial progenitor"),
    "EP3": entry("CL:0000066", "coarsened", "pancreatic epithelial progenitor"),
    "EP4": entry("CL:0000066", "coarsened", "pancreatic epithelial progenitor"),
}

#: Adult_Human_MTG -- labels are per-cluster, e.g. 'Pvalb_12', 'L2/3 IT_5'. The base label
#: (cluster suffix stripped) is mapped here; lib_common applies the rule.
MTG_BASES = {
    "Endo": entry("CL:0000115", "synonym", "Endo = endothelial cell"),
    "Micro-PVM": entry("CL:0000113", "coarsened", "microglia + perivascular macrophages; both are mononuclear phagocytes"),
    "VLMC": entry("CL:0000057", "coarsened", "vascular leptomeningeal cells are fibroblast-lineage; the register has no VLMC term"),
    # Every cortical neuron subtype collapses onto the register's single neuronal catch-all.
    **{
        base: entry("CL:0000540", "coarsened", f"'{base}' is a cortical neuron subtype; the register's only neuronal term here is neuron")
        for base in (
            "Chandelier", "L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP", "L6 CT", "L6 IT",
            "L6 IT Car3", "L6b", "Lamp5", "Lamp5_Lhx6", "Pax6", "Pvalb", "Sncg", "Sst",
            "Sst Chodl", "Vip",
        )
    },
}

ADDITIONS = {
    "Adult_Human_PancreaticIslet.pkl": PANCREATIC_ISLET,
    "Fetal_Human_Pancreas.pkl": FETAL_PANCREAS,
    "Adult_Human_MTG.pkl": MTG_BASES,
}

NOTES = {
    "Adult_Human_MTG.pkl": [
        "'Astro_*' (astrocyte), 'Oligo_*' (oligodendrocyte) and 'OPC_*' (oligodendrocyte precursor) are",
        "left unmapped: the corpus register contains no astrocyte, oligodendrocyte or OPC term, so a",
        "prediction of one cannot be expressed in the corpus's label space. In siletti these cells are",
        "largely in the out_of_register family for the same reason.",
    ],
    "Fetal_Human_Pancreas.pkl": [
        "'Acinar cell' is left unmapped on purpose: the register's only acinar term is CL:0002623",
        "'acinar cell of salivary gland', which is a different cell, so mapping pancreatic acinar to it",
        "would be a false match. 'Tip', 'Trunk', 'ε' and 'α/PP-Pro' are fetal progenitor / ghrelin-cell",
        "states with no register term.",
    ],
    "Mouse_Whole_Brain.pkl": [
        "Not used as an arm. Its 334 labels are cluster codes of the form '263 CS-RPO Meis2 Gaba' and the",
        "register contains essentially no mouse brain cell types, so predictions would be almost entirely",
        "unmapped and the arm would carry no information. Recorded here rather than silently dropped.",
    ],
    "Cells_Intestinal_Tract.pkl": [
        "Not used as an arm. 130 of its 134 labels are fine-grained states ('Activated CD4 T',",
        "'Fetal arterial EC', 'Stromal 1 (ADAMDEC1+)'); mapping them would be a large hand-curation",
        "exercise for a dataset that already has the immune model and the gut model applied.",
    ],
}


def main() -> None:
    known = {t["cl_id"] for t in json.loads(CL_TERMS.read_text(encoding="utf-8"))["terms"]}
    payload = json.loads(MAP.read_text(encoding="utf-8"))
    added = 0
    for model, table in ADDITIONS.items():
        for label, item in table.items():
            if item["cl_id"] not in known:
                raise SystemExit(f"{model}: {label!r} -> {item['cl_id']} is not a CL term in the snapshot")
        target = payload["models"].setdefault(model, {})
        for label, item in table.items():
            if label not in target:
                target[label] = item
                added += 1
            elif target[label] != item:
                raise SystemExit(f"{model}: {label!r} already present with a different mapping -- refusing to overwrite")
    payload["_model_notes"] = NOTES
    MAP.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"added {added} label entries; models now in the map: {sorted(payload['models'])}")


if __name__ == "__main__":
    main()
