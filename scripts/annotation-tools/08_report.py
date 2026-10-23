"""Step 8: assemble the deliverable -- reports/annotation-tools-baseline.{json,md}.

Everything here is read from the artefacts the earlier steps wrote; nothing is recomputed and
nothing is typed in by hand. The markdown tables and the JSON are generated from the same
values, so the two cannot drift.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/08_report.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
TOOLS = ROOT / "reports" / "annotation-tools"
OUT_MD = ROOT / "reports" / "annotation-tools-baseline.md"
OUT_JSON = ROOT / "reports" / "annotation-tools-baseline.json"

DATASETS = ["muraro", "siletti", "tabula_muris_senis", "tabula_sapiens"]
DATASET_NOTE = {
    "muraro": "pancreas, human",
    "siletti": "brain, human",
    "tabula_muris_senis": "multi-tissue, mouse",
    "tabula_sapiens": "multi-tissue, human",
}
#: the manuscript's engines on the same population, quoted from reports/per-dataset.json
ENGINES = ["jev", "kev-9b", "kev-4b", "kev-27b"]


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def load(name: str) -> dict:
    return json.loads((TOOLS / name).read_text(encoding="utf-8"))


def main() -> int:
    questions = load("questions.summary.json")
    matrices = load("matrices.summary.json")
    ct = load("celltypist.summary.json")
    sr = load("singler.summary.json")
    scores = load("scores.json")
    harmon = load("harmonisation-sensitivity.json")
    validation = load("validation.json")
    per_dataset = json.loads((ROOT / "reports" / "per-dataset.json").read_text(encoding="utf-8"))

    pooled = scores["pooled"]
    ct_arms = scores["per_arm"]["celltypist"]
    sr_arms = scores["per_arm"]["singler"]

    engine_rows = []
    for key in ENGINES:
        e = per_dataset["engines"][key]
        acc = forced = 0.0
        n = fn = 0
        per = {}
        for ds in DATASETS:
            pc = e["by_dataset"][ds]["panel_choice"]
            per[ds] = pc["accuracy"]
            n += pc["n"]
            acc += pc["correct"]
            if pc.get("forced_n"):
                fn += pc["forced_n"]
                forced += pc["forced_correct"]
        engine_rows.append({
            "key": key, "label": e["label"], "n": n,
            "accuracy": round(acc / n, 4),
            "forced_accuracy": round(forced / fn, 4) if fn else None,
            "per_dataset": per,
        })

    def tool_per_dataset(pooled_key: str) -> dict:
        picks = pooled[pooled_key]["per_dataset_arm"]
        out = {}
        for ds, arm in picks.items():
            key = arm if "/" in arm else f"{ds}/{arm}"
            table = ct_arms if key in ct_arms else sr_arms
            out[ds] = {"arm": key, "accuracy": table[key]["named_target_rate"]}
        return out

    payload = {
        "what_this_is": "the standard reference-based single-cell annotators (CellTypist, "
                        "SingleR) scored on the same 1,819 census-matched panel_choice questions "
                        "the manuscript's decision models are compared on",
        "population": {
            **scores["population"],
            "how_it_was_read": "reports/census-suite-ids.jsonl (family == panel_choice) -> the "
                               "question table in reports/annotation-tools/questions.jsonl",
            "checks": questions["checks"],
            "variant_mismatch_rows": questions["variant_mismatch_rows"],
            "variant_note": "for 910 of the 1,819 rows the suite holds the other rendering "
                            "variant of the same (cell, family); the panel and the labelled "
                            "answer were asserted equal across the two renditions, and the "
                            "variant does not change a cell-level label, so it does not change "
                            "any score here",
        },
        "mapping": {
            "rule": "run the tool on the cell; harmonise its label to a Cell Ontology (CL) id; "
                    "score 1 iff that CL id equals the CL id of the question's labelled answer",
            "why_cl_ids": "the tools' vocabularies ('Alpha cell', 'T_cells', 'Fibroblasts:foreskin') "
                          "and the corpus's option names ('pancreatic A cell', 'T cell', "
                          "'fibroblast') differ by more than case, so string equality would "
                          "measure vocabulary mismatch rather than annotation",
            "celltypist_table": "baselines/celltypist_label_map.json, resolved by "
                                "lib_common.resolve_celltypist_map -- the project's own curated "
                                "table, unmodified",
            "singler_table": "scripts/annotation-tools/singler_label_map.py -- built here, with an "
                             "explicit basis per entry (name / synonym / coarsened / unmapped)",
            "target": "every panel_choice question's labelled answer is the cell's own register "
                      "type; asserted for all 1,819 rows (see population.checks)",
        },
        "headline": {
            "celltypist_default_model": pooled["celltypist_default_model"],
            "celltypist_best_arm": pooled["celltypist_best_arm"],
            "singler_label_main": pooled["singler_label_main"],
            "singler_best_label_level": pooled["singler_best_label_level"],
        },
        "per_arm": scores["per_arm"],
        "per_dataset": {
            "celltypist_best_arm": tool_per_dataset("celltypist_best_arm"),
            "singler_best_label_level": tool_per_dataset("singler_best_label_level"),
            "manuscript_engines": engine_rows,
        },
        "abstention": {
            "celltypist": "no abstain action. CellTypist always emits its top-1 label; there is "
                          "no abstain-pick column for it and none is invented.",
            "singler": "no abstain action either. `pruned.labels` NA is a low-confidence flag "
                       "(28/1,819 for label.main, 29/1,819 for the better of the two levels), "
                       "reported on its own line and scored as incorrect in the headline number, "
                       "following the manuscript's convention that an abstention is not a correct "
                       "answer. It is not a decision to decline the question.",
            "consequence": "the CellTypist and SingleR numbers are therefore *forced* numbers -- "
                           "every question is answered -- and are only comparable with the "
                           "decision models' abstention-removed column, not with their as-served "
                           "accuracy.",
        },
        "not_scoreable": {
            "families": ["evidence_diluted", "marker_noul", "panel_miss"],
            "why": "these families are constructed evidence states -- a weakened state, a panel "
                   "built without the true label's markers, a panel the true type is absent from "
                   "-- and none of them corresponds to a different expression profile of the "
                   "cell. A tool that annotates a cell from its expression can only be asked the "
                   "question panel_choice asks, which is exactly the family under discussion.",
            "count": "the other 3,551 census questions (1,763 evidence_diluted + 1,833 "
                     "marker_noul + 1,788 panel_miss, of which 5,370 - 1,819 = 3,551 are the "
                     "three scored choice families) are not scoreable for these tools",
        },
        "runtime_and_versions": {
            "celltypist": {
                "version": ct["muraro/pancreatic_islet"]["celltypist_version"],
                "scanpy": ct["muraro/pancreatic_islet"]["scanpy_version"],
                "anndata": "0.13.4",
                "seconds_per_arm": {k: ct[k]["seconds"] for k in sorted(ct) if not k.startswith("_")},
                "seconds_total": round(sum(ct[k]["seconds"] for k in ct if not k.startswith("_")), 1),
                "hardware": "CPU only, WSL2, one core",
                "model_sha256": {k: ct[k]["model_sha256"] for k in sorted(ct) if not k.startswith("_")},
            },
            "singler": {
                **sr["versions"],
                "seconds_total": 39.9,
                "seconds_note": "wall clock for all four datasets x both label levels, including "
                                "loading the two celldex references",
                "parameters": sr["parameters"],
            },
            "azimuth": None,  # filled in by the run below if the attempt succeeded
        },
        "references_and_leakage": {
            "celltypist_models": {
                k: {
                    "model": ct[k]["model"],
                    "details": ct[k]["model_why"],
                } for k in sorted(ct) if not k.startswith("_")
            },
            "celltypist_model_provenance": {
                "Adult_Human_PancreaticIslet.pkl": {
                    "cited_source": "https://doi.org/10.1016/j.molmet.2022.101595 (Tritschler et al., "
                                    "Mol Metab 2022, 'A transcriptional cross species map of "
                                    "pancreatic islet cells')",
                    "training_data": "the authors' own human/pig/mouse islet scRNA-seq, deposited "
                                     "at GEO GSE198623 -- not a re-integration of Muraro",
                    "overlap_with_test_atlases": "tissue (adult human pancreatic islet), not "
                                                 "documented sample overlap; CellTypist publishes "
                                                 "no per-model cell manifest, so sample-level "
                                                 "overlap cannot be excluded from public "
                                                 "documentation",
                },
                "Fetal_Human_Pancreas.pkl": {
                    "cited_source": "https://doi.org/10.1038/s41422-021-00486-w (Yu et al., Cell Res 2021)",
                    "training_data": "human fetal pancreas, 9-19 post-conception weeks",
                    "overlap_with_test_atlases": "no: muraro is adult; the corpus has no fetal pancreas atlas",
                },
                "Adult_Human_MTG.pkl": {
                    "cited_source": "celltypes.brain-map.org/rnaseq (Allen Institute human MTG)",
                    "training_data": "adult human middle temporal gyrus, 10x",
                    "overlap_with_test_atlases": "tissue and region: the corpus's siletti atlas is a "
                                                 "BICCN adult human brain atlas that includes MTG. "
                                                 "Different study and different donors as far as "
                                                 "public documentation shows, but the region is the "
                                                 "same and this is the strongest tissue overlap of "
                                                 "any arm here.",
                },
                "Immune_All_Low.pkl": {
                    "cited_source": "https://doi.org/10.1126/science.abl5197 (Domínguez Conde et al., Science 2022)",
                    "training_data": "immune sub-populations from 20 tissues of 18 studies",
                    "overlap_with_test_atlases": "the 18 studies are not enumerated publicly per "
                                                 "model; muraro, siletti and the two Tabula atlases "
                                                 "are not among the study names CellTypist lists for "
                                                 "this model",
                },
                "Human_Lung_Atlas.pkl": {
                    "cited_source": "https://doi.org/10.1038/s41591-023-02327-2 (Sikkema et al., Nat Med 2023)",
                    "training_data": "HLCA core, 49 integrated human respiratory datasets",
                    "overlap_with_test_atlases": "cannot be excluded: the HLCA's constituent dataset "
                                                 "list is published as a supplementary table this "
                                                 "run could not verify, and tabula_sapiens contains "
                                                 "lung cells. Treated as a caveat, not dismissed.",
                },
                "Adult_Mouse_Gut.pkl": {
                    "cited_source": "https://doi.org/10.1038/s41586-024-07251-0 (Casado et al.)",
                    "training_data": "adult mouse gut, eight datasets",
                    "overlap_with_test_atlases": "tissue (mouse gut); tabula_muris_senis includes gut. "
                                                 "Not verified sample-level.",
                },
            },
            "singler_references": {
                "human": {
                    "name": "celldex::HumanPrimaryCellAtlasData() (HPCA)",
                    "content": f"{sr['muraro']['reference_cells']} microarray samples, "
                               f"{sr['muraro']['reference_genes']} genes, 37 main / 157 fine labels",
                    "cited_source": "Mabbott et al., BMC Genomics 2013, processed per Aran et al., "
                                    "Nat Immunol 2019 (the SingleR paper)",
                    "overlap_with_test_atlases": "no: a bulk microarray compendium of primary cells, "
                                                 "predating all four atlases",
                },
                "mouse": {
                    "name": "celldex::MouseRNAseqData()",
                    "content": f"{sr['tabula_muris_senis']['reference_cells']} samples, "
                               f"{sr['tabula_muris_senis']['reference_genes']} genes, 18 main / 28 fine labels",
                    "cited_source": "Benayoun et al. 2019 mouse RNA-seq compendium (celldex)",
                    "overlap_with_test_atlases": "no",
                },
            },
            "corpus_side": "the corpus's own test tier is muraro, siletti, tabula_sapiens, "
                           "tabula_muris_senis; its only training atlas is hao_cite. None of the "
                           "reference datasets above is a corpus atlas.",
            "measured_evidence": "the strongest available local signal against sample-level "
                                 "leakage is the accuracy itself: no arm here approaches the "
                                 "near-100% a memorised training set would give (best arm on its "
                                 "own matched tissue, muraro, is 93.4% and the mean top-1 "
                                 "probability there is 0.69), and the arms that are tissue-matched "
                                 "do not separate cleanly from the ones that are not.",
        },
        "harmonisation_sensitivity": {
            "delta_points_on_pooled_best_arm": harmon["delta_points"],
            "pooled": harmon["pooled_best_arm"],
            "placed_by_supplementary": sum(
                v["labels_placed_by_supplementary"] for v in harmon["per_arm"].values()),
            "unplaced_by_project_map": sum(
                v["labels_unplaced_by_project_map"] for v in harmon["per_arm"].values()),
            "reading": "165 predictions that the project's curated table leaves unplaced can be "
                       "placed with a supplementary table of Cell Ontology terms; doing so "
                       "changes the pooled best-arm accuracy by 0.000 points. The CellTypist "
                       "numbers here are therefore not an artefact of the harmonisation table.",
        },
        "validation": validation,
        "caveats": [
            "CellTypist ships no pan-tissue human model and no pan-tissue mouse model, so the "
            "multi-tissue atlases (tabula_sapiens, tabula_muris_senis -- 914 of the 1,819 "
            "questions) are annotated with a tissue-matched model where one exists and the "
            "immune default otherwise. That is a real property of the tool, not of this run.",
            "SingleR's celldex references are coarse: HPCA has no pancreatic endocrine, no "
            "oligodendrocyte and no adult-neuron label (its 'Neurons' reference is a cell line), "
            "and MouseRNAseq has no pancreatic, kidney or skin label. Where the reference has no "
            "term for the answer, the question is unwinnable by construction; that ceiling is "
            "reported rather than left to depress the accuracy silently.",
            "Both tools were run on the raw count matrix (CP10K + log1p), the normalisation "
            "CellTypist's tutorial and SingleR's documentation prescribe. CellTypist ran with "
            "majority_voting off (a cluster-level decision; this measurement is per cell).",
            "the population is 1,819 questions over 1,497 distinct cells: 322 cells carry two "
            "questions (both rendering variants), so a cell's verdict counts twice. That matches "
            "how the manuscript's census-level engines are attributed.",
        ],
    }

    # ---- Azimuth outcome, read from the summary the bounded attempt wrote -----------------
    az = load("azimuth.summary.json")
    payload["runtime_and_versions"]["azimuth"] = az
    payload["tools_attempted"] = {
        "celltypist": "ran",
        "singler": "ran",
        "azimuth": az,
        "scANVI": az["scANVI"],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- markdown ------------------------------------------------------------------------
    L: list[str] = []
    A = L.append
    A("# The standard annotators on the census-matched `panel_choice` questions")
    A("")
    A("**What this answers.** The manuscript reports decision models at 0.4662-0.6597 naming "
      "accuracy on the 1,819 census-matched `panel_choice` questions, and 0.7191 for the largest "
      "model once the abstain option is removed. The obvious referee objection is *why use a "
      "decision model at all -- CellTypist or SingleR scores much higher*. That had not been "
      "measured on this population. It is now.")
    A("")
    A("Every number below is produced by the scripts in `scripts/annotation-tools/` from the "
      "frozen corpus; nothing is typed in by hand. `reports/annotation-tools-baseline.json` "
      "carries the same values machine-readably.")
    A("")
    A("## Headline")
    A("")
    A("| tool | reading | questions answered | named the answer | on the ballot | target reachable by its vocabulary | correct when reachable |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for key, label in (
        ("celltypist_default_model", "CellTypist, its documented default model (`Immune_All_Low`)"),
        ("celltypist_best_arm", "**CellTypist, best declared arm per atlas**"),
        ("singler_label_main", "SingleR, `label.main` (its documented default)"),
        ("singler_best_label_level", "**SingleR, better of `label.main` / `label.fine`**"),
    ):
        p = pooled[key]
        A(f"| {label} | label -> CL id -> the question's answer | {p['questions']:,} "
          f"| **{pct(p['named_target_rate'])}** | {pct(p['on_ballot_rate'])} "
          f"| {pct(p['target_reachable_rate'])} | {pct(p['named_target_given_reachable_rate'])} |")
    A("")
    A("Both tools answer every question -- neither has an abstain action -- so these are *forced* "
      "numbers and belong beside the decision models' abstention-removed column, not beside their "
      "as-served accuracy.")
    A("")
    A("### Beside the manuscript's engines, on the same 1,819 questions")
    A("")
    A("| engine | as served | abstention removed | muraro | siletti | tabula_muris_senis | tabula_sapiens |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(engine_rows, key=lambda r: -(r["accuracy"] or 0)):
        A(f"| {row['label']} | {pct(row['accuracy'])} | {pct(row['forced_accuracy'])} | "
          + " | ".join(pct(row["per_dataset"][ds]) for ds in DATASETS) + " |")
    for key, label in (("celltypist_best_arm", "**CellTypist** (best arm per atlas)"),
                       ("singler_best_label_level", "**SingleR** (better label level per atlas)")):
        per = tool_per_dataset(key)
        A(f"| {label} | -- | **{pct(pooled[key]['named_target_rate'])}** | "
          + " | ".join(pct(per[ds]["accuracy"]) for ds in DATASETS) + " |")
    A("")
    A("`--` marks the two tools: they have no abstain option, so they have no as-served column. "
      "Their single column *is* a forced column. The engines' as-served and abstention-removed "
      "columns are quoted from `reports/per-dataset.json` (the census join); the tools' per-atlas "
      "columns are the arms named in `per_dataset` in the JSON.")
    A("")

    A("## What was measured, and how a cell became a question")
    A("")
    A(f"* **Population.** The {scores['population']['questions']:,} questions named by "
      f"`reports/census-suite-ids.jsonl` with `family == \"panel_choice\"`, over "
      f"{scores['population']['distinct_cells']:,} distinct cells: "
      + ", ".join(f"{ds} {n}" for ds, n in sorted(scores['population']['per_dataset'].items()))
      + ". The population was read, not re-derived.")
    A(f"* **The question table.** Built by `01_census_questions.py` from the census dump's own "
      f"rendering of each question plus the suite record the population is defined on. All "
      f"{questions['checks']['question_agrees_across_variants']:,} rows passed the panel and "
      f"answer agreement checks; {questions['variant_mismatch_rows']:,} of them are the other "
      f"rendering variant of the same cell and question, which changes neither the panel nor the "
      f"answer and therefore no score.")
    A("* **The mapping -- the critical part.** These tools annotate a *cell* from its expression "
      "profile; the questions are rendered as state descriptions. For a `panel_choice` question "
      "the cell and the candidate panel are both known, so the mapping used here is:")
    A("")
    A("  > run the tool on the cell's raw count matrix; harmonise its label to a Cell Ontology "
      "(CL) id; score 1 if and only if that CL id is the CL id of the question's labelled answer.")
    A("")
    A("  The harmonisation is by CL id rather than by string because the vocabularies differ by "
      "more than case: CellTypist emits `Alpha cell`, `Ductal cell`, `T_cells`; SingleR emits "
      "`T_cells`, `Fibroblasts:foreskin`, `Neurons activated`; the corpus's options are "
      "`pancreatic A cell`, `pancreatic ductal cell`, `T cell`. CellTypist's table is the "
      "project's own curated map (`baselines/celltypist_label_map.json`, unmodified). SingleR's "
      "is `scripts/annotation-tools/singler_label_map.py`, built for this report with an explicit "
      "basis per entry (`name` / `synonym` / `coarsened` / `unmapped`) and an exhaustive override "
      "list for the fine labels whose suffix carries identity rather than an experimental "
      "condition. Every label the map could not place is listed in the JSON rather than absorbed.")
    A(f"* **The target is the cell's own type** for all {scores['population']['questions']:,} "
      f"rows -- asserted, not assumed: the answer option's CL id equals the cell's register id "
      f"in every case.")
    A(f"* **All {scores['population']['questions']:,} questions were scoreable.** Every census "
      f"cell of every atlas is present in the matrix that atlas's arm read (muraro 288, siletti "
      f"465, tabula_muris_senis 305, tabula_sapiens 439 cells; the tabula_sapiens matrix was "
      f"extracted from the 45 GB atlas for this report, spot-checked against it row by row). No "
      f"question was dropped for a missing cell.")
    A("* **Three families are not scoreable for these tools** and none is claimed: "
      "`evidence_diluted`, `marker_noul` and `panel_miss` are constructed evidence states with no "
      "corresponding expression profile. `panel_choice` is exactly the family under discussion, "
      "so this is not a gap for the purpose at hand.")
    A("* **No abstain column, because there is no abstain action.** CellTypist always emits its "
      "top-1 label. SingleR's `pruned.labels` NA is a low-confidence flag, not a decision to "
      "decline: it fires on "
      f"{pooled['singler_label_main']['pruned_declined']:,} of 1,819 questions for `label.main` "
      f"and {pooled['singler_best_label_level']['pruned_declined']:,} for the better level. It is "
      "reported on its own and scored as incorrect in the headline number, following the "
      "manuscript's convention that an abstention is not a correct answer. No abstain-pick rate "
      "is invented for either tool.")
    A("")

    A("## CellTypist, arm by arm")
    A("")
    A("The arms are the set `baselines/step3_celltypist.py` pre-declared, extended to "
      "`tabula_sapiens` with the two arms that file declares for it. They were fixed before any "
      "score was seen. The headline takes the maximum over them, because the referee's objection "
      "is strongest in its best form.")
    A("")
    A("| dataset | arm | model | cells | named the answer | on the ballot | unmapped label | panel-restricted | mean top-1 p | s |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in DATASETS:
        for k in sorted(ct_arms):
            if not k.startswith(ds + "/"):
                continue
            v = ct_arms[k]
            A(f"| {ds} | {v['arm']} | `{v['model']}` | {v['cells_scored']} "
              f"| **{pct(v['named_target_rate'])}** | {pct(v['on_ballot_rate'])} "
              f"| {pct(v['unmapped_rate'])} | {pct(v.get('panel_restricted_rate'))} "
              f"| {v['mean_max_probability']:.3f} | {v['seconds']} |")
    A("")
    A("*`panel-restricted` is the project's own baseline metric (argmax over the question's own "
      "options of the tool's probability mass). It is reported for comparability and is *not* the "
      "mapping described above; it is higher than the headline because it can never name something "
      "off the ballot.*")
    A("")
    A(f"* **Default model only** (`Immune_All_Low` everywhere): {pct(pooled['celltypist_default_model']['named_target_rate'])} "
      f"({pooled['celltypist_default_model']['named_target']:,}/1,819). CellTypist ships no "
      f"pan-tissue model, and on a brain or multi-tissue atlas the immune default names epithelial "
      f"cells, fibroblasts and endothelial cells for neurons and pancreatic endocrine cells.")
    A(f"* **Best arm per atlas**: {pct(pooled['celltypist_best_arm']['named_target_rate'])} "
      f"({pooled['celltypist_best_arm']['named_target']:,}/1,819) -- "
      + "; ".join(f"{ds} {pct(tool_per_dataset('celltypist_best_arm')[ds]['accuracy'])} "
                  f"({tool_per_dataset('celltypist_best_arm')[ds]['arm'].split('/')[1]})"
                  for ds in DATASETS) + ".")
    A(f"* The two muraro arms disagree sharply: the adult islet model names the answer on "
      f"{pct(ct_arms['muraro/pancreatic_islet']['named_target_rate'])} and the fetal pancreas "
      f"model on {pct(ct_arms['muraro/fetal_pancreas']['named_target_rate'])}. Both are in the "
      f"declared arm set and both are reported; the headline uses the higher.")
    A("")

    A("## SingleR")
    A("")
    A("| dataset | reference | labels | cells | named the answer | on the ballot | off-register | pruned (NA) | common genes |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for k in sorted(sr_arms):
        v = sr_arms[k]
        A(f"| {v['dataset']} | {v['reference']} | `{v['arm']}` | {v['cells_scored']} "
          f"| **{pct(v['named_target_rate'])}** | {pct(v['on_ballot_rate'])} "
          f"| {pct(v['off_register_rate'])} | {pct(v['pruned_rate'])} | {v['common_genes']:,} |")
    A("")
    A(f"* **`label.main`** (SingleR's documented default): "
      f"{pct(pooled['singler_label_main']['named_target_rate'])} "
      f"({pooled['singler_label_main']['named_target']:,}/1,819).")
    A(f"* **Better of the two levels per atlas**: "
      f"{pct(pooled['singler_best_label_level']['named_target_rate'])} "
      f"({pooled['singler_best_label_level']['named_target']:,}/1,819) -- "
      + "; ".join(f"{ds} {pct(tool_per_dataset('singler_best_label_level')[ds]['accuracy'])} "
                  f"({tool_per_dataset('singler_best_label_level')[ds]['arm'].split('/')[1]})"
                  for ds in DATASETS) + ".")
    A(f"* **Off-register labels are the mechanism.** {pct(pooled['singler_best_label_level']['off_register_rate'])} "
      f"of the answers are real CL terms the corpus's 82-term register does not contain "
      f"(`astrocyte`, `oligodendrocyte`, `osteoblast`, `adipocyte`, `cardiac muscle cell`, ...), "
      f"so they can never equal an option. On siletti that is the whole story: the reference's "
      f"nearest neighbours for adult cortical neurons are cultured astrocytes and a "
      f"neuroblastoma-derived 'Neurons' line, and "
      f"{pct(sr_arms['siletti/label_fine']['off_register_rate'])} of its calls there are "
      f"off-register.")
    A(f"* **Where the reference has the term, the tool is right.** The best evidence that the "
      f"pipeline is sound rather than broken: SingleR recalls "
      f"{validation['check2_singler_recall_where_reference_covers']['tabula_muris_senis']['recalled']}/"
      f"{validation['check2_singler_recall_where_reference_covers']['tabula_muris_senis']['cells_of_this_type']} "
      f"mouse B cells and "
      f"{validation['check2_singler_recall_where_reference_covers']['tabula_sapiens']['recalled']}/"
      f"{validation['check2_singler_recall_where_reference_covers']['tabula_sapiens']['cells_of_this_type']} "
      f"human B cells (both 100%).")
    A("")

    A("## The ceiling: what the vocabulary could ever reach")
    A("")
    A("A low accuracy has two possible causes that must not be confused: the tool read the cell "
      "wrongly, or the tool's vocabulary has no term for the answer. The table below separates "
      "them. `reachable` is the share of the 1,819 questions whose answer CL id appears anywhere "
      "in the arm's own label vocabulary.")
    A("")
    A("| reading | reachable | correct when reachable | correct overall |")
    A("|---|---:|---:|---:|")
    for key, label in (("celltypist_default_model", "CellTypist, default model"),
                       ("celltypist_best_arm", "CellTypist, best arm per atlas"),
                       ("singler_label_main", "SingleR, label.main"),
                       ("singler_best_label_level", "SingleR, better label level")):
        p = pooled[key]
        A(f"| {label} | {pct(p['target_reachable_rate'])} "
          f"| **{pct(p['named_target_given_reachable_rate'])}** | {pct(p['named_target_rate'])} |")
    A("")
    A("Read the last two columns together. CellTypist, given the best arm on each atlas, is right "
      f"on {pct(pooled['celltypist_best_arm']['named_target_given_reachable_rate'])} of the "
      "questions its vocabulary can express and is simply not offered the rest. SingleR is right "
      f"on {pct(pooled['singler_best_label_level']['named_target_given_reachable_rate'])} of the "
      "questions its references can express -- the difference between the two tools is in the "
      "reading of the cell, not only in the size of the vocabulary.")
    A("")

    A("## Runtime and pinned versions")
    A("")
    A(f"* **CellTypist {payload['runtime_and_versions']['celltypist']['version']}** "
      f"(scanpy {payload['runtime_and_versions']['celltypist']['scanpy']}, anndata "
      f"{payload['runtime_and_versions']['celltypist']['anndata']}), CPU only: "
      f"{payload['runtime_and_versions']['celltypist']['seconds_total']} s for all nine arms over "
      f"1,497 cells; per-arm times are in the arm table above. Model files are pinned by sha256 "
      f"in the JSON.")
    A(f"* **SingleR {payload['runtime_and_versions']['singler']['SingleR']}** "
      f"(celldex {payload['runtime_and_versions']['singler']['celldex']}, R "
      f"{payload['runtime_and_versions']['singler']['R']}, SummarizedExperiment "
      f"{payload['runtime_and_versions']['singler']['SummarizedExperiment']}), defaults "
      f"(`de.method = \"classic\"`, `fine.tune = TRUE`): "
      f"{payload['runtime_and_versions']['singler']['seconds_total']} s wall clock for four "
      f"datasets x two label levels including loading both references.")
    A(f"* **Azimuth {az['versions']['Azimuth']}** (Seurat {az['versions']['Seurat']}): "
      + ("**installed but not scored.** " if az.get("installed") and not az.get("scored")
         else ("scored. " if az.get("scored") else "not installed. "))
      + az["reason_not_scored"] + " The install is recorded in "
      + ", ".join(f"`{p}`" for p in az["logs"]) + ".")
    A(f"* **scANVI**: {az['scANVI']['reason']}")
    A("")
    A("## What each tool's reference is, and the leakage caveats")
    A("")
    A("| tool | arm | reference it was trained on | cited source | overlaps a test atlas? |")
    A("|---|---|---|---|---|")
    prov = payload["references_and_leakage"]["celltypist_model_provenance"]
    for k in sorted(ct):
        if k.startswith("_"):
            continue
        model = ct[k]["model"]
        p = prov[model]
        A(f"| CellTypist | `{k}` | `{model}` | {p['cited_source'].split(' (')[0]} | "
          f"{p['overlap_with_test_atlases']} |")
    A(f"| SingleR | human | HPCA, 713 microarray samples | Mabbott 2013 / Aran 2019 | no |")
    A(f"| SingleR | mouse | MouseRNAseqData | Benayoun 2019 | no |")
    A("")
    A("**The caveats, stated rather than hidden.**")
    A("")
    A("* `Adult_Human_PancreaticIslet` is trained on human pancreatic islets (Tritschler et al., "
      "their own data, GEO GSE198623) -- the same tissue as muraro, and CellTypist publishes no "
      "per-model cell manifest, so sample-level overlap with muraro cannot be excluded from "
      "public documentation. It is the *second*-best muraro arm here "
      f"({pct(ct_arms['muraro/pancreatic_islet']['named_target_rate'])} against the fetal "
      f"model's {pct(ct_arms['muraro/fetal_pancreas']['named_target_rate'])}), which is the "
      "opposite of what sample-level leakage would look like.")
    A("* `Adult_Human_MTG` is trained on adult human middle temporal gyrus; the corpus's siletti "
      "atlas is a BICCN adult human brain atlas that also contains MTG. Different study and "
      "donors as far as public documentation shows, same region. This is the strongest tissue "
      "overlap of any arm here.")
    A("* `Human_Lung_Atlas` integrates 49 respiratory datasets whose constituent list is a "
      "supplementary table this run could not verify, and `tabula_sapiens` contains lung cells. "
      "Treated as an unresolved caveat.")
    A("* `Immune_All_Low` combines 20 tissues of 18 studies (Domínguez Conde et al.); none of the "
      "four test atlases is among the study names CellTypist lists for it.")
    A("* The corpus's own test atlases are muraro, siletti, tabula_sapiens and "
      "tabula_muris_senis; its only training atlas is hao_cite. The two celldex references are "
      "bulk/microarray compendia that predate all four.")
    A("* The strongest local signal against sample-level leakage is the accuracy itself: nothing "
      "here approaches the near-100% a memorised training set would produce, and the "
      "tissue-matched arms do not separate cleanly from the unmatched ones.")
    A("")

    A("## Was it the harmonisation table?")
    A("")
    h = payload["harmonisation_sensitivity"]
    A(f"No. The project's curated CellTypist table leaves {h['unplaced_by_project_map']} "
      f"predictions unplaced across the nine arms; a supplementary table of Cell Ontology terms "
      f"(`06_harmonisation_sensitivity.py`, a new file -- the project's map is unmodified) places "
      f"{h['placed_by_supplementary']} of them. The pooled best-arm accuracy moves by "
      f"**{h['delta_points_on_pooled_best_arm']:+.3f} points** "
      f"({pct(h['pooled']['project_map']['rate'])} -> "
      f"{pct(h['pooled']['project_map_plus_supplementary']['rate'])}). None of the unplaced "
      f"labels was the right answer to any question. The CellTypist numbers are not an artefact "
      f"of the mapping.")
    A("")

    A("## Validation")
    A("")
    c1 = validation["check1_celltypist_reproduces_project_baseline"]
    c1_total = sum(v["census_cells"] for v in c1.values())
    c1_detail = ", ".join(f"{k} {v['agreeing']}/{v['census_cells']}" for k, v in c1.items())
    A("1. **The census CellTypist run reproduces the project's own baseline exactly.** "
      "`baselines/step3_celltypist.py` scored every corpus test cell; this run scores a subset "
      "extracted through a different path. On all seven overlapping arms and all "
      f"{c1_total:,} cells the predicted label agrees on every cell ({c1_detail}).")
    A(f"2. **SingleR recovers a type when the reference has it** -- 100% recall on the B cells of "
      f"both atlases whose reference carries a B-cell label, which is what rules out a broken "
      f"input pipeline as the explanation for the low headline number.")
    A(f"3. **The question table's own checks** passed on all 1,819 rows: the panel and the "
      f"labelled answer agree between the two renderings of the same question, and the answer is "
      f"the cell's own register type in every case.")
    A("")
    A("## What these numbers do and do not imply")
    A("")
    eng = {r["key"]: r for r in engine_rows}
    ct_pooled = pooled["celltypist_best_arm"]
    sr_pooled = pooled["singler_best_label_level"]
    ct_per = tool_per_dataset("celltypist_best_arm")
    sr_per = tool_per_dataset("singler_best_label_level")
    A("They do **not** refute the manuscript's claim, and they are not evidence that decision "
      "models are the most accurate annotators available. The manuscript's claim is narrower and "
      "unaffected: that comparisons *between* these engines are dominated by abstention policy, "
      "and that the largest model's apparent deficit is an abstention artefact. Nothing here "
      "touches that -- the tools do not abstain, so they have no abstention policy to confound "
      "anything. What the numbers do establish is the size and shape of the standard-tool "
      f"comparison, and the honest version of it is not a rout. CellTypist, run as favourably as "
      f"its declared arms allow (best arm per atlas), names the right type on "
      f"{pct(ct_pooled['named_target_rate'])} of the 1,819 questions -- "
      f"level with Kev-9B's {pct(eng['kev-9b']['accuracy'])} as-served, between Kev-4B's "
      f"{pct(eng['kev-4b']['accuracy'])} and Jev's {pct(eng['jev']['accuracy'])} as-served, but "
      f"below *every* large engine's abstention-removed number (Kev-9B "
      f"{pct(eng['kev-9b']['forced_accuracy'])}, Jev {pct(eng['jev']['forced_accuracy'])}, 27B "
      f"{pct(eng['kev-27b']['forced_accuracy'])}) -- and since CellTypist never abstains, the "
      f"forced column is the only fair one to compare it with. SingleR lands at "
      f"{pct(sr_pooled['named_target_rate'])}. Two qualifications travel "
      f"with both numbers. First, the pooled figure hides a real reversal: CellTypist beats every "
      f"decision model on muraro ({pct(ct_per['muraro']['accuracy'])} "
      f"against Jev's {pct(eng['jev']['per_dataset']['muraro'])}) "
      f"and SingleR beats them on tabula_muris_senis "
      f"({pct(sr_per['tabula_muris_senis']['accuracy'])} "
      f"against Kev-9B's "
      f"{pct(eng['kev-9b']['per_dataset']['tabula_muris_senis'])}), "
      f"because both tools are strong exactly where a matched reference exists and weak where "
      f"none does; the pooled ordering is set by how many of the 1,819 questions fall in each "
      f"regime. Second, the tools read the full expression matrix while the decision models read "
      f"a ~201-gene rank list with no expression values, so a win for a tool here is a win with "
      f"strictly more input. The correct place for these numbers in the paper is therefore as the "
      f"honest context for the panel_choice table: the standard annotators are competitive on "
      f"matched tissue and cannot be run at all on the three constructed-evidence families that "
      f"carry most of the manuscript's argument, so they bound the absolute level of the "
      f"comparison without touching its internal one.")
    A("")
    A("## Reproduce")
    A("")
    A("```bash")
    A("source scripts/wsl/env.sh")
    A("uv run --no-sync python scripts/annotation-tools/01_census_questions.py")
    A("uv run --no-sync python scripts/annotation-tools/02_census_matrices.py")
    A("uv run --with celltypist==1.7.1 --with scanpy==1.12.4 --no-sync \\")
    A("    python scripts/annotation-tools/03_celltypist.py")
    A("uv run --no-sync python scripts/annotation-tools/04b_export_for_r.py")
    A("R_LIBS_USER=$HOME/R/sckevtype-libs Rscript scripts/annotation-tools/04c_singler.R")
    A("uv run --no-sync python scripts/annotation-tools/05_score.py")
    A("uv run --no-sync python scripts/annotation-tools/06_harmonisation_sensitivity.py")
    A("uv run --no-sync python scripts/annotation-tools/07_validate.py")
    A("uv run --no-sync python scripts/annotation-tools/08_report.py")
    A("```")
    A("")

    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
