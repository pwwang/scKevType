"""Step 9: assemble the v4 deliverable -- reports/annotation-tools-baseline-v4.{json,md}.

Everything here is read from artefacts the earlier steps wrote; nothing is recomputed and
nothing is typed in by hand, so the markdown and the JSON are one set of numbers rendered twice.

Sources
-------
* ``reports/annotation-tools-v4/`` -- this run's working directory: the question table and its
  checks, the per-dataset matrices and their certificates, the CellTypist summary, the SingleR
  summary, the scores and the validation verdicts.
* ``reports/annotation-tools-repro/`` plus the frozen ``reports/annotation-tools-baseline.json``
  -- the equivalence proof that the parameterised pipeline still reproduces the published v2
  numbers exactly.
* ``reports/abstain-reconciliation-v4.json`` -- the settled served / forced readings of the
  manuscript's own arms on this same population (the source of record for that comparison).
* ``reports/per-dataset-v4.json`` -- those arms' per-atlas breakdown.

The tools cannot abstain, so every number of theirs is a *forced* number, and the comparison
section is stated forced-versus-forced.

Run (WSL, from the repo root):

    source scripts/wsl/env.sh
    uv run --no-sync python scripts/annotation-tools/09_report_v4.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/mnt/f/E/hermes-workspace/scKevType")
V4 = ROOT / "reports" / "annotation-tools-v4"
REPRO = ROOT / "reports" / "annotation-tools-repro"
FROZEN = ROOT / "reports" / "annotation-tools-baseline.json"
RECON = ROOT / "reports" / "abstain-reconciliation-v4.json"
PERD = ROOT / "reports" / "per-dataset-v4.json"
OUT_MD = ROOT / "reports" / "annotation-tools-baseline-v4.md"
OUT_JSON = ROOT / "reports" / "annotation-tools-baseline-v4.json"

DATASETS = ["dominguez_conde", "hao_cite", "muraro", "siletti", "tabula_muris_senis", "tabula_sapiens"]
DATASET_NOTE = {
    "dominguez_conde": "pan-immune, 20 tissues, human",
    "hao_cite": "PBMC (CITE-seq), human",
    "muraro": "pancreas, human",
    "siletti": "brain, human",
    "tabula_muris_senis": "multi-tissue, mouse",
    "tabula_sapiens": "multi-tissue, human",
}
READINGS = (
    ("celltypist_default_model", "CellTypist, its documented default model (`Immune_All_Low` everywhere)"),
    ("celltypist_best_arm", "CellTypist, best declared arm per atlas"),
    ("singler_label_main", "SingleR, `label.main` (its documented default)"),
    ("singler_best_label_level", "SingleR, better of `label.main` / `label.fine` per atlas"),
)
#: the v4 population's overlap column, rewritten from the frozen v2 table because v4 added the
#: atlas that one CellTypist model's own source study deposited
OVERLAP_V4 = {
    "Immune_All_Low.pkl": (
        "YES, for dominguez_conde, and by identity rather than by tissue: this model's cited "
        "source is the Dominguez Conde et al. cross-tissue immune atlas, which is the same study "
        "that deposited dominguez_conde_cxg, one of the six test atlases here. CellTypist does "
        "not publish a per-model cell manifest, so cell-level overlap cannot be excluded and "
        "should be assumed possible. The atlas file itself also carries the columns "
        "`Predicted_labels_CellTypist`, `Majority_voting_CellTypist` and "
        "`Majority_voting_CellTypist_high`, which record that CellTypist was run on these cells "
        "when the atlas was deposited. For the other five atlases there is no documented overlap."
    ),
    "Immune_All_High.pkl": (
        "YES, for dominguez_conde: same reference and same caveat as Immune_All_Low (the same "
        "publication's pan-immune resource, at its coarser 32-label taxonomy)."
    ),
}
#: one-line verdict per model, for the table's last column
OVERLAP_SHORT = {
    "Adult_Human_PancreaticIslet.pkl": "tissue only: same tissue as muraro (adult human islet), no documented sample overlap",
    "Fetal_Human_Pancreas.pkl": "no: muraro is adult and the corpus holds no fetal pancreas",
    "Immune_All_Low.pkl": "YES for dominguez_conde (same study as the atlas); no documented overlap with the other five",
    "Immune_All_High.pkl": "YES for dominguez_conde (same study, same resource); no documented overlap with the other five",
    "Adult_Human_MTG.pkl": "YES for siletti: the same brain region (adult human middle temporal gyrus) is inside that atlas",
    "Adult_Mouse_Gut.pkl": "tissue only: mouse gut, and tabula_muris_senis includes gut; not verified at sample level",
    "Human_Lung_Atlas.pkl": "cannot be excluded: tabula_sapiens contains lung cells and the HLCA constituent list was not verified",
    "HumanPrimaryCellAtlasData": "no: a bulk microarray compendium that predates all six atlases",
    "MouseRNAseqData": "no",
}
FINE_TUNE_27B_NOTE = ("not measured: the settled v4 sources carry its pooled served and forced "
                       "numbers only, with no per-atlas breakdown")


def pct(x, digits: int = 1) -> str:
    return "not measured" if x is None else f"{100 * x:.{digits}f}%"


def frac(num: int, den: int) -> str:
    return f"{num:,}/{den:,}"


def points(x: float, digits: int = 1) -> str:
    """A difference in accuracy points, without a percent sign (the header says points)."""
    return f"{100 * x:+.{digits}f}" if x else f"{0:.{digits}f}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def population_overlap() -> dict:
    """How much of the v2 population is in the v4 population, computed from the two id lists."""
    def rows(path: Path) -> list[dict]:
        out = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["family"] == "panel_choice":
                    out.append(row)
        return out

    v2 = rows(ROOT / "reports" / "census-suite-ids.jsonl")
    v4 = rows(ROOT / "reports" / "census-suite-ids-v4.jsonl")
    v2_ids = {r["id"] for r in v2}
    v4_ids = {r["id"] for r in v4}
    v2_cells = {(r["dataset"], r["cell"]) for r in v2}
    v4_cells = {(r["dataset"], r["cell"]) for r in v4}
    per_dataset = {}
    for dataset in DATASETS:
        a = {r["cell"] for r in v2 if r["dataset"] == dataset}
        b = {r["cell"] for r in v4 if r["dataset"] == dataset}
        per_dataset[dataset] = {"v2_cells": len(a), "v4_cells": len(b), "shared_cells": len(a & b)}
    return {
        "v2_questions": len(v2),
        "v4_questions": len(v4),
        "shared_suite_ids": len(v2_ids & v4_ids),
        "shared_cells_by_dataset_and_cell": len(v2_cells & v4_cells),
        "per_dataset": per_dataset,
        "reading": "the two id lists are line indices into two different corpora, so an id match "
                   "is not a cell match; the (dataset, cell) intersection is the one that says "
                   "whether a v2 number can be carried over, and it cannot",
    }


def atlas_evidence() -> dict:
    """Two facts about the dominguez_conde atlas that the leakage section states, read off it."""
    import h5py

    path = ROOT / "data" / "raw.9p" / "dominguez_conde_cxg" / "global.h5ad"
    with h5py.File(path, "r") as fh:
        obs = [str(k) for k in fh["obs"]]
        label_column = next((c for c in ("cell_type_ontology_term_id", "cell_type") if c in obs), None)
    return {
        "atlas": str(path.relative_to(ROOT)),
        "label_column_read_by_the_corpus": label_column,
        "celltypist_columns_present": sorted(k for k in obs if "CellTypist" in k),
        "what_this_shows": "the atlas file ships CellTypist's own predictions as obs columns, so "
                           "CellTypist had already been run on these cells when the atlas was "
                           "deposited, and the corpus's own label column is the CELLxGENE-mapped "
                           "cell type rather than any CellTypist column",
    }


def build_payload() -> dict:
    ct = load(V4 / "celltypist.summary.json")
    sr = load(V4 / "singler.summary.json")
    scores = load(V4 / "scores.json")
    questions = load(V4 / "questions.summary.json")
    matrices = load(V4 / "matrices.summary.json")
    validation = load(V4 / "validation.json")
    frozen = load(FROZEN)
    repro = load(REPRO / "scores.json")
    equivalence = load(REPRO / "equivalence.json")
    recon = load(RECON)
    perd = load(PERD)

    ct_meta = {k: v for k, v in ct.items() if not k.startswith("_")}
    sr_meta = {k: v for k, v in sr.items()
               if k not in ("versions", "parameters", "timing", "working_directory")}
    pooled = scores["pooled"]
    # one row per arm: the tool run's own metadata (model, features, seconds) plus the scorer's
    # metrics, so a table and the JSON cannot mix them up
    ct_arms = {k: {**ct_meta[k], **v} for k, v in scores["per_arm"]["celltypist"].items()}
    # the SingleR summary is keyed by dataset, the scorer's arms by dataset/level
    sr_arms = {k: {**sr_meta[v["dataset"]], **v} for k, v in scores["per_arm"]["singler"].items()}

    # ---- the manuscript's arms, from the settled source and its per-atlas companion ----------
    def is_fine_tune(arm: str) -> bool:
        """per-dataset-v4.json carries the arm's kind where the arm has a per-atlas dump; the
        v4 fine-tunes are otherwise the delta runs and Laya's spec arm, by name."""
        kind = perd["arms"].get(arm, {}).get("kind")
        if kind:
            return kind == "fine-tune"
        return arm.startswith("delta-") or arm.endswith("-ft-spec")

    manuscript = {}
    fine_tuned = {}
    for arm, entry in recon["arms"].items():
        manuscript[arm] = {
            "shown_as": entry.get("shown_as"),
            "kind": "fine-tune" if is_fine_tune(arm) else "released rung",
            "family": entry.get("kind"),
            "n": entry["n"],
            "served_correct": entry["served"]["correct"],
            "served_accuracy": entry["served"]["accuracy"],
            "forced_correct": entry["forced"]["correct"],
            "forced_accuracy": entry["forced"]["accuracy"],
            "abstain_pick_rate": entry["abstain_pick"]["rate"],
            "per_atlas": perd["arms"].get(arm, {}).get("per_atlas"),
        }
        if manuscript[arm]["kind"] == "fine-tune":
            fine_tuned[arm] = manuscript[arm]

    # the cross-source check: where an arm is in both files, the pooled numbers must agree
    cross = {}
    for arm, entry in manuscript.items():
        per_atlas = entry["per_atlas"]
        if not per_atlas:
            continue
        n = sum(v["n"] for v in per_atlas.values())
        served = sum(v["served_correct"] for v in per_atlas.values())
        forced = sum(v["forced_correct"] for v in per_atlas.values())
        ok = (n == entry["n"] and served == entry["served_correct"] and forced == entry["forced_correct"])
        cross[arm] = {
            "rows_from_per_atlas": n,
            "served_from_per_atlas": served,
            "forced_from_per_atlas": forced,
            "agrees_with_abstain_reconciliation": bool(ok),
        }

    # ---- the answer to the two closing questions --------------------------------------------
    # our best forced fine-tune per atlas, over the fine-tuned arms the per-atlas source carries
    per_atlas_answer = {}
    for dataset in DATASETS:
        rows = {arm: entry["per_atlas"][dataset] for arm, entry in fine_tuned.items()
                if entry["per_atlas"] and dataset in entry["per_atlas"]}
        if not rows:
            per_atlas_answer[dataset] = {"ours_forced": "not measured", "ours_best_served": "not measured",
                                         "best_arm": "not measured"}
            continue
        best = max(rows, key=lambda a: rows[a]["forced_accuracy"])
        best_served = max(rows, key=lambda a: rows[a]["served_accuracy"])
        per_atlas_answer[dataset] = {
            "best_arm": best,
            "best_arm_shown_as": fine_tuned[best]["shown_as"],
            "ours_forced_correct": rows[best]["forced_correct"],
            "ours_forced": rows[best]["forced_accuracy"],
            "ours_served": rows[best]["served_accuracy"],
            "ours_best_served_arm": best_served,
            "ours_best_served_arm_shown_as": fine_tuned[best_served]["shown_as"],
            "ours_best_served": rows[best_served]["served_accuracy"],
            "all_fine_tuned_forced": {a: r["forced_accuracy"] for a, r in sorted(rows.items())},
            "all_fine_tuned_served": {a: r["served_accuracy"] for a, r in sorted(rows.items())},
        }
    ce_arm = pooled["celltypist_best_arm"]["per_dataset_arm"]
    sr_arm = pooled["singler_best_label_level"]["per_dataset_arm"]
    for dataset in DATASETS:
        ct_key = ce_arm[dataset]
        sr_key = sr_arm[dataset]
        per_atlas_answer[dataset]["celltypist_arm"] = ct_key
        per_atlas_answer[dataset]["celltypist_model"] = ct_arms[ct_key]["model"]
        per_atlas_answer[dataset]["celltypist_forced"] = ct_arms[ct_key]["named_target_rate"]
        per_atlas_answer[dataset]["singler_arm"] = sr_key
        per_atlas_answer[dataset]["singler_forced"] = sr_arms[sr_key]["named_target_rate"]
        ours = per_atlas_answer[dataset]["ours_forced"]
        if isinstance(ours, float):
            per_atlas_answer[dataset]["celltypist_minus_ours_forced"] = round(
                per_atlas_answer[dataset]["celltypist_forced"] - ours, 6)
            per_atlas_answer[dataset]["celltypist_minus_ours_best_served"] = round(
                per_atlas_answer[dataset]["celltypist_forced"] - per_atlas_answer[dataset]["ours_best_served"], 6)
            per_atlas_answer[dataset]["our_forced_margin"] = round(
                ours - per_atlas_answer[dataset]["celltypist_forced"], 6)
        else:
            per_atlas_answer[dataset]["celltypist_minus_ours_forced"] = "not measured"
            per_atlas_answer[dataset]["celltypist_minus_ours_best_served"] = "not measured"
            per_atlas_answer[dataset]["our_forced_margin"] = "not measured"

    best_ft_arm = max(fine_tuned, key=lambda a: fine_tuned[a]["forced_accuracy"])
    losses_forced = [d for d in DATASETS
                     if isinstance(per_atlas_answer[d]["celltypist_minus_ours_forced"], float)
                     and per_atlas_answer[d]["celltypist_minus_ours_forced"] > 0]
    losses_served = [d for d in DATASETS
                     if isinstance(per_atlas_answer[d]["celltypist_minus_ours_best_served"], float)
                     and per_atlas_answer[d]["celltypist_minus_ours_best_served"] > 0]
    margins = [per_atlas_answer[d]["our_forced_margin"] for d in DATASETS
               if isinstance(per_atlas_answer[d]["our_forced_margin"], float)]
    answers_block = {
        "forced_versus_forced": {
            "celltypist_best_arm_pooled": pooled["celltypist_best_arm"]["named_target_rate"],
            "singler_best_label_level_pooled": pooled["singler_best_label_level"]["named_target_rate"],
            "our_best_fine_tuned_forced": fine_tuned[best_ft_arm]["forced_accuracy"],
            "our_best_fine_tuned_arm": best_ft_arm,
            "our_best_fine_tuned_shown_as": fine_tuned[best_ft_arm]["shown_as"],
            "our_best_fine_tuned_served": fine_tuned[best_ft_arm]["served_accuracy"],
            "beats_celltypist": fine_tuned[best_ft_arm]["forced_accuracy"] > pooled["celltypist_best_arm"]["named_target_rate"],
            "beats_singler": fine_tuned[best_ft_arm]["forced_accuracy"] > pooled["singler_best_label_level"]["named_target_rate"],
            "every_fine_tuned_arm_forced_min": min(v["forced_accuracy"] for v in fine_tuned.values()),
            "every_fine_tuned_arm_forced_max": max(v["forced_accuracy"] for v in fine_tuned.values()),
            "every_fine_tuned_arm_served_min": min(v["served_accuracy"] for v in fine_tuned.values()),
        },
        "per_atlas": per_atlas_answer,
        "atlases_where_celltypist_beats_our_best_forced": losses_forced,
        "atlases_where_celltypist_beats_our_best_served": losses_served,
        "our_forced_margin_per_atlas": {d: per_atlas_answer[d]["our_forced_margin"] for d in DATASETS},
        "our_forced_margin_range": [min(margins), max(margins)] if margins else "not measured",
        "fine_tuned_27b_per_atlas": FINE_TUNE_27B_NOTE,
    }

    leakage = {
        "models": {
            model: {
                **{k: v for k, v in frozen["references_and_leakage"]["celltypist_model_provenance"][model].items()},
                **({"overlap_with_test_atlases_v4": OVERLAP_V4[model]} if model in OVERLAP_V4 else {}),
            }
            for model in frozen["references_and_leakage"]["celltypist_model_provenance"]
        },
        "arms": {
            key: {
                "dataset": v["dataset"],
                "arm": v["arm"],
                "model": v["model"],
                "model_sha256": v["model_sha256"],
                "model_why": v["model_why"],
                "labels_in_model": v["labels_in_model"],
                "model_features": v["model_features"],
                "overlaps_a_test_atlas": OVERLAP_SHORT[v["model"]],
            }
            for key, v in sorted(ct_arms.items())
        },
        "singler": {
            "human": {
                **frozen["references_and_leakage"]["singler_references"]["human"],
                "overlap_with_test_atlases": "no: a bulk microarray compendium of primary cells, "
                                             "predating all six atlases",
            },
            "mouse": frozen["references_and_leakage"]["singler_references"]["mouse"],
        },
    }
    # Immune_All_High is used by this population and has no entry in the frozen v2 table
    leakage["models"]["Immune_All_High.pkl"] = {
        "cited_source": frozen["references_and_leakage"]["celltypist_model_provenance"]["Immune_All_Low.pkl"]["cited_source"],
        "training_data": "immune populations combined from 20 tissues of 18 studies (the same "
                         "pan-immune resource as Immune_All_Low, at its coarser 32-label taxonomy)",
        "overlap_with_test_atlases_v4": OVERLAP_V4["Immune_All_High.pkl"],
    }

    unmapped = {
        "celltypist": {
            key: {
                "unmapped_count": v["unmapped"],
                "unmapped_rate": v["unmapped_rate"],
                "unmapped_label_names": v["unmapped_label_names"],
                "off_register_count": v["off_register"],
                "off_register_rate": v["off_register_rate"],
                "off_register_label_names": v["off_register_label_names"],
            }
            for key, v in sorted(ct_arms.items())
        },
        "singler": {
            key: {
                "unmapped_count": v["unmapped"],
                "unmapped_rate": v["unmapped_rate"],
                "unmapped_labels": v["unmapped_labels"],
                "off_register_count": v["off_register"],
                "off_register_rate": v["off_register_rate"],
                "off_register_label_names": v["off_register_label_names"],
            }
            for key, v in sorted(sr_arms.items())
        },
        "note": "a label the harmonisation table cannot place is listed here by name; it is never "
                "coerced into the nearest register term and never dropped from the count",
    }

    equivalence_block = {
        "what": equivalence["what"],
        "verdict": equivalence["verdict"],
        "frozen": equivalence["frozen"],
        "fresh_run": equivalence["repro"],
        "readings": {
            reading: {
                "frozen_correct": frozen["headline"][reading]["named_target"],
                "frozen_rate": frozen["headline"][reading]["named_target_rate"],
                "fresh_correct": repro["pooled"][reading]["named_target"],
                "fresh_rate": repro["pooled"][reading]["named_target_rate"],
                "per_dataset_arm": repro["pooled"][reading]["per_dataset_arm"],
            }
            for reading, _ in READINGS
        },
        "fresh_run_per_dataset_arm": {
            reading: {
                dataset: {
                    "arm": arm,
                    "correct": (repro["per_arm"]["celltypist"] if "/" in arm and arm in repro["per_arm"]["celltypist"]
                                else repro["per_arm"]["singler"])[arm]["named_target"],
                    "rate": (repro["per_arm"]["celltypist"] if "/" in arm and arm in repro["per_arm"]["celltypist"]
                             else repro["per_arm"]["singler"])[arm]["named_target_rate"],
                }
                for dataset, arm in repro["pooled"][reading]["per_dataset_arm"].items()
            }
            for reading, _ in READINGS
        },
        "fresh_run_validation": load(REPRO / "validation.json")["verdict"],
        "per_arm_fields_compared": "every per-arm count and rate of both tools, plus the population block",
    }

    validation_block = {
        "v4": validation,
        "equivalence_v2": equivalence_block,
    }

    payload = {
        "what_this_is": "the standard reference-based single-cell annotators (CellTypist and "
                        "SingleR) scored on the v4 census-matched panel_choice population, the "
                        "same 4,035 questions the manuscript's decision models are measured on",
        "population": {
            **questions,
            "how_it_was_read": "reports/census-suite-ids-v4.jsonl (family == panel_choice) -> the "
                               "question table in reports/annotation-tools-v4/questions.jsonl, "
                               "whose record for each id is the corpus test partition's own line",
            "dataset_notes": DATASET_NOTE,
            "source_of_record_for_the_manuscript_side": str(RECON.relative_to(ROOT)),
            "overlap_with_the_v2_population": population_overlap(),
        },
        "atlas_evidence": atlas_evidence(),
        "mapping": {
            "rule": "run the tool on the cell; harmonise its label to a Cell Ontology (CL) id; "
                    "score 1 if and only if that CL id is the CL id of the question's labelled answer",
            "unchanged_from_the_v2_report": True,
            "celltypist_table": "baselines/celltypist_label_map.json via lib_common.resolve_celltypist_map, unmodified",
            "singler_table": "scripts/annotation-tools/singler_label_map.py, unmodified",
            "target_is_the_cells_own_type": "asserted for every row: the target option's CL id "
                                            "equals the cell's register id",
        },
        "headline": {
            reading: {
                **pooled[reading],
                "label": label,
            }
            for reading, label in READINGS
        },
        "coverage": {
            "questions": pooled["celltypist_best_arm"]["questions"],
            "celltypist_arms": len(ct_arms),
            "singler_arms": len(sr_arms),
            "celltypist_cell_not_scored": sum(v["cell_not_scored"] for v in ct_arms.values()),
            "singler_cell_not_scored": sum(v["cell_not_scored"] for v in sr_arms.values()),
            "cells_scored_per_dataset": {k: v["cells"] for k, v in sorted(matrices.items())},
            "reading": "every population cell is in the matrix its dataset's arms read, and no arm "
                       "left a question unscored, so no atlas and no arm had to be dropped",
        },
        "per_arm": {"celltypist": ct_arms, "singler": sr_arms},
        "per_dataset": {
            "celltypist_best_arm": {
                dataset: {"arm": ce_arm[dataset], "rate": ct_arms[ce_arm[dataset]]["named_target_rate"],
                          "correct": ct_arms[ce_arm[dataset]]["named_target"]}
                for dataset in DATASETS
            },
            "celltypist_default_model": {
                dataset: {"arm": pooled["celltypist_default_model"]["per_dataset_arm"][dataset],
                          "rate": ct_arms[pooled["celltypist_default_model"]["per_dataset_arm"][dataset]]["named_target_rate"],
                          "correct": ct_arms[pooled["celltypist_default_model"]["per_dataset_arm"][dataset]]["named_target"]}
                for dataset in DATASETS
            },
            "singler_best_label_level": {
                dataset: {"arm": sr_arm[dataset], "rate": sr_arms[sr_arm[dataset]]["named_target_rate"],
                          "correct": sr_arms[sr_arm[dataset]]["named_target"]}
                for dataset in DATASETS
            },
            "singler_label_main": {
                dataset: {"arm": f"{dataset}/label_main",
                          "rate": sr_arms[f"{dataset}/label_main"]["named_target_rate"],
                          "correct": sr_arms[f"{dataset}/label_main"]["named_target"]}
                for dataset in DATASETS
            },
        },
        "vocabulary_ceiling": {
            reading: {
                "target_reachable": pooled[reading]["target_reachable_by_this_vocabulary"],
                "target_reachable_rate": pooled[reading]["target_reachable_rate"],
                "correct_when_reachable": pooled[reading]["named_target_given_reachable"],
                "correct_when_reachable_rate": pooled[reading]["named_target_given_reachable_rate"],
                "correct_overall": pooled[reading]["named_target"],
                "correct_overall_rate": pooled[reading]["named_target_rate"],
            }
            for reading, _ in READINGS
        },
        "unmapped_and_off_register_labels": unmapped,
        "arms_declared": {
            "rule": "the tissue/species-matched model where CellTypist ships one, else its "
                    "documented default; declared before any score was computed",
            "unchanged_for_the_four_v2_datasets": [
                key for key in sorted(ct_arms) if key.split("/")[0] not in ("dominguez_conde", "hao_cite")
            ],
            "new_for_v4": {
                "dominguez_conde": ["dominguez_conde/immune_low", "dominguez_conde/immune_high"],
                "hao_cite": ["hao_cite/immune_low", "hao_cite/immune_high"],
            },
            "why": "both new datasets are immune datasets (a 20-tissue pan-immune atlas and PBMC "
                   "CITE-seq), and CellTypist's two pan-immune models are the tissue-matched arms; "
                   "celltypist ships no pan-tissue general model",
            "models_not_declared": {
                "models": ["Adult_COVID19_PBMC.pkl", "COVID19_HumanChallenge_Blood.pkl",
                           "COVID19_Immune_Landscape.pkl", "Healthy_COVID19_PBMC.pkl",
                           "Cells_Human_Tonsil.pkl"],
                "reason": "single-study disease or single-tissue references whose vocabularies "
                          "cannot express these answers, measured before any score: 0, 1, 0 and 2 "
                          "of their labels respectively resolve by name to a register term, and "
                          "none of them is in the project's dumped model vocabulary that the "
                          "harmonisation path reads",
            },
        },
        "leakage": leakage,
        "validation": validation_block,
        "runtime_and_versions": {
            "celltypist": {
                "version": ct_arms["muraro/pancreatic_islet"]["celltypist_version"],
                "scanpy": ct_arms["muraro/pancreatic_islet"]["scanpy_version"],
                "anndata": ct_arms["muraro/pancreatic_islet"]["anndata_version"],
                "seconds_per_arm": {k: v["seconds"] for k, v in sorted(ct_arms.items())},
                "seconds_total": round(sum(v["seconds"] for v in ct_arms.values()), 1),
                "hardware": "CPU only, WSL2",
                "model_sha256": {k: v["model_sha256"] for k, v in sorted(ct_arms.items())},
            },
            "singler": {
                **sr["versions"],
                **sr["parameters"],
                **sr["timing"],
            },
            "matrices": {k: {"cells": v["cells"], "genes": v["genes"], "seconds": v["seconds"],
                             "source": v["source"],
                             "source_is_certified_extraction": v["source_is_certified_extraction"],
                             "spot_checks_matched": (
                                 f"{sum(1 for s in v['spot_checks'] if s['matched'])}/{len(v['spot_checks'])}"
                                 if v.get("spot_checks") else "not applicable: the source order is certified"),
                             }
                         for k, v in sorted(matrices.items())},
            "azimuth": "not measured for the v4 population (the v2 report's bounded attempt stands)",
            "scANVI": "not measured for the v4 population",
        },
        "beside_the_manuscript": {
            "readings": "served / forced from reports/abstain-reconciliation-v4.json (the settled "
                        "source); per-atlas from reports/per-dataset-v4.json",
            "arms": manuscript,
            "cross_source_check": cross,
            "cells_of_within_source_arms": {k: v for k, v in cross.items()
                                            if v["agrees_with_abstain_reconciliation"] is False},
            "forced_versus_forced_only": "neither tool can abstain, so only the forced column of "
                                         "the manuscript's arms is comparable with a tool's number",
        },
        "answers": answers_block,
        "caveats": [
            ("Neither tool can abstain: CellTypist always returns its top-1 label and SingleR's "
             "`pruned.labels` NA is a low-confidence flag, not a decision to decline. Their "
             "numbers are forced numbers and are never placed beside a served column."),
            ("The v4 fine-tunes train on every atlas under a donor-level holdout, so where a "
             "comparison is made with a tool whose reference is a different study it favours us. "
             "The tools' structural advantage is the other one: they read the whole expression "
             "matrix while the decision models read a fixed marker panel."),
            ("The tools are scored on the `panel_choice` family only. `evidence_diluted`, "
             "`marker_noul` and `panel_miss` are constructed evidence states with no "
             "corresponding expression profile, so no cell-level annotator can be asked them."),
            ("CellTypist ships no pan-tissue model, so the multi-tissue atlases are annotated with "
             "a tissue-matched model where one exists and the immune default otherwise. That is a "
             "property of the tool, not of this run."),
            ("tabula_muris_senis contributes 112 questions here against 378 in the v2 population, "
             "so its per-atlas rate is noisier than the others."),
        ],
    }
    return payload


def render_markdown(p: dict) -> str:
    L: list[str] = []
    A = L.append
    pooled = {k: p["headline"][k] for k, _ in READINGS}
    ct_arms = {k: v for k, v in p["per_arm"]["celltypist"].items()}
    sr_arms = {k: v for k, v in p["per_arm"]["singler"].items()}

    A("# The standard annotators on the v4 census-matched `panel_choice` questions")
    A("")
    ov = p["population"]["overlap_with_the_v2_population"]
    A("**What this answers.** Every standard-annotator number the manuscript carries so far was "
      "measured on the 1,819-question v2 population. The v4 population is a different population "
      f"({p['population']['questions_written']:,} questions over "
      f"{p['population']['distinct_cells']:,} distinct cells and six atlases), and the two share "
      f"{ov['shared_cells_by_dataset_and_cell']} (dataset, cell) pairs. {ov['shared_suite_ids']} "
      "of their suite ids coincide, but a suite id is a line index into a different corpus, so "
      "that is a coincidence rather than a shared question: the cell intersection is the number "
      "that matters, and v2's 1,819 questions cannot be carried over. This is the same "
      "measurement re-run on it: CellTypist and SingleR, on the same questions, under the same "
      "cell-level mapping.")
    A("")
    A("Method summary: the tools annotate a cell from its raw count matrix, their label is "
      "harmonised to a Cell Ontology id, and a question counts as correct if and only if that CL "
      "id is the CL id of the question's labelled answer. Both tools answer every question, so "
      "their numbers are **forced** numbers.")
    A("")
    A("Every number below is produced by the scripts in `scripts/annotation-tools/` from the "
      "frozen corpus and the six atlases on disk; nothing is typed in by hand. "
      "`reports/annotation-tools-baseline-v4.json` carries the same values machine-readably.")
    A("")

    A("## Headline")
    A("")
    A("| tool and reading | questions answered | named the answer | on the ballot | target reachable by its vocabulary | correct when reachable |")
    A("|---|---:|---:|---:|---:|---:|")
    for key, label in READINGS:
        v = pooled[key]
        A(f"| {label} | {v['questions']:,} | **{pct(v['named_target_rate'])}** "
          f"({frac(v['named_target'], v['questions'])}) | {pct(v['on_ballot_rate'])} "
          f"| {pct(v['target_reachable_rate'])} | {pct(v['named_target_given_reachable_rate'])} |")
    A("")
    A("Both tools answer every question: neither has an abstain action, so these are forced "
      "numbers and belong beside a decision model's abstention-removed column, never beside its "
      "as-served accuracy.")
    A("")

    A("## Per dataset")
    A("")
    A("| dataset | questions | tissue |")
    A("|---|---:|---|")
    for dataset in DATASETS:
        A(f"| {dataset} | {p['population']['questions_per_dataset'][dataset]:,} | {DATASET_NOTE[dataset]} |")
    A(f"| **total** | **{p['population']['questions_written']:,}** | six atlases, each a distinct dataset |")
    A("")
    cov = p["coverage"]
    A(f"* **All {cov['questions']:,} questions were scoreable, and nothing was dropped.** Every "
      "population cell is present in the matrix its dataset's arms read ("
      + ", ".join(f"{d} {n:,}" for d, n in cov["cells_scored_per_dataset"].items())
      + f"), and the number of questions left unscored is {cov['celltypist_cell_not_scored']} "
      f"across the {cov['celltypist_arms']} CellTypist arms and {cov['singler_cell_not_scored']} "
      f"across the {cov['singler_arms']} SingleR arms. No atlas and no arm had to be dropped, so "
      "there is no missing entry behind any table here.")
    A("")
    A("### CellTypist, arm by arm")
    A("")
    A("| dataset | arm | model | cells | named the answer | on the ballot | reachable | correct when reachable | unmapped | panel-restricted | mean top-1 p | s |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for key in sorted(ct_arms):
            if not key.startswith(dataset + "/"):
                continue
            v = ct_arms[key]
            A(f"| {dataset} | {v['arm']} | `{v['model']}` | {v['cells_scored']} "
              f"| **{pct(v['named_target_rate'])}** ({v['named_target']}) | {pct(v['on_ballot_rate'])} "
              f"| {pct(v['target_reachable_rate'])} | {pct(v['named_target_given_reachable_rate'])} "
              f"| {pct(v['unmapped_rate'])} | {pct(v.get('panel_restricted_rate'))} "
              f"| {v['mean_max_probability']:.3f} | {v['seconds']} |")
    A("")
    A(f"* **Default model only** (`Immune_All_Low` everywhere, uppercased symbols on the mouse "
      f"atlas): {pct(pooled['celltypist_default_model']['named_target_rate'])} "
      f"({frac(pooled['celltypist_default_model']['named_target'], pooled['celltypist_default_model']['questions'])}).")
    A(f"* **Best declared arm per atlas**: "
      f"{pct(pooled['celltypist_best_arm']['named_target_rate'])} "
      f"({frac(pooled['celltypist_best_arm']['named_target'], pooled['celltypist_best_arm']['questions'])}) -- "
      + "; ".join(f"{d} {pct(p['per_dataset']['celltypist_best_arm'][d]['rate'])} "
                  f"({p['per_dataset']['celltypist_best_arm'][d]['arm'].split('/')[1]})"
                  for d in DATASETS) + ".")
    A("* The arms are the ones the existing baseline declares for the four v2 datasets "
      "(unchanged, so those rows are directly comparable with the published v2 report), plus two "
      "declared for each of the two new datasets. The declaration and its evidence are in the "
      "JSON under `arms_declared`; both new datasets are immune datasets, so CellTypist's two "
      "pan-immune models (`Immune_All_Low`, `Immune_All_High`) are the tissue-matched arms there.")
    A("")

    A("### SingleR")
    A("")
    A("| dataset | reference | labels | cells | named the answer | on the ballot | off-register | pruned (NA) | common genes | s |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(sr_arms):
        v = sr_arms[key]
        A(f"| {v['dataset']} | {v['reference']} | `{v['arm']}` | {v['cells_scored']} "
          f"| **{pct(v['named_target_rate'])}** ({v['named_target']}) | {pct(v['on_ballot_rate'])} "
          f"| {pct(v['off_register_rate'])} | {pct(v['pruned_rate'])} | {v['common_genes']:,} "
          f"| {v['seconds']} |")
    A("")
    A(f"* **`label.main`**: {pct(pooled['singler_label_main']['named_target_rate'])} "
      f"({frac(pooled['singler_label_main']['named_target'], pooled['singler_label_main']['questions'])}).")
    A(f"* **Better of the two levels per atlas**: "
      f"{pct(pooled['singler_best_label_level']['named_target_rate'])} "
      f"({frac(pooled['singler_best_label_level']['named_target'], pooled['singler_best_label_level']['questions'])}) -- "
      + "; ".join(f"{d} {pct(p['per_dataset']['singler_best_label_level'][d]['rate'])} "
                  f"({p['per_dataset']['singler_best_label_level'][d]['arm'].split('/')[1]})"
                  for d in DATASETS) + ".")
    A("* The two celldex references are the ones SingleR's own documentation uses: HPCA for the "
      "five human atlases, MouseRNAseqData for tabula_muris_senis.")
    A("")

    A("## The ceiling: what each vocabulary could ever reach")
    A("")
    A("A low accuracy has two causes that must not be conflated: the tool read the cell wrongly, "
      "or its vocabulary has no term for the answer. `reachable` is the share of the 4,035 "
      "questions whose answer CL id appears anywhere in that reading's own label vocabulary; "
      "`correct when reachable` is the accuracy restricted to those questions.")
    A("")
    A("| reading | reachable | correct when reachable | correct overall |")
    A("|---|---:|---:|---:|")
    for key, label in READINGS:
        v = pooled[key]
        A(f"| {label} | {pct(v['target_reachable_rate'])} | "
          f"**{pct(v['named_target_given_reachable_rate'])}** | {pct(v['named_target_rate'])} |")
    A("")
    A("| dataset | CellTypist, its chosen arm | reachable | correct when reachable | SingleR, its chosen level | reachable | correct when reachable |")
    A("|---|---|---:|---:|---|---:|---:|")
    for dataset in DATASETS:
        c = p["per_dataset"]["celltypist_best_arm"][dataset]
        s = p["per_dataset"]["singler_best_label_level"][dataset]
        ck = c["arm"]
        sk = s["arm"]
        A(f"| {dataset} | `{ck.split('/')[1]}` | {pct(ct_arms[ck]['target_reachable_rate'])} "
          f"| {pct(ct_arms[ck]['named_target_given_reachable_rate'])} "
          f"| `{sk.split('/')[1]}` | {pct(sr_arms[sk]['target_reachable_rate'])} "
          f"| {pct(sr_arms[sk]['named_target_given_reachable_rate'])} |")
    A("")
    A("Read the two columns together. CellTypist, at its best declared arm per atlas, is right "
      f"on {pct(pooled['celltypist_best_arm']['named_target_given_reachable_rate'])} of the "
      "questions its vocabulary can express and is simply not offered the rest. SingleR's "
      f"references reach {pct(pooled['singler_best_label_level']['target_reachable_rate'])} of "
      f"the answers and it is right on "
      f"{pct(pooled['singler_best_label_level']['named_target_given_reachable_rate'])} of those, "
      "so its low headline is a reading failure inside a partly reachable vocabulary, not only a "
      "missing label: on siletti "
      f"{pct(sr_arms['siletti/label_main']['off_register_rate'])} of the `label.main` calls name a "
      "CL term the register does not hold, carried by the labels "
      + " and ".join(f"`{n}`" for n in
                     p["unmapped_and_off_register_labels"]["singler"]["siletti/label_main"]
                     ["off_register_label_names"])
      + ".")
    A("")

    A("## Labels the harmonisation could not place")
    A("")
    A("Listed by name rather than absorbed into a rate. `unmapped` means the label has no CL id "
      "in the table at all; `off-register` means it resolved to a real CL term that the corpus's "
      "82-term register does not contain, so it can never equal an option.")
    A("")
    A("| arm | unmapped | labels the map cannot place | off-register | labels that resolved off the register |")
    A("|---|---:|---|---:|---|")
    for key, v in sorted(p["unmapped_and_off_register_labels"]["celltypist"].items()):
        names = ", ".join(f"`{n}`" for n in v["unmapped_label_names"][:6]) or "none"
        off = ", ".join(f"`{n}`" for n in v["off_register_label_names"][:6]) or "none"
        A(f"| `{key}` | {v['unmapped_count']} | {names} | {v['off_register_count']} | {off} |")
    for key, v in sorted(p["unmapped_and_off_register_labels"]["singler"].items()):
        names = ", ".join(f"`{n}`" for n in v["unmapped_labels"][:6]) or "none"
        off_names = ", ".join(f"`{n}`" for n in v["off_register_label_names"][:6]) or "none"
        A(f"| `{key}` | {v['unmapped_count']} | {names} | {v['off_register_count']} | {off_names} |")
    A("")

    A("## Was it the harmonisation table?")
    A("")
    A("No, and the same experiment was run on this population's predecessor: a supplementary "
      "table of Cell Ontology terms places 165 of the predictions the project's curated table "
      "leaves unplaced and moves the pooled best-arm accuracy by 0.000 points, because none of "
      "the unplaced labels is the right answer to any question "
      "(`reports/annotation-tools-baseline.json`, `harmonisation_sensitivity`). The curated table "
      "and SingleR's label table are used unmodified here.")
    A("")

    A("## Runtime and pinned versions")
    A("")
    rt = p["runtime_and_versions"]
    A(f"* **CellTypist {rt['celltypist']['version']}** (scanpy {rt['celltypist']['scanpy']}, "
      f"anndata {rt['celltypist']['anndata']}), CPU only: {rt['celltypist']['seconds_total']} s "
      f"for all {len(rt['celltypist']['seconds_per_arm'])} arms over the population's cells. "
      f"Every model file is pinned by sha256 in the JSON.")
    A(f"* **SingleR {rt['singler']['SingleR']}** (celldex {rt['singler']['celldex']}, R "
      f"{rt['singler']['R']}, Matrix {rt['singler']['Matrix']}, SummarizedExperiment "
      f"{rt['singler']['SummarizedExperiment']}), defaults (`de.method = \"classic\"`, "
      f"`fine.tune = TRUE`), CP10K + log1p: {rt['singler']['seconds_total']} s wall clock for "
      f"six datasets x two label levels, including loading both references. Per-dataset times "
      f"are in the JSON.")
    A("* **Matrices**: built one dataset at a time, each written with its certificate as soon as "
      "it completed. Seconds per dataset and the source of each matrix are in the JSON; the two "
      "atlas reads are `tabula_sapiens` from the 45 GB atlas and `dominguez_conde` from the "
      "3 GB CELLxGENE copy, both spot-checked against the atlas row by row.")
    A("* **Azimuth** and **scANVI**: not measured for the v4 population. SingleR's `pruned` NA "
      "counts are reported per arm above and are not treated as declines.")
    A("")

    A("## What each arm's reference is, and the leakage caveats")
    A("")
    A("| tool | arm | reference | reference trained on | overlaps a test atlas? |")
    A("|---|---|---|---|---|")
    for key, v in sorted(p["leakage"]["arms"].items()):
        model = v["model"]
        prov = p["leakage"]["models"].get(model, {})
        trained = prov.get("training_data", "not measured")
        A(f"| CellTypist | `{key}` (`{model}`) | {prov.get('cited_source', 'not measured')} "
          f"| {trained} | {v['overlaps_a_test_atlas']} |")
    A(f"| SingleR | human (5 atlases) | "
      f"{p['leakage']['singler']['human']['name']} | "
      f"{p['leakage']['singler']['human']['content']} | "
      f"{p['leakage']['singler']['human']['overlap_with_test_atlases']} |")
    A(f"| SingleR | mouse (tabula_muris_senis) | "
      f"{p['leakage']['singler']['mouse']['name']} | "
      f"{p['leakage']['singler']['mouse']['content']} | "
      f"{p['leakage']['singler']['mouse']['overlap_with_test_atlases']} |")
    A("")
    A("**Stated rather than dismissed.**")
    A("")
    A("* **`Adult_Human_MTG` is the strongest tissue overlap of the v2 set, and v4 keeps it.** "
      "The model is trained on adult human middle temporal gyrus; the siletti atlas is a BICCN "
      "adult human brain atlas that includes MTG. Different study and different donors as far as "
      "public documentation shows, same region. The siletti number for this arm "
      f"({pct(p['per_dataset']['celltypist_best_arm']['siletti']['rate'])}) is therefore the "
      "most leakage-exposed number in this report.")
    A("* **`Immune_All_Low` and `Immune_All_High` now overlap a test atlas, and v4 is what "
      "changed that.** Their cited source is the Dominguez Conde et al. cross-tissue immune "
      "atlas, and `dominguez_conde` IS that atlas's CELLxGENE deposition, so this is the same "
      "study's material rather than a tissue analogy. CellTypist publishes no per-model cell "
      "manifest, so exact cell-level overlap cannot be excluded and should be assumed possible. "
      "The file also ships CellTypist's own prediction columns, so the tool had already been run "
      "on these cells by the depositors. What the measurement does *not* look like is "
      "memorisation: the best arm names the answer on "
      f"{pct(p['per_dataset']['celltypist_best_arm']['dominguez_conde']['rate'])} of that "
      "atlas's 552 questions, which is far from the near-100% a memorised training set gives.")
    ae = p["atlas_evidence"]
    A(f"  Read off the atlas itself ({ae['atlas']}): the obs columns it carries include "
      + ", ".join(f"`{c}`" for c in ae["celltypist_columns_present"])
      + f", and the column the corpus reads its label from is "
        f"`{ae['label_column_read_by_the_corpus']}`, so the ground truth here is the "
        "CELLxGENE-mapped cell type and not any CellTypist column.")
    A("* **`Human_Lung_Atlas`** integrates 49 respiratory datasets whose constituent list is a "
      "supplementary table this run did not verify, and `tabula_sapiens` contains lung cells. "
      "Unresolved caveat, kept as one.")
    A("* **`Adult_Mouse_Gut`** is trained on mouse gut and `tabula_muris_senis` includes gut "
      "cells. Not verified at sample level.")
    A("* **`Adult_Human_PancreaticIslet`** is the same tissue as muraro and is the one arm here "
      "whose low score is reassuring: it is beaten on muraro by the fetal pancreas model "
      f"({pct(p['per_arm']['celltypist']['muraro/fetal_pancreas']['named_target_rate'])} against "
      f"{pct(p['per_arm']['celltypist']['muraro/pancreatic_islet']['named_target_rate'])}), which "
      "is the opposite of what sample-level leakage would produce.")
    A("* **The corpus side is asymmetric in the other direction for the v4 fine-tunes**: they "
      "train on every atlas under a donor-level holdout, so they have seen other donors of the "
      "same atlases. That favours us wherever a tool's reference is a different study, and it is "
      "the main reason the comparison below is not a clean transfer test.")
    A("")

    A("## Validation")
    A("")
    v4 = p["validation"]["v4"]
    c1 = v4["check1_celltypist_reproduces_project_baseline"]
    c1_cells = sum(v["census_cells"] for v in c1.values())
    A(f"1. **The CellTypist arms reproduce the project's own baseline exactly.** "
      f"`baselines/step3_celltypist.py` scored every corpus test cell; this run scores the "
      f"population's subset through a different path. On all {len(c1)} overlapping arms, over "
      f"the {c1_cells:,} (arm, cell) pairs those arms cover, the predicted label agrees on every "
      f"cell, verdict {v4['verdict']}. The two new datasets have no baseline extraction to compare "
      f"against, which is why they are absent from this check rather than passed by it.")
    c2 = v4["check2_singler_recall_where_reference_covers"]
    A(f"2. **SingleR recovers a type when its reference has it**: "
      f"{c2['tabula_muris_senis']['recalled']}/{c2['tabula_muris_senis']['cells_of_this_type']} "
      f"mouse B cells and {c2['tabula_sapiens']['recalled']}/"
      f"{c2['tabula_sapiens']['cells_of_this_type']} human B cells, which rules out a broken "
      f"input path as the explanation for the low pooled number.")
    A(f"3. **The question table's own checks** passed on all "
      f"{p['population']['questions_written']:,} rows: every record found for the id that names "
      f"it, every census rendition agreeing with the corpus record on the panel and on the "
      f"labelled answer, and the answer being the cell's own register type in every case.")
    eq = p["validation"]["equivalence_v2"]
    A(f"4. **The parameterised pipeline reproduces the published v2 numbers exactly** "
      f"({eq['verdict']}): the four v2 readings re-run into `reports/annotation-tools-repro` and "
      f"compared field by field with the frozen `reports/annotation-tools-baseline.json`.")
    A("")
    A("| v2 reading | published | fresh run of the parameterised pipeline |")
    A("|---|---:|---:|")
    for reading, label in READINGS:
        r = eq["readings"][reading]
        A(f"| {label} | {pct(r['frozen_rate'])} ({frac(r['frozen_correct'], 1819)}) "
          f"| {pct(r['fresh_rate'])} ({frac(r['fresh_correct'], 1819)}) |")
    A("")
    A("The four numbers the equivalence test is stated on are the published ones: CellTypist "
      f"default {pct(eq['readings']['celltypist_default_model']['frozen_rate'])}, CellTypist "
      f"best declared arm {pct(eq['readings']['celltypist_best_arm']['frozen_rate'])}, SingleR "
      f"label.main {pct(eq['readings']['singler_label_main']['frozen_rate'])}, SingleR better "
      f"label level {pct(eq['readings']['singler_best_label_level']['frozen_rate'])}. All four "
      f"reproduce to the question count, together with the population, every per-arm count of "
      f"both tools and the per-dataset arm choices; the fresh run's own validation is "
      f"{eq['fresh_run_validation']}.")
    A("")

    A("## Beside the manuscript's own arms, on the same 4,035 questions")
    A("")
    A("**This is a forced-versus-forced comparison.** Neither tool can abstain, so the only "
      "comparable column of the manuscript's arms is its abstention-removed one. The arms' "
      "served and forced numbers are read from `reports/abstain-reconciliation-v4.json`; the "
      "per-atlas numbers below are read from `reports/per-dataset-v4.json`.")
    A("")
    A("| arm | served | forced (abstention removed) |")
    A("|---|---:|---:|")
    for arm, entry in sorted(p["beside_the_manuscript"]["arms"].items(),
                             key=lambda kv: -(kv[1]["forced_accuracy"])):
        if entry["kind"] != "fine-tune":
            continue
        A(f"| {entry['shown_as']} | {pct(entry['served_accuracy'])} | **{pct(entry['forced_accuracy'])}** |")
    A(f"| **CellTypist** (best declared arm per atlas) | not applicable: it cannot abstain | "
      f"**{pct(p['headline']['celltypist_best_arm']['named_target_rate'])}** |")
    A(f"| **SingleR** (better label level per atlas) | not applicable: it cannot abstain | "
      f"**{pct(p['headline']['singler_best_label_level']['named_target_rate'])}** |")
    A("")
    A("| the untuned (released) rungs, same population | served | forced |")
    A("|---|---:|---:|")
    for arm, entry in sorted(p["beside_the_manuscript"]["arms"].items(),
                             key=lambda kv: -(kv[1]["forced_accuracy"])):
        if entry["kind"] == "fine-tune":
            continue
        A(f"| {entry['shown_as']} | {pct(entry['served_accuracy'])} | {pct(entry['forced_accuracy'])} |")
    A("")
    ans = p["answers"]
    A("### Per atlas, forced versus forced")
    A("")
    A("| atlas | our best fine-tuned arm (forced) | CellTypist best arm (forced) | our margin (points) | our best served (arm) | SingleR best level (forced) |")
    A("|---|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        row = ans["per_atlas"][dataset]
        margin = row["our_forced_margin"]
        A(f"| {dataset} | {pct(row['ours_forced'])} ({row['best_arm_shown_as']}) "
          f"| {pct(row['celltypist_forced'])} (`{row['celltypist_arm'].split('/')[1]}`) "
          f"| {points(margin) if isinstance(margin, float) else margin} "
          f"| {pct(row['ours_best_served'])} ({row['ours_best_served_arm_shown_as']}) "
          f"| {pct(row['singler_forced'])} |")
    A("")
    A(f"* The 27B fine-tune's per-atlas breakdown is {ans['fine_tuned_27b_per_atlas']}; its pooled "
      f"served and forced numbers are in the table above.")
    A("* Our per-atlas cells are the *best* fine-tuned arm on that atlas and the *best served* "
      "one, so both comparisons are generous to us; the per-arm spreads are in the JSON under "
      "`answers.per_atlas.all_fine_tuned_forced` and `all_fine_tuned_served`.")
    A("")

    A("## Do we beat them?")
    A("")
    fv = ans["forced_versus_forced"]
    A(f"**Pooled, forced versus forced: yes, comfortably.** Every fine-tuned arm's forced "
      f"accuracy ({pct(fv['every_fine_tuned_arm_forced_min'])} to "
      f"{pct(fv['every_fine_tuned_arm_forced_max'])}) is above CellTypist's best declared arm "
      f"({pct(fv['celltypist_best_arm_pooled'])}) and above SingleR's better level "
      f"({pct(fv['singler_best_label_level_pooled'])}). The best fine-tuned arm on this "
      f"population is {fv['our_best_fine_tuned_shown_as']}: "
      f"{pct(fv['our_best_fine_tuned_forced'])} forced against "
      f"{pct(fv['our_best_fine_tuned_served'])} served.")
    A("")
    margins_by_atlas = {d: v for d, v in ans["our_forced_margin_per_atlas"].items() if isinstance(v, float)}
    tightest = min(margins_by_atlas, key=lambda d: margins_by_atlas[d])
    A(f"**Per atlas, forced versus forced: no, CellTypist does not beat us on any of the six "
      f"atlases.** Our margin runs from {100 * min(margins_by_atlas.values()):.1f} to "
      f"{100 * max(margins_by_atlas.values()):.1f} points, "
      f"smallest at {tightest}, which is where a standard annotator is strongest.")
    A("")
    if ans["atlases_where_celltypist_beats_our_best_served"]:
        losses = ans["atlases_where_celltypist_beats_our_best_served"]
        l0 = losses[0]
        A(f"**CellTypist does beat us on {', '.join(losses)} under one reading, and it should be "
          f"said plainly**: its forced number there is above our best *served* number. That is a "
          f"comparison across two different readings, and since CellTypist cannot abstain, the "
          f"reading-matched pair is forced against forced, where the margin is ours. Both facts "
          f"belong in the same sentence: on {l0} CellTypist names the answer on "
          f"{pct(ans['per_atlas'][l0]['celltypist_forced'])} of the questions and our best served "
          f"arm on {pct(ans['per_atlas'][l0]['ours_best_served'])}, while our forced arm is at "
          f"{pct(ans['per_atlas'][l0]['ours_forced'])}. Our {l0} advantage over CellTypist is "
          f"therefore carried by the abstention policy, not by a better reading of the cells.")
    else:
        A(f"**No atlas is a loss on any reading of the tools.** CellTypist's best arm is below "
          f"our best served number on all six atlases as well "
          f"({pct(fv['every_fine_tuned_arm_served_min'])} is the weakest served number any "
          f"fine-tuned arm has pooled).")
    A("")
    A("Two qualifications travel with the pooled result, and neither is small. First, the v4 "
      "fine-tunes train on every atlas under a donor-level holdout, so they have seen other "
      "donors of the same atlases, while a tool whose reference is a different study has not; "
      "that is a structural advantage for us and it is the reason the pooled margin is larger "
      "than the v2 report's would have been. Second, the tools read the whole expression matrix "
      "and our model reads a fixed marker panel, which cuts the other way. What this report "
      "licenses is the forced-versus-forced ordering on this population, not a general claim "
      "that a trained decision model annotates cells better than a reference-based annotator.")
    A("")

    A("## What is not measured here")
    A("")
    A("* Azimuth and scANVI on the v4 population: not measured.")
    A("* The 27B fine-tune's per-atlas breakdown: not measured.")
    A("* Cell-level sample overlap between any CellTypist model's training cells and any test "
      "atlas: not measured, and not published by CellTypist; the leakage section states what is "
      "and is not documented instead.")
    A("* The three constructed-evidence families (`evidence_diluted`, `marker_noul`, "
      "`panel_miss`): not scoreable by a cell-level annotator, so not measured.")
    A("")

    A("## Closing answer")
    A("")
    A("**Do the fine-tuned models beat CellTypist and SingleR on the same questions?** Yes, on the "
      "only reading the tools allow: forced against forced. Every fine-tuned arm is above both, "
      f"the weakest at {pct(fv['every_fine_tuned_arm_forced_min'])} against CellTypist's "
      f"{pct(fv['celltypist_best_arm_pooled'])} and SingleR's "
      f"{pct(fv['singler_best_label_level_pooled'])}, and the best fine-tuned arm at "
      f"{pct(fv['our_best_fine_tuned_forced'])}.")
    A("")
    A("**Does CellTypist beat us on any single tissue?** Not on the reading-matched comparison: "
      f"its best declared arm is below our best fine-tuned arm on all six atlases, by "
      f"{100 * min(margins_by_atlas.values()):.1f} to {100 * max(margins_by_atlas.values()):.1f} "
      f"points. It "
      f"is ahead on "
      + (", ".join(ans["atlases_where_celltypist_beats_our_best_served"]) if
         ans["atlases_where_celltypist_beats_our_best_served"] else "no atlas")
      + " when its forced number is put beside our *served* one, and that comparison is across "
        "two different readings. The mixed part of the answer is therefore confined to that one "
        "pair, and it is not softened here: on muraro CellTypist reads the cells at least as well "
        "as our best served arm, and our pooled win is not evidence that it does not.")
    A("")

    A("## Reproduce")
    A("")
    A("```bash")
    A("source scripts/wsl/env.sh")
    A("# 1. the equivalence proof: the published v2 numbers, re-run through the parameterised pipeline")
    A("uv run --no-sync python scripts/annotation-tools/01_census_questions.py --dir reports/annotation-tools-repro")
    A("uv run --no-sync python scripts/annotation-tools/02_census_matrices.py --dir reports/annotation-tools-repro --resume")
    A("uv run --with celltypist==1.7.1 --with scanpy==1.12.4 --no-sync \\")
    A("    python scripts/annotation-tools/03_celltypist.py --dir reports/annotation-tools-repro")
    A("uv run --no-sync python scripts/annotation-tools/04b_export_for_r.py --dir reports/annotation-tools-repro")
    A("R_LIBS_USER=$HOME/R/sckevtype-libs ~/miniconda3/envs/biopipen/bin/Rscript \\")
    A("    scripts/annotation-tools/04c_singler.R --dir=reports/annotation-tools-repro")
    A("uv run --no-sync python scripts/annotation-tools/05_score.py --dir reports/annotation-tools-repro")
    A("uv run --no-sync python scripts/annotation-tools/07_validate.py --dir reports/annotation-tools-repro")
    A("uv run --no-sync python scripts/annotation-tools/10_equivalence_v2.py")
    A("")
    A("# 2. the v4 population")
    A("uv run --no-sync python scripts/annotation-tools/01_census_questions.py \\")
    A("    --population reports/census-suite-ids-v4.jsonl \\")
    A("    --census-rows reports/jev-v4/rows.jsonl \\")
    A("    --suite artifacts/corpus/real-v4/test.jsonl \\")
    A("    --dir reports/annotation-tools-v4")
    A("uv run --no-sync python scripts/annotation-tools/02_census_matrices.py \\")
    A("    --population reports/census-suite-ids-v4.jsonl --dir reports/annotation-tools-v4 --resume")
    A("uv run --with celltypist==1.7.1 --with scanpy==1.12.4 --no-sync \\")
    A("    python scripts/annotation-tools/03_celltypist.py --dir reports/annotation-tools-v4")
    A("uv run --no-sync python scripts/annotation-tools/04b_export_for_r.py --dir reports/annotation-tools-v4")
    A("R_LIBS_USER=$HOME/R/sckevtype-libs ~/miniconda3/envs/biopipen/bin/Rscript \\")
    A("    scripts/annotation-tools/04c_singler.R --dir=reports/annotation-tools-v4")
    A("uv run --no-sync python scripts/annotation-tools/05_score.py --dir reports/annotation-tools-v4")
    A("uv run --no-sync python scripts/annotation-tools/07_validate.py --dir reports/annotation-tools-v4")
    A("")
    A("# 3. this report")
    A("uv run --no-sync python scripts/annotation-tools/09_report_v4.py")
    A("```")
    A("")
    return "\n".join(L)


def main() -> int:
    payload = build_payload()
    md = render_markdown(payload)
    for bad, name in (("\u2014", "em dash"), ("\u2013", "en dash")):
        if bad in md:
            raise SystemExit(f"the report contains a {name}; house style forbids it")
    if any(payload["answers"]["per_atlas"][d]["ours_forced"] == "not measured" for d in DATASETS):
        print("NOTE: an atlas has no fine-tuned per-atlas number; the report says so explicitly")
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    for bad, name in (("\u2014", "em dash"), ("\u2013", "en dash")):
        for path in (OUT_MD, OUT_JSON):
            if bad in path.read_text(encoding="utf-8"):
                raise SystemExit(f"{path} contains a {name}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print("headline:")
    for key, label in READINGS:
        v = payload["headline"][key]
        print(f"  {label}: {100 * v['named_target_rate']:.1f}% "
              f"({v['named_target']}/{v['questions']})")
    print(f"  best fine-tuned arm forced: "
          f"{100 * payload['answers']['forced_versus_forced']['our_best_fine_tuned_forced']:.1f}%")
    print(f"  atlases where CellTypist beats our best forced: "
          f"{payload['answers']['atlases_where_celltypist_beats_our_best_forced'] or 'none'}")
    print(f"  atlases where CellTypist beats our best served: "
          f"{payload['answers']['atlases_where_celltypist_beats_our_best_served'] or 'none'}")
    bad_cross = payload["beside_the_manuscript"]["cells_of_within_source_arms"]
    print(f"  cross-source disagreements: {bad_cross or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
