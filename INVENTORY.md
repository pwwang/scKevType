# INVENTORY

What this deposit contains, what it does not, and why. All paths are relative to this
repository's root.

**Scope.** The deposit is the code that produces the paper's numbers: the package, the training /
evaluation / inference runners, the harness that submits requests to the Jev API and to frontier
models, the Slurm plumbing that actually trains and scores, and the analysis that computes the
paper's statistics from scored outputs — with the tests for each layer. Data and results are not
part of it; `README.md` says what a run has to supply.

## Included

### sckevtype/ — the package itself: contracts (frozen artifacts, canonical state renderer, vocabulary, CL ontology maps, definitions, feature budget), data (atlas adapters, supervision factory, corpus builder), train (model, suite builder, training CLI, sharding, capacity), eval (metrics, conformal sets, triage gate, scoring CLI), jev (API client, ask/evaluate, usage), laya (render into Laya's item format, fine-tune, score). Every module any of these imports is inside this tree.  (49 files)

- `sckevtype/__init__.py`
- `sckevtype/contracts/__init__.py`
- `sckevtype/contracts/budget.py`
- `sckevtype/contracts/build.py`
- `sckevtype/contracts/definitions.py`
- `sckevtype/contracts/fetch.py`
- `sckevtype/contracts/freeze.py`
- `sckevtype/contracts/ontology.py`
- `sckevtype/contracts/paths.py`
- `sckevtype/contracts/register.py`
- `sckevtype/contracts/render.py`
- `sckevtype/contracts/schema.py`
- `sckevtype/contracts/vocab.py`
- `sckevtype/data/__init__.py`
- `sckevtype/data/adapters/__init__.py`
- `sckevtype/data/adapters/baron_archive.py`
- `sckevtype/data/adapters/cellxgene_h5ad.py`
- `sckevtype/data/adapters/csv_expr.py`
- `sckevtype/data/adapters/tenx_mtx.py`
- `sckevtype/data/armb.py`
- `sckevtype/data/augment.py`
- `sckevtype/data/build.py`
- `sckevtype/data/depth.py`
- `sckevtype/data/families.py`
- `sckevtype/data/fetch.py`
- `sckevtype/data/panels.py`
- `sckevtype/data/pull.py`
- `sckevtype/data/qc.py`
- `sckevtype/data/registry.py`
- `sckevtype/data/splits.py`
- `sckevtype/eval/__init__.py`
- `sckevtype/eval/conformal.py`
- `sckevtype/eval/metrics.py`
- `sckevtype/eval/run.py`
- `sckevtype/eval/triage.py`
- `sckevtype/jev/__init__.py`
- `sckevtype/jev/client.py`
- `sckevtype/jev/probe.py`
- `sckevtype/jev/request.py`
- `sckevtype/laya/__init__.py`
- `sckevtype/laya/render.py`
- `sckevtype/laya/score.py`
- `sckevtype/laya/train_ddp.py`
- `sckevtype/train/__init__.py`
- `sckevtype/train/compute.py`
- `sckevtype/train/model.py`
- `sckevtype/train/shard.py`
- `sckevtype/train/suite.py`
- `sckevtype/train/train.py`

### tests/ — pytest suites for the layers above (contracts / data / eval / jev / laya / train). Tests needing a GPU or the train extra are marked `slow`/`network` in `tests/conftest.py`.  (58 files) (`tests/analysis/` — the ten modules covering the statistics scripts — is listed in its own section below.)

- `tests/conftest.py`
- `tests/contracts/test_definitions.py`
- `tests/contracts/test_fetch.py`
- `tests/contracts/test_freeze.py`
- `tests/contracts/test_import_guard.py`
- `tests/contracts/test_ontology.py`
- `tests/contracts/test_register.py`
- `tests/contracts/test_render.py`
- `tests/contracts/test_schema.py`
- `tests/contracts/test_vocab.py`
- `tests/data/conftest.py`
- `tests/data/test_adapters.py`
- `tests/data/test_armb.py`
- `tests/data/test_atlas_access.py`
- `tests/data/test_augment.py`
- `tests/data/test_baron_archive.py`
- `tests/data/test_build.py`
- `tests/data/test_depth.py`
- `tests/data/test_families.py`
- `tests/data/test_fetch.py`
- `tests/data/test_panel_vocabulary.py`
- `tests/data/test_panels.py`
- `tests/data/test_qc.py`
- `tests/data/test_real_config.py`
- `tests/data/test_registry.py`
- `tests/data/test_sources.py`
- `tests/data/test_splits.py`
- `tests/data/test_step8.py`
- `tests/data/test_tissue_map_covers_atlas_labels.py`
- `tests/data/test_v4_corpus.py`
- `tests/eval/conftest.py`
- `tests/eval/test_axis1_rows.py`
- `tests/eval/test_conformal.py`
- `tests/eval/test_forced_rows_rescore.py`
- `tests/eval/test_guard.py`
- `tests/eval/test_metrics.py`
- `tests/eval/test_noul_row_dump.py`
- `tests/eval/test_row_dump_full.py`
- `tests/eval/test_run_cli.py`
- `tests/eval/test_triage.py`
- `tests/jev/test_client.py`
- `tests/jev/test_probe.py`
- `tests/laya/test_count_jsonl.py`
- `tests/laya/test_finetune_reconcile.py`
- `tests/laya/test_render.py`
- `tests/laya/test_render_geometry.py`
- `tests/laya/test_score.py`
- `tests/train/conftest.py`
- `tests/train/test_config_provenance.py`
- `tests/train/test_fullft_load.py`
- `tests/train/test_import_guard.py`
- `tests/train/test_init_from.py`
- `tests/train/test_model.py`
- `tests/train/test_resume.py`
- `tests/train/test_shard.py`
- `tests/train/test_suite.py`
- `tests/train/test_token_budget.py`
- `tests/train/test_train.py`

### configs/ — run configuration: the dataset registry and model registry (read by `sckevtype.data.*` and `sckevtype.train.compute`), the corpus specs (`real-v1..v4`, `baron-external`, `smoke`) and the register seeds `sckevtype.contracts.build` requires (`SEED_FILES`), for every version the corpus configs still name. Inputs, not results.  (29 files)

The two measured atlas inventories per seed version (`atlas_labels.json`, `atlas_tissue_labels.json`) are written by `scripts/curate_register_v2.py` and `scripts/derive_tissue_map.py` against the atlases, and the generated smoke-fixture data by `configs/corpus/fixtures/smoke/generate.py`; both are regenerable from those inputs rather than carried as files.


- `configs/corpus/baron-external.yaml`
- `configs/corpus/cellxgene-assets.yaml`
- `configs/corpus/fixtures/smoke/generate.py`
- `configs/corpus/real-v1.yaml`
- `configs/corpus/real-v2.yaml`
- `configs/corpus/real-v3.yaml`
- `configs/corpus/real-v4.yaml`
- `configs/corpus/smoke.yaml`
- `configs/datasets.yaml`
- `configs/models.yaml`
- `configs/seed/cl_terms.seed.yaml`
- `configs/seed/cl_terms.v2.yaml`
- `configs/seed/definitions.seed.yaml`
- `configs/seed/filter.seed.yaml`
- `configs/seed/genes.seed.txt`
- `configs/seed/label_map.seed.yaml`
- `configs/seed/tissue_map.seed.yaml`
- `configs/seed/v2/cl_terms.seed.yaml`
- `configs/seed/v2/definitions.seed.yaml`
- `configs/seed/v2/filter.seed.yaml`
- `configs/seed/v2/genes.seed.txt`
- `configs/seed/v2/label_map.seed.yaml`
- `configs/seed/v2/tissue_map.seed.yaml`
- `configs/seed/v3/cl_terms.seed.yaml`
- `configs/seed/v3/definitions.seed.yaml`
- `configs/seed/v3/filter.seed.yaml`
- `configs/seed/v3/genes.seed.txt`
- `configs/seed/v3/label_map.seed.yaml`
- `configs/seed/v3/tissue_map.seed.yaml`

### baselines/ — the standard-annotator baselines the paper compares against: CellTypist, logistic regression on HVGs, and logistic regression on the corpus's own rank-list task shape — run step by step and scored through one shared scorer — plus their self-tests and two A/B scripts that justify the LR's configuration. `baselines/celltypist_label_map.json` is the CellTypist-label → CL-id harmonisation table those steps read (a mapping input, not a measurement).  (13 files)

- `baselines/README.md`
- `baselines/ab_lr_config.py`
- `baselines/ab_scaling.py`
- `baselines/celltypist_label_map.json`
- `baselines/lib_common.py`
- `baselines/models_info.py`
- `baselines/selftest.py`
- `baselines/step1_cells.py`
- `baselines/step2_expr.py`
- `baselines/step3_celltypist.py`
- `baselines/step4_lr.py`
- `baselines/step5_score.py`
- `baselines/step6_panel_lr.py`

### scripts/llm/ — the general-LLM baseline's serving harness: render a decision-model question into a chat prompt and parse the answer back (`render.py`), ask a local instruct model (`run_llm.py`) or a frontier model through OpenRouter with the key read from `OPENROUTER_API_KEY` (`run_frontier.py`), lift the ask set out of an existing raw ask tree (`build_asks.py`), prove an ask set is the decision models' own questions (`verify-asks.py`), score answers exactly as the decision models are scored (`score.py`), fold a pass into one comparable record / summary (`make_row.py`, `assemble.py`).  (8 files)

- `scripts/llm/assemble.py`
- `scripts/llm/build_asks.py`
- `scripts/llm/make_row.py`
- `scripts/llm/render.py`
- `scripts/llm/run_frontier.py`
- `scripts/llm/run_llm.py`
- `scripts/llm/score.py`
- `scripts/llm/verify-asks.py`

### scripts/jev/ — Jev API submission and scoring: draw the census populations the asks are taken on (`census-sample-v4.py`, `census-sample-baron.py`), the end-to-end ask pipeline with the key read from `JEV_KEY` (`run-smoke.sh`), score Jev's answers on the census population under the settled forced-abstention rules (`score-census-v4.py`, `score-noul-v4.py`), test whether batching families changes the answers (`equivalence.py`), and the probe/census superset check (`subset-check.py`).  (8 files)

- `scripts/jev/census-sample-baron.py`
- `scripts/jev/census-sample-v4.py`
- `scripts/jev/equivalence.py`
- `scripts/jev/jev-v4-ask.sh`
- `scripts/jev/run-smoke.sh`
- `scripts/jev/score-census-v4.py`
- `scripts/jev/score-noul-v4.py`
- `scripts/jev/subset-check.py`

### scripts/kev/ — the pinned reference rung: run the released reconstruction through the same encoder and pointer head our questions use and dump axis-1 rows (`kev_probe.py`), the full-weight load path (`kev_fullft.py`), LoRA fine-tuning on a full-weight checkpoint (`kev_fullft_train.py`), self-tests proving both paths without a GPU (`selftest.py`, `kev_fullft_selftest.py`), and the 27B feasibility analysis (memory/time).  (6 files)

- `scripts/kev/27b-feasibility-analysis.py`
- `scripts/kev/kev_fullft.py`
- `scripts/kev/kev_fullft_selftest.py`
- `scripts/kev/kev_fullft_train.py`
- `scripts/kev/kev_probe.py`
- `scripts/kev/selftest.py`

### scripts/annotation-tools/ — the census baseline pipeline that runs the standard annotators (CellTypist, SingleR, Azimuth) on the census cells and scores them on the corpus's own questions: probes, question/matrix extraction, the annotator runs including the R references, scoring, the harmonisation sensitivity check and the validation step.  (13 files)

- `scripts/annotation-tools/00_probe_tabula_sapiens.py`
- `scripts/annotation-tools/00b_probe_tissues.py`
- `scripts/annotation-tools/01_census_questions.py`
- `scripts/annotation-tools/02_census_matrices.py`
- `scripts/annotation-tools/03_celltypist.py`
- `scripts/annotation-tools/04a_singler_references.R`
- `scripts/annotation-tools/04b_export_for_r.py`
- `scripts/annotation-tools/04c_singler.R`
- `scripts/annotation-tools/05_score.py`
- `scripts/annotation-tools/06_harmonisation_sensitivity.py`
- `scripts/annotation-tools/07_validate.py`
- `scripts/annotation-tools/09_azimuth_siletti.R`
- `scripts/annotation-tools/singler_label_map.py`

### scripts/ — analysis and measurement (top level) — statistics computed from scored rows, and measurements taken on the training path: the cross-engine accuracy table, the forced-naming rows, the axis-1 audit and recompute, confusion structure, the noul column, per-seed spread and the frozen-configuration check, the token-budget measurements, the hardware/memory probes, and the panel-swap extraction.  (27 files)

- `scripts/9b-finetune-score.py`
- `scripts/accuracy-table.py`
- `scripts/arms-forced.py`
- `scripts/axis1-audit.py`
- `scripts/axis1-recompute.py`
- `scripts/bench_train.py`
- `scripts/cl-depth-report.py`
- `scripts/confusion-structure.py`
- `scripts/curate_register_v2.py`
- `scripts/depth-register-extension.py`
- `scripts/derive_tissue_map.py`
- `scripts/dump-manifest.py`
- `scripts/extra-seeds-score.py`
- `scripts/gpu-bench.py`
- `scripts/kev-init-score.py`
- `scripts/measure-branch-tokens.py`
- `scripts/measure-row-form.py`
- `scripts/measure-row-tokens.py`
- `scripts/measure-state-tokens.py`
- `scripts/noul-accuracy.py`
- `scripts/probe-activation-memory.py`
- `scripts/score-forced-rows.py`
- `scripts/seedvar-config-diff.py`
- `scripts/seedvar-spread.py`
- `scripts/step10-acceptance.sh`
- `scripts/step10-train-verify.sh`
- `scripts/verify-real-v2.sh`

### scripts/ — remaining subdirectories — learning-curve suite builder, the speed/cost measurement harness, the panel-swap builder, and the WSL/Linux environment bootstrap the README names.  (15 files)

- `scripts/baselines/capped_population.py`
- `scripts/baselines/extend_label_map.py`
- `scripts/eval/build_panel_swaps.py`
- `scripts/learning-curve/build-suites.py`
- `scripts/speed-cost/census-crosscheck.sh`
- `scripts/speed-cost/collect_timing.py`
- `scripts/speed-cost/dump_stats.py`
- `scripts/speed-cost/io-probe.sh`
- `scripts/speed-cost/jev-timing.sh`
- `scripts/speed-cost/kev-timing-task.sh`
- `scripts/speed-cost/laya-timing-task.sh`
- `scripts/speed-cost/laya-timing.py`
- `scripts/speed-cost/submit-speed-cost.sh`
- `scripts/wsl/bootstrap.sh`
- `scripts/wsl/env.sh`

### scripts/ — corpus/register builders — build the corpus's own inputs: curate the v2 register (coverage, grounded definitions, seeds), derive the tissue map by rule, and extend the register for the depth layer.  (3 files)

- `scripts/curate_register_v2.py`
- `scripts/depth-register-extension.py`
- `scripts/derive_tissue_map.py`

### Repository metadata — build metadata, lock file, reference requirements

- `pyproject.toml`, `uv.lock`, `.python-version`, `cluster/reqs-cluster.txt`

## analysis/ — the statistics behind the paper

`analysis/` holds the fifteen scripts that compute the paper's statistics from scored rows, and
`tests/analysis/` the ten pytest modules that pin their arithmetic and their refusals. Between them
they cover: the reconciliation of every served / abstain-pick / forced reading on the census-matched
populations and on the external-validation (Baron) suite; the paired tests and the p-values the paper
quotes, including the exact two-sided binomial on discordant pairs the headline comparisons use; the
calibration (ECE / Brier) tables; the per-dataset and per-family breakdowns; the census-versus-corpus
checks; the Laya joins; the learning-curve scoring; and the variant-redundancy audit that asks
whether a duplicated `panel_choice` cell is load-bearing.

### Running them

```bash
uv run --no-sync python analysis/baron-reconciliation.py --help
uv run --no-sync python analysis/paired-tests-v4.py
uv run pytest tests/analysis -q
```

Every script is standalone: it reads its inputs by path relative to the repository root and exits on
the first one it cannot find — there is no fallback and no bundled sample.

**Inputs.** The scripts read the material a scoring run produces: the KeV and fine-tuned arm dumps
(`forced-rows-*.jsonl`, `rows-full-*.jsonl`), the census id lists (`census-suite-ids*.jsonl`), the
Jev and general-LLM row dumps (`jev-*/rows.jsonl`, `llm-baseline-*.jsonl`), the corpus and register
artefacts, and the summary JSONs of the passes that ran before them. In the working repository those
live under `reports/` and `runs/` alongside the scripts, and the scripts write their tables and their
markdown back into `reports/`. A run therefore needs that tree in place first — it is what steps 1-5
of *Reproducing a run* in `README.md` produce.

**How they read each other.** Several of these scripts extend a settled one rather than restate it:
the statistical rule and the arm roster live in a single script, and the scripts that build on it
parse that sibling's *source text* and execute the functions it defines instead of importing it —
a settled script re-runs its own aggregation at import time, so importing it is not an option. Those
references are paths resolved from the repository root into `analysis/` (`ROOT / "analysis" /
"<name>.py"`), so they resolve wherever the repository is checked out.

### analysis/ — the statistics scripts  (15 files)

- `analysis/abstain-reconciliation.py`
- `analysis/abstain-reconciliation-v4.py`
- `analysis/baron-reconciliation.py`
- `analysis/calibration-v4-baron.py`
- `analysis/family-reconciliation-v4.py`
- `analysis/laya-finetune-reconcile.py`
- `analysis/laya-finetune-table.py`
- `analysis/laya-v4-noul-reconcile.py`
- `analysis/laya-v4-reconcile.py`
- `analysis/learning-curve-score.py`
- `analysis/paired-tests-baron.py`
- `analysis/paired-tests-v4.py`
- `analysis/per-dataset-v4.py`
- `analysis/real-v3-corpus.py`
- `analysis/variant-redundancy.py`

### tests/analysis/ — their tests  (10 files)

- `tests/analysis/test_abstain_reconciliation_binom.py`
- `tests/analysis/test_accuracy_table.py`
- `tests/analysis/test_arms_forced.py`
- `tests/analysis/test_baron_reconciliation.py`
- `tests/analysis/test_calibration_v4_baron.py`
- `tests/analysis/test_confusion_structure.py`
- `tests/analysis/test_laya_v4_noul_reconcile.py`
- `tests/analysis/test_noul_accuracy.py`
- `tests/analysis/test_paired_tests_baron.py`
- `tests/analysis/test_paired_tests_v4.py`
