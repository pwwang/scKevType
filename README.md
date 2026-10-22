# scKevType — code deposit

Code for **scKevType**, a definition-conditioned decision model for single-cell cell-type
annotation: given a cell as a rank-ordered gene list and a candidate panel supplied at request
time (every option carrying a marker definition), it returns a calibrated probability per
option, a trained *insufficient evidence* answer, and a risk-controlled triage decision.

This repository is the code that produces the paper's numbers. Every number in the paper comes from
a run of these programs against the public atlases and corpora named in `configs/`, which a
reproduction fetches and builds first — see *Reproducing a run* and *What a run needs*.

**Model weights.** The fine-tuned model weights this code trains live in a separate deposit on
Hugging Face, <https://huggingface.co/pwwang/scKevType>; this repository is the code deposit.

## Layout

```
sckevtype/                the package — the only thing that produces results
  contracts/              frozen artifacts: record schema, the canonical state renderer,
                          vocabulary, CL ontology maps, definition resource, feature budget
  data/                   dataset adapters (CELLxGENE h5ad, 10x MTX, CSV, Baron archive),
                          the supervision factory (question families, panel sampler,
                          splits, sampling caps, QC probes), the corpus builder
  train/                  thin adapter over the pinned reference method (`kev`): model,
                          suite builder, the training CLI, sharding, capacity measurement
  eval/                   metrics (AUROC, ECE, NLL, Brier, AURC), panel-stratified conformal
                          sets, the risk-controlled triage gate, the scoring CLI
  jev/                    client for the Jev decision-model API (ask / evaluate / usage)
  laya/                   the Laya arm: render the corpus into Laya's item format, fine-tune,
                          score
tests/                    pytest: contracts, data, eval, jev, laya, train
tests/analysis/           pytest: what the analysis scripts compute
analysis/                 the statistics the paper's numbers come from: reconciliations, paired
                          tests and their p-values, calibration, per-dataset and per-family tables
configs/                  run configuration: dataset registry, model registry, corpus specs
                          (`real-v1..v4`, `baron-external`, `smoke`), register seeds
baselines/                the standard-annotator baselines (CellTypist, LR on HVGs, LR on the
                          corpus's own rank-list task shape)
scripts/
  llm/                    the general-LLM baseline: render a question into a chat prompt, ask a
                          local instruct model or a frontier model through OpenRouter, score
  jev/                    ask the Jev API, then score the answers on the census population
  kev/                    the pinned reference rung: probe a released `kev` checkpoint, LoRA
                          fine-tune a full-weight one, self-tests that need no GPU
  annotation-tools/       the CellTypist / SingleR census baseline pipeline
  9b-finetune-score.py    statistics from scored rows ...
  accuracy-table.py       the cross-engine accuracy table, one number at a time
  arms-forced.py          the fine-tuned arms' rows in the forced-naming table
  axis1-*.py              the axis-1 (panel adequacy) statistic, audit and recompute
  confusion-structure.py  what an engine names when it is wrong
  noul-accuracy.py        the noul column: binary marker-absence accuracy
  score-forced-rows.py    re-score one arm on a named population
  seedvar-*.py            seed-variance spread and the frozen-configuration check
  measure-*.py            token-budget measurements with the pinned tokenizer
  speed-cost/             the speed/cost measurement harness
```

## Install

```bash
uv sync                     # contracts / data / eval layers: no GPU, no model stack
uv sync --extra train       # + torch, transformers, peft, accelerate, datasets and `kev`
uv sync --extra data        # + anndata, h5py, requests (dataset adapters)
uv sync --extra eval        # + scikit-learn, pandas, pyarrow, matplotlib
uv sync --extra annotators  # + celltypist, scanpy, scipy (the standard-annotator baselines)
uv sync --extra laya        # + the `laya` package and its runtime
```

`import sckevtype.contracts` is deliberately cheap: the contracts layer must stay inspectable
on a CPU-only box, and there is a guard test that proves no torch/transformers/anndata is
pulled in. `kev` is likewise imported inside the functions that need it, which is why the base
install imports even without it.

The study ran under Python 3.12 on Linux (WSL2 / Slurm). Training needs CUDA: the Qwen3.5
bases are hybrid (Gated DeltaNet) and want `flash-linear-attention`.

## Reproducing a run

The corpus is built from public atlases that this deposit does not ship. In outline:

```bash
# 1. fetch the atlases named in configs/datasets.yaml into the data root
uv run python -m sckevtype.data.fetch

# 2. build the corpus for a config in configs/corpus/  ->  records + manifest
uv run python -m sckevtype.data.build --config configs/corpus/real-v4.yaml

# 3. build a trainable suite from it
uv run python -m sckevtype.train.suite  --corpus <corpus-dir> --out <suite-dir>

# 4. train (needs the train extra and a GPU)
uv run python -m sckevtype.train.train  --suite <suite-dir> --out <run-dir>

# 5. evaluate / score, writing report.json + rows.jsonl
uv run python -m sckevtype.eval.run     --suite <suite-dir> --run <run-dir> --out <eval-dir>

# 6. statistics on top of the scored rows
uv run python scripts/arms-forced.py --help
uv run python analysis/paired-tests-v4.py --help      # the statistics the paper quotes
```

Inference and API submission:

```bash
# ask the Jev API (key from the environment: JEV_KEY, never a file)
uv run python -m sckevtype.jev.probe ask --help

# ask a local instruct model the same questions, then score the answers
uv run python scripts/llm/run_llm.py --help
uv run python scripts/llm/score.py   --help

# ask frontier models through OpenRouter (key from OPENROUTER_API_KEY)
uv run python scripts/llm/run_frontier.py --help
```

Training and evaluation ran on a Slurm cluster with a shared filesystem. The submission scripts are not part of this deposit.

## What a run needs

* **Data.** The atlases come from `sckevtype.data.fetch` (public sources, named in
  `configs/datasets.yaml`), the corpora are built from them by `sckevtype.data.build`, and the
  frozen resources the package reads — vocabularies, CL terms, definitions, suites, the per-version
  corpora — are rebuilt from the seeds under `configs/seed/` by the builders in
  `sckevtype/contracts/`. Nothing here is downloaded from this repository.
* **Scored rows, for the statistics.** The scripts in `analysis/` read the row dumps and summary
  JSONs that steps 4-5 write under `reports/` and `runs/`, and write their tables back into
  `reports/`; run them from the repository root, after those steps, against a tree that has them.
  `INVENTORY.md` lists the inputs and the commands.
* **`kev/`, the reference method.** The architecture this deposit adapts is third-party code, pinned
  upstream at `2ba81661a7875e13295739a14c73578c1832705a` and declared as a git dependency in
  `pyproject.toml` (`[tool.uv.sources]`); it is imported lazily, so a base install runs without it.
  A local checkout can be dropped in as `kev/` with that source line switched to `path = "kev"`.

`INVENTORY.md` lists every file in the repository, section by section.

## License

MIT. See `LICENSE`.
