# Workflows

This page documents the main repository workflows using the current commands and repo-relative paths. Run commands from the repository root unless stated otherwise.

For the technical contract behind these workflows, see the [Technical Spec](spec.md).

## Installation

Install the base package:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Install the SLM stack:

```bash
pip install -e ".[slm]"
```

Install the optional MLOps extra:

```bash
pip install -e ".[mlops]"
```

Current extras from `pyproject.toml`:

- base: `requests`, `tqdm`
- `dev`: `pytest`
- `slm`: `torch`, `transformers`, `accelerate`, `peft`, `datasets`, `trl`
- `mlops`: `mlflow`

## Canonical Data Preparation

Create the canonical processed dataset and split manifest:

```bash
python -m recipe_mpr_qa.cli prepare-data \
  --input data/500QA.json \
  --output data/processed/recipe_mpr_qa.jsonl \
  --split-output data/processed/primary_split.json
```

Validate the raw dataset:

```bash
python -m recipe_mpr_qa.cli validate-data \
  --input data/500QA.json \
  --kind raw
```

Validate the canonical processed dataset:

```bash
python -m recipe_mpr_qa.cli validate-data \
  --input data/processed/recipe_mpr_qa.jsonl \
  --kind prepared
```

Print dataset metadata:

```bash
python -m recipe_mpr_qa.cli dataset-stats \
  --input data/processed/recipe_mpr_qa.jsonl \
  --kind prepared
```

## Split Export

Export one split from the committed split manifest:

```bash
python -m recipe_mpr_qa.cli export-split \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --split train \
  --output data/processed/train.jsonl
```

Valid split names are `train`, `validation`, and `test`.

## Train-Only Augmentation

Create a train-only augmentation artifact:

```bash
python -m recipe_mpr_qa.cli augment-train \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/train_augmented.jsonl \
  --max-variants 2
```

Current behavior:

- reads the canonical processed dataset
- selects only train parents from the split manifest
- rewrites queries conservatively
- preserves options, labels, query-type flags, and correctness explanations
- writes only synthetic `RecipeExample` rows

The augmentation artifact is optional. If you do not pass it later, the rest of the pipeline remains unchanged.

## LLM Evaluation

The repository contains a multiple-choice evaluation script in `llm_evaluation/mc_eval.py`.

Run evaluation from the repository root with explicit paths:

```bash
python llm_evaluation/mc_eval.py \
  --model deepseek-r1:7b \
  --backend ollama \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Evaluate a different split:

```bash
python llm_evaluation/mc_eval.py \
  --model deepseek-r1:7b \
  --backend ollama \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json \
  --split validation
```

Limit the run length:

```bash
python llm_evaluation/mc_eval.py \
  --model deepseek-r1:7b \
  --backend ollama \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json \
  --limit 10
```

Run with the Hugging Face backend:

```bash
python llm_evaluation/mc_eval.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backend huggingface \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Notes:

- `--model` is required
- default `--backend` is `ollama`
- default `--split` is `test`
- if `--output` is omitted, the script writes `results/<Model>_<Split>_<N>.json` relative to the current working directory
- `llm_evaluation/config.json` currently carries the local Ollama URL and default temperature

## SLM Fine-Tuning

The fine-tuning entrypoint is `finetuning/finetune.py`.

Run with defaults:

```bash
python finetuning/finetune.py
```

Run with an explicit model and output directory:

```bash
python finetuning/finetune.py \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --output-dir outputs/smollm2_recipe_mpr
```

Include the optional train-only augmentation artifact:

```bash
python finetuning/finetune.py \
  --augmented-train-path data/processed/train_augmented.jsonl
```

Important current behavior:

- fine-tuning reads the canonical processed dataset and split manifest by default
- `--augmented-train-path` appends extra examples only to the training split
- validation and test splits remain unchanged
- model-facing prompts use deterministic per-example option shuffling
- the script trains a causal LM in prompt-completion form, where the completion is the correct answer letter

## Tracked MLOps Wrappers

The repository also exposes tracked wrappers through `recipe-mpr-qa`. These commands add run manifests, registries, and artifact lineage on top of the existing training and evaluation workflow without changing the direct scripts.

Run a tracked training job:

```bash
recipe-mpr-qa run-train \
  --stage candidate \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-path data/processed/recipe_mpr_qa.jsonl \
  --split-manifest-path data/processed/primary_split.json
```

Run a tracked evaluation job:

```bash
recipe-mpr-qa run-eval \
  --stage baseline \
  --backend ollama \
  --model deepseek-r1:7b \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

List tracked runs:

```bash
recipe-mpr-qa list-runs
```

Compare tracked runs:

```bash
recipe-mpr-qa compare-runs --run-id train-... --run-id eval-...
```

Promote a tracked run:

```bash
recipe-mpr-qa promote-run --run-id train-... --stage validated
```

Optional MLflow mirroring:

```bash
recipe-mpr-qa run-eval \
  --enable-mlflow \
  --mlflow-experiment recipe-mpr-qa \
  --backend ollama \
  --model deepseek-r1:7b \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

## Common Output Locations

Source-of-truth artifacts:

- `data/processed/recipe_mpr_qa.jsonl`
- `data/processed/primary_split.json`

Optional generated artifacts:

- `data/processed/train_augmented.jsonl`
- `data/processed/train.jsonl`
- `llm_evaluation/results/*.json`
- `mlops/runs/*/run_manifest.json`
- `mlops/registry/runs.json`
- `mlops/registry/models.json`
- `outputs/smollm2_recipe_mpr/`

Interpretation notes:

- the canonical processed dataset and split manifest are the primary stable artifacts
- augmentation output is derived and optional
- evaluation results and training outputs are run artifacts and must be interpreted in light of the current experiment-status caveats
