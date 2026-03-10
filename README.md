# Recipe-MPR QA

This repository implements an end-to-end experiment stack for comparing specialized small language models against general LLMs on the Recipe-MPR multiple-choice recipe recommendation dataset.

The codebase is structured so the full workflow can be developed and validated locally, then executed on a GPU machine for actual experiments. That includes deterministic data preparation, augmentation artifact generation, DistilBERT baselines, SmolLM2-style causal SLM baselines, fine-tuning, Ollama inference, LLM-as-a-judge evaluation, and MLflow logging.

## What Is In Scope

- Canonical processed dataset and fixed split manifest
- Config-driven experiment commands
- Persisted synthetic augmentation datasets
- Vanilla DistilBERT embedding-similarity baseline
- DistilBERT fine-tuning as a query-option scorer
- SmolLM2-style chat-template SLM baseline
- LoRA-ready causal SLM fine-tuning with letter-only supervision
- Ollama-backed multiple-choice baseline inference
- Ollama-backed LLM-as-a-judge evaluation
- Structured run artifacts and summary manifests
- Optional MLflow integration

## What Gets Run Where

Local development:

- validate schemas and contracts
- regenerate processed artifacts
- run dry-run and mocked integration tests
- verify CLI orchestration without training or live model servers

GPU or model-serving environment:

- execute DistilBERT training runs
- execute SmolLM2 causal SLM baselines and LoRA fine-tuning runs
- evaluate real checkpoints
- generate synthetic data with an Ollama model
- run LLM baseline inference and judge evaluation

## Quickstart

Install the package in editable mode before using the CLI from a fresh clone:

```powershell
python -m pip install -e ".[dev]"
```

For the full experiment stack on a GPU or model-serving machine:

```powershell
python -m pip install -e ".[train,llm,tracking,dev]"
```

Prepare the canonical dataset and split manifest:

```powershell
recipe-mpr-qa prepare-data `
  --input data/500QA.json `
  --output data/processed/recipe_mpr_qa.jsonl `
  --split-output data/processed/primary_split.json
```

Run the vanilla DistilBERT baseline with the default config:

```powershell
recipe-mpr-qa evaluate-slm --config configs/slm_vanilla.toml
```

Generate augmentation artifacts through Ollama:

```powershell
recipe-mpr-qa generate-augmentation --config configs/augmentation.toml
```

Train the fine-tuned DistilBERT model:

```powershell
recipe-mpr-qa train-slm --config configs/slm_finetune_original.toml
```

Run the SmolLM2-style causal SLM baseline:

```powershell
recipe-mpr-qa evaluate-slm --config configs/slm_smollm2_baseline.toml
```

Run the general LLM baseline:

```powershell
recipe-mpr-qa run-llm --config configs/llm_baseline.toml
```

Judge model predictions:

```powershell
recipe-mpr-qa judge-predictions `
  --config configs/judge.toml `
  --predictions artifacts/runs/llm-baseline/llm/test_predictions.jsonl
```

Run the non-training test suite:

```powershell
pytest
```

## Repo Structure

- `data/500QA.json`: raw dataset source of truth
- `data/processed/`: committed normalized dataset and split manifest
- `configs/`: reusable TOML experiment configs
- `docs/spec.md`: detailed project specification and contracts
- `src/recipe_mpr_qa/data`: schemas, validation, loaders, and split generation
- `src/recipe_mpr_qa/slm`: DistilBERT pipelines plus causal SmolLM2-style SLM support
- `src/recipe_mpr_qa/llm`: prompts, Ollama client, inference, and judge logic
- `src/recipe_mpr_qa/evaluation`: prediction and judgment records, metrics, summaries
- `src/recipe_mpr_qa/tracking`: MLflow adapter
- `artifacts/runs/<run_id>/`: generated configs, checkpoints, predictions, metrics, and summaries

## Current Experiment Matrix

- `SLM vanilla`: pretrained DistilBERT embedding similarity, no task training
- `SLM finetune`: DistilBERT cross-encoder over query-option pairs
- `SLM finetune + augmentation`: same model trained with original plus synthetic queries
- `Causal SLM baseline`: SmolLM2-style instruct model prompted through a chat template and parsed as a single letter
- `Causal SLM finetune`: chat-template supervised fine-tuning with optional LoRA adapters
- `General LLM`: Ollama-served multiple-choice baseline using the shared prompt format
- `LLM judge`: Ollama-served rubric scorer over model predictions

## Artifacts

Each run writes to `artifacts/runs/<run_id>/` with a stable layout:

- `configs/`: resolved config snapshots
- `augmentation/`: synthetic dataset artifacts and augmentation metrics
- `slm/`: SLM predictions, metrics, and checkpoints
- `llm/`: baseline LLM predictions and metrics
- `judge/`: judgment records and metrics
- `manifests/run_summary.json`: machine-readable run summary

## Dependencies

Base install:

- standard library only for data preparation, configs, and schema tooling

Optional extras:

- `.[train]` for `torch`, `transformers`, `datasets`, `accelerate`, and `peft`
- `.[llm]` for `requests`
- `.[tracking]` for `mlflow`

## Notes

- The processed dataset and primary split manifest are committed and treated as stable contracts.
- The branch is designed to be GPU-ready without forcing heavy training or live model calls in the local test suite.
- The SLM stack now supports both the original DistilBERT path and a SmolLM2-style causal chat path inspired by the notebook work on `main`.
- DVC is intentionally not implemented in this pass.
