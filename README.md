# Recipe-MPR QA

## Overview

This repository contains the current foundation for the Recipe-MPR project: dataset preparation, deterministic splits, tokenizer-ready loading, and standardized prompt/output formatting for model runs.

The broader project compares specialized small language models against general LLMs on the Recipe-MPR multiple-choice recipe recommendation dataset. The code currently in the repo establishes the shared data contract and workflow for dataset preparation and model-facing inputs.

## Current Scope

- Raw dataset validation and normalization
- Canonical processed JSONL artifact with stable example ids
- Deterministic 70/15/15 split manifest stratified by query-type signature
- Tokenizer-ready option-scoring loader for downstream consumers
- Standardized multiple-choice prompt formatting and output record schema
- CLI commands for prepare, validate, stats, and split export
- Regression tests for all implemented functionality

## Working With The Data

### Quickstart

Create the processed dataset and split manifest:

```powershell
python -m recipe_mpr_qa.cli prepare-data `
  --input data/500QA.json `
  --output data/processed/recipe_mpr_qa.jsonl `
  --split-output data/processed/primary_split.json
```

Validate the raw dataset:

```powershell
python -m recipe_mpr_qa.cli validate-data --input data/500QA.json --kind raw
```

Export the training split:

```powershell
python -m recipe_mpr_qa.cli export-split `
  --dataset data/processed/recipe_mpr_qa.jsonl `
  --split-manifest data/processed/primary_split.json `
  --split train `
  --output data/processed/train.jsonl
```

Create an optional augmented training artifact:

```powershell
python -m recipe_mpr_qa.cli augment-train `
  --dataset data/processed/recipe_mpr_qa.jsonl `
  --split-manifest data/processed/primary_split.json `
  --output data/processed/train_augmented.jsonl `
  --max-variants 2
```

Run the test suite:

```powershell
pytest
```

### LLM Evaluation

Evaluate any Ollama model on the prepared dataset splits. Requires Ollama running locally and the package installed (`pip install -e .`).

```bash
cd llm_evaluation

# Evaluate on the test split — output auto-named results/smollm2_135m_test_75.json
python mc_eval.py --model smollm2:135m

# Evaluate on a different split
python mc_eval.py --model smollm2:135m --split train

# Use a different model
python mc_eval.py --model deepseek-r1:7b

# Run only the first 10 questions (useful for quick testing)
python mc_eval.py --model smollm2:135m --limit 10

# Explicit output path overrides auto-naming
python mc_eval.py --model smollm2:135m --output results/custom_name.json
```

Output files are auto-named as `results/<Model>_<Split>_<N>.json` (e.g. `deepseek-r1_7b_test_10.json`). Use `--output` to override.

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Ollama model name (e.g. `smollm2:135m`, `deepseek-r1:7b`) |
| `--output` | auto | Output JSON path (default: `results/<Model>_<Split>_<N>.json`) |
| `--split` | `test` | Which split to evaluate: `train`, `validation`, or `test` |
| `--limit` | all | Limit the number of questions to evaluate |
| `--data` | `../data/processed/recipe_mpr_qa.jsonl` | Path to prepared dataset |
| `--split-manifest` | `../data/processed/primary_split.json` | Path to split manifest |
| `--config` | `config.json` | Path to Ollama config (URL, temperature) |

**Outputs:** A JSON file with per-question results and accuracy metrics breakdown by query type.

### Current Data Outputs

- `data/processed/recipe_mpr_qa.jsonl`: canonical normalized dataset artifact
- `data/processed/primary_split.json`: deterministic 70/15/15 split manifest
- `data/processed/train_augmented.jsonl`: optional synthetic train-only artifact generated from the canonical dataset

## Project Structure

- `data/500QA.json`: raw Recipe-MPR dataset
- `docs/spec.md`: current project foundation specification
- `src/recipe_mpr_qa/data`: preparation, validation, splits, and loader interfaces
- `src/recipe_mpr_qa/formats.py`: prompt formatting, response parsing, and prediction record schema
- `src/recipe_mpr_qa/cli.py`: command-line entrypoints for dataset preparation and inspection
- `llm_evaluation`: original Ollama-based evaluation helpers preserved from the initial repo
- `tests`: regression coverage for the current implementation

## Current Implementation

- Canonical example schema with stable `example_id` values
- Deterministic stratified split manifest
- Optional training-only query augmentation artifact with conservative query rewrites
- Tokenizer-ready option-scoring loader for query-option scoring workflows
- Standardized multiple-choice prompt format and parser for LLM outputs
- Canonical prediction record schema for consistent model outputs
- CLI workflow for preparing, validating, inspecting, and exporting dataset artifacts

## Interfaces And Contracts

### Core Data Types

- `RecipeExample`: one normalized Recipe-MPR question with five ordered options
- `PreparedDataset`: validated collection of canonical examples
- `SplitManifest`: deterministic train/validation/test partition
- `OptionScoringExample`: one query-option pair for downstream scoring models
- `PromptSpec`: shared multiple-choice prompt contract
- `PredictionRecord`: canonical serialized output format for model predictions

## Project Notes

### Ownership

- Pedram: data preparation, loading, and formatting foundation

### Installation

```bash
# Core package (includes requests, tqdm)
pip install -e .

# Development (adds pytest)
pip install -e ".[dev]"

# SLM experiments (adds torch, transformers, datasets, trl)
pip install -e ".[slm]"

# Fine-tuning with optional training augmentation
python finetuning/finetune.py --augmented-train-path data/processed/train_augmented.jsonl

# Results dashboard (adds streamlit, plotly)
pip install -e ".[dashboard]"

# Everything
pip install -e ".[dev,slm,dashboard]"
```
