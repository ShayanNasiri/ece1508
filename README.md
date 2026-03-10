# Recipe-MPR QA

This repository contains the current foundation for the Recipe-MPR project: dataset preparation, deterministic splits, tokenizer-ready loading, and standardized prompt/output formatting for model runs.

The broader project compares specialized small language models against general LLMs on the Recipe-MPR multiple-choice recipe recommendation dataset. The code currently in the repo establishes the shared data contract and workflow that later model training and evaluation work will build on.

## Current Status

- Raw dataset validation and normalization
- Canonical processed JSONL artifact with stable example ids
- Deterministic 70/15/15 split manifest stratified by query-type signature
- Tokenizer-ready option-scoring loader for downstream consumers
- Standardized multiple-choice prompt formatting and output record schema
- CLI commands for prepare, validate, stats, and split export
- Regression tests for all implemented functionality

## Quickstart

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

Run the test suite:

```powershell
pytest
```

## Repo Layout

- `data/500QA.json`: raw Recipe-MPR dataset
- `data/processed/recipe_mpr_qa.jsonl`: canonical normalized dataset artifact
- `data/processed/primary_split.json`: deterministic 70/15/15 split manifest
- `docs/spec.md`: current project foundation specification
- `src/recipe_mpr_qa/data`: preparation, validation, splits, and loader interfaces
- `src/recipe_mpr_qa/formats.py`: prompt formatting, response parsing, and prediction record schema
- `src/recipe_mpr_qa/cli.py`: command-line entrypoints for dataset preparation and inspection
- `tests`: regression coverage for the current implementation

## Implemented So Far

- Canonical example schema with stable `example_id` values
- Deterministic stratified split manifest
- Tokenizer-ready option-scoring loader for query-option scoring workflows
- Standardized multiple-choice prompt format and parser for LLM outputs
- Canonical prediction record schema for consistent model outputs
- CLI workflow for preparing, validating, inspecting, and exporting dataset artifacts

## Data Contracts

- `RecipeExample`: one normalized Recipe-MPR question with five ordered options
- `PreparedDataset`: validated collection of canonical examples
- `SplitManifest`: deterministic train/validation/test partition
- `OptionScoringExample`: one query-option pair for downstream scoring models
- `PromptSpec`: shared multiple-choice prompt contract
- `PredictionRecord`: canonical serialized output format for model predictions

## Ownership

- Pedram: data preparation, loading, and formatting foundation

## Dependencies

The current implementation runs with the Python standard library. `pytest` is the only declared development dependency.
