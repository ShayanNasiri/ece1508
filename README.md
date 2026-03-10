# Recipe-MPR QA

This repository currently contains the Phase 1 foundation for the Recipe-MPR project: dataset preparation, deterministic splits, tokenizer-ready loading, and standardized prompt/output formatting for later model runs.

The focus here is Pedram's scope only. The repo is set up so later training and evaluation work can consume a stable data contract, but this branch of `main` does not implement later-phase modeling, API integration, judging, or tracking.

## Phase 1 Scope

- Raw dataset validation and normalization
- Canonical processed JSONL artifact with stable example ids
- Deterministic 70/15/15 split manifest stratified by query-type signature
- Tokenizer-ready option-scoring loader for downstream consumers
- Standardized multiple-choice prompt formatting and output record schema
- CLI commands for prepare, validate, stats, and split export
- Regression tests for all Phase 1 functionality

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

Run the Phase 1 test suite:

```powershell
pytest
```

## Repo Layout

- `data/500QA.json`: raw Recipe-MPR dataset
- `data/processed/recipe_mpr_qa.jsonl`: canonical normalized Phase 1 dataset artifact
- `data/processed/primary_split.json`: deterministic 70/15/15 split manifest
- `docs/spec.md`: Phase 1 specification
- `src/recipe_mpr_qa/data`: preparation, validation, splits, and loader interfaces
- `src/recipe_mpr_qa/formats.py`: prompt formatting, response parsing, and prediction record schema
- `src/recipe_mpr_qa/cli.py`: Phase 1 command-line entrypoints
- `tests`: Phase 1 regression coverage

## Deliverables

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

- Pedram: Phase 1 data preparation, loading, and formatting

## Dependencies

The Phase 1 implementation runs with the Python standard library. `pytest` is the only declared development dependency.
