# Recipe-MPR QA

This repository compares specialized small language models against general LLMs on the Recipe-MPR multiple-choice recipe recommendation dataset.

The project objective is to test whether a task-specific DistilBERT pipeline can match or outperform a general-purpose LLM baseline while keeping the workflow reproducible and easy to extend. Phase 1 establishes the data contracts, deterministic splits, model I/O schemas, and repo structure that later phases build on.

## Project Status

- Phase 1: implemented in this repo
- Phase 2: vanilla DistilBERT baseline and fine-tuning
- Phase 3: Ollama-backed LLM inference and provider expansion
- Phase 4: LLM-as-a-judge evaluation and MLflow-backed experiment tracking

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
- `data/processed/recipe_mpr_qa.jsonl`: canonical normalized dataset for all phases
- `data/processed/primary_split.json`: deterministic 70/15/15 split manifest
- `docs/spec.md`: end-to-end project specification
- `src/recipe_mpr_qa/data`: preparation, validation, splits, and loader interfaces
- `src/recipe_mpr_qa/llm`: prompt contracts and Ollama integration points
- `src/recipe_mpr_qa/evaluation`: prediction and judge record schemas
- `src/recipe_mpr_qa/tracking`: MLflow-facing experiment metadata helpers
- `tests`: Phase 1 regression coverage

## Phase 1 Deliverables

- Canonical example schema with stable `example_id` values
- Deterministic stratified split manifest
- Tokenizer-ready option-scoring loader for DistilBERT-style training
- Standardized multiple-choice prompt format and parser for LLM outputs
- Canonical prediction record schema for all later phases
- CLI workflow for preparing, validating, inspecting, and exporting dataset artifacts

## Planned Evaluation

Primary metric:

- exact-match accuracy on the held-out test split

Secondary metrics:

- per-query-type accuracy
- breakdowns by query-type signature
- LLM-as-a-judge scores for ingredient alignment, dietary/constraint satisfaction, reasoning quality, and overall verdict

## Team Ownership

- Pedram: Phase 1 data preparation, loading, and repo foundation
- Jagrit: Phase 2 DistilBERT baseline and fine-tuning
- Shayan: Phase 3 LLM inference plus Phase 4 judge integration

## Dependencies

The base Phase 1 implementation is intentionally light and runs with only the Python standard library.

Optional extras:

- `.[train]` for DistilBERT training and Hugging Face tooling
- `.[llm]` for `requests`-based Ollama inference
- `.[tracking]` for MLflow integration

## Roadmap

1. Use the fixed split manifest for all baseline and fine-tuning runs.
2. Train and evaluate the vanilla DistilBERT option scorer.
3. Add teacher-generated query augmentation while keeping the same five-option structure.
4. Run Ollama-backed LLM baselines through the shared prompt and prediction record schema.
5. Compare all systems with exact-match metrics and an LLM-as-a-judge pipeline.
