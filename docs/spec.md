# Recipe-MPR QA Specification

## 1. Objective

The project evaluates whether a specialized, fine-tuned DistilBERT option scorer can match or exceed a general LLM baseline on Recipe-MPR. The operational baseline target is to beat the documented 65% accuracy reference on the task while maintaining a reproducible experimental workflow.

## 2. Dataset and Task Definition

- Source dataset: `data/500QA.json`
- Task: five-way multiple-choice recipe selection
- Example count: 500
- Query types: `Specific`, `Commonsense`, `Negated`, `Analogical`, `Temporal`
- Gold label: one correct option id per query

### Canonical Phase 1 Example Schema

Each prepared example contains:

- `example_id`: stable identifier `rmpr-0001` through `rmpr-0500`
- `query`: normalized query text
- `options`: ordered list of five candidate options with `option_id` and `text`
- `answer_option_id`: gold option id
- `query_type_flags`: boolean map for the five query categories
- `correctness_explanation`: original gold evidence map from the raw dataset, preserving string or string-list evidence values
- `source_metadata`: raw dataset path, raw row index, and any normalization applied

Normalization policy:

- Strip outer whitespace from `query`
- Preserve option text, explanations, and `<INFERRED>` markers
- Preserve raw option ordering

Validation policy:

- Required keys must be present
- Exactly five options per example
- Option ids must be unique within an example
- `answer_option_id` must match one of the option ids
- Query-type keys must match the expected five-category schema
- Text fields must be non-empty after trimming for validation purposes

## 3. Split Policy

The project uses one deterministic primary split committed to the repo:

- Train: 350 examples
- Validation: 75 examples
- Test: 75 examples
- Strategy: stratified by query-type signature
- Seed: `1508`

The split manifest is the shared contract for all later phases. No phase should create ad hoc splits unless explicitly documented as auxiliary analysis.

## 4. Phase Plan

### Phase 1: Data Preparation and Loading

Deliverables:

- canonical JSONL dataset artifact
- deterministic split manifest
- CLI for prepare, validate, stats, and export
- tokenizer-ready option-scoring loader
- shared multiple-choice prompt format and response parser
- canonical prediction and judgment record schemas
- tests covering data, loaders, prompt parsing, serialization, and CLI behavior

Interfaces:

- `RecipeExample`
- `PreparedDataset`
- `OptionScoringExample`
- `PredictionRecord`
- `PromptSpec`

### Phase 2: DistilBERT Baselines

Model formulation:

- DistilBERT is used as an option scorer over query-option pairs.
- Each original question expands into five examples with one positive label and four negatives.
- Question-level prediction is the option with the highest score among its five candidates.

Expected work:

- vanilla pretrained DistilBERT baseline
- fine-tuned DistilBERT on the fixed split
- optional augmented fine-tuning using teacher-generated query variants over the same option sets

### Phase 3: General LLM Inference

Provider architecture:

- Ollama-first provider interface
- shared prompt format and prediction record schema across providers
- model-specific settings captured in per-run metadata

Baseline expectation:

- at least one open model served through Ollama
- room to add more providers without changing downstream evaluation formats

### Phase 4: Evaluation, Judge, and Tracking

Evaluation modes:

- exact-match accuracy against the gold answer
- per-query-type and per-signature breakdowns
- LLM-as-a-judge evaluation on `query + chosen option + short rationale + gold evidence`

Judged dimensions:

- ingredient alignment
- constraint satisfaction
- reasoning quality
- overall verdict

Tracking policy:

- Phase 1 defines run naming and metadata contracts
- MLflow is the intended experiment backend once training/inference phases start logging runs
- DVC is a planned future addition for expanded data/version control, not a Phase 1 requirement

## 5. Canonical Output Schemas

### PredictionRecord

Every baseline or model run should emit JSONL records with:

- `run_id`
- `phase`
- `provider`
- `model_name`
- `split`
- `example_id`
- `prompt_version`
- `raw_response`
- `parsed_choice`
- `predicted_option_id`
- `gold_option_id`
- `is_correct`
- `latency_ms`
- `metadata`

### JudgmentRecord

Reserved for the Phase 4 judge pipeline:

- `run_id`
- `phase`
- `provider`
- `model_name`
- `split`
- `example_id`
- `judge_model_name`
- `ingredient_alignment`
- `constraint_satisfaction`
- `reasoning_quality`
- `overall_verdict`
- `rationale`
- `metadata`

## 6. CLI Contract

Supported commands:

- `prepare-data`: create the canonical processed dataset and split manifest
- `validate-data`: validate raw or prepared dataset inputs
- `dataset-stats`: print dataset summary counts
- `export-split`: export one manifest-defined split to JSONL

The CLI must remain dependency-light and runnable without training or tracking extras installed.

## 7. Testing Requirements

Phase 1 tests must cover:

- malformed records and validation failures
- whitespace normalization behavior
- deterministic ids and split generation
- split sizes, coverage, and disjointness
- option-scoring expansion and tokenizer passthrough
- prompt rendering and noisy response parsing
- prediction record JSONL round-trips
- CLI smoke behavior and artifact generation

The committed processed artifacts must regenerate exactly from the raw dataset and split seed.
