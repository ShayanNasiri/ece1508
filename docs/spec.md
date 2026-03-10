# Recipe-MPR QA Foundation Specification

## 1. Overview

The project studies question answering on Recipe-MPR using a five-way recipe selection task. The current repository implementation defines the canonical dataset contract together with the preparation, loading, and formatting interfaces used by the codebase today.

## 2. Current Scope

The current implementation covers the data-facing foundation of the project:

- raw dataset validation
- canonical example normalization
- deterministic split generation
- loader interfaces for downstream consumers
- prompt and prediction formatting
- CLI support for preparation and inspection

## 3. Dataset Definition

- Source dataset: `data/500QA.json`
- Task: five-way multiple-choice recipe selection
- Example count: 500
- Query types: `Specific`, `Commonsense`, `Negated`, `Analogical`, `Temporal`
- Gold label: one correct option id per query

### 3.1 Canonical Example Schema

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

## 4. Data Artifacts And Splits

### 4.1 Canonical Processed Dataset

- Path: `data/processed/recipe_mpr_qa.jsonl`
- Contents: normalized canonical examples derived from `data/500QA.json`
- Identifier policy: stable `example_id` values `rmpr-0001` through `rmpr-0500`

### 4.2 Split Policy

The project uses one deterministic primary split committed to the repo:

- Train: 350 examples
- Validation: 75 examples
- Test: 75 examples
- Strategy: stratified by query-type signature
- Seed: `1508`

The split manifest is the shared contract for all later phases. No phase should create ad hoc splits unless explicitly documented as auxiliary analysis.

## 5. Implementation Structure

The repository currently ships:

- canonical JSONL dataset artifact
- deterministic split manifest
- CLI for prepare, validate, stats, and export
- tokenizer-ready option-scoring loader
- shared multiple-choice prompt format and response parser
- canonical prediction record schema
- tests covering data, loaders, prompt parsing, serialization, and CLI behavior

### 5.1 Interfaces

- `RecipeExample`
- `PreparedDataset`
- `OptionScoringExample`
- `PredictionRecord`
- `PromptSpec`

### 5.2 Module Layout

- `src/recipe_mpr_qa/data`: dataset constants, schemas, preparation, and loaders
- `src/recipe_mpr_qa/formats.py`: prompt and prediction record formatting
- `src/recipe_mpr_qa/cli.py`: command-line entrypoints

## 6. Output Formats

### 6.1 PredictionRecord

Every model or prompt run should emit JSONL records with:

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

## 7. CLI Contract

Supported commands:

- `prepare-data`: create the canonical processed dataset and split manifest
- `validate-data`: validate raw or prepared dataset inputs
- `dataset-stats`: print dataset summary counts
- `export-split`: export one manifest-defined split to JSONL

The CLI must remain dependency-light and runnable without non-standard-library runtime dependencies.

## 8. Testing Requirements

The test suite must cover:

- malformed records and validation failures
- whitespace normalization behavior
- deterministic ids and split generation
- split sizes, coverage, and disjointness
- option-scoring expansion and tokenizer passthrough
- prompt rendering and noisy response parsing
- prediction record JSONL round-trips
- CLI smoke behavior and artifact generation

The committed processed artifacts must regenerate exactly from the raw dataset and split seed.
