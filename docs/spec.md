# Recipe-MPR QA Technical Specification

This document is the canonical technical contract for the repository. It defines the task artifacts, public data types, evaluation modes, command surface, and the invariants that the current codebase relies on.

For project motivation and live status, see [Project Overview](project_overview.md) and [Experiment Status](experiments_status.md).

## Overview

The repository studies Recipe-MPR as a five-way multiple-choice recipe recommendation task.

- raw dataset: `data/500QA.json`
- canonical processed dataset: `data/processed/recipe_mpr_qa.jsonl`
- canonical split manifest: `data/processed/primary_split.json`
- query-type flags: `Specific`, `Commonsense`, `Negated`, `Analogical`, `Temporal`

Each example contains one user query, five candidate recipe options, and one correct option id.

## Artifact Contracts

### Canonical processed dataset

The canonical processed dataset is the normalized source-of-truth artifact used by the rest of the repository.

- path: `data/processed/recipe_mpr_qa.jsonl`
- format: JSONL, one `RecipeExample` per line
- source: derived from `data/500QA.json`
- size: 500 examples
- stable ids: `rmpr-0001` through `rmpr-0500`

Normalization and preservation rules:

- `query` is stripped of outer whitespace
- option text is preserved
- correctness explanations are preserved, including `<INFERRED>` markers
- raw option ordering is preserved in the canonical processed dataset

The canonical processed dataset is intentionally not rewritten to reflect model-facing prompt shuffling.

### Split manifest

The split manifest is the shared train, validation, and test partition used throughout the project.

- path: `data/processed/primary_split.json`
- train: 350 examples
- validation: 75 examples
- test: 75 examples
- seed: `1508`
- strategy: stratified by query-type signature

Evaluation and fine-tuning should not create ad hoc splits unless explicitly documented as auxiliary analysis.

### Train-only augmentation artifact

The optional augmentation artifact reuses the `RecipeExample` schema and contains only synthetic training examples.

- example output path: `data/processed/train_augmented.jsonl`
- source: generated from train split parents in the canonical processed dataset
- scope: training only
- allowed changes: query rewrite only
- preserved fields: `options`, `answer_option_id`, `query_type_flags`, `correctness_explanation`

Augmented examples add provenance into `source_metadata`:

- `parent_example_id`
- `augmentation_strategy`
- `variant_index`

### Query-only synthetic artifacts

The query-only synthetic workflow also reuses the `RecipeExample` schema, but it is stricter than the legacy augmentation path.

- example paths:
  - `data/processed/synthetic/query_candidates.jsonl`
  - `data/processed/synthetic/query_reviewed.jsonl`
  - `data/processed/synthetic/query_approved.jsonl`
- scope: training only
- allowed changes: query only
- preserved fields: `options`, `answer_option_id`, `query_type_flags`, `correctness_explanation`

Required provenance fields in `source_metadata`:

- `synthetic_mode`
- `generator_model`
- `generation_prompt_version`
- `approval_batch_id`
- `review_status`
- `review_scores`
- `created_at`
- `intended_query_type_target`
- `parent_example_id`

Lifecycle meaning:

- candidate: generated but not reviewed
- reviewed: scored and labeled by the review step
- approved: reviewed and also admitted by repo-side filters

Reviewed artifacts are not automatically training-eligible.

### Full-generation synthetic artifacts

The full-generation workflow uses a separate nested record schema because it creates new multiple-choice items rather than rewriting existing queries.

- example paths:
  - `data/processed/synthetic/full_candidates.jsonl`
  - `data/processed/synthetic/full_reviewed.jsonl`
  - `data/processed/synthetic/full_approved.jsonl`
- each row stores:
  - `recipe_example`
  - `provenance`

Required provenance fields:

- `synthetic_mode`
- `generator_model`
- `generation_prompt_version`
- `approval_batch_id`
- `review_status`
- `review_scores`
- `created_at`
- `intended_query_type_target`
- `seed_example_ids`
- `distractor_generation_method`
- `distribution_fit_score`

### Train-ready synthetic artifact

The training-admission step materializes one train-only `RecipeExample` JSONL artifact that can be passed to fine-tuning.

- example path: `data/processed/synthetic/train_synthetic_025.jsonl`
- source: approved query-only artifacts, approved full-generation artifacts, or both
- use: append only to the training split through `--augmented-train-path`

This is the only synthetic output that should be handed to the training script.

### Historical run artifacts

These files exist in the repo but are not part of the stable task contract:

- `llm_evaluation/results/*.json`
- `outputs/*`

They should be interpreted using the caveats in [Experiment Status](experiments_status.md).

## Public Data Types

### `RecipeOption`

Represents one candidate option inside a multiple-choice example.

- `option_id`: string identifier from the raw dataset
- `text`: option text

Invariant:

- both fields must be non-empty strings

### `RecipeExample`

Represents one normalized Recipe-MPR example.

- `example_id`
- `query`
- `options`
- `answer_option_id`
- `query_type_flags`
- `correctness_explanation`
- `source_metadata`

Invariants:

- exactly five options
- option ids must be unique within an example
- `answer_option_id` must match one of the option ids
- `query_type_flags` must contain exactly the five expected query categories
- `correctness_explanation` must be a non-empty mapping of strings to non-empty strings or non-empty string lists
- `source_metadata` must be a mapping

### `PreparedDataset`

Represents a validated collection of canonical or derived `RecipeExample` rows.

- `examples`
- `metadata`

Invariant:

- example ids must be unique within the dataset

### `SyntheticFullRecord`

Represents one full-generation synthetic item plus its review provenance.

- `recipe_example`
- `provenance`

Invariants:

- `recipe_example` must satisfy the `RecipeExample` contract
- `provenance.synthetic_mode` must be `full_generation`
- `seed_example_ids` must be non-empty
- `distribution_fit_score` must be null or a number between `0` and `1`

### `SyntheticFullDataset`

Represents a validated collection of `SyntheticFullRecord` rows.

- `records`
- `metadata`

Invariant:

- synthetic example ids must be unique within the dataset

### `SplitManifest`

Represents the shared partition of example ids across `train`, `validation`, and `test`.

- `splits`
- `metadata`

Invariants:

- only `train`, `validation`, and `test` are valid split names
- no example id may appear in more than one split

### `OptionScoringExample`

Represents one query-option pair for option-scoring style consumers.

- `example_id`
- `option_id`
- `option_index`
- `group_size`
- `query`
- `option_text`
- `label`
- `tokenized_inputs`

Invariant:

- `label` is binary and indicates whether the option is correct for that query

### `PromptSpec`

Represents the shared multiple-choice prompt contract.

- `version`
- `template`

The default prompt version is `recipe-mpr-mc-v1`.

### `PredictionRecord`

Represents a normalized prediction output row for model runs.

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

## Prompt And Evaluation Contract

### Model-facing prompt

The model-facing prompt is built from `PromptSpec` and rendered through `build_multiple_choice_prompt()`.

Key rules:

- prompts present options as letters `A` through `E`
- model-facing option order is shuffled deterministically per example
- canonical dataset ordering is not mutated
- the same prompt contract is shared between evaluation and fine-tuning

### Response parsing

The response parser is intentionally conservative.

Expected behavior:

- prefer clear answer signals over loose mentions
- avoid over-crediting chain-of-thought outputs that enumerate several choices
- return `None` when a defensible answer letter cannot be recovered

### Evaluation modes

The public evaluation surface supports two modes:

- `generative`
- `loglikelihood`

Mode contract:

- `generative` queries the backend for text and parses an answer letter
- `loglikelihood` scores answer letters directly and requires the Hugging Face backend
- both modes share the same dataset, split, prompt, and answer-mapping contract

## Public Command Surface

The package CLI is implemented in `src/recipe_mpr_qa/cli.py` and exposed as `recipe-mpr-qa` after editable installation.

The current public commands are:

- `prepare-data`
- `validate-data`
- `dataset-stats`
- `export-split`
- `augment-train`
- `generate-synthetic-query`
- `review-synthetic-query`
- `approve-synthetic-query`
- `generate-synthetic-full`
- `review-synthetic-full`
- `approve-synthetic-full`
- `build-synthetic-train`
- `run-train`
- `run-eval`
- `list-runs`
- `compare-runs`
- `promote-run`

Command behavior summary:

- `prepare-data` writes the canonical processed dataset and split manifest
- `validate-data` validates either raw or prepared data
- `dataset-stats` prints dataset metadata
- `export-split` writes one split-specific JSONL artifact
- `augment-train` writes a derived train-only query-rewrite artifact
- `generate-synthetic-query` writes query-only synthetic candidates through the OpenAI API
- `review-synthetic-query` adds structured review decisions to query-only candidates
- `approve-synthetic-query` filters reviewed query-only candidates into an approved artifact
- `generate-synthetic-full` writes full-generation synthetic candidates through the OpenAI API
- `review-synthetic-full` adds structured review decisions to full-generation candidates
- `approve-synthetic-full` filters reviewed full-generation candidates into an approved artifact
- `build-synthetic-train` combines approved synthetic artifacts into a train-ready `RecipeExample` JSONL file
- `run-train` wraps the existing fine-tuning path with tracked manifests and registries
- `run-eval` wraps the existing evaluation path with tracked manifests and registries
- `list-runs`, `compare-runs`, and `promote-run` operate on the local run registry under `mlops/`

Tracked wrapper note:

- `run-train` and `run-eval` forward unknown flags to the underlying training or evaluation script

## Runtime And Environment Contract

### Editable install expectation

These package-level commands expect an editable install first:

- `recipe-mpr-qa`
- `python -m recipe_mpr_qa.cli`

### OpenAI environment loading

The synthetic-data client resolves the API key in this order:

1. explicit `api_key` argument
2. `OPENAI_API_KEY` from the process environment
3. repo-local `.env` discovery through upward directory search

The supported `.env` key name is:

- `OPENAI_API_KEY`

### Package extras

Declared extras in `pyproject.toml`:

- `dev`
- `slm`
- `mlops`
- `dashboard`

Current documentation support level:

- `dev`, `slm`, and `mlops` are active documented extras
- `dashboard` is currently packaging-only and does not have a first-class documented workflow

## Key Output Locations

Stable source-of-truth artifacts:

- `data/processed/recipe_mpr_qa.jsonl`
- `data/processed/primary_split.json`

Derived artifacts:

- `data/processed/train_augmented.jsonl`
- `data/processed/synthetic/query_candidates*.jsonl`
- `data/processed/synthetic/query_reviewed*.jsonl`
- `data/processed/synthetic/query_approved*.jsonl`
- `data/processed/synthetic/full_candidates*.jsonl`
- `data/processed/synthetic/full_reviewed*.jsonl`
- `data/processed/synthetic/full_approved*.jsonl`
- `data/processed/synthetic/train_*.jsonl`

Tracked artifacts:

- `mlops/runs/*/run_manifest.json`
- `mlops/registry/runs.json`
- `mlops/registry/models.json`

Historical outputs:

- `llm_evaluation/results/*.json`
- `outputs/*`

## Current Regression Coverage

The repo currently includes regression tests covering:

- dataset preparation and validation behavior
- prompt and parser behavior
- synthetic provenance validation
- synthetic candidate-to-approved-to-train-ready workflow behavior
- tracked wrapper behavior and registry updates
