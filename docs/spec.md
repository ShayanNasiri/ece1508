# Recipe-MPR QA Technical Specification

This document is the canonical technical contract for the repository. It defines the task artifacts, public data types, prompt and parsing behavior, and the invariants that the current codebase relies on.

For project motivation and status, see [Project Overview](project_overview.md) and [Experiment Status](experiments_status.md).

## Overview

The repository studies Recipe-MPR as a five-way multiple-choice recipe recommendation task.

- Raw dataset: `data/500QA.json`
- Canonical processed dataset: `data/processed/recipe_mpr_qa.jsonl`
- Canonical split manifest: `data/processed/primary_split.json`
- Query-type flags: `Specific`, `Commonsense`, `Negated`, `Analogical`, `Temporal`

Each example contains one user query, five candidate recipe options, and one correct option id.

## Canonical Data Artifacts

### Canonical processed dataset

The canonical processed dataset is the normalized source-of-truth artifact used by the rest of the repository.

- Path: `data/processed/recipe_mpr_qa.jsonl`
- Format: JSONL, one `RecipeExample` per line
- Source: derived from `data/500QA.json`
- Size: 500 examples
- Stable ids: `rmpr-0001` through `rmpr-0500`

Normalization and preservation rules:

- `query` is stripped of outer whitespace
- option text is preserved
- correctness explanations are preserved, including `<INFERRED>` markers
- raw option ordering is preserved in the canonical processed dataset

The canonical processed dataset is intentionally not rewritten to reflect model-facing prompt shuffling. It remains a stable representation of the source data.

### Split manifest

The split manifest is the shared train, validation, and test partition used throughout the project.

- Path: `data/processed/primary_split.json`
- Train: 350 examples
- Validation: 75 examples
- Test: 75 examples
- Seed: `1508`
- Strategy: stratified by query-type signature

The split manifest is the repo-wide contract. Evaluation and fine-tuning should not create ad hoc splits unless explicitly documented as auxiliary analysis.

### Train-only augmentation artifact

The optional augmentation artifact reuses the `RecipeExample` schema and contains only synthetic training examples.

- Example output path: `data/processed/train_augmented.jsonl`
- Source: generated from train split parents in the canonical processed dataset
- Scope: training only
- Allowed changes: query rewrite only
- Preserved fields: `options`, `answer_option_id`, `query_type_flags`, `correctness_explanation`

Augmented examples add provenance into `source_metadata`:

- `parent_example_id`
- `augmentation_strategy`
- `variant_index`

Augmented example ids are unique synthetic ids such as `rmpr-0123-aug-01`.

## Public Data Types

### `RecipeOption`

Represents one candidate option inside a multiple-choice example.

- `option_id`: string identifier from the raw dataset
- `text`: option text

Invariants:

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

Represents a validated collection of canonical or augmented `RecipeExample` rows.

- `examples`
- `metadata`

Invariant:

- example ids must be unique within the dataset

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

## Prompt And Parsing Contract

### Model-facing prompt

The model-facing prompt is built from `PromptSpec` and rendered through `build_multiple_choice_prompt()`.

Key rules:

- the prompt always presents exactly five answer letters, `A` through `E`
- the canonical processed dataset keeps source ordering
- model-facing prompts use deterministic per-example option shuffling when `shuffle_key` is provided
- the fine-tuning and LLM evaluation paths pass `shuffle_key=example.example_id`

This separation is intentional:

- canonical processed data stays stable and faithful to source order
- model-facing prompts avoid answer-position leakage by using reproducible shuffling

### Response parsing

`parse_multiple_choice_response()` converts a raw model response into a choice letter when possible.

Current high-level behavior:

- accepts exact answer-only outputs like `A` or `Option C`
- prefers explicit answer markers such as `Final answer: B`
- allows a final trailing standalone letter when the response clearly ends with one
- avoids treating chain-of-thought enumeration of multiple options as a valid answer by default
- may fall back to a limited option-text overlap heuristic when option texts are provided

The parser is designed to be conservative. Unclear outputs should fail to parse instead of being over-credited.

## CLI Contract

The repository CLI is implemented in `src/recipe_mpr_qa/cli.py` and exposed as `recipe-mpr-qa`.

Supported commands:

- `prepare-data`
- `validate-data`
- `dataset-stats`
- `export-split`
- `augment-train`

Expected behavior:

- `prepare-data` writes the canonical processed dataset and split manifest
- `validate-data` validates either raw or prepared inputs
- `dataset-stats` prints dataset metadata
- `export-split` writes one manifest-defined split to JSONL
- `augment-train` writes a train-only augmentation artifact in `RecipeExample` format

The CLI should remain dependency-light relative to the rest of the repo and should not require the SLM stack to prepare or inspect data artifacts.

## Output Locations

Important repo-relative output locations:

- `data/processed/recipe_mpr_qa.jsonl`: canonical processed dataset
- `data/processed/primary_split.json`: split manifest
- `data/processed/train_augmented.jsonl`: optional train-only augmentation artifact
- `llm_evaluation/results/`: evaluation JSON outputs
- `outputs/`: saved fine-tuning artifacts from prior runs

## Testing Expectations

The test suite should continue to cover the core repository contracts:

- dataset validation failures
- deterministic ids and split generation
- split sizes, coverage, and disjointness
- option-scoring expansion
- prompt rendering and response parsing
- prediction record serialization
- CLI smoke behavior
- augmentation provenance and train-only behavior

The committed canonical processed dataset and split manifest should remain reproducible from the raw dataset and split seed.
