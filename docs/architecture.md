# Architecture

This page summarizes how data moves through the repository and where the major implementation responsibilities live.

For the public contract behind these components, see the [Technical Spec](spec.md).

## High-Level Data Flow

```text
data/500QA.json
  -> prepare-data
  -> data/processed/recipe_mpr_qa.jsonl
  -> primary_split.json

recipe_mpr_qa.jsonl + primary_split.json
  -> export-split
  -> split-specific JSONL artifacts

recipe_mpr_qa.jsonl + primary_split.json
  -> augment-train
  -> train_augmented.jsonl

recipe_mpr_qa.jsonl + primary_split.json
  -> model-facing prompt rendering
  -> LLM evaluation
  -> llm_evaluation/results/*.json

recipe_mpr_qa.jsonl + primary_split.json + optional train_augmented.jsonl
  -> prompt-completion dataset builder
  -> SLM fine-tuning
  -> outputs/*
```

## Module Responsibilities

### `src/recipe_mpr_qa/data`

This package owns the canonical data layer:

- constants such as default artifact paths and query-type names
- typed models including `RecipeExample`, `PreparedDataset`, and `SplitManifest`
- raw-data preparation and normalization
- split generation and serialization
- dataset loading and option-scoring expansion
- train-only augmentation utilities

### `src/recipe_mpr_qa/formats.py`

This module owns the shared model I/O contract:

- `PromptSpec`
- multiple-choice prompt rendering
- deterministic per-example option shuffling for model-facing prompts
- response parsing from raw model text back to answer letters
- `PredictionRecord` serialization helpers

This is where the separation between canonical data ordering and model-facing prompt ordering is enforced.

### `src/recipe_mpr_qa/cli.py`

This module exposes the dependency-light command-line surface for:

- `prepare-data`
- `validate-data`
- `dataset-stats`
- `export-split`
- `augment-train`

The CLI is the main way to produce and inspect source-of-truth data artifacts.

### `llm_evaluation`

This directory contains the local evaluation stack:

- model backends for Ollama and Hugging Face
- the multiple-choice evaluation script
- result JSON artifacts from prior runs

The evaluation script loads the canonical processed dataset and split manifest, renders a model-facing prompt, queries a model, parses the response, and computes accuracy.

### `finetuning`

This directory contains the current SLM fine-tuning path:

- prompt-completion dataset construction
- Hugging Face and TRL training configuration
- optional inclusion of a train-only augmentation artifact
- plotting and prior notebook material

The fine-tuning script uses the same model-facing prompt contract as evaluation so that both paths share prompt order and answer mapping behavior.

## Key Design Boundaries

### Canonical processed data vs. model-facing prompt

The repository intentionally separates two concerns:

- the canonical processed dataset preserves source ordering and acts as the stable data contract
- the model-facing prompt applies deterministic per-example option shuffling to avoid answer-position leakage

That means the dataset artifact stays stable while the benchmark-facing prompt remains fairer.

### Augmentation boundary

The train-only augmentation artifact is not part of the canonical processed dataset. It is a derived artifact that can be added to fine-tuning but omitted entirely without changing the rest of the repository pipeline.

### Parser boundary

The response parser is intentionally conservative. It should convert clear answer signals into letters while avoiding over-crediting chain-of-thought text that merely mentions several options.
