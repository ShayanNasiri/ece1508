# Architecture

This page summarizes how data moves through the repository and where the major implementation responsibilities live.

For the public contract behind these components, see the [Technical Spec](spec.md). For runnable commands, see [Workflows](workflows.md).

## High-Level Flows

### Canonical data flow

```text
data/500QA.json
  -> recipe-mpr-qa prepare-data
  -> data/processed/recipe_mpr_qa.jsonl
  -> data/processed/primary_split.json
```

This is the stable source-of-truth layer for the repo.

### Optional augmentation flow

```text
recipe_mpr_qa.jsonl + primary_split.json
  -> recipe-mpr-qa augment-train
  -> data/processed/train_augmented.jsonl
```

This path rewrites training queries only and preserves authentic options and labels.

### Synthetic-data R&D flow

```text
recipe_mpr_qa.jsonl + primary_split.json
  -> generate-synthetic-query
  -> review-synthetic-query
  -> approve-synthetic-query
  -> query-only approved artifact

recipe_mpr_qa.jsonl + primary_split.json
  -> generate-synthetic-full
  -> review-synthetic-full
  -> approve-synthetic-full
  -> full-generation approved artifact

approved synthetic artifacts
  -> build-synthetic-train
  -> train-ready synthetic RecipeExample artifact
```

Candidate, reviewed, approved, and train-ready artifacts are intentionally separate. Only the train-ready output is intended to be passed into fine-tuning.

### Evaluation flow

```text
recipe_mpr_qa.jsonl + primary_split.json
  -> model-facing prompt rendering
  -> evaluation backend
  -> parsed predictions + metrics
  -> llm_evaluation/results/*.json
```

The evaluation path supports:

- `generative` mode: generate text, then parse the answer letter
- `loglikelihood` mode: score `A` through `E` directly by logits through the Hugging Face backend

### Fine-tuning flow

```text
recipe_mpr_qa.jsonl + primary_split.json + optional train-only artifact
  -> prompt-completion dataset builder
  -> SLM fine-tuning
  -> outputs/*
```

The optional train-only artifact can come from either:

- `augment-train`
- `build-synthetic-train`

### Tracked wrapper flow

```text
recipe-mpr-qa run-train / run-eval
  -> underlying training or evaluation entrypoint
  -> tracked manifest
  -> registry update
  -> optional MLflow mirror
```

The tracked layer wraps existing scripts. It does not replace them.

## Package Responsibilities

### `src/recipe_mpr_qa/data`

Owns the canonical data layer:

- constants such as default artifact paths and query-type names
- typed models including `RecipeExample`, `PreparedDataset`, and `SplitManifest`
- raw-data preparation and normalization
- split generation and serialization
- dataset loading and option-scoring expansion
- train-only augmentation utilities

### `src/recipe_mpr_qa/formats.py`

Owns the shared task I/O contract:

- `PromptSpec`
- multiple-choice prompt rendering
- deterministic per-example option shuffling for model-facing prompts
- response parsing from raw model text back to answer letters
- `PredictionRecord` serialization helpers

This is where the separation between canonical data ordering and model-facing prompt ordering is enforced.

### `src/recipe_mpr_qa/evaluation`

Owns the package-level evaluation implementation:

- argument parsing and evaluation runner
- backend dispatch to Ollama or Hugging Face
- prompt rendering and prediction parsing
- metrics computation
- support for both `generative` and `loglikelihood` evaluation modes

The repo-root script `llm_evaluation/mc_eval.py` is a thin wrapper over this package module.

### `src/recipe_mpr_qa/slm`

Owns the package-level fine-tuning implementation:

- prompt-completion dataset construction
- Hugging Face and TRL training configuration
- optional inclusion of train-only augmentation or train-ready synthetic artifacts
- serialization of run config and output metadata

The repo-root script `finetuning/finetune.py` is the corresponding top-level entrypoint.

### `src/recipe_mpr_qa/synthetic`

Owns the synthetic-data R&D layer:

- query-only and full-generation artifact schemas
- provenance validation helpers
- duplicate and near-duplicate filtering helpers
- OpenAI Responses API integration for structured generation and review
- candidate generation, review, approval, and training-admission helpers

### `src/recipe_mpr_qa/tracking`

Owns the tracked run layer:

- run-manifest models
- artifact reference helpers
- local run and model registries
- comparison/report helpers
- optional MLflow mirroring
- wrappers around the existing training and evaluation entrypoints

### `src/recipe_mpr_qa/cli.py`

Owns the repo-level command-line surface. The CLI now covers:

- base data commands
- augmentation commands
- synthetic-data commands
- tracked train/eval wrapper commands
- local run registry operations

It is not just a thin data helper anymore; it is the main package entrypoint for repository workflows after installation.

## Key Design Boundaries

### Canonical data vs model-facing prompt

The canonical processed dataset preserves source ordering and acts as the stable data contract. The model-facing prompt applies deterministic per-example option shuffling to reduce answer-position leakage. This keeps the source artifact stable while making the benchmark path fairer.

### Derived-train-only boundary

Both augmentation and synthetic artifacts are derived, train-only inputs. They are not part of the canonical processed dataset and can be omitted without changing the base evaluation path.

### Review vs approval boundary

Synthetic artifacts pass through two distinct gates:

- review: the reviewer model scores and labels a generated candidate
- approval: repo-side filters decide whether that reviewed candidate is admissible for training

Only approved artifacts are eligible for `build-synthetic-train`.

### Tracking boundary

Tracked wrappers record manifests, stages, and lineage for runs. They do not create a second model-training implementation or a second evaluation implementation.

### Historical output boundary

Files under `llm_evaluation/results/`, `outputs/`, and the proposal/report artifacts under `docs/` remain useful context, but they are not the authoritative description of the current codebase.
