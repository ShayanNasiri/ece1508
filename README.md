# Recipe-MPR QA

Recipe-MPR QA is a course project and research-style repository centered on one question: can a fine-tuned small language model outperform or remain competitive with a larger general-purpose language model on Recipe-MPR while staying lightweight enough for local use?

Recipe-MPR is a five-way multiple-choice recipe recommendation task. Each example contains a natural-language food preference query, five candidate recipe descriptions, and one correct answer. The task is harder than keyword matching because many queries depend on commonsense reasoning, negation, analogical cues, or temporal constraints.

## Current Repository Status

The repository currently provides the shared project foundation:

- canonical processed dataset preparation from the raw Recipe-MPR source
- deterministic train, validation, and test split generation
- typed dataset contracts and loaders for downstream consumers
- standardized model-facing prompt rendering and response parsing
- local LLM evaluation utilities
- SLM fine-tuning scaffolding for prompt-completion training
- optional train-only query augmentation with conservative rule-based rewrites
- local-first MLOps wrappers for tracked train and eval runs
- regression tests covering the implemented data and workflow surfaces

## Quickstart

Install the package from the repository root:

```bash
pip install -e .
```

Install development tooling:

```bash
pip install -e ".[dev]"
```

Install the SLM stack:

```bash
pip install -e ".[slm]"
```

Install the optional MLOps extra:

```bash
pip install -e ".[mlops]"
```

Prepare the canonical processed dataset and split manifest:

```bash
python -m recipe_mpr_qa.cli prepare-data \
  --input data/500QA.json \
  --output data/processed/recipe_mpr_qa.jsonl \
  --split-output data/processed/primary_split.json
```

Create an optional train-only augmentation artifact:

```bash
python -m recipe_mpr_qa.cli augment-train \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/train_augmented.jsonl \
  --max-variants 2
```

Run the test suite:

```bash
pytest
```

Run LLM evaluation from the repository root with explicit paths:

```bash
python llm_evaluation/mc_eval.py \
  --model deepseek-r1:7b \
  --backend ollama \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Run SLM fine-tuning with the canonical train split only:

```bash
python finetuning/finetune.py
```

Run SLM fine-tuning with the optional train-only augmentation artifact:

```bash
python finetuning/finetune.py \
  --augmented-train-path data/processed/train_augmented.jsonl
```

## Documentation Map

- [Docs Hub](docs/index.md)
- [Project Overview](docs/project_overview.md)
- [Technical Spec](docs/spec.md)
- [Workflows](docs/workflows.md)
- [Architecture](docs/architecture.md)
- [Experiment Status](docs/experiments_status.md)
- [MLOps Layer](docs/mlops.md)

## Repository Structure

- `data/`: raw and generated dataset artifacts
- `docs/`: project overview, technical contract, workflows, and status notes
- `src/recipe_mpr_qa/`: canonical data, formatting, and CLI implementation
- `llm_evaluation/`: evaluation scripts and result artifacts
- `finetuning/`: SLM fine-tuning and related utilities
- `outputs/`: saved training artifacts from prior runs
- `tests/`: regression coverage for the current implementation

## Current Caveat

The committed evaluation JSON files under `llm_evaluation/results/` and the saved training artifacts under `outputs/` should be treated as provisional historical outputs, not final benchmark evidence. The benchmark logic changed materially after the answer-position leakage fix and parser hardening, so experiments should be rerun before any reported numbers are treated as authoritative.
