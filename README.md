# Recipe-MPR QA

Recipe-MPR QA is a course project and research-style repository centered on one question: can a fine-tuned small language model stay competitive with, or outperform, a larger general-purpose language model on Recipe-MPR while remaining lightweight enough for local use?

Recipe-MPR is a five-way multiple-choice recipe recommendation task. Each example contains a natural-language food preference query, five candidate recipe descriptions, and one correct answer. The task is harder than keyword matching because many queries depend on commonsense reasoning, negation, analogical cues, or temporal constraints.

## Current Repository Status

The repository currently provides:

- stable source-of-truth data preparation and deterministic train, validation, and test splits
- a shared prompt/parsing contract for model-facing multiple-choice evaluation
- local LLM evaluation utilities, including both generative and loglikelihood scoring modes
- SLM fine-tuning scaffolding for prompt-completion training
- optional train-only rule-based query augmentation
- experimental dual-track synthetic-data generation, review, approval, and training-admission workflows through the OpenAI API
- optional local-first tracked wrappers for train and eval runs
- regression tests for the current data, synthetic, evaluation, and tracking surfaces

Support levels in the current repo:

- stable: canonical dataset, split manifest, prompt/parsing contract, direct evaluation and direct fine-tuning
- optional but supported: train-only augmentation and tracked MLOps wrappers
- implemented but experimental: synthetic-data generation and the pilot artifacts under `data/processed/synthetic/`
- historical only: old JSON result files under `llm_evaluation/results/`, saved model outputs under `outputs/`, and the proposal/report artifacts in `docs/`

## Quickstart

Install the package from the repository root. This editable install is the expected starting point if you want to use `recipe-mpr-qa` or `python -m recipe_mpr_qa.cli`.

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[slm]"
pip install -e ".[mlops]"
```

Current extras declared in `pyproject.toml`:

- `dev`: `pytest`
- `slm`: `torch`, `transformers`, `accelerate`, `peft`, `datasets`, `trl`
- `mlops`: `mlflow`
- `dashboard`: declared packaging extra only; not yet a first-class documented workflow

Prepare the canonical processed dataset and split manifest:

```bash
recipe-mpr-qa prepare-data \
  --input data/500QA.json \
  --output data/processed/recipe_mpr_qa.jsonl \
  --split-output data/processed/primary_split.json
```

Run local multiple-choice evaluation in generative mode:

```bash
python llm_evaluation/mc_eval.py \
  --model deepseek-r1:7b \
  --backend ollama \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Run evaluation in loglikelihood mode:

```bash
python llm_evaluation/mc_eval.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backend huggingface \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json \
  --eval-mode loglikelihood
```

Run fine-tuning on the canonical train split only:

```bash
python finetuning/finetune.py
```

Use an existing train-only augmentation or approved synthetic artifact:

```bash
python finetuning/finetune.py \
  --augmented-train-path data/processed/synthetic/train_query_pilot_ratio010.jsonl
```

For the synthetic-data R&D workflow, see [docs/synthetic_data.md](docs/synthetic_data.md). That workflow requires an `OPENAI_API_KEY`, either in the shell environment or in a repo-root `.env` file. A template lives at [.env.example](.env.example).

Run the regression suite:

```bash
pytest -q
```

## Current Experimental State

The repo now contains real pilot synthetic artifacts under `data/processed/synthetic/`.

As of April 1, 2026:

- query-only pilot: 75 candidates, 75 reviewed, 60 approved
- full-generation pilot: 40 candidates, 40 reviewed, 15 approved
- train-ready pilot outputs exist for query-only, full-generation, and mixed handoff artifacts

Those files are useful for future training and evaluation handoff, but they are not benchmark evidence. No training or held-out evaluation conclusions have been drawn from them yet.

The old committed evaluation JSON files under `llm_evaluation/results/` and the saved training outputs under `outputs/` remain historical context only. The benchmark path changed materially after answer-position leakage mitigation and parser hardening, so any final benchmark claims still require fresh reruns.

## Documentation Map

- [Docs Hub](docs/index.md)
- [Project Overview](docs/project_overview.md)
- [Technical Spec](docs/spec.md)
- [Workflows](docs/workflows.md)
- [Architecture](docs/architecture.md)
- [Experiment Status](docs/experiments_status.md)
- [Synthetic Data R&D](docs/synthetic_data.md)
- [MLOps Layer](docs/mlops.md)
- [Documentation Audit](docs/documentation_audit.md)

## Repository Structure

- `data/`: raw, processed, and derived dataset artifacts
- `docs/`: canonical project docs plus historical proposal/report material
- `src/recipe_mpr_qa/`: canonical data, formatting, synthetic, evaluation, and tracking implementation
- `llm_evaluation/`: repo-root evaluation wrapper and result artifacts
- `finetuning/`: repo-root fine-tuning wrapper and related materials
- `outputs/`: saved historical training outputs
- `tests/`: regression coverage for the current implementation
