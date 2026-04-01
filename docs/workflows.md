# Workflows

This page documents the main repository workflows using the current commands and repo-relative paths. Run commands from the repository root unless stated otherwise.

For the artifact and interface contract behind these workflows, see the [Technical Spec](spec.md).

## Installation And Runtime Expectations

Install the base package first:

```bash
pip install -e .
```

That editable install is the expected prerequisite for:

- `recipe-mpr-qa`
- `python -m recipe_mpr_qa.cli`
- imports from `src/recipe_mpr_qa/*`

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[slm]"
pip install -e ".[mlops]"
```

Current extras from `pyproject.toml`:

- base: `requests`, `tqdm`
- `dev`: `pytest`
- `slm`: `torch`, `transformers`, `accelerate`, `peft`, `datasets`, `trl`
- `mlops`: `mlflow`
- `dashboard`: declared extra only; no first-class workflow is documented yet

## Base Data Commands

Prepare the canonical processed dataset and split manifest:

```bash
recipe-mpr-qa prepare-data \
  --input data/500QA.json \
  --output data/processed/recipe_mpr_qa.jsonl \
  --split-output data/processed/primary_split.json
```

Validate the raw dataset:

```bash
recipe-mpr-qa validate-data \
  --input data/500QA.json \
  --kind raw
```

Validate the canonical processed dataset:

```bash
recipe-mpr-qa validate-data \
  --input data/processed/recipe_mpr_qa.jsonl \
  --kind prepared
```

Print dataset metadata:

```bash
recipe-mpr-qa dataset-stats \
  --input data/processed/recipe_mpr_qa.jsonl \
  --kind prepared
```

Export one split from the committed split manifest:

```bash
recipe-mpr-qa export-split \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --split train \
  --output data/processed/train.jsonl
```

Valid split names are `train`, `validation`, and `test`.

## Legacy Train-Only Augmentation

Create a train-only augmentation artifact:

```bash
recipe-mpr-qa augment-train \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/train_augmented.jsonl \
  --max-variants 2
```

Current behavior:

- reads the canonical processed dataset
- selects only train parents from the split manifest
- rewrites queries conservatively
- preserves options, labels, query-type flags, and correctness explanations
- writes only derived `RecipeExample` rows intended for training use

## Synthetic Data R&D Workflow

The synthetic-data workflow is approval-gated and requires the OpenAI API.

Preferred setup: create a repo-root `.env` file from `.env.example` and set the key there:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The synthetic-data client will search for `.env` starting from the current working directory and walk upward through parent directories until it finds an `OPENAI_API_KEY`.

PowerShell alternative for the current shell only:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

### Query-only track

Generate candidate queries from train parents:

```bash
recipe-mpr-qa generate-synthetic-query \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/query_candidates.jsonl \
  --limit 75 \
  --max-candidates-per-parent 3
```

Review the candidates:

```bash
recipe-mpr-qa review-synthetic-query \
  --input data/processed/synthetic/query_candidates.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --output data/processed/synthetic/query_reviewed.jsonl
```

Approve the reviewed candidates:

```bash
recipe-mpr-qa approve-synthetic-query \
  --input data/processed/synthetic/query_reviewed.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/query_approved.jsonl \
  --approval-batch-id pilot-q
```

### Full-generation track

Generate candidate full synthetic items from train seeds:

```bash
recipe-mpr-qa generate-synthetic-full \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/full_candidates.jsonl \
  --limit 40 \
  --max-candidates-per-seed 1
```

Review the candidates:

```bash
recipe-mpr-qa review-synthetic-full \
  --input data/processed/synthetic/full_candidates.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --output data/processed/synthetic/full_reviewed.jsonl
```

Approve the reviewed candidates:

```bash
recipe-mpr-qa approve-synthetic-full \
  --input data/processed/synthetic/full_reviewed.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/full_approved.jsonl \
  --approval-batch-id pilot-f
```

### Training admission

Build one train-ready synthetic artifact from approved sources:

```bash
recipe-mpr-qa build-synthetic-train \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --query-approved-path data/processed/synthetic/query_approved.jsonl \
  --full-approved-path data/processed/synthetic/full_approved.jsonl \
  --target-ratio 0.25 \
  --full-share 0.30 \
  --output data/processed/synthetic/train_synthetic_025.jsonl
```

Notes:

- query-only and full-generation artifacts stay separate until this step
- reviewed artifacts are not automatically training-eligible
- `build-synthetic-train` is the only synthetic step that produces a file intended for fine-tuning
- the command also supports explicit caps through `--max-query-examples` and `--max-full-examples`

Current checked-in handoff files:

- `data/processed/synthetic/query_approved_merged.jsonl`
- `data/processed/synthetic/full_approved_merged.jsonl`
- `data/processed/synthetic/train_query_ratio025.jsonl`
- `data/processed/synthetic/train_full_ratio010.jsonl`
- `data/processed/synthetic/train_mixed_ratio025.jsonl`

Important note for multi-batch synthetic work:

- approved artifacts from different generation batches should not be concatenated blindly because synthetic example ids are currently batch-local
- the checked-in `*_merged.jsonl` pools already resolve those collisions for the current repo state

## LLM Evaluation

The public evaluation entrypoint is the repo-root wrapper:

- `python llm_evaluation/mc_eval.py`

That script wraps `src/recipe_mpr_qa/evaluation/mc_eval.py`.

### Generative mode

```bash
python llm_evaluation/mc_eval.py \
  --model deepseek-r1:7b \
  --backend ollama \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

### Loglikelihood mode

```bash
python llm_evaluation/mc_eval.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backend huggingface \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json \
  --eval-mode loglikelihood
```

Notes:

- `--model` is required
- default `--backend` is `ollama`
- default `--split` is `test`
- `--eval-mode generative` is the default
- `--eval-mode loglikelihood` requires `--backend huggingface`
- if `--output` is omitted, results are written under `results/<Model>_<Split>_<N>.json` relative to the current working directory

## SLM Fine-Tuning

The fine-tuning entrypoint is:

- `python finetuning/finetune.py`

Run with defaults:

```bash
python finetuning/finetune.py
```

Run with an explicit model and output directory:

```bash
python finetuning/finetune.py \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --output-dir outputs/smollm2_recipe_mpr
```

Include a train-only augmentation artifact:

```bash
python finetuning/finetune.py \
  --augmented-train-path data/processed/train_augmented.jsonl
```

Use a reviewed train-ready synthetic artifact instead:

```bash
python finetuning/finetune.py \
  --augmented-train-path data/processed/synthetic/train_query_ratio025.jsonl
```

Important behavior:

- fine-tuning reads the canonical processed dataset and split manifest by default
- `--augmented-train-path` appends extra examples only to the training split
- validation and test splits remain unchanged
- model-facing prompts use deterministic per-example option shuffling
- the script reads an existing derived artifact; it does not generate synthetic data automatically

## Tracked MLOps Wrappers

The `recipe-mpr-qa` CLI also exposes tracked wrappers. These commands add run manifests, registries, and artifact lineage on top of the direct scripts.

Unknown flags are forwarded to the underlying training or evaluation entrypoint, so wrapper-specific flags and downstream script flags can be combined in one command.

Run a tracked training job:

```bash
recipe-mpr-qa run-train \
  --stage candidate \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-path data/processed/recipe_mpr_qa.jsonl \
  --split-manifest-path data/processed/primary_split.json
```

Run a tracked training job that uses a train-ready synthetic artifact:

```bash
recipe-mpr-qa run-train \
  --stage candidate \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-path data/processed/recipe_mpr_qa.jsonl \
  --split-manifest-path data/processed/primary_split.json \
  --augmented-train-path data/processed/synthetic/train_query_ratio025.jsonl
```

Run a tracked evaluation job:

```bash
recipe-mpr-qa run-eval \
  --stage baseline \
  --backend ollama \
  --model deepseek-r1:7b \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Run tracked evaluation in loglikelihood mode:

```bash
recipe-mpr-qa run-eval \
  --stage baseline \
  --backend huggingface \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json \
  --eval-mode loglikelihood
```

Evaluate a tracked fine-tuned model by linking to a parent training run:

```bash
recipe-mpr-qa run-eval \
  --parent-run-id train-... \
  --backend huggingface \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Registry and comparison commands:

```bash
recipe-mpr-qa list-runs
recipe-mpr-qa compare-runs --run-id train-... --run-id eval-...
recipe-mpr-qa promote-run --run-id train-... --stage validated
```

## Common Output Locations

Stable source-of-truth artifacts:

- `data/processed/recipe_mpr_qa.jsonl`
- `data/processed/primary_split.json`

Derived train-only artifacts:

- `data/processed/train_augmented.jsonl`
- `data/processed/synthetic/query_candidates*.jsonl`
- `data/processed/synthetic/query_reviewed*.jsonl`
- `data/processed/synthetic/query_approved*.jsonl`
- `data/processed/synthetic/full_candidates*.jsonl`
- `data/processed/synthetic/full_reviewed*.jsonl`
- `data/processed/synthetic/full_approved*.jsonl`
- `data/processed/synthetic/train_*.jsonl`

Run artifacts and registries:

- `llm_evaluation/results/*.json`
- `mlops/runs/*/run_manifest.json`
- `mlops/registry/runs.json`
- `mlops/registry/models.json`
- `outputs/*`

Interpretation notes:

- canonical processed data and the split manifest are the primary stable artifacts
- augmentation output is optional derived training input
- synthetic artifacts are derived, experimental, and review-gated
- run artifacts are meaningful only in light of the current caveats documented in [Experiment Status](experiments_status.md)
