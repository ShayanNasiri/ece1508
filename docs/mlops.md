# MLOps Layer

This repository includes a local-first MLOps layer for experiment management. It is designed to stay lightweight: the tracked layer wraps the existing train and eval entrypoints instead of introducing a second experiment stack.

For the direct commands, see [Workflows](workflows.md). For the artifact and run contract, see [Technical Spec](spec.md).

## Design Intent

The tracked layer is optional but supported.

It does not replace:

- `python finetuning/finetune.py`
- `python llm_evaluation/mc_eval.py`

Instead, it adds:

- tracked run manifests
- local run and model registries
- artifact lineage
- run comparison
- promotion stages
- optional MLflow mirroring

Filesystem manifests remain the source of truth. MLflow is optional and acts only as a mirror.

## Local Layout

Tracked artifacts live under `mlops/`:

- `mlops/runs/<run_id>/run_manifest.json`
- `mlops/runs/<run_id>/summary.json`
- `mlops/registry/runs.json`
- `mlops/registry/models.json`
- `mlops/reports/`

The layer stores references to existing dataset, evaluation, and training artifacts. It does not duplicate large model directories.

## Wrapper Scope

The package-level wrappers live under `src/recipe_mpr_qa/tracking/` and are exposed through:

- `recipe-mpr-qa run-train`
- `recipe-mpr-qa run-eval`
- `recipe-mpr-qa list-runs`
- `recipe-mpr-qa compare-runs`
- `recipe-mpr-qa promote-run`

The wrappers forward unknown flags to the underlying training or evaluation entrypoint. That means you can combine wrapper flags such as `--stage` or `--enable-mlflow` with downstream flags such as `--model-name`, `--split`, `--eval-mode`, or `--augmented-train-path`.

## Tracked Training

Example:

```bash
recipe-mpr-qa run-train \
  --stage candidate \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-path data/processed/recipe_mpr_qa.jsonl \
  --split-manifest-path data/processed/primary_split.json
```

If `--output-dir` is omitted, the wrapper allocates one automatically under `outputs/tracked/<run_id>`.

Input lineage recorded for tracked training includes:

- canonical dataset path
- split-manifest path
- optional `augmented_train` artifact path when `--augmented-train-path` is provided

That means approved synthetic train artifacts are already first-class lineage inputs once they are handed to fine-tuning.

## Tracked Evaluation

Example:

```bash
recipe-mpr-qa run-eval \
  --stage baseline \
  --backend ollama \
  --model deepseek-r1:7b \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Tracked evaluation can also:

- link to a parent training run through `--parent-run-id`
- infer the model path from that parent training run if `--model` is omitted
- pass through `--eval-mode loglikelihood` to the underlying evaluation stack

Example:

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

## Registry Stages

The tracked registry uses these stages:

- `baseline`
- `candidate`
- `validated`
- `archived`

Typical usage:

- use `baseline` for reference runs
- use `candidate` for new fine-tuned models or experimental runs
- use `validated` for runs you want to keep as trusted reference points
- use `archived` for runs that should remain recorded but not treated as active candidates

## Comparison And Promotion

List tracked runs:

```bash
recipe-mpr-qa list-runs
```

Compare tracked runs:

```bash
recipe-mpr-qa compare-runs --run-id train-... --run-id eval-...
```

Promote a tracked run:

```bash
recipe-mpr-qa promote-run --run-id train-... --stage validated
```

## Optional MLflow

Install the extra:

```bash
pip install -e ".[mlops]"
```

Enable mirroring on a tracked run:

```bash
recipe-mpr-qa run-train \
  --enable-mlflow \
  --mlflow-experiment recipe-mpr-qa \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-path data/processed/recipe_mpr_qa.jsonl \
  --split-manifest-path data/processed/primary_split.json
```

MLflow mirroring is optional. If it is unavailable or disabled, the local manifests under `mlops/` remain the canonical record.

## Boundaries And Caveats

- the tracked layer records experiments; it does not validate benchmark correctness by itself
- historical outputs under `llm_evaluation/results/` and `outputs/` remain subject to the caveats in [Experiment Status](experiments_status.md)
- synthetic pilot artifacts only become tracked lineage once they are passed into `run-train` through `--augmented-train-path`
