# MLOps Layer

This repository now includes a local-first MLOps layer for experiment management. It is designed to be runnable immediately on a local machine while remaining lightweight enough to stay out of the way when you just want to use the direct scripts.

For the regular data and experiment commands, see [Workflows](workflows.md). For the technical contract, see [Technical Spec](spec.md).

## Design Intent

The MLOps layer is additive. It does not replace:

- `python finetuning/finetune.py`
- `python llm_evaluation/mc_eval.py`

Instead, it wraps them with:

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

The layer stores references to existing training and evaluation artifacts. It does not duplicate large model directories.

## Wrapper Commands

Tracked training:

```bash
recipe-mpr-qa run-train \
  --stage candidate \
  --model-name HuggingFaceTB/SmolLM2-135M-Instruct \
  --data-path data/processed/recipe_mpr_qa.jsonl \
  --split-manifest-path data/processed/primary_split.json
```

Tracked evaluation:

```bash
recipe-mpr-qa run-eval \
  --stage baseline \
  --backend ollama \
  --model deepseek-r1:7b \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
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

MLflow mirroring is optional. If it is unavailable or disabled, the local manifests under `mlops/` remain the canonical record of the run.
