# Benchmark Run Inventory

This document captures the benchmark-relevant runs that existed locally on `phase1-repo-foundation` as of April 4, 2026, after quota-driven artifact cleanup.

It serves two purposes:

- preserve the important benchmark metadata even when large local artifact directories have been pruned
- distinguish between actively retained artifact directories and runs that are only preserved as documented summaries

## Benchmark Contract

All runs listed here were recorded against the current fair benchmark contract introduced on this branch:

- dataset: `data/processed/recipe_mpr_qa.jsonl`
- split: `data/processed/primary_split.json`
- split sizes: `350 / 75 / 75`
- contract version: `recipe-mpr-benchmark-v1`
- prompt version: `recipe-mpr-chat-mc-v2`
- parser version: `recipe-mpr-mc-parser-v2`
- option shuffle strategy: deterministic per example
- option shuffle seed: `1508`
- code commit at run time: `05cd41814bacd36099dbf595e39edb9261bbaa9b`

## Retained Artifact Roots

These runs still have live artifact directories under `artifacts/` after cleanup. For the large fine-tuning runs, only the best checkpoint is intended to be retained long term; intermediate checkpoints can be pruned without losing the benchmark summary, prediction files, or reloadable best adapter.

| Run ID | Model | Interface | Train data | Val acc | Test acc | Correct | Parse failures | Slurm job |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-360m-baseline` | `HuggingFaceTB/SmolLM2-360M-Instruct` | generate | base model | n/a | `0.1867` | `14/75` | `0` | `111633` |
| `smollm2-135m-finetune-original-highcap` | `HuggingFaceTB/SmolLM2-135M-Instruct` | generate | original only | `0.2667` | `0.1733` | `13/75` | `0` | `111638` |
| `smollm2-360m-finetune-original-highcap` | `HuggingFaceTB/SmolLM2-360M-Instruct` | generate | original only | `0.5867` | `0.5867` | `44/75` | `0` | `111637` |
| `smollm2-135m-finetune-original-highcap-loglikelihood` | `HuggingFaceTB/SmolLM2-135M-Instruct` | loglikelihood | original only | `0.2800` | `0.2267` | `17/75` | `0` | `111714` |

## Documented But Pruned Runs

These runs were observed in the local benchmark registries before quota cleanup, but their full artifact directories were already removed or are no longer being retained as active artifact roots.

| Run ID | Model | Interface | Train data | Test acc | Correct | Parse failures | Slurm job | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-135m-baseline-generate` | `HuggingFaceTB/SmolLM2-135M-Instruct` | generate | base model | `0.0000` | `0/75` | `71` | `111639` | artifact dir removed |
| `smollm2-135m-baseline-loglikelihood` | `HuggingFaceTB/SmolLM2-135M-Instruct` | loglikelihood | base model | `0.2000` | `15/75` | `0` | `111711` | artifact dir removed |
| `smollm2-360m-baseline-loglikelihood` | `HuggingFaceTB/SmolLM2-360M-Instruct` | loglikelihood | base model | `0.2000` | `15/75` | `0` | `111712` | artifact dir removed |

## Cleanup Notes

- The canonical benchmark dataset and split remain the source of truth and should not be deleted during artifact cleanup.
- Active local registries should only list runs whose artifact roots still exist.
- For fine-tuning runs, the checkpoint directories dominate disk use. The benchmark-critical minimum to retain is:
  - `benchmark/benchmark_manifest.json`
  - `manifests/run_summary.json`
  - `slm/metrics.json`
  - `slm/test_predictions.jsonl`
  - `slm/validation_predictions.jsonl` for fine-tunes
  - `slm/checkpoint_manifest.json`
  - the best checkpoint directory referenced by the checkpoint manifest
  - the reloadable top-level adapter and tokenizer files in `slm/checkpoints/`

## Current Headline Result

The strongest currently retained original-only result on this branch is:

- `smollm2-360m-finetune-original-highcap`: `44/75` test accuracy (`58.67%`) with zero parse failures

That result is materially stronger than the smaller-model runs and the base-model baselines currently documented here, but it still needs independent reruns and the rest of the benchmark matrix before it should be treated as a final claim.
