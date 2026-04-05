# Benchmark Run Inventory

This document captures the benchmark-relevant runs and validations retained locally on `phase1-repo-foundation` as of April 5, 2026, after quota-driven artifact cleanup and the final branch wrap-up pass.

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
- primary benchmark evidence currently on disk was produced from commit `05cd41814bacd36099dbf595e39edb9261bbaa9b`
- later branch work added quota-safe Slurm launchers, branch documentation, and retained-artifact cleanup, but the final April 4-5 rerun batch never actually started

## Retained Artifact Roots

These runs still have live artifact directories under `artifacts/` after cleanup. For the large fine-tuning runs, only the best checkpoint is intended to be retained long term; intermediate checkpoints can be pruned without losing the benchmark summary, prediction files, or reloadable best adapter.

| Run ID | Model | Interface | Train data | Val acc | Test acc | Correct | Parse failures | Slurm job |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-360m-baseline` | `HuggingFaceTB/SmolLM2-360M-Instruct` | generate | base model | n/a | `0.1867` | `14/75` | `0` | `111633` |
| `smollm2-135m-finetune-original-highcap` | `HuggingFaceTB/SmolLM2-135M-Instruct` | generate | original only | `0.2667` | `0.1733` | `13/75` | `0` | `111638` |
| `smollm2-360m-finetune-original-highcap` | `HuggingFaceTB/SmolLM2-360M-Instruct` | generate | original only | `0.5867` | `0.5867` | `44/75` | `0` | `111637` |
| `smollm2-135m-finetune-original-highcap-loglikelihood` | `HuggingFaceTB/SmolLM2-135M-Instruct` | loglikelihood | original only | `0.2800` | `0.2267` | `17/75` | `0` | `111714` |

These are the only runs that currently have both retained artifact roots and useful benchmark summaries after cleanup.

## Documented But Pruned Runs

These runs were observed in the local benchmark registries before quota cleanup, but their full artifact directories were already removed or are no longer being retained as active artifact roots.

| Run ID | Model | Interface | Train data | Test acc | Correct | Parse failures | Slurm job | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `smollm2-135m-baseline-generate` | `HuggingFaceTB/SmolLM2-135M-Instruct` | generate | base model | `0.0000` | `0/75` | `71` | `111639` | artifact dir removed |
| `smollm2-135m-baseline-loglikelihood` | `HuggingFaceTB/SmolLM2-135M-Instruct` | loglikelihood | base model | `0.2000` | `15/75` | `0` | `111711` | artifact dir removed |
| `smollm2-360m-baseline-loglikelihood` | `HuggingFaceTB/SmolLM2-360M-Instruct` | loglikelihood | base model | `0.2000` | `15/75` | `0` | `111712` | artifact dir removed |

## Validation That Actually Ran

The following validation steps are backed by either local execution during the wrap-up pass or retained cluster logs:

- local static validation:
  - `python3 -m compileall src tests slurm/prune_run_checkpoints.py`
  - `bash -n` over the Slurm launcher set
- cluster environment canary:
  - `logs/smollm-canary_111630.out` completed and reported `torch=2.6.0+cu124`, `cuda_available=True`, and `device_name=NVIDIA GeForce RTX 4090`
- completed benchmark runs with retained summaries:
  - `111633` 360M generate baseline
  - `111637` 360M original-only generate fine-tune
  - `111638` 135M original-only generate fine-tune
  - `111714` 135M original-only loglikelihood fine-tune

Validation that did not happen on this branch:

- no `pytest` run, because `pytest` was not installed in the local shell environment used for wrap-up
- no synthetic-data benchmark run
- no hosted reference-LLM rerun under the fair contract

## April 4-5 Rerun Attempt

An additional clean-commit rerun batch was queued under the quota-safe launch path with these job IDs:

- `111732`
- `111733`
- `111734`
- `111735`
- `111736`
- `111737`

Those jobs never started and produced no benchmark artifacts. Slurm held them with:

- `Reason=user_env_retrieval_failed_requeued_held`

The jobs were canceled during wrap-up so the branch would not be left with stale pending work.

Implication:

- the retained benchmark evidence on this branch still comes from the earlier completed runs listed above
- there is not yet an independent clean-commit reproduction of the `44/75` 360M original-only result

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

That result is materially stronger than the smaller-model runs and the base-model baselines currently documented here, but it still needs an independent rerun and the rest of the benchmark matrix before it should be treated as a final claim.

## Wrap-Up Takeaway

This branch now contains:

- a fairer and better governed benchmark pipeline than the original scaffold
- a retained local benchmark inventory with provenance and cleanup discipline
- a quota-safe Slurm launch path for future reruns

What it does not yet contain is a fully replicated final benchmark matrix. The strongest defensible statement from the retained evidence is that the branch can produce a plausible 360M original-only result of `44/75`, but that result has not yet been independently reproduced on the final cleaned-up code state.
