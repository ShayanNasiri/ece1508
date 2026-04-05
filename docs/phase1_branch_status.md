# Phase 1 Branch Status

This document is the branch-level wrap-up for `phase1-repo-foundation`.

It summarizes what was implemented, what was actually validated, what benchmark evidence currently exists, and what remains unfinished.

## What This Branch Added

Relative to the original scaffold, this branch now includes:

- a shared benchmark contract with deterministic per-example option shuffling for generative multiple-choice evaluation
- stricter response parsing and richer prediction provenance
- benchmark manifests, registry entries, and report generation utilities
- a governed synthetic-query workflow with candidate, review, approval, and train-ready stages
- SmolLM2 benchmark configs for 135M and 360M baseline and fine-tuning runs
- Slurm launchers for canaries, base-model evaluations, and original-only fine-tunes
- quota-safe Slurm defaults:
  - Hugging Face cache and shared venv default to `/tmp`
  - post-train checkpoint pruning keeps only the best retained checkpoint
- retained-run documentation and cleanup discipline

## What Actually Ran

The following evidence exists and completed successfully on this branch:

### Local validation

- `python3 -m compileall src tests slurm/prune_run_checkpoints.py`
- `bash -n` over the Slurm launcher scripts

### Cluster environment validation

- `logs/smollm-canary_111630.out`
  - dependency install completed
  - `torch=2.6.0+cu124`
  - `cuda_available=True`
  - `device_name=NVIDIA GeForce RTX 4090`

### Retained benchmark runs

- `111633` `smollm2-360m-baseline`
  - test accuracy `14/75` (`18.67%`)
- `111637` `smollm2-360m-finetune-original-highcap`
  - validation accuracy `44/75` (`58.67%`)
  - test accuracy `44/75` (`58.67%`)
- `111638` `smollm2-135m-finetune-original-highcap`
  - validation accuracy `20/75` (`26.67%`)
  - test accuracy `13/75` (`17.33%`)
- `111714` `smollm2-135m-finetune-original-highcap-loglikelihood`
  - validation accuracy `21/75` (`28.00%`)
  - test accuracy `17/75` (`22.67%`)

See [benchmark_run_inventory.md](/u/yazdinip/ece1508/docs/benchmark_run_inventory.md) for the retained artifact roots and cleanup state.

## What Did Not Actually Run

An additional rerun batch was queued late in the branch under the quota-safe launch path:

- `111732`
- `111733`
- `111734`
- `111735`
- `111736`
- `111737`

Those jobs never launched. Slurm kept them in:

- `Reason=user_env_retrieval_failed_requeued_held`

They were canceled during wrap-up and produced no benchmark artifacts.

Meaning:

- there is no clean-commit rerun of the 360M original-only result yet
- the retained benchmark evidence still comes from the earlier completed runs on this branch

## Current Best Result

The strongest retained result is:

- `smollm2-360m-finetune-original-highcap`
  - test accuracy `44/75`
  - `58.67%`
  - zero parse failures

This is the branch’s current best empirical result, but it should still be treated as provisional until it is rerun on the final cleaned-up code state.

## Current Limitations

- No synthetic-data experiment has been run end to end on this branch.
- No hosted reference-LLM row has been rerun under the current contract.
- No full final benchmark matrix exists yet.
- `pytest` was not run in the wrap-up environment because the local shell did not have `pytest` installed.
- Push to GitHub was not completed from this shell because no credentials were available for the HTTPS remote.

## Recommended Next Step After This Branch

If work resumes later, the next technically defensible move is:

1. fix the Slurm `user_env_retrieval_failed_requeued_held` issue
2. rerun the 360M generate baseline and 360M original-only fine-tune on the current cleaned-up commit
3. compare those fresh results against the retained `14/75` and `44/75` rows
4. only then extend to synthetic-data experiments and the final comparison table
