# Experiment Status

This page documents the live state of the experiment stack and the main caveats that matter when interpreting repository outputs.

For project framing, see [Project Overview](project_overview.md). For runnable commands, see [Workflows](workflows.md).

## Live Snapshot

Date of this snapshot: April 1, 2026.

The repository currently supports these experiment surfaces:

- canonical processed dataset preparation and deterministic splits
- standardized prompt rendering and conservative answer parsing
- direct local LLM evaluation through `llm_evaluation/mc_eval.py`
- direct SLM fine-tuning through `finetuning/finetune.py`
- optional train-only rule-based augmentation
- experimental dual-track synthetic-data generation, review, approval, and train-admission commands
- optional tracked train/eval wrappers with run manifests and registries

## Current Support Tiers

- stable: canonical dataset, split manifest, prompt/parsing contract, direct evaluation, direct fine-tuning
- optional but supported: `augment-train` and the tracked MLOps layer
- implemented but experimental: synthetic-data generation and the current synthetic handoff artifacts
- historical only: committed old result JSON files, saved model outputs, and the proposal/report artifacts

## Synthetic Data Status

The synthetic-data workflow is no longer hypothetical. The repo now contains pilot outputs, second-pass outputs, merged approved pools, and train-ready ratio artifacts under `data/processed/synthetic/`.

Current approved pools:

- query-only:
  - pilot approved: `60`
  - second-pass approved: `94`
  - merged approved pool: `154`
- full-generation:
  - pilot approved: `15`
  - second-pass approved: `24`
  - merged approved pool: `39`

Current validated train-ready handoff artifacts:

- `train_query_ratio025.jsonl`: `88`
- `train_full_ratio010.jsonl`: `35`
- `train_mixed_ratio025.jsonl`: `88`
  - query-only share: `62`
  - full-generation share: `26`

Interpretation:

- dual-track synthetic capability exists in code and has been exercised on real runs
- a second pass materially improved the handoff pool, especially for query-only
- query-only still has the stronger approval yield under the present gates
- full-generation remains materially weaker and should still be treated as the riskier experimental track
- no training or held-out evaluation conclusions have been drawn from these artifacts yet

Operational note:

- the current generator produces ids that are unique within a batch, not automatically across batches
- the checked-in merged approved pools already resolve those collisions and are the right source files for future multi-batch training handoff
- future multi-batch generation should either reuse those merged pools or patch id generation before another expansion pass

Smoke artifacts also exist in the same directory, but they are debugging/probing outputs and should not be treated as the main project narrative.

## Recent Corrective Changes That Still Matter

### Answer-position leakage mitigation

The canonical processed dataset still preserves source option ordering, but model-facing prompts now use deterministic per-example option shuffling in evaluation and fine-tuning paths. This removes the earlier shortcut where the correct answer position could be learned from raw ordering.

### Parser hardening

The response parser now prefers explicit answer signals and is less willing to over-credit chain-of-thought outputs that merely enumerate options. This makes multiple-choice scoring more conservative and more defensible.

### Expanded evaluation surface

The evaluation stack now supports both:

- `generative` mode
- `loglikelihood` mode through the Hugging Face backend

Any documentation or interpretation that treats generative parsing as the only evaluation mode is now stale.

## Current Caveats

### Committed result files are provisional

The JSON outputs currently committed under `llm_evaluation/results/` should be treated as historical run artifacts, not final benchmark evidence. They predate important benchmark-correctness changes and need to be regenerated before they are cited as authoritative.

### Saved training artifacts are historical

The artifacts under `outputs/` reflect prior training runs and should be interpreted the same way: useful as historical context, but not as final project evidence after the benchmark corrections.

### Synthetic artifacts are experimental handoff material

The synthetic files now checked into the repo are real handoff outputs, but they are still pre-training artifacts. They establish that the workflow works and that the repo now has candidate training inputs for future study. They do not establish that synthetic data improves the task.

## What Still Requires Fresh Evidence

These points still matter for any final benchmark story:

- benchmark numbers need reruns after the recent fairness and parser corrections
- query-only synthetic data still requires training/eval evidence before adoption
- full-generation synthetic data requires even stronger evidence because answer validity and distractor quality are higher-risk
- mixed synthetic training should only be considered if both tracks survive individual evaluation

## Recommended Reading Of Repo Outputs

Treat the repository in this order:

1. trust the canonical data artifacts, split manifest, code, and tests as the current source of truth
2. trust the documented workflows as the current reproduction path
3. treat the merged approved pools and train-ready synthetic artifacts as experimental handoff material for future training/eval
4. treat committed result JSON files and saved model outputs as historical context until experiments are rerun
