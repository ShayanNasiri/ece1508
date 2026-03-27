# Experiment Status

This page documents the current state of the experiment stack and the main caveats that matter when interpreting the repository's outputs.

For project framing, see [Project Overview](project_overview.md). For runnable commands, see [Workflows](workflows.md).

## What Is Implemented

The repository currently supports these experiment surfaces:

- canonical processed dataset preparation from the raw Recipe-MPR source
- deterministic train, validation, and test split generation
- standardized model-facing prompt rendering
- response parsing back into answer letters
- local LLM evaluation through `llm_evaluation/mc_eval.py`
- SLM fine-tuning through `finetuning/finetune.py`
- optional train-only query augmentation for the fine-tuning path

This means the project has a reproducible data and workflow foundation, plus runnable evaluation and fine-tuning entrypoints.

## Recent Corrective Changes

The benchmark path has already been tightened by several important fixes:

### Answer-position leakage mitigation

The canonical processed dataset still preserves source option ordering, but model-facing prompts now use deterministic per-example option shuffling in the evaluation and fine-tuning paths. This removes the earlier shortcut where the correct answer position could be learned from raw ordering.

### Parser hardening

The response parser now prefers explicit answer signals and is less willing to over-credit chain-of-thought outputs that merely enumerate options. This makes multiple-choice scoring more conservative and more defensible.

### Optional train-only augmentation

The repo now supports a separate augmentation artifact that rewrites only training queries. This expands the SLM training input space without modifying the canonical processed dataset, split manifest, evaluation workflow, or default fine-tuning behavior.

## Current Caveats

### Committed result files are provisional

The JSON outputs currently committed under `llm_evaluation/results/` should be treated as historical run artifacts, not final benchmark evidence. They predate important benchmark-correctness changes and need to be regenerated before they are cited as authoritative.

### Saved training artifacts are historical

The artifacts under `outputs/` reflect prior training runs and should be interpreted the same way: useful as historical context, but not as final project evidence after the benchmark corrections.

### Workflow is more trustworthy than the old numbers

The repository should currently be read as:

- a reliable workflow and data foundation
- a runnable evaluation and fine-tuning stack
- a benchmark path that still requires fresh experiment runs for final reporting

## Known Interpretation Limits

These points still matter when discussing experimental conclusions:

- the canonical processed dataset intentionally preserves source ordering, so fairness depends on using the model-facing prompt path rather than inspecting raw option position
- the current evaluation path still uses free-form generation plus answer parsing rather than direct option scoring
- fine-tuning and evaluation commands are implemented and documented, but reproducible final benchmark claims require rerunning them after the recent fixes

## Recommended Reading Of Repo Outputs

Treat the repository in this order:

1. trust the canonical data artifacts, split manifest, code, and tests as the current source of truth
2. trust the documented workflows as the current reproduction path
3. treat committed result JSON files and saved model outputs as historical context until the experiments are rerun
