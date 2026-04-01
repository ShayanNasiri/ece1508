# Project Overview

Recipe-MPR QA studies whether a fine-tuned small language model can solve the Recipe-MPR task competitively against a larger general-purpose language model while remaining cheaper to run and easier to inspect.

For the repository landing page, see the [README](../README.md). For exact contracts and runnable commands, see [Technical Spec](spec.md) and [Workflows](workflows.md).

## Why Recipe-MPR Matters

Recipe-MPR is not a simple keyword-matching benchmark. Each example asks for the best recipe match among five candidates, but the user request may depend on several kinds of reasoning:

- direct constraints such as ingredients, cooking methods, or dish type
- commonsense interpretation
- negation such as excluding an ingredient or preparation style
- analogical framing
- temporal cues

The repository tracks these patterns through five query-type flags:

- `Specific`
- `Commonsense`
- `Negated`
- `Analogical`
- `Temporal`

## Core Repository Goal

The central project question is:

Can a fine-tuned small language model outperform, or at least remain competitive with, a larger language model on Recipe-MPR while staying practical for local use?

That requires more than a model checkpoint. The repository first needs:

- a stable and reproducible dataset and split contract
- a fair prompt/parsing and evaluation path
- a fine-tuning path that reuses the same task contract
- clear experimental boundaries so exploratory extensions do not get mistaken for benchmark evidence

## What The Repository Currently Implements

The current repository already provides:

- raw dataset validation and normalization
- canonical processed dataset creation
- deterministic train, validation, and test split generation
- typed dataset loaders and option-scoring expansion utilities
- standardized model-facing prompt rendering and response parsing
- local LLM evaluation scripts
- SLM fine-tuning scaffolding for prompt-completion training
- optional train-only query augmentation through conservative rule-based rewrites
- experimental dual-track synthetic-data workflows for query-only and full-generation artifacts
- optional tracked train/eval wrappers with local registries and artifact lineage

## Support Levels

The repo now needs to be read in layers:

- stable: canonical processed dataset, split manifest, prompt/parsing contract, direct evaluation, and direct fine-tuning
- optional but supported: train-only augmentation and tracked MLOps wrappers
- implemented but experimental: synthetic-data generation, review, approval, and train-admission workflows
- historical only: older result JSON files, saved model outputs, and the proposal/report artifacts

## Current Experimental Snapshot

As of April 1, 2026:

- the synthetic-data extension is no longer just planned; the repo contains real pilot artifacts under `data/processed/synthetic/`
- query-only pilot outputs currently look stronger than full-generation outputs under the existing gates
- no training or held-out evaluation conclusions have yet been drawn from those pilot files
- committed evaluation results and saved training outputs remain provisional historical artifacts rather than final project evidence

The repository should therefore be read as a strong experimental foundation with active R&D extensions, not as a finished benchmark report.

## What Still Needs Fresh Evidence

Important limits still matter for the final story:

- benchmark numbers should be rerun after the recent fairness and parser corrections
- synthetic-data tracks must still earn inclusion through fresh training and evaluation
- full-generation synthetic data has a higher burden of proof than query-only data
- the historical outputs under `llm_evaluation/results/` and `outputs/` should not be cited as final claims

## Course Context

The proposal artifacts remain available for the original course framing:

- [Project Briefing PDF](project_briefing.pdf)
- [Project Briefing TeX](project_briefing.tex)

Use those files for historical context. Use the current Markdown docs for live repository behavior, workflow, and status.
