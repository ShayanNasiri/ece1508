# Project Overview

Recipe-MPR QA studies whether a fine-tuned small language model can solve the Recipe-MPR task competitively against a larger general-purpose language model while remaining cheaper to run and easier to inspect.

For the repository landing page, see the [README](../README.md). For the technical contract, see the [Technical Spec](spec.md).

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

## Task Definition

Each Recipe-MPR example contains:

- one natural-language query
- five candidate recipe options
- one correct option id
- query-type flags
- correctness explanations from the source dataset

The repo treats this as a standardized five-way multiple-choice task. The canonical processed dataset preserves the source example structure, while model-facing prompts render the options as letters `A` through `E`.

## Core Project Goal

The central project question is:

Can a fine-tuned small language model outperform, or at least remain competitive with, a larger language model on Recipe-MPR while staying practical for local use?

This leads to two project needs:

- a stable and reproducible dataset and evaluation foundation
- a fair comparison path for local LLM evaluation and SLM fine-tuning

## What The Repository Currently Covers

The current repository already implements the project foundation and experiment scaffolding:

- raw dataset validation and normalization
- canonical processed dataset creation
- deterministic train, validation, and test split generation
- typed dataset loaders and option-scoring expansion utilities
- standardized model-facing prompt rendering and response parsing
- local LLM evaluation scripts
- SLM fine-tuning scaffolding for prompt-completion training
- optional train-only query augmentation through conservative rule-based rewrites

## What Remains Experimental

The repository is not yet at the stage where committed benchmark numbers should be treated as final project conclusions.

Important experimental boundaries:

- committed evaluation result files are historical outputs and must be treated as provisional
- experiment runs need to be rerun after recent benchmark-correctness fixes
- the repo currently provides the evaluation and fine-tuning workflow, but not a finalized benchmark report

## Work Completed So Far

This summary aligns the current repo with the project proposal while reflecting the code that now exists:

- the shared data foundation is implemented and tested
- the split contract is deterministic and committed
- prompt formatting and response parsing are standardized
- local LLM evaluation is runnable through the current scripts
- SLM fine-tuning is scaffolded and can optionally consume a train-only augmentation artifact
- the benchmark path has been tightened through prompt-order leakage mitigation and parser hardening

## Course Context

The proposal artifacts remain available for the original course framing:

- [Project Briefing PDF](project_briefing.pdf)
- [Project Briefing TeX](project_briefing.tex)

Use those documents for the historical proposal narrative. Use the current docs set for the repository's actual behavior, workflow, and status.
