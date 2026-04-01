# Synthetic Data R&D

This page is the source of truth for the repository's synthetic-data expansion phase. It treats synthetic data as a research and development extension of the project, not as a default training shortcut.

The implementation currently supports two separate tracks:

- `query_only`: keep authentic options and the authentic gold label fixed, then generate new train-only queries anchored to real train examples
- `full_generation`: generate new Recipe-MPR-style multiple-choice examples with a new query, five new options, one correct answer, and four distractors

The two tracks do not sit at the same evidence tier:

- query-only is lower risk because answer validity is checked against authentic options
- full generation is higher risk because it must justify distractor quality, uniqueness of the correct answer, and distributional fit to Recipe-MPR

## Research Questions

The synthetic-data phase should answer these questions before training decisions are locked:

1. What concrete gaps in the authentic train split are worth targeting: sparse query-type signatures, lexical narrowness, or repeatable model error clusters?
2. Which query-only methods preserve the original answer while still adding useful diversity?
3. Can full-generation methods create new Recipe-MPR-like items with one defensible answer and four plausible distractors?
4. What quality thresholds are strong enough for training admission?
5. How much synthetic data can be added before it starts to distort the benchmark or dominate authentic signal?
6. For each track, should the final status be `adopt`, `keep experimental`, or `reject`?

## Literature Review

### Query-Only And Controlled Generation

| Source | Main Use | Why It Matters Here |
| --- | --- | --- |
| Ding et al. 2024, Findings ACL: <https://aclanthology.org/2024.findings-acl.97/> | survey of LLM-based data augmentation | frames method choices and common failure modes |
| Self-Instruct: <https://arxiv.org/abs/2212.10560> | synthetic supervision from model-generated data | supports structured synthetic generation with filtering instead of naive expansion |
| Promptagator: <https://arxiv.org/abs/2209.11755> | targeted synthetic query generation | supports using synthetic queries to improve retrieval-like behavior from small supervision |
| Targeted Data Generation: <https://arxiv.org/abs/2305.17804> | generate data around known model weaknesses | supports error-cluster-driven generation instead of uniform generation |
| LLM2LLM: <https://aclanthology.org/2024.findings-acl.388/> | iterative data augmentation in low-data regimes | supports staged augmentation with method comparison |
| ConQuest: <https://aclanthology.org/2021.wnut-1.25/> | answer-aware question generation | close analogue for anchored, answer-preserving query generation |
| CoDa: <https://arxiv.org/abs/2404.00415> | conditioned data generation | supports conditioning synthetic examples on task structure rather than free-form prompting |

Default interpretation:

- query-only synthetic data is defensible when it is constrained, answer-aware, filtered, and kept subordinate to authentic data
- uniform, unfiltered generation is not defensible enough for this repo

### Full Generation And Distractor Quality

| Source | Main Use | Why It Matters Here |
| --- | --- | --- |
| Alhazmi et al. 2024, EMNLP survey: <https://aclanthology.org/2024.emnlp-main.799.pdf> | distractor-generation survey | frames distractor plausibility and MCQ-specific quality concerns |
| Liang et al. 2018: <https://aclanthology.org/W18-0533/> | distractor generation for MCQ tasks | early grounding for MCQ-specific generation quality |
| LLM-based distractor prompting: <https://arxiv.org/abs/2307.16338> | prompt-driven distractor generation | supports model-assisted MCQ construction with strong review requirements |

Default interpretation:

- full generation must be treated as benchmark-preserving item construction, not just stronger augmentation
- the burden of proof is higher because the repository must justify option quality, answer uniqueness, and fit to Recipe-MPR style

### Caution On Synthetic Mixtures

| Source | Risk | Why It Matters Here |
| --- | --- | --- |
| Shumailov et al. 2023: <https://arxiv.org/abs/2305.17493> | model collapse under recursive synthetic training | supports keeping authentic data as the majority signal |
| Dohmatob et al. 2024: <https://arxiv.org/abs/2404.01413> | recursion and synthetic-distribution drift | supports conservative mixture caps and strong filtering |

Default interpretation:

- this phase should keep a real-data majority
- higher synthetic ratios need stronger evidence, especially for full generation

## Artifact Lifecycle

The repository uses separate artifacts for the two tracks and keeps lifecycle states distinct.

### Candidate

Generated output that has not yet been judged by the reviewer model.

### Reviewed

Generated output that has gone through structured model review and now carries:

- `review_status`
- `review_scores`
- a review summary
- failure-mode metadata when present

Reviewed does not mean training-eligible.

### Approved

Reviewed output that also survived repo-side admission filters such as:

- train-only parent or seed constraints
- duplicate and near-duplicate checks
- threshold checks on the review scores
- synthetic-mode-specific safety and quality gates

Approved is the first stage that can feed training admission.

### Train-ready

The output of `build-synthetic-train`, which converts approved synthetic artifacts into one train-only `RecipeExample` JSONL file that can be passed to fine-tuning.

## Artifact Model

### Query-only artifacts

- candidates, reviewed artifacts, and approved artifacts reuse `RecipeExample`
- required provenance lives in `source_metadata`
- required fields include:
  - `synthetic_mode=query_only`
  - `generator_model`
  - `generation_prompt_version`
  - `approval_batch_id`
  - `review_status`
  - `review_scores`
  - `created_at`
  - `intended_query_type_target`
  - `parent_example_id`

### Full-generation artifacts

- candidates, reviewed artifacts, and approved artifacts use a separate nested schema
- each record stores:
  - `recipe_example`
  - `provenance`
- full-generation provenance additionally requires:
  - `seed_example_ids`
  - `distractor_generation_method`
  - `distribution_fit_score`

### Training-admission artifact

- `build-synthetic-train` converts approved query-only and/or full-generation artifacts into one train-only `RecipeExample` JSONL artifact
- this artifact is the only synthetic file that should be passed to fine-tuning

## Validation And Approval Gates

Shared hard gates:

- schema validity
- train-only isolation
- provenance completeness
- duplicate and near-duplicate filtering
- no id collisions with authentic examples

Pilot defaults for this phase:

- query-only pilot: 75 train parents
- full-generation pilot: 40 train seeds
- do not expand a track beyond pilot size until its audit passes

Query-only review gates:

- semantic preservation
- constraint preservation
- answer preservation against the authentic options
- low leakage risk
- acceptable language quality

Query-only approval defaults:

- `answer_preservation >= 0.95`
- `semantic_preservation >= 0.90`
- `constraint_preservation >= 0.90`
- `language_quality >= 0.90`
- `leakage_risk <= 0.20`

Full-generation review gates:

- exactly one defensible correct answer
- plausible but still incorrect distractors
- low leakage risk
- acceptable language quality
- acceptable distributional fit to Recipe-MPR

Full-generation approval defaults:

- `single_answer_validity >= 0.95`
- `distractor_plausibility >= 0.90`
- `distribution_fit_score >= 0.70`
- `language_quality >= 0.90`
- `leakage_risk <= 0.20`

Manual audit policy:

- inspect every approved pilot item
- after pilot, inspect at least 25% of approved query-only items
- after pilot, inspect at least 50% of approved full-generation items, with a floor of 25 items when available

## Current Pilot State

The repo now contains real pilot artifacts under `data/processed/synthetic/`.

As of April 1, 2026:

- query-only pilot:
  - `query_candidates_pilot.jsonl`: 75
  - `query_reviewed_pilot.jsonl`: 75
  - `query_approved_pilot.jsonl`: 60
  - `train_query_pilot_all.jsonl`: 60
  - `train_query_pilot_ratio010.jsonl`: 35
- full-generation pilot:
  - `full_candidates_pilot.jsonl`: 40
  - `full_reviewed_pilot.jsonl`: 40
  - `full_approved_pilot.jsonl`: 15
  - `train_full_pilot_all.jsonl`: 15
- mixed train-ready pilot handoff:
  - `train_mixed_pilot_all.jsonl`: 75

Interpretation:

- the workflow has been exercised with a real OpenAI-backed run
- query-only currently has the stronger approval yield
- full-generation remains the more fragile and more experimental path
- these artifacts support future training and evaluation, but they do not establish usefulness yet

Smoke files in the same directory should be treated as debug/probing artifacts, not as the main pilot record.

## Ratio Study And Decision Rules

The ratio study remains staged.

1. Re-establish the no-synthetic baseline with fresh runs.
2. Test query-only alone.
3. Test full-generation alone.
4. Test mixed synthetic training only if both tracks pass individually.

Recommended ratios for this phase:

- query-only: `0.10`, `0.25`, `0.50`
- full-generation: `0.05`, `0.10`, `0.25`
- mixed: total `0.25` with `70%` query-only and `30%` full-generation inside the synthetic pool

Adoption defaults:

- query-only needs either at least `+1.5` absolute validation accuracy or `+3.0` targeted query-type macro accuracy without meaningful regressions
- full-generation needs either at least `+2.0` absolute validation accuracy or `+4.0` targeted query-type macro accuracy plus clean audits
- any track that is neutral or harmful on held-out test should be documented as investigated but not adopted

## OpenAI-Only Practical Path

The implementation uses the OpenAI Responses API with structured JSON outputs.

Default repo model split:

- `gpt-5.4-mini` for bulk candidate generation
- `gpt-5.4` for structured review and adjudication

Environment requirement:

Preferred repo-local setup:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Put that line in a `.env` file at the repository root. The synthetic-data client will look for `.env` from the current working directory upward through parent directories.

PowerShell alternative:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

## CLI Workflow

Query-only:

```bash
recipe-mpr-qa generate-synthetic-query \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/query_candidates.jsonl

recipe-mpr-qa review-synthetic-query \
  --input data/processed/synthetic/query_candidates.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --output data/processed/synthetic/query_reviewed.jsonl

recipe-mpr-qa approve-synthetic-query \
  --input data/processed/synthetic/query_reviewed.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/query_approved.jsonl \
  --approval-batch-id pilot-q
```

Full-generation:

```bash
recipe-mpr-qa generate-synthetic-full \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/full_candidates.jsonl

recipe-mpr-qa review-synthetic-full \
  --input data/processed/synthetic/full_candidates.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --output data/processed/synthetic/full_reviewed.jsonl

recipe-mpr-qa approve-synthetic-full \
  --input data/processed/synthetic/full_reviewed.jsonl \
  --dataset data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --output data/processed/synthetic/full_approved.jsonl \
  --approval-batch-id pilot-f
```

Training admission:

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

Then pass the built training artifact into fine-tuning:

```bash
python finetuning/finetune.py \
  --augmented-train-path data/processed/synthetic/train_synthetic_025.jsonl
```

## Reporting Expectations

Any final report or benchmark narrative should answer these points explicitly:

- why synthetic data was investigated
- why the repo kept separate query-only and full-generation tracks
- what literature justified the chosen methods
- what quality thresholds were used
- how mixture ratios were chosen
- whether each track ended in `adopt`, `keep experimental`, or `reject`

An acceptable rigorous outcome is:

- both capabilities were implemented
- only one passed validation strongly enough for training use
- the other remained documented as experimental
