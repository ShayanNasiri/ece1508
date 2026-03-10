# Recipe-MPR QA Specification

## 1. Project Objective

The project evaluates whether specialized SLM pipelines can match or exceed a general LLM baseline on Recipe-MPR while keeping the workflow reproducible, inspectable, and easy to rerun on dedicated hardware.

The implementation target in this repository is the full experiment stack up to the point where the team can launch real augmentation, training, inference, and judge runs on a GPU or local model-serving machine and collect results through consistent artifacts.

## 2. Task Definition

- Dataset source: `data/500QA.json`
- Task: five-way multiple-choice recipe recommendation
- Example count: 500
- Query type flags:
  - `Specific`
  - `Commonsense`
  - `Negated`
  - `Analogical`
  - `Temporal`
- Gold answer: exactly one option id per example

## 3. Canonical Dataset Contract

The canonical processed dataset is written to `data/processed/recipe_mpr_qa.jsonl`.

Each example follows the `RecipeExample` schema:

- `example_id`: stable id from `rmpr-0001` to `rmpr-0500`
- `query`: normalized query text
- `options`: ordered list of 5 `RecipeOption` entries
- `answer_option_id`: gold option id
- `query_type_flags`: boolean map over the five task categories
- `correctness_explanation`: evidence map preserved from the raw dataset
- `source_metadata`: raw source index, path, and normalization metadata

Normalization policy:

- strip only outer whitespace from `query`
- preserve option text and explanation content
- preserve `<INFERRED>` markers and raw option ordering

Validation policy:

- required keys must exist
- query and option text must be non-empty
- exactly 5 options per example
- option ids must be unique within an example
- `answer_option_id` must exist in the options
- query-type keys must exactly match the expected schema
- `correctness_explanation` values must be non-empty strings or non-empty string lists

## 4. Split Policy

The project uses one committed split manifest: `data/processed/primary_split.json`.

- train: 350 examples
- validation: 75 examples
- test: 75 examples
- seed: `1508`
- strategy: stratified by query-type signature

All baselines and training runs are expected to use this split unless an explicit alternative analysis is documented separately.

## 5. Additional Data Artifacts

### 5.1 Augmentation Artifacts

Synthetic query augmentation is persisted as JSONL under `artifacts/runs/<run_id>/augmentation/augmented_examples.jsonl`.

Synthetic examples reuse the `RecipeExample` schema with:

- `example_id`: `<source_example_id>-aug-<NNN>`
- identical options and gold answer as the source example
- inherited query-type flags and correctness explanation
- `source_metadata.synthetic = true`
- `source_metadata.source_example_id`
- `source_metadata.teacher_model_name`
- `source_metadata.prompt_version`

Augmentation constraints:

- generate only from the train split
- do not alter the option set
- do not alter the gold answer
- each synthetic query must be non-empty
- ids must be stable and unique within the artifact

## 6. Experiment Config Contracts

Configs are TOML files under `configs/` and are loaded through typed config models:

- `DataConfig`
- `OutputConfig`
- `TrackingConfig`
- `AugmentationConfig`
- `VanillaSLMConfig`
- `FineTuneConfig`
- `CausalBaselineConfig`
- `CausalFineTuneConfig`
- `LLMRunConfig`
- `JudgeConfig`

Command-specific bundles:

- `AugmentationRunConfig`
- `SLMExperimentConfig`
- `LLMExperimentConfig`
- `JudgeExperimentConfig`

Shared config behavior:

- `output.run_id` is required
- `output.artifacts_root` defaults to `artifacts/runs`
- `data.dataset_path` defaults to the committed processed dataset
- `data.split_manifest_path` defaults to the committed split manifest
- tracking is optional and disabled by default

## 7. Experiment Pipelines

### 7.1 Vanilla SLM Baseline

Definition:

- model: pretrained `distilbert-base-uncased`
- no task fine-tuning
- encode query and each option independently
- mean-pool token embeddings with attention masking
- score query-option similarity with cosine similarity
- choose the highest-scoring option

Outputs:

- `artifacts/runs/<run_id>/slm/<split>_predictions.jsonl`
- `artifacts/runs/<run_id>/slm/<split>_metrics.json`
- run summary manifest

### 7.2 Fine-Tuned SLM

Definition:

- model: DistilBERT sequence classifier over query-option pairs
- training view: one positive and four negative rows per original question
- training runtime: Hugging Face Trainer
- evaluation view: reconstruct question-level predictions by taking the highest option score per example

Supported training modes:

- original-only fine-tuning
- original-plus-augmented fine-tuning

Outputs:

- `artifacts/runs/<run_id>/slm/checkpoints/`
- `artifacts/runs/<run_id>/slm/validation_predictions.jsonl`
- `artifacts/runs/<run_id>/slm/test_predictions.jsonl`
- `artifacts/runs/<run_id>/slm/metrics.json`
- run summary manifest

### 7.3 Causal SLM Baseline and Fine-Tuning

Definition:

- model family: instruct-style causal LM, with `HuggingFaceTB/SmolLM2-135M-Instruct` as the default config target
- prompt shape: single user message with `User request`, five `A-E` options, and an instruction to reply with only one letter
- baseline evaluation: generate a short completion, parse the returned letter, and map it back to the original option id
- fine-tuning: supervise the model on `prompt -> gold letter` sequences using the same chat-oriented format
- adapter strategy: LoRA-ready by default for causal fine-tuning

Outputs:

- `artifacts/runs/<run_id>/slm/<split>_predictions.jsonl`
- `artifacts/runs/<run_id>/slm/checkpoints/`
- `artifacts/runs/<run_id>/slm/validation_predictions.jsonl`
- `artifacts/runs/<run_id>/slm/test_predictions.jsonl`
- `artifacts/runs/<run_id>/slm/metrics.json`
- run summary manifest

### 7.4 General LLM Baseline

Definition:

- provider: Ollama
- task prompt: shared multiple-choice template
- parser: extract `A` through `E` from raw model output
- prediction: map parsed letter back to the original option id

Operational behavior:

- retry failed requests
- capture latency
- optionally resume from partially written prediction JSONL

Outputs:

- `artifacts/runs/<run_id>/llm/<split>_predictions.jsonl`
- `artifacts/runs/<run_id>/llm/<split>_metrics.json`
- run summary manifest

### 7.5 LLM as Judge

Definition:

- provider: Ollama
- input: query, predicted option, gold option, gold evidence, and optional model rationale
- output: strict JSON

Required judge fields:

- `ingredient_alignment`: integer 1-5
- `constraint_satisfaction`: integer 1-5
- `reasoning_quality`: integer 1-5
- `overall_verdict`: `correct | partially_correct | incorrect`
- `rationale`: non-empty string

Outputs:

- `artifacts/runs/<run_id>/judge/judgments.jsonl`
- `artifacts/runs/<run_id>/judge/metrics.json`
- run summary manifest

## 8. Prompt Contracts

### 8.1 Multiple-Choice Prompt

Used for baseline LLM inference.

- query shown once
- options rendered as `A` through `E`
- model instructed to return only the letter

### 8.2 Augmentation Prompt

Used for teacher-generated rewrites.

- request JSON only
- request a fixed number of rewrites
- preserve task semantics and answer choice

Expected JSON shape:

```json
{"rewrites": ["query one", "query two"]}
```

### 8.3 Causal SLM Prompt

Used for SmolLM2-style baseline prompting and supervised fine-tuning.

- rendered as a chat-oriented user message
- includes `User request`
- lists options as `A` through `E`
- instructs the model to reply with only one letter

Training view:

- the supervised target is the gold letter only
- the fine-tuning sequence is effectively `prompt -> assistant: <gold_letter>`

### 8.4 Judge Prompt

Used for LLM-as-a-judge evaluation.

- request JSON only
- provide query, predicted option, gold option, gold evidence, and rationale
- require numeric rubric scores plus verdict and rationale

## 9. Output Schemas

### 9.1 PredictionRecord

Every model inference path writes `PredictionRecord` JSONL rows:

- `run_id`
- `phase`
- `provider`
- `model_name`
- `split`
- `example_id`
- `prompt_version`
- `raw_response`
- `parsed_choice`
- `predicted_option_id`
- `gold_option_id`
- `is_correct`
- `latency_ms`
- `response_rationale`
- `metadata`

### 9.2 JudgmentRecord

Judge runs write `JudgmentRecord` JSONL rows:

- `run_id`
- `phase`
- `provider`
- `model_name`
- `split`
- `example_id`
- `prediction_run_id`
- `predicted_option_id`
- `gold_option_id`
- `judge_model_name`
- `ingredient_alignment`
- `constraint_satisfaction`
- `reasoning_quality`
- `overall_verdict`
- `rationale`
- `metadata`

### 9.3 Run Summary

Each run can emit `artifacts/runs/<run_id>/manifests/run_summary.json` containing:

- resolved config snapshot
- dataset metadata
- artifact paths
- prediction metrics
- judgment metrics
- optional component-specific metadata

## 10. Metrics

Primary metric:

- exact-match accuracy on question-level predictions

Secondary metrics:

- per-query-type accuracy
- per-query-signature accuracy
- average judge rubric scores
- judge verdict distribution

## 11. CLI Contract

Supported commands:

- `prepare-data`
- `validate-data`
- `dataset-stats`
- `export-split`
- `generate-augmentation`
- `train-slm`
- `evaluate-slm`
- `run-llm`
- `judge-predictions`
- `summarize-run`

Behavioral requirements:

- legacy data commands remain lightweight
- experiment commands are config-first
- commands write resolved configs and structured artifacts
- failures return non-zero exit codes with human-readable errors

## 12. Tracking

MLflow integration is optional and activated only when the tracking extra is installed and configs enable it.

The tracking adapter logs:

- experiment name
- run name
- tags
- parameters
- flattened numeric metrics
- artifact files and directories

DVC is out of scope for this implementation pass.

## 13. Testing Requirements

The non-training test suite must cover:

- data validation and deterministic artifacts
- config parsing and default resolution
- augmentation prompt parsing and artifact writing
- vanilla SLM scoring on toy embeddings
- causal SLM prompt rendering and letter parsing
- causal SLM orchestration with mocked Hugging Face components
- fine-tune orchestration with mocked Hugging Face components
- LLM inference retries, parsing, resume behavior, and record writing
- judge prompt parsing and judgment serialization
- MLflow adapter behavior with a mocked backend
- CLI dry-run behavior across augmentation, SLM, LLM, judge, and summary commands

The local test suite is not required to perform real training or live model queries. The repository is considered experiment-ready when these contracts pass locally and the commands can be pointed at a proper GPU/model-serving environment without further code changes.
