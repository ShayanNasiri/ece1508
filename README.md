# Recipe-MPR QA

Recipe-MPR QA is a course project and research-style repository centered on one question: can a fine-tuned small language model stay competitive with, or outperform, a larger general-purpose language model on Recipe-MPR while remaining lightweight enough for local use?

Recipe-MPR is a five-way multiple-choice recipe recommendation task. Each example contains a natural-language food preference query, five candidate recipe descriptions, and one correct answer. The task is harder than keyword matching because many queries depend on commonsense reasoning, negation, analogical cues, or temporal constraints.

## Final Results

The full scaling and fine-tuning study uncovered a sharp **capability threshold** between 360M and 1.7B parameters: below the threshold, fine-tuning and synthetic-data augmentation produce no meaningful gains over random chance, while above it both yield large, additive improvements.

| Model | Mode | Test accuracy |
|---|---|---|
| SmolLM2-135M-Instruct | zero-shot | 18.7% |
| SmolLM2-360M-Instruct | zero-shot | 25.3% |
| SmolLM2-1.7B-Instruct | zero-shot | 18.7% |
| SmolLM2-135M-Instruct | LoRA fine-tune (Run C) | 22.7% |
| SmolLM2-360M-Instruct | LoRA fine-tune (Run G) | 17.3% |
| SmolLM2-360M-Instruct | LoRA fine-tune + 25% synthetic (Run H) | 21.3% |
| **SmolLM2-1.7B-Instruct** | **LoRA fine-tune (Run I)** | **78.7%** |
| **SmolLM2-1.7B-Instruct** | **LoRA fine-tune + 25% synthetic (Run K)** | **82.7%** |
| Qwen2.5-3B-Instruct | zero-shot | 86.7% |
| DeepSeek-R1-Distill-Qwen-7B | zero-shot | 97.3% |

Headline finding: a 1.7B SmolLM2 fine-tuned with LoRA on the train split plus a 25% synthetic-data ratio reaches **82.7%** test accuracy, closing most of the gap to a 3B-parameter zero-shot baseline at roughly half the parameter count.

- **Final report:** [docs/final-report/report.tex](docs/final-report/report.tex)
- **Bar chart:** [llm_evaluation/results/results_bar_chart.png](llm_evaluation/results/results_bar_chart.png)
- **All result JSONs:** [llm_evaluation/results/](llm_evaluation/results/)

## Demo

For a guided walkthrough of the input/output path end to end (load data, build a multiple-choice prompt, parse a model response, load a saved evaluation result), open [demo.ipynb](demo.ipynb) in the repo root and run the cells in order. The notebook is self-contained and only requires `pip install -r requirements.txt` plus `pip install -e .`.

## Current Repository Status

The repository currently provides:

- stable source-of-truth data preparation and deterministic train, validation, and test splits
- a shared prompt/parsing contract for model-facing multiple-choice evaluation
- local LLM evaluation utilities, including both generative and loglikelihood scoring modes
- SLM fine-tuning scaffolding for prompt-completion training
- optional train-only rule-based query augmentation
- experimental dual-track synthetic-data generation, review, approval, and training-admission workflows through the OpenAI API
- optional local-first tracked wrappers for train and eval runs
- regression tests for the current data, synthetic, evaluation, and tracking surfaces

Support levels in the current repo:

- stable: canonical dataset, split manifest, prompt/parsing contract, direct evaluation and direct fine-tuning
- optional but supported: train-only augmentation and tracked MLOps wrappers
- implemented but experimental: synthetic-data generation plus the current handoff artifacts under `data/processed/synthetic/`
- historical only: old JSON result files under `llm_evaluation/results/`, saved model outputs under `outputs/`, and the proposal/report artifacts in `docs/`

## Quickstart

Install the package from the repository root. The editable install is the expected starting point if you want to use `recipe-mpr-qa` or `python -m recipe_mpr_qa.cli`.

```bash
pip install -e .
```

If you would rather pin loose dependencies without the editable install, [`requirements.txt`](requirements.txt) mirrors the dependencies declared in `pyproject.toml` (core + the `slm` extras + Jupyter for the demo notebook):

```bash
pip install -r requirements.txt
```

Optional extras (recommended path for active development):

```bash
pip install -e ".[dev]"
pip install -e ".[slm]"
pip install -e ".[mlops]"
```

Current extras declared in `pyproject.toml`:

- `dev`: `pytest`
- `slm`: `torch`, `transformers`, `accelerate`, `peft`, `datasets`, `trl`
- `mlops`: `mlflow`

Prepare the canonical processed dataset and split manifest:

```bash
recipe-mpr-qa prepare-data \
  --input data/500QA.json \
  --output data/processed/recipe_mpr_qa.jsonl \
  --split-output data/processed/primary_split.json
```

Run local multiple-choice evaluation in generative mode:

```bash
python llm_evaluation/mc_eval.py \
  --model deepseek-r1:7b \
  --backend ollama \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json
```

Run evaluation in loglikelihood mode:

```bash
python llm_evaluation/mc_eval.py \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --backend huggingface \
  --data data/processed/recipe_mpr_qa.jsonl \
  --split-manifest data/processed/primary_split.json \
  --config llm_evaluation/config.json \
  --eval-mode loglikelihood
```

Run fine-tuning on the canonical train split only:

```bash
python finetuning/finetune.py
```

Use an existing train-only augmentation or train-ready synthetic artifact:

```bash
python finetuning/finetune.py \
  --augmented-train-path data/processed/synthetic/train_query_ratio025.jsonl
```

For the synthetic-data R&D workflow, see [docs/synthetic_data.md](docs/synthetic_data.md). That workflow requires an `OPENAI_API_KEY`, either in the shell environment or in a repo-root `.env` file. A template lives at [.env.example](.env.example).

Run the regression suite:

```bash
pytest -q
```

## Final Experimental State

All planned evaluations are complete and the result files in `llm_evaluation/results/` are the source of truth for the final report. The 1.7B + 25%-synthetic configuration (Run K) is the headline result at **82.7%** test accuracy; see the table in [Final Results](#final-results) above for the full landscape.

The synthetic-data handoff artifacts under `data/processed/synthetic/` are still in the repo and were used to train Run H (360M) and Run K (1.7B):

- `data/processed/synthetic/query_approved_merged.jsonl`
- `data/processed/synthetic/full_approved_merged.jsonl`
- `data/processed/synthetic/train_query_ratio025.jsonl`
- `data/processed/synthetic/train_full_ratio010.jsonl`
- `data/processed/synthetic/train_mixed_ratio025.jsonl`

The SLURM scripts that produced the final results live in [`slurm/`](slurm/). The bar chart and per-query-type breakdowns are reproducible from the result JSONs via the helper exposed in `recipe_mpr_qa.evaluation.results` (see [demo.ipynb](demo.ipynb)).

## Documentation Map

- [Docs Hub](docs/index.md)
- [Project Overview](docs/project_overview.md)
- [Technical Spec](docs/spec.md)
- [Workflows](docs/workflows.md)
- [Architecture](docs/architecture.md)
- [Experiment Status](docs/experiments_status.md)
- [Synthetic Data R&D](docs/synthetic_data.md)
- [MLOps Layer](docs/mlops.md)

## Repository Structure

- `data/`: raw, processed, and derived dataset artifacts
- `docs/`: canonical project docs plus the final report under `docs/final-report/`
- `src/recipe_mpr_qa/`: canonical data, formatting, synthetic, evaluation, and tracking implementation
- `llm_evaluation/`: repo-root evaluation wrapper and result artifacts (incl. `results_bar_chart.png`)
- `finetuning/`: repo-root fine-tuning wrapper and related materials
- `slurm/`: SLURM batch scripts used to run training and evaluation on the cluster
- `outputs/`: saved training checkpoints and run configs
- `tests/`: regression coverage for the current implementation
- `demo.ipynb`: end-to-end demo notebook (input \u2192 prompt \u2192 parse \u2192 result summary)
- `requirements.txt`: pinned-loose dependency list mirroring `pyproject.toml`
