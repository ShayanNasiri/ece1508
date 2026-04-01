# Documentation Audit

This page is a maintainer-facing reconciliation matrix between the live repository and the documentation set. It is intentionally operational: the goal is to make it obvious which doc owns each surface and how stale assumptions were resolved.

Snapshot date: April 1, 2026.

## Reconciliation Matrix

| Surface | Live Repo Truth | Canonical Doc Owner | Action |
| --- | --- | --- | --- |
| Package CLI in `src/recipe_mpr_qa/cli.py` | CLI now spans base data, augmentation, synthetic workflows, tracked wrappers, and registry commands | [Workflows](workflows.md), [Technical Spec](spec.md) | rewrite docs to cover full command surface and stop presenting CLI as data-only |
| Evaluation in `src/recipe_mpr_qa/evaluation/mc_eval.py` | supports both `generative` and `loglikelihood`; loglikelihood requires Hugging Face backend | [Workflows](workflows.md), [Technical Spec](spec.md), [Experiment Status](experiments_status.md) | add explicit evaluation-mode coverage and remove stale implication that generative parsing is the only mode |
| Repo-root eval wrapper `llm_evaluation/mc_eval.py` | thin wrapper over package evaluation module | [Workflows](workflows.md), [Architecture](architecture.md) | document wrapper/package split clearly |
| Fine-tuning in `src/recipe_mpr_qa/slm/finetune.py` | accepts `--augmented-train-path` for legacy augmentation or reviewed train-ready synthetic artifacts | [Workflows](workflows.md), [Technical Spec](spec.md) | clarify training-input contract and handoff boundary |
| Synthetic artifacts in `src/recipe_mpr_qa/synthetic` | separate query-only and full-generation schemas, plus candidate, reviewed, approved, and train-ready states | [Synthetic Data R&D](synthetic_data.md), [Technical Spec](spec.md), [Architecture](architecture.md) | expand lifecycle docs and review-vs-approval semantics |
| OpenAI client in `src/recipe_mpr_qa/synthetic/openai.py` | loads `OPENAI_API_KEY` from env or upward `.env` search | [Workflows](workflows.md), [Synthetic Data R&D](synthetic_data.md), [Technical Spec](spec.md) | document `.env` behavior consistently |
| Tracking layer in `src/recipe_mpr_qa/tracking` | tracked wrappers record lineage and optionally mirror to MLflow | [MLOps Layer](mlops.md), [Architecture](architecture.md), [Technical Spec](spec.md) | rewrite MLOps docs to match live wrapper scope |
| Pilot synthetic outputs under `data/processed/synthetic/` | real pilot artifacts exist for query-only, full-generation, and mixed train-ready handoff | [Experiment Status](experiments_status.md), [Synthetic Data R&D](synthetic_data.md) | add dated live-status snapshot and treat files as experimental handoff material, not benchmark evidence |
| Historical outputs under `llm_evaluation/results/` and `outputs/` | still present, but predate current benchmark-correctness assumptions | [Experiment Status](experiments_status.md), [README](../README.md) | explicitly mark as historical/provisional |
| Package extras in `pyproject.toml` | `dev`, `slm`, `mlops` are active; `dashboard` is declared but not yet a documented workflow | [README](../README.md), [Workflows](workflows.md), [Technical Spec](spec.md) | document support level clearly |
| Historical proposal/report artifacts | useful for course context, not for live repo truth | [Docs Hub](index.md), `project_briefing.tex` | keep historical boundary explicit and update TeX minimally where needed |

## Current Ownership Rule

Each page should own one kind of truth:

- `README.md`: landing page and support-level summary
- `docs/index.md`: map of canonical vs historical docs
- `docs/project_overview.md`: project framing and implemented scope
- `docs/spec.md`: exact contracts and command surface
- `docs/workflows.md`: runnable commands and prerequisites
- `docs/architecture.md`: module boundaries and data flow
- `docs/experiments_status.md`: live state and caveats
- `docs/synthetic_data.md`: synthetic methodology and pilot interpretation
- `docs/mlops.md`: tracked wrapper behavior and registries

This page exists to keep that ownership model from drifting again as the repo expands.
