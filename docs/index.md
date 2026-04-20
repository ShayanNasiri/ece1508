# Documentation Hub

This directory contains the canonical project documentation for Recipe-MPR QA. The root [README](../README.md) is the landing page; the pages here divide ownership by purpose so the docs can stay aligned with the live repository.

## Read This First

Read based on what you need:

- New collaborator or evaluator: start with [Project Overview](project_overview.md)
- Running commands and reproducing artifacts: go to [Workflows](workflows.md)
- Understanding artifact invariants and public interfaces: read [Technical Spec](spec.md)
- Understanding package boundaries and end-to-end flow: read [Architecture](architecture.md)
- Understanding the live repo state and current caveats: read [Experiment Status](experiments_status.md)
- Understanding the synthetic-data methodology and current pilot state: read [Synthetic Data R&D](synthetic_data.md)
- Understanding tracked wrappers, registries, and lineage: read [MLOps Layer](mlops.md)

## Canonical Page Ownership

- [Project Overview](project_overview.md): project motivation, task framing, implemented scope, support tiers, and current narrative
- [Technical Spec](spec.md): artifact contracts, public types, evaluation modes, command surface, and invariants
- [Workflows](workflows.md): install prerequisites, runnable commands, required environment setup, and output expectations
- [Architecture](architecture.md): module responsibilities, data flow, wrapper boundaries, and package layout
- [Experiment Status](experiments_status.md): live snapshot of repo state, synthetic pilot outcomes, and what remains unvalidated
- [Synthetic Data R&D](synthetic_data.md): literature grounding, method rationale, artifact lifecycle, approval rules, and pilot interpretation
- [MLOps Layer](mlops.md): tracked run commands, manifest/registry layout, lineage behavior, and MLflow mirroring

## Canonical Vs Historical

These pages are the current source of truth for repository behavior:

- `README.md`
- `docs/*.md`

These files remain historical course context:

- [Project Briefing PDF](project_briefing.pdf)
- [Project Briefing TeX](project_briefing.tex)

The proposal/report artifacts are still useful for the original project framing, but they should not be treated as the authoritative description of the current repo. Use the Markdown docs and the live code for that.
