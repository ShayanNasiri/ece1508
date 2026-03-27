# Documentation Hub

This directory contains the canonical project documentation for Recipe-MPR QA. The root [README](../README.md) is the landing page; the pages here provide the deeper project, workflow, and technical detail.

## Read This First

Read based on what you need:

- New to the project: start with [Project Overview](project_overview.md)
- Running the repo: go to [Workflows](workflows.md)
- Understanding data contracts and artifact invariants: read [Technical Spec](spec.md)
- Understanding how the codebase fits together: read [Architecture](architecture.md)
- Understanding current benchmark caveats and recent fixes: read [Experiment Status](experiments_status.md)
- Understanding tracked runs and registries: read [MLOps Layer](mlops.md)

## Documentation Set

- [Project Overview](project_overview.md): project motivation, task definition, scope, and work completed so far
- [Technical Spec](spec.md): canonical artifact contracts, core types, prompt behavior, parser expectations, and CLI contract
- [Workflows](workflows.md): installation, data preparation, augmentation, evaluation, and fine-tuning commands
- [Architecture](architecture.md): high-level data flow and module responsibilities
- [Experiment Status](experiments_status.md): current state of the experiment stack, corrective changes, and important caveats
- [MLOps Layer](mlops.md): tracked wrapper commands, local artifact layout, run registries, and optional MLflow mirroring

## Historical Project Context

The proposal artifacts remain in the repository as historical course context:

- [Project Briefing PDF](project_briefing.pdf)
- [Project Briefing TeX](project_briefing.tex)

These proposal files are useful for the original framing, but they are not the main source of truth for the current repository behavior. For that, use the docs pages in this directory and the root README.
