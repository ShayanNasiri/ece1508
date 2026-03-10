from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.artifacts import write_json
from recipe_mpr_qa.data.models import PreparedDataset
from recipe_mpr_qa.evaluation.metrics import summarize_judgment_metrics, summarize_prediction_metrics
from recipe_mpr_qa.evaluation.records import JudgmentRecord, PredictionRecord


def build_run_summary(
    *,
    run_id: str,
    component: str,
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    prediction_records: Sequence[PredictionRecord] = (),
    judgment_records: Sequence[JudgmentRecord] = (),
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "run_id": run_id,
        "component": component,
        "dataset_metadata": dict(dataset.metadata),
        "config": dict(config),
        "artifact_paths": dict(artifact_paths),
        "prediction_metrics": summarize_prediction_metrics(prediction_records, dataset),
        "judgment_metrics": summarize_judgment_metrics(judgment_records),
    }
    if extra_metadata:
        summary["metadata"] = dict(extra_metadata)
    return summary


def write_run_summary(summary: Mapping[str, Any], output_path: Path | str) -> None:
    write_json(summary, output_path)
