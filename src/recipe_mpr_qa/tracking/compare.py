from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from recipe_mpr_qa.tracking.models import RegistryEntry, RunManifest
from recipe_mpr_qa.tracking.registry import (
    DEFAULT_MLOPS_ROOT,
    get_run_stage,
    read_run_manifest,
)


def _comparison_row(manifest: RunManifest, stage: str | None) -> dict[str, Any]:
    metrics = dict(manifest.metrics)
    return {
        "run_id": manifest.run_id,
        "run_type": manifest.run_type,
        "stage": stage or "",
        "status": manifest.status,
        "model_name": str(manifest.model.get("name", "")),
        "parent_run_id": manifest.parent_run_id,
        "overall": metrics.get("overall"),
        "parse_failures": metrics.get("parse_failures"),
        "total": metrics.get("total"),
        "best_metric": metrics.get("best_metric"),
        "finished_at": manifest.finished_at,
    }


def build_run_comparison(
    run_ids: Sequence[str],
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> list[dict[str, Any]]:
    rows = []
    for run_id in run_ids:
        manifest = read_run_manifest(run_id, mlops_root=mlops_root)
        rows.append(_comparison_row(manifest, get_run_stage(run_id, mlops_root=mlops_root)))
    return rows


def format_run_table(entries: Sequence[RegistryEntry]) -> str:
    rows = [
        {
            "run_id": entry.run_id,
            "run_type": entry.run_type,
            "stage": entry.stage,
            "status": entry.status,
            "model_name": entry.model_name,
        }
        for entry in entries
    ]
    return _format_table(rows, ("run_id", "run_type", "stage", "status", "model_name"))


def format_run_comparison_table(rows: Sequence[dict[str, Any]]) -> str:
    return _format_table(
        rows,
        ("run_id", "run_type", "stage", "status", "model_name", "overall", "parse_failures"),
    )


def write_comparison_report(rows: Sequence[dict[str, Any]], output_path: Path | str) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(list(rows), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return resolved_path


def _format_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    header = {column: column for column in columns}
    normalized_rows = [header, *rows]
    widths = {
        column: max(len(str(row.get(column, ""))) for row in normalized_rows)
        for column in columns
    }
    lines = []
    lines.append("  ".join(column.ljust(widths[column]) for column in columns))
    lines.append("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        lines.append(
            "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        )
    return "\n".join(lines)
