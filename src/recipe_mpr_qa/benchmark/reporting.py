from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.benchmark.provenance import BENCHMARK_CONTRACT_VERSION, BenchmarkRunManifest, read_run_manifest


def collect_benchmark_manifests(root_dir: Path | str) -> tuple[BenchmarkRunManifest, ...]:
    root = Path(root_dir)
    manifests = []
    for path in sorted(root.rglob("benchmark_manifest.json")):
        manifests.append(read_run_manifest(path))
    return tuple(manifests)


def _load_summary(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    resolved_path = Path(path)
    if not resolved_path.exists():
        return {}
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def build_benchmark_table_rows(
    manifests: Sequence[BenchmarkRunManifest],
    *,
    required_contract_version: str = BENCHMARK_CONTRACT_VERSION,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        if manifest.status != "completed":
            continue
        if manifest.contract.version != required_contract_version:
            continue
        summary = _load_summary(manifest.artifact_paths.get("summary"))
        prediction_metrics = summary.get("prediction_metrics", manifest.metrics)
        accuracy = prediction_metrics.get("accuracy")
        if accuracy is None:
            continue
        rows.append(
            {
                "run_id": manifest.run_id,
                "component": manifest.component,
                "status": manifest.status,
                "split": manifest.contract.split_name,
                "model_name": manifest.model.get("name"),
                "model_interface": manifest.model.get("interface"),
                "decoding_mode": manifest.model.get("decoding_mode"),
                "provider": manifest.model.get("provider"),
                "train_data": manifest.model.get("train_data"),
                "synthetic_policy": manifest.model.get("synthetic_policy"),
                "accuracy": accuracy,
                "correct_count": prediction_metrics.get("correct_count"),
                "example_count": prediction_metrics.get("example_count"),
                "parse_failure_count": prediction_metrics.get("parse_failure_count"),
                "parse_failure_rate": prediction_metrics.get("parse_failure_rate"),
                "accuracy_ci95_low": prediction_metrics.get("accuracy_ci95_low"),
                "accuracy_ci95_high": prediction_metrics.get("accuracy_ci95_high"),
                "prompt_version": manifest.contract.prompt_version,
                "parser_version": manifest.contract.parser_version,
                "contract_version": manifest.contract.version,
                "summary_path": manifest.artifact_paths.get("summary"),
                "notes": "; ".join(manifest.notes),
            }
        )
    return rows


def write_benchmark_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | str,
    format: str | None = None,
) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = format or resolved_path.suffix.lstrip(".") or "json"
    if resolved_format == "json":
        resolved_path.write_text(
            json.dumps(list(rows), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if resolved_format == "csv":
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with resolved_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return
    if resolved_format == "md":
        fieldnames = [
            "run_id",
            "model_name",
            "model_interface",
            "decoding_mode",
            "accuracy",
            "correct_count",
            "example_count",
            "parse_failure_count",
            "accuracy_ci95_low",
            "accuracy_ci95_high",
            "synthetic_policy",
            "notes",
        ]
        header = "| " + " | ".join(fieldnames) + " |"
        separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
        lines = [header, separator]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
        resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    raise ValueError(f"Unsupported benchmark table format: {resolved_format}")
