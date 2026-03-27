from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from recipe_mpr_qa.tracking.artifacts import utc_now_iso
from recipe_mpr_qa.tracking.models import (
    REGISTRY_STAGES,
    RegistryEntry,
    RunManifest,
)

DEFAULT_MLOPS_ROOT = Path("mlops")


def get_run_dir(run_id: str, mlops_root: Path | str = DEFAULT_MLOPS_ROOT) -> Path:
    return Path(mlops_root) / "runs" / run_id


def get_run_manifest_path(run_id: str, mlops_root: Path | str = DEFAULT_MLOPS_ROOT) -> Path:
    return get_run_dir(run_id, mlops_root) / "run_manifest.json"


def get_run_summary_path(run_id: str, mlops_root: Path | str = DEFAULT_MLOPS_ROOT) -> Path:
    return get_run_dir(run_id, mlops_root) / "summary.json"


def get_registry_path(
    name: str,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> Path:
    return Path(mlops_root) / "registry" / f"{name}.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_manifest(
    manifest: RunManifest,
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> Path:
    output_path = get_run_manifest_path(manifest.run_id, mlops_root)
    _write_json(output_path, manifest.to_dict())
    return output_path


def read_run_manifest(
    run_id: str,
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> RunManifest:
    payload = _read_json(get_run_manifest_path(run_id, mlops_root), None)
    if payload is None:
        raise FileNotFoundError(f"No run manifest found for run_id={run_id}")
    return RunManifest.from_dict(payload)


def list_run_manifests(
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> list[RunManifest]:
    root = Path(mlops_root) / "runs"
    if not root.exists():
        return []
    manifests: list[RunManifest] = []
    for manifest_path in sorted(root.glob("*/run_manifest.json")):
        manifests.append(RunManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8"))))
    return manifests


def _select_metric_summary(manifest: RunManifest) -> dict[str, Any]:
    metrics = dict(manifest.metrics)
    summary: dict[str, Any] = {}
    for key in ("overall", "parse_failures", "total_correct", "total", "best_metric", "epoch", "global_step"):
        if key in metrics:
            summary[key] = metrics[key]
    return summary


def _primary_artifact_path(manifest: RunManifest) -> str | None:
    preferred_names = ("final_model_dir", "result_json", "output_dir")
    for name in preferred_names:
        for artifact in manifest.output_artifacts:
            if artifact.name == name:
                return artifact.path
    return manifest.output_artifacts[0].path if manifest.output_artifacts else None


def build_run_summary(manifest: RunManifest, *, stage: str) -> dict[str, Any]:
    if stage not in REGISTRY_STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    return {
        "run_id": manifest.run_id,
        "run_type": manifest.run_type,
        "status": manifest.status,
        "stage": stage,
        "model_name": manifest.model.get("name", ""),
        "created_at": manifest.created_at,
        "finished_at": manifest.finished_at,
        "parent_run_id": manifest.parent_run_id,
        "metrics": _select_metric_summary(manifest),
        "artifact_path": _primary_artifact_path(manifest),
    }


def write_run_summary(
    manifest: RunManifest,
    *,
    stage: str,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> Path:
    output_path = get_run_summary_path(manifest.run_id, mlops_root)
    _write_json(output_path, build_run_summary(manifest, stage=stage))
    return output_path


def load_registry_entries(
    name: str,
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> list[RegistryEntry]:
    payload = _read_json(get_registry_path(name, mlops_root), [])
    return [RegistryEntry.from_dict(item) for item in payload]


def write_registry_entries(
    name: str,
    entries: Iterable[RegistryEntry],
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> Path:
    output_path = get_registry_path(name, mlops_root)
    _write_json(output_path, [entry.to_dict() for entry in entries])
    return output_path


def _upsert_registry_entry(entries: list[RegistryEntry], entry: RegistryEntry) -> list[RegistryEntry]:
    updated_entries = [existing for existing in entries if existing.entry_id != entry.entry_id]
    updated_entries.append(entry)
    updated_entries.sort(key=lambda item: item.created_at)
    return updated_entries


def _build_run_registry_entry(manifest: RunManifest, *, stage: str) -> RegistryEntry:
    return RegistryEntry(
        entry_id=manifest.run_id,
        run_id=manifest.run_id,
        run_type=manifest.run_type,
        stage=stage,
        status=manifest.status,
        model_name=str(manifest.model.get("name", "")),
        created_at=manifest.created_at,
        updated_at=manifest.finished_at or manifest.created_at,
        parent_run_id=manifest.parent_run_id,
        artifact_path=_primary_artifact_path(manifest),
        metrics=_select_metric_summary(manifest),
        metadata={
            "entry_kind": "run",
        },
    )


def _build_model_registry_entry(manifest: RunManifest, *, stage: str) -> RegistryEntry | None:
    if manifest.run_type != "train":
        return None
    artifact_path = _primary_artifact_path(manifest)
    return RegistryEntry(
        entry_id=manifest.run_id,
        run_id=manifest.run_id,
        run_type=manifest.run_type,
        stage=stage,
        status=manifest.status,
        model_name=str(manifest.model.get("name", "")),
        created_at=manifest.created_at,
        updated_at=manifest.finished_at or manifest.created_at,
        parent_run_id=manifest.parent_run_id,
        artifact_path=artifact_path,
        metrics=_select_metric_summary(manifest),
        metadata={
            "entry_kind": "model",
            "artifact_path": artifact_path,
        },
    )


def register_run(
    manifest: RunManifest,
    *,
    stage: str,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> None:
    write_run_manifest(manifest, mlops_root=mlops_root)
    write_run_summary(manifest, stage=stage, mlops_root=mlops_root)

    run_entries = load_registry_entries("runs", mlops_root=mlops_root)
    updated_run_entries = _upsert_registry_entry(run_entries, _build_run_registry_entry(manifest, stage=stage))
    write_registry_entries("runs", updated_run_entries, mlops_root=mlops_root)

    model_entry = _build_model_registry_entry(manifest, stage=stage)
    if model_entry is not None:
        model_entries = load_registry_entries("models", mlops_root=mlops_root)
        updated_model_entries = _upsert_registry_entry(model_entries, model_entry)
        write_registry_entries("models", updated_model_entries, mlops_root=mlops_root)


def list_registered_runs(
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
    run_type: str | None = None,
    status: str | None = None,
    stage: str | None = None,
) -> list[RegistryEntry]:
    entries = load_registry_entries("runs", mlops_root=mlops_root)
    filtered = []
    for entry in entries:
        if run_type is not None and entry.run_type != run_type:
            continue
        if status is not None and entry.status != status:
            continue
        if stage is not None and entry.stage != stage:
            continue
        filtered.append(entry)
    return filtered


def get_run_stage(
    run_id: str,
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> str | None:
    for entry in load_registry_entries("runs", mlops_root=mlops_root):
        if entry.run_id == run_id:
            return entry.stage
    return None


def promote_run(
    run_id: str,
    stage: str,
    *,
    mlops_root: Path | str = DEFAULT_MLOPS_ROOT,
) -> RegistryEntry:
    if stage not in REGISTRY_STAGES:
        raise ValueError(f"Unsupported stage: {stage}")

    run_entries = load_registry_entries("runs", mlops_root=mlops_root)
    target_run_entry = None
    updated_run_entries: list[RegistryEntry] = []
    for entry in run_entries:
        if entry.run_id == run_id:
            updated_entry = replace(entry, stage=stage, updated_at=utc_now_iso())
            updated_run_entries.append(updated_entry)
            target_run_entry = updated_entry
        else:
            updated_run_entries.append(entry)
    if target_run_entry is None:
        raise FileNotFoundError(f"No tracked run found for run_id={run_id}")
    write_registry_entries("runs", updated_run_entries, mlops_root=mlops_root)

    model_entries = load_registry_entries("models", mlops_root=mlops_root)
    updated_model_entries: list[RegistryEntry] = []
    for entry in model_entries:
        if entry.run_id == run_id:
            updated_model_entries.append(replace(entry, stage=stage, updated_at=utc_now_iso()))
        else:
            updated_model_entries.append(entry)
    if updated_model_entries:
        write_registry_entries("models", updated_model_entries, mlops_root=mlops_root)

    manifest = read_run_manifest(run_id, mlops_root=mlops_root)
    write_run_summary(manifest, stage=stage, mlops_root=mlops_root)
    return target_run_entry
