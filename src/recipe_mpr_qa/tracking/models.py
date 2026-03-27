from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

RUN_TYPES = ("train", "eval")
RUN_STATUSES = ("running", "completed", "failed")
REGISTRY_STAGES = ("baseline", "candidate", "validated", "archived")
SCHEMA_VERSION = "mlops-run-v1"


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: str
    artifact_type: str
    exists: bool
    sha256: str | None = None
    size_bytes: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "artifact_type": self.artifact_type,
            "exists": self.exists,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            name=str(payload["name"]),
            path=str(payload["path"]),
            artifact_type=str(payload["artifact_type"]),
            exists=bool(payload["exists"]),
            sha256=payload.get("sha256"),
            size_bytes=payload.get("size_bytes"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    run_type: str
    status: str
    created_at: str
    finished_at: str | None
    entrypoint: str
    command: tuple[str, ...]
    git_commit: str | None
    git_dirty: bool | None
    environment: Mapping[str, Any]
    input_artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    output_artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    model: Mapping[str, Any] = field(default_factory=dict)
    prompt: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    parent_run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_type not in RUN_TYPES:
            raise ValueError(f"Unsupported run_type: {self.run_type}")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "entrypoint": self.entrypoint,
            "command": list(self.command),
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "environment": dict(self.environment),
            "input_artifacts": [artifact.to_dict() for artifact in self.input_artifacts],
            "output_artifacts": [artifact.to_dict() for artifact in self.output_artifacts],
            "model": dict(self.model),
            "prompt": dict(self.prompt),
            "metrics": dict(self.metrics),
            "parent_run_id": self.parent_run_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        return cls(
            schema_version=str(payload["schema_version"]),
            run_id=str(payload["run_id"]),
            run_type=str(payload["run_type"]),
            status=str(payload["status"]),
            created_at=str(payload["created_at"]),
            finished_at=payload.get("finished_at"),
            entrypoint=str(payload["entrypoint"]),
            command=tuple(payload.get("command", [])),
            git_commit=payload.get("git_commit"),
            git_dirty=payload.get("git_dirty"),
            environment=payload.get("environment", {}),
            input_artifacts=tuple(
                ArtifactRef.from_dict(item) for item in payload.get("input_artifacts", [])
            ),
            output_artifacts=tuple(
                ArtifactRef.from_dict(item) for item in payload.get("output_artifacts", [])
            ),
            model=payload.get("model", {}),
            prompt=payload.get("prompt", {}),
            metrics=payload.get("metrics", {}),
            parent_run_id=payload.get("parent_run_id"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class RegistryEntry:
    entry_id: str
    run_id: str
    run_type: str
    stage: str
    status: str
    model_name: str
    created_at: str
    updated_at: str
    parent_run_id: str | None = None
    artifact_path: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in REGISTRY_STAGES:
            raise ValueError(f"Unsupported stage: {self.stage}")
        if self.run_type not in RUN_TYPES:
            raise ValueError(f"Unsupported run_type: {self.run_type}")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "run_id": self.run_id,
            "run_type": self.run_type,
            "stage": self.stage,
            "status": self.status,
            "model_name": self.model_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_run_id": self.parent_run_id,
            "artifact_path": self.artifact_path,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegistryEntry":
        return cls(
            entry_id=str(payload["entry_id"]),
            run_id=str(payload["run_id"]),
            run_type=str(payload["run_type"]),
            stage=str(payload["stage"]),
            status=str(payload["status"]),
            model_name=str(payload["model_name"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            parent_run_id=payload.get("parent_run_id"),
            artifact_path=payload.get("artifact_path"),
            metrics=payload.get("metrics", {}),
            metadata=payload.get("metadata", {}),
        )
