from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

BENCHMARK_SCHEMA_VERSION = "benchmark.v1"
BENCHMARK_CONTRACT_VERSION = "recipe-mpr-benchmark-v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_path(path: Path | str) -> str:
    resolved_path = Path(path)
    digest = hashlib.sha256()
    with resolved_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: Mapping[str, Any] | Sequence[Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def collect_git_metadata(repo_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    metadata = {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        metadata["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def build_environment_summary() -> dict[str, Any]:
    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": Path.cwd().as_posix(),
        "hostname": platform.node(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CONDA_DEFAULT_ENV",
                "VIRTUAL_ENV",
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
            )
            if os.environ.get(key)
        },
    }
    return summary


@dataclass(frozen=True)
class BenchmarkContract:
    version: str
    dataset_path: str
    split_manifest_path: str
    prompt_version: str
    parser_version: str | None
    option_shuffle_seed: int | None
    option_shuffle_strategy: str
    split_name: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_path": self.dataset_path,
            "split_manifest_path": self.split_manifest_path,
            "prompt_version": self.prompt_version,
            "parser_version": self.parser_version,
            "option_shuffle_seed": self.option_shuffle_seed,
            "option_shuffle_strategy": self.option_shuffle_strategy,
            "split_name": self.split_name,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkContract":
        return cls(
            version=str(payload["version"]),
            dataset_path=str(payload["dataset_path"]),
            split_manifest_path=str(payload["split_manifest_path"]),
            prompt_version=str(payload["prompt_version"]),
            parser_version=(
                str(payload["parser_version"]) if payload.get("parser_version") is not None else None
            ),
            option_shuffle_seed=(
                int(payload["option_shuffle_seed"])
                if payload.get("option_shuffle_seed") is not None
                else None
            ),
            option_shuffle_strategy=str(payload.get("option_shuffle_strategy", "none")),
            split_name=(str(payload["split_name"]) if payload.get("split_name") is not None else None),
            notes=tuple(str(item) for item in payload.get("notes", [])),
        )


@dataclass(frozen=True)
class BenchmarkRunManifest:
    schema_version: str
    run_id: str
    component: str
    status: str
    created_at: str
    updated_at: str
    contract: BenchmarkContract
    dataset_sha256: str
    split_sha256: str
    config_hash: str | None
    model: Mapping[str, Any]
    environment: Mapping[str, Any]
    git: Mapping[str, Any]
    artifact_paths: Mapping[str, str]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    slurm: Mapping[str, Any] = field(default_factory=dict)
    parent_artifacts: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "component": self.component,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "contract": self.contract.to_dict(),
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "config_hash": self.config_hash,
            "model": dict(self.model),
            "environment": dict(self.environment),
            "git": dict(self.git),
            "artifact_paths": dict(self.artifact_paths),
            "metrics": dict(self.metrics),
            "slurm": dict(self.slurm),
            "parent_artifacts": dict(self.parent_artifacts),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkRunManifest":
        return cls(
            schema_version=str(payload["schema_version"]),
            run_id=str(payload["run_id"]),
            component=str(payload["component"]),
            status=str(payload["status"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            contract=BenchmarkContract.from_dict(payload["contract"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            split_sha256=str(payload["split_sha256"]),
            config_hash=(str(payload["config_hash"]) if payload.get("config_hash") is not None else None),
            model=payload.get("model", {}),
            environment=payload.get("environment", {}),
            git=payload.get("git", {}),
            artifact_paths=payload.get("artifact_paths", {}),
            metrics=payload.get("metrics", {}),
            slurm=payload.get("slurm", {}),
            parent_artifacts=payload.get("parent_artifacts", {}),
            notes=tuple(str(item) for item in payload.get("notes", [])),
        )


def build_benchmark_contract(
    *,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    prompt_version: str,
    parser_version: str | None,
    option_shuffle_seed: int | None,
    option_shuffle_strategy: str,
    split_name: str | None = None,
    notes: tuple[str, ...] = (),
) -> BenchmarkContract:
    return BenchmarkContract(
        version=BENCHMARK_CONTRACT_VERSION,
        dataset_path=Path(dataset_path).as_posix(),
        split_manifest_path=Path(split_manifest_path).as_posix(),
        prompt_version=prompt_version,
        parser_version=parser_version,
        option_shuffle_seed=option_shuffle_seed,
        option_shuffle_strategy=option_shuffle_strategy,
        split_name=split_name,
        notes=notes,
    )


def build_run_manifest(
    *,
    run_id: str,
    component: str,
    status: str,
    contract: BenchmarkContract,
    dataset_path: Path | str,
    split_manifest_path: Path | str,
    config_payload: Mapping[str, Any] | None,
    model: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    metrics: Mapping[str, Any] | None = None,
    slurm: Mapping[str, Any] | None = None,
    parent_artifacts: Mapping[str, str] | None = None,
    notes: tuple[str, ...] = (),
    created_at: str | None = None,
) -> BenchmarkRunManifest:
    now = utc_now_iso()
    created = created_at or now
    return BenchmarkRunManifest(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        run_id=run_id,
        component=component,
        status=status,
        created_at=created,
        updated_at=now,
        contract=contract,
        dataset_sha256=sha256_path(dataset_path),
        split_sha256=sha256_path(split_manifest_path),
        config_hash=hash_payload(config_payload) if config_payload is not None else None,
        model=dict(model),
        environment=build_environment_summary(),
        git=collect_git_metadata(Path.cwd()),
        artifact_paths={key: str(value) for key, value in artifact_paths.items()},
        metrics=dict(metrics or {}),
        slurm=dict(slurm or {}),
        parent_artifacts={key: str(value) for key, value in (parent_artifacts or {}).items()},
        notes=notes,
    )


def write_run_manifest(manifest: BenchmarkRunManifest, output_path: Path | str) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_run_manifest(input_path: Path | str) -> BenchmarkRunManifest:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return BenchmarkRunManifest.from_dict(payload)
