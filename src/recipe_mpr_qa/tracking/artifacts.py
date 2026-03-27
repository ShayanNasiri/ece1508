from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from recipe_mpr_qa.tracking.models import ArtifactRef


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(run_type: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{run_type}-{timestamp}-{uuid4().hex[:8]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative_path(path: Path, repo_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = repo_root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def build_artifact_ref(
    *,
    name: str,
    path: Path | str,
    repo_root: Path,
    artifact_type: str | None = None,
    include_hash: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    resolved_path = Path(path)
    exists = resolved_path.exists()
    normalized_type = artifact_type
    if normalized_type is None:
        if resolved_path.is_file():
            normalized_type = "file"
        elif resolved_path.is_dir():
            normalized_type = "dir"
        else:
            normalized_type = "missing"

    sha256 = None
    size_bytes = None
    if exists and resolved_path.is_file():
        size_bytes = resolved_path.stat().st_size
        if include_hash:
            sha256 = file_sha256(resolved_path)

    return ArtifactRef(
        name=name,
        path=repo_relative_path(resolved_path, repo_root),
        artifact_type=normalized_type,
        exists=exists,
        sha256=sha256,
        size_bytes=size_bytes,
        metadata=metadata or {},
    )


def collect_git_metadata(repo_root: Path) -> dict[str, Any]:
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "git_commit": commit_proc.stdout.strip() or None,
            "git_dirty": bool(status_proc.stdout.strip()),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "git_commit": None,
            "git_dirty": None,
        }


def build_environment_summary() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }
