from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SlurmJobSpec:
    job_name: str
    command: tuple[str, ...]
    partition: str = "gpu"
    time_limit: str = "04:00:00"
    gpus: int = 1
    cpus_per_task: int = 4
    mem_gb: int = 32
    workdir: str | None = None
    output_path: str | None = None
    error_path: str | None = None
    dependency: str | None = None
    array: str | None = None
    environment: Mapping[str, str] | None = None


def render_sbatch_script(spec: SlurmJobSpec) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={spec.job_name}",
        f"#SBATCH --partition={spec.partition}",
        f"#SBATCH --time={spec.time_limit}",
        f"#SBATCH --gpus={spec.gpus}",
        f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
        f"#SBATCH --mem={spec.mem_gb}G",
    ]
    if spec.output_path:
        lines.append(f"#SBATCH --output={spec.output_path}")
    if spec.error_path:
        lines.append(f"#SBATCH --error={spec.error_path}")
    if spec.dependency:
        lines.append(f"#SBATCH --dependency={spec.dependency}")
    if spec.array:
        lines.append(f"#SBATCH --array={spec.array}")
    lines.extend(
        [
            "set -euo pipefail",
            "",
        ]
    )
    if spec.workdir:
        lines.append(f"cd {spec.workdir}")
    for key, value in sorted((spec.environment or {}).items()):
        lines.append(f"export {key}={value}")
    lines.extend(
        [
            "",
            " ".join(spec.command),
            "",
        ]
    )
    return "\n".join(lines)


def parse_sacct_rows(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = [item.strip() for item in lines[0].split("|")]
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        values = [item.strip() for item in line.split("|")]
        rows.append({header[index]: values[index] for index in range(min(len(header), len(values)))})
    return rows
