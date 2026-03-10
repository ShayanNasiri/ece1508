from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentContext:
    experiment_name: str
    run_name: str
    tags: Mapping[str, str] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tags": dict(self.tags),
            "params": dict(self.params),
        }


def build_mlflow_tags(
    *,
    phase: str,
    provider: str,
    model_name: str,
    split: str,
    prompt_version: str | None = None,
) -> dict[str, str]:
    tags = {
        "phase": phase,
        "provider": provider,
        "model_name": model_name,
        "split": split,
    }
    if prompt_version is not None:
        tags["prompt_version"] = prompt_version
    return tags
