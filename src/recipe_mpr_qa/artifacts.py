from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    root_dir: Path
    run_dir: Path
    configs_dir: Path
    augmentation_dir: Path
    slm_dir: Path
    llm_dir: Path
    judge_dir: Path
    checkpoints_dir: Path
    manifests_dir: Path

    def component_metrics_path(self, component: str) -> Path:
        return self.run_dir / component / "metrics.json"

    def component_predictions_path(self, component: str, split: str) -> Path:
        return self.run_dir / component / f"{split}_predictions.jsonl"

    def resolved_config_path(self, config_name: str = "resolved_config.json") -> Path:
        return self.configs_dir / config_name

    def summary_path(self) -> Path:
        return self.manifests_dir / "run_summary.json"


def ensure_run_layout(run_id: str, root_dir: Path | str = Path("artifacts/runs")) -> RunLayout:
    base_dir = Path(root_dir)
    run_dir = base_dir / run_id
    configs_dir = run_dir / "configs"
    augmentation_dir = run_dir / "augmentation"
    slm_dir = run_dir / "slm"
    llm_dir = run_dir / "llm"
    judge_dir = run_dir / "judge"
    checkpoints_dir = slm_dir / "checkpoints"
    manifests_dir = run_dir / "manifests"
    for path in (
        run_dir,
        configs_dir,
        augmentation_dir,
        slm_dir,
        llm_dir,
        judge_dir,
        checkpoints_dir,
        manifests_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return RunLayout(
        run_id=run_id,
        root_dir=base_dir,
        run_dir=run_dir,
        configs_dir=configs_dir,
        augmentation_dir=augmentation_dir,
        slm_dir=slm_dir,
        llm_dir=llm_dir,
        judge_dir=judge_dir,
        checkpoints_dir=checkpoints_dir,
        manifests_dir=manifests_dir,
    )


def write_json(data: Mapping[str, Any], output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
