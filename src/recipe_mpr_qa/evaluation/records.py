from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    phase: str
    provider: str
    model_name: str
    split: str
    example_id: str
    prompt_version: str
    raw_response: str
    parsed_choice: str | None
    predicted_option_id: str | None
    gold_option_id: str
    is_correct: bool
    latency_ms: float | None
    response_rationale: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "provider": self.provider,
            "model_name": self.model_name,
            "split": self.split,
            "example_id": self.example_id,
            "prompt_version": self.prompt_version,
            "raw_response": self.raw_response,
            "parsed_choice": self.parsed_choice,
            "predicted_option_id": self.predicted_option_id,
            "gold_option_id": self.gold_option_id,
            "is_correct": self.is_correct,
            "latency_ms": self.latency_ms,
            "response_rationale": self.response_rationale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionRecord":
        return cls(
            run_id=payload["run_id"],
            phase=payload["phase"],
            provider=payload["provider"],
            model_name=payload["model_name"],
            split=payload["split"],
            example_id=payload["example_id"],
            prompt_version=payload["prompt_version"],
            raw_response=payload["raw_response"],
            parsed_choice=payload.get("parsed_choice"),
            predicted_option_id=payload.get("predicted_option_id"),
            gold_option_id=payload["gold_option_id"],
            is_correct=bool(payload["is_correct"]),
            latency_ms=payload.get("latency_ms"),
            response_rationale=payload.get("response_rationale"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class JudgmentRecord:
    run_id: str
    phase: str
    provider: str
    model_name: str
    split: str
    example_id: str
    prediction_run_id: str
    predicted_option_id: str | None
    gold_option_id: str
    judge_model_name: str
    ingredient_alignment: float
    constraint_satisfaction: float
    reasoning_quality: float
    overall_verdict: str
    rationale: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "provider": self.provider,
            "model_name": self.model_name,
            "split": self.split,
            "example_id": self.example_id,
            "prediction_run_id": self.prediction_run_id,
            "predicted_option_id": self.predicted_option_id,
            "gold_option_id": self.gold_option_id,
            "judge_model_name": self.judge_model_name,
            "ingredient_alignment": self.ingredient_alignment,
            "constraint_satisfaction": self.constraint_satisfaction,
            "reasoning_quality": self.reasoning_quality,
            "overall_verdict": self.overall_verdict,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JudgmentRecord":
        return cls(
            run_id=payload["run_id"],
            phase=payload["phase"],
            provider=payload["provider"],
            model_name=payload["model_name"],
            split=payload["split"],
            example_id=payload["example_id"],
            prediction_run_id=payload["prediction_run_id"],
            predicted_option_id=payload.get("predicted_option_id"),
            gold_option_id=payload["gold_option_id"],
            judge_model_name=payload["judge_model_name"],
            ingredient_alignment=float(payload["ingredient_alignment"]),
            constraint_satisfaction=float(payload["constraint_satisfaction"]),
            reasoning_quality=float(payload["reasoning_quality"]),
            overall_verdict=payload["overall_verdict"],
            rationale=payload["rationale"],
            metadata=payload.get("metadata", {}),
        )


def write_prediction_records(
    records: list[PredictionRecord] | tuple[PredictionRecord, ...], output_path: Path | str
) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=True, separators=(",", ":")) for record in records]
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_prediction_records(input_path: Path | str) -> tuple[PredictionRecord, ...]:
    resolved_path = Path(input_path)
    lines = [line for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(PredictionRecord.from_dict(json.loads(line)) for line in lines)


def write_judgment_records(
    records: list[JudgmentRecord] | tuple[JudgmentRecord, ...], output_path: Path | str
) -> None:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=True, separators=(",", ":")) for record in records]
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_judgment_records(input_path: Path | str) -> tuple[JudgmentRecord, ...]:
    resolved_path = Path(input_path)
    lines = [line for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(JudgmentRecord.from_dict(json.loads(line)) for line in lines)
