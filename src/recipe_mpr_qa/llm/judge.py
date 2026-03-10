from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.data.models import PreparedDataset
from recipe_mpr_qa.evaluation.records import (
    JudgmentRecord,
    PredictionRecord,
    read_judgment_records,
    write_judgment_records,
)
from recipe_mpr_qa.llm.prompts import JUDGE_PROMPT_SPEC, build_judge_prompt, parse_judge_response


def judge_predictions(
    *,
    dataset: PreparedDataset,
    prediction_records: Sequence[PredictionRecord],
    client,
    run_id: str,
    model_name: str,
    output_path: Path | str,
    temperature: float = 0.0,
    resume: bool = True,
) -> tuple[JudgmentRecord, ...]:
    output_path = Path(output_path)
    existing_records: dict[str, JudgmentRecord] = {}
    if resume and output_path.exists():
        existing_records = {record.example_id: record for record in read_judgment_records(output_path)}

    examples_by_id = {example.example_id: example for example in dataset.examples}
    completed = dict(existing_records)
    for record in prediction_records:
        if record.example_id in completed:
            continue
        example = examples_by_id[record.example_id]
        predicted_text = next(
            (
                option.text
                for option in example.options
                if option.option_id == record.predicted_option_id
            ),
            "",
        )
        prompt = build_judge_prompt(
            example=example,
            predicted_option_text=predicted_text,
            model_rationale=record.response_rationale,
        )
        started = time.perf_counter()
        response_text = client.generate(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
        )
        _latency_ms = (time.perf_counter() - started) * 1000.0
        parsed = parse_judge_response(response_text)
        completed[record.example_id] = JudgmentRecord(
            run_id=run_id,
            phase="phase4",
            provider="ollama",
            model_name=record.model_name,
            split=record.split,
            example_id=record.example_id,
            prediction_run_id=record.run_id,
            predicted_option_id=record.predicted_option_id,
            gold_option_id=record.gold_option_id,
            judge_model_name=model_name,
            ingredient_alignment=float(parsed["ingredient_alignment"]),
            constraint_satisfaction=float(parsed["constraint_satisfaction"]),
            reasoning_quality=float(parsed["reasoning_quality"]),
            overall_verdict=parsed["overall_verdict"],
            rationale=parsed["rationale"],
            metadata={
                "prompt_version": JUDGE_PROMPT_SPEC.version,
                "raw_response": response_text,
                "latency_ms": _latency_ms,
            },
        )
        ordered_records = tuple(completed[key] for key in sorted(completed))
        write_judgment_records(ordered_records, output_path)
    return tuple(completed[key] for key in sorted(completed))
