from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.evaluation.records import (
    PredictionRecord,
    read_prediction_records,
    write_prediction_records,
)
from recipe_mpr_qa.llm.prompts import (
    DEFAULT_PROMPT_SPEC,
    build_multiple_choice_prompt,
    parse_multiple_choice_response,
)


def run_llm_predictions(
    *,
    examples: Sequence[RecipeExample],
    client,
    run_id: str,
    provider: str,
    model_name: str,
    split: str,
    output_path: Path | str,
    prompt_version: str = DEFAULT_PROMPT_SPEC.version,
    temperature: float = 0.0,
    resume: bool = True,
) -> tuple[PredictionRecord, ...]:
    output_path = Path(output_path)
    existing_records: dict[str, PredictionRecord] = {}
    if resume and output_path.exists():
        existing_records = {
            record.example_id: record for record in read_prediction_records(output_path)
        }

    completed = dict(existing_records)
    for example in examples:
        if example.example_id in completed:
            continue
        prompt, letter_to_option_id = build_multiple_choice_prompt(
            query=example.query,
            options=example.options,
        )
        started = time.perf_counter()
        response_text = client.generate(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        parsed_choice = parse_multiple_choice_response(response_text)
        predicted_option_id = (
            letter_to_option_id[parsed_choice] if parsed_choice in letter_to_option_id else None
        )
        completed[example.example_id] = PredictionRecord(
            run_id=run_id,
            phase="phase3",
            provider=provider,
            model_name=model_name,
            split=split,
            example_id=example.example_id,
            prompt_version=prompt_version,
            raw_response=response_text,
            parsed_choice=parsed_choice,
            predicted_option_id=predicted_option_id,
            gold_option_id=example.answer_option_id,
            is_correct=predicted_option_id == example.answer_option_id,
            latency_ms=latency_ms,
            metadata={"option_mapping": letter_to_option_id},
        )
        ordered_records = tuple(completed[key] for key in sorted(completed))
        write_prediction_records(ordered_records, output_path)
    return tuple(completed[key] for key in sorted(completed))
