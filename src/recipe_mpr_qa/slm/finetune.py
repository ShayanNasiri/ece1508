from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.data.loaders import build_option_scoring_examples
from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.evaluation.records import PredictionRecord, write_prediction_records


def _require_transformers():
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] to use fine-tuning") from exc
    return AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


def _require_torch_dataset():
    return object


class _RowDataset(_require_torch_dataset()):
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


def _build_rows(
    examples: Sequence[RecipeExample],
    tokenizer,
    *,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scoring_examples = build_option_scoring_examples(
        examples,
        tokenizer=tokenizer,
        tokenizer_kwargs={"truncation": True, "max_length": max_length},
    )
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for item in scoring_examples:
        row = dict(item.tokenized_inputs or {})
        row["labels"] = item.label
        rows.append(row)
        metadata.append(
            {
                "example_id": item.example_id,
                "option_id": item.option_id,
                "option_index": item.option_index,
                "group_size": item.group_size,
                "label": item.label,
            }
        )
    return rows, metadata


def _prediction_records_from_logits(
    *,
    logits,
    metadata: Sequence[Mapping[str, Any]],
    examples: Sequence[RecipeExample],
    run_id: str,
    model_name: str,
    split: str,
) -> tuple[PredictionRecord, ...]:
    grouped_scores: dict[str, list[tuple[str, float]]] = {}
    for index, row in enumerate(metadata):
        logit_row = logits[index]
        if isinstance(logit_row, (list, tuple)):
            score = float(logit_row[-1])
        else:
            score = float(logit_row)
        grouped_scores.setdefault(row["example_id"], []).append(
            (row["option_index"], row["option_id"], score)
        )

    examples_by_id = {example.example_id: example for example in examples}
    records: list[PredictionRecord] = []
    for example_id, option_scores in grouped_scores.items():
        option_scores = sorted(option_scores, key=lambda item: item[0])
        predicted_option_id = max(option_scores, key=lambda item: item[2])[1]
        example = examples_by_id[example_id]
        records.append(
            PredictionRecord(
                run_id=run_id,
                phase="phase2",
                provider="slm",
                model_name=model_name,
                split=split,
                example_id=example_id,
                prompt_version="distilbert-cross-encoder-v1",
                raw_response=str([(option_id, score) for _, option_id, score in option_scores]),
                parsed_choice=None,
                predicted_option_id=predicted_option_id,
                gold_option_id=example.answer_option_id,
                is_correct=predicted_option_id == example.answer_option_id,
                latency_ms=None,
                metadata={
                    "option_scores": [
                        {"option_index": index, "option_id": option_id, "score": score}
                        for index, option_id, score in option_scores
                    ]
                },
            )
        )
    return tuple(sorted(records, key=lambda item: item.example_id))


def _build_trainer(
    *,
    model_name: str,
    checkpoint_dir: Path,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    tokenizer,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    num_train_epochs: float,
):
    (
        AutoModelForSequenceClassification,
        _AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    ) = _require_transformers()
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir=checkpoint_dir.as_posix(),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_RowDataset(train_rows),
        eval_dataset=_RowDataset(eval_rows),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    return trainer


def train_finetuned_model(
    *,
    train_examples: Sequence[RecipeExample],
    validation_examples: Sequence[RecipeExample],
    test_examples: Sequence[RecipeExample],
    run_id: str,
    output_dir: Path | str,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    learning_rate: float = 5e-5,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    num_train_epochs: float = 3.0,
):
    (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        _DataCollatorWithPadding,
        _Trainer,
        _TrainingArguments,
    ) = _require_transformers()
    del AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_rows, _train_meta = _build_rows(train_examples, tokenizer, max_length=max_length)
    validation_rows, validation_meta = _build_rows(
        validation_examples, tokenizer, max_length=max_length
    )
    test_rows, test_meta = _build_rows(test_examples, tokenizer, max_length=max_length)
    trainer = _build_trainer(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        train_rows=train_rows,
        eval_rows=validation_rows,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
    )
    trainer.train()
    trainer.save_model(checkpoint_dir.as_posix())
    tokenizer.save_pretrained(checkpoint_dir.as_posix())
    validation_output = trainer.predict(_RowDataset(validation_rows))
    validation_records = _prediction_records_from_logits(
        logits=validation_output.predictions,
        metadata=validation_meta,
        examples=validation_examples,
        run_id=run_id,
        model_name=model_name,
        split="validation",
    )
    write_prediction_records(
        validation_records,
        checkpoint_dir.parent / "validation_predictions.jsonl",
    )
    test_output = trainer.predict(_RowDataset(test_rows))
    test_records = _prediction_records_from_logits(
        logits=test_output.predictions,
        metadata=test_meta,
        examples=test_examples,
        run_id=run_id,
        model_name=model_name,
        split="test",
    )
    write_prediction_records(test_records, checkpoint_dir.parent / "test_predictions.jsonl")
    return {
        "trainer": trainer,
        "validation_records": validation_records,
        "test_records": test_records,
        "checkpoint_dir": checkpoint_dir,
    }


def evaluate_finetuned_model(
    *,
    examples: Sequence[RecipeExample],
    run_id: str,
    split: str,
    checkpoint_dir: Path | str,
    output_path: Path | str,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
):
    (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    ) = _require_transformers()
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir.as_posix())
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir.as_posix())
    rows, metadata = _build_rows(examples, tokenizer, max_length=max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=checkpoint_dir.as_posix(),
            per_device_eval_batch_size=8,
            report_to=[],
        ),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    output = trainer.predict(_RowDataset(rows))
    records = _prediction_records_from_logits(
        logits=output.predictions,
        metadata=metadata,
        examples=examples,
        run_id=run_id,
        model_name=model_name,
        split=split,
    )
    write_prediction_records(records, output_path)
    return records
