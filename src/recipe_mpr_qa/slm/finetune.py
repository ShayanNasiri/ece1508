from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe_mpr_qa.benchmark.provenance import BENCHMARK_CONTRACT_VERSION
from recipe_mpr_qa.data.loaders import build_option_scoring_examples
from recipe_mpr_qa.data.models import RecipeExample
from recipe_mpr_qa.evaluation.metrics import summarize_prediction_metrics
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
        try:
            score = float(logit_row[-1])
        except (TypeError, IndexError, KeyError):
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
                model_interface="classifier",
                decoding_mode="pairwise_logits",
                parse_status="not_applicable",
                contract_version=BENCHMARK_CONTRACT_VERSION,
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
    training_kwargs = {
        "output_dir": checkpoint_dir.as_posix(),
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "report_to": [],
    }
    training_args = TrainingArguments(
        **_add_eval_strategy_kwargs(TrainingArguments, training_kwargs)
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
    best_checkpoint_dir, checkpoint_scores = _select_best_finetune_checkpoint(
        checkpoint_root=checkpoint_dir,
        validation_examples=validation_examples,
        run_id=run_id,
        model_name=model_name,
        max_length=max_length,
    )
    validation_records = evaluate_finetuned_model(
        examples=validation_examples,
        run_id=run_id,
        split="validation",
        checkpoint_dir=best_checkpoint_dir,
        output_path=checkpoint_dir.parent / "validation_predictions.jsonl",
        model_name=model_name,
        max_length=max_length,
    )
    test_records = evaluate_finetuned_model(
        examples=test_examples,
        run_id=run_id,
        split="test",
        checkpoint_dir=best_checkpoint_dir,
        output_path=checkpoint_dir.parent / "test_predictions.jsonl",
        model_name=model_name,
        max_length=max_length,
    )
    (checkpoint_dir.parent / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "best_checkpoint_dir": best_checkpoint_dir.as_posix(),
                "checkpoints": checkpoint_scores,
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "trainer": trainer,
        "validation_records": validation_records,
        "test_records": test_records,
        "checkpoint_dir": checkpoint_dir,
        "best_checkpoint_dir": best_checkpoint_dir,
        "checkpoint_scores": checkpoint_scores,
    }


def evaluate_finetuned_model(
    *,
    examples: Sequence[RecipeExample],
    run_id: str,
    split: str,
    checkpoint_dir: Path | str,
    output_path: Path | str | None,
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
    tokenizer_source = _resolve_tokenizer_source(checkpoint_dir, model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir.as_posix())
    rows, metadata = _build_rows(examples, tokenizer, max_length=max_length)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            **_add_eval_strategy_kwargs(
                TrainingArguments,
                {
                    "output_dir": checkpoint_dir.as_posix(),
                    "per_device_eval_batch_size": 8,
                    "report_to": [],
                },
            )
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
    if output_path is not None:
        write_prediction_records(records, output_path)
    return records


def _add_eval_strategy_kwargs(training_arguments_cls, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    init_parameters = training_arguments_cls.__init__.__code__.co_varnames
    resolved = dict(kwargs)
    if "evaluation_strategy" in init_parameters:
        resolved["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in init_parameters:
        resolved["eval_strategy"] = "epoch"
    return resolved


def _resolve_tokenizer_source(checkpoint_dir: Path, *, model_name: str) -> str:
    for candidate in (checkpoint_dir, checkpoint_dir.parent):
        if any((candidate / file_name).exists() for file_name in ("tokenizer.json", "tokenizer_config.json")):
            return candidate.as_posix()
    return model_name


def _list_checkpoint_dirs(checkpoint_root: Path) -> list[Path]:
    checkpoint_dirs = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ] if checkpoint_root.exists() else []
    def sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.split("-", 1)[-1]
        return (int(suffix) if suffix.isdigit() else -1, path.name)
    ordered = sorted(checkpoint_dirs, key=sort_key)
    if checkpoint_root.exists():
        ordered.append(checkpoint_root)
    return ordered or [checkpoint_root]


def _select_best_finetune_checkpoint(
    *,
    checkpoint_root: Path,
    validation_examples: Sequence[RecipeExample],
    run_id: str,
    model_name: str,
    max_length: int,
) -> tuple[Path, list[dict[str, Any]]]:
    checkpoint_scores: list[dict[str, Any]] = []
    best_checkpoint_dir = checkpoint_root
    best_accuracy = float("-inf")
    for checkpoint_dir in _list_checkpoint_dirs(checkpoint_root):
        records = evaluate_finetuned_model(
            examples=validation_examples,
            run_id=run_id,
            split="validation",
            checkpoint_dir=checkpoint_dir,
            output_path=None,
            model_name=model_name,
            max_length=max_length,
        )
        metrics = summarize_prediction_metrics(
            records,
            type("Dataset", (), {"examples": tuple(validation_examples)})(),
        )
        checkpoint_scores.append(
            {
                "checkpoint_dir": checkpoint_dir.as_posix(),
                "accuracy": metrics["accuracy"],
                "correct_count": metrics["correct_count"],
                "parse_failure_count": metrics["parse_failure_count"],
            }
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_checkpoint_dir = checkpoint_dir
    return best_checkpoint_dir, checkpoint_scores
