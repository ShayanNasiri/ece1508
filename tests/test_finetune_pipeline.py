from __future__ import annotations

from pathlib import Path

from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.evaluation.records import read_prediction_records
from recipe_mpr_qa.slm import finetune as finetune_module


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, path: str):
        del path
        return cls()

    def __call__(self, query: str, option_text: str, **kwargs):
        del kwargs
        return {"input_ids": [len(query), len(option_text)], "attention_mask": [1, 1]}

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeModel:
    @classmethod
    def from_pretrained(cls, path: str, num_labels: int | None = None):
        del path, num_labels
        return cls()


class FakeDataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class FakeTrainingArguments:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTrainer:
    def __init__(self, *, model, args, train_dataset=None, eval_dataset=None, tokenizer=None, data_collator=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def train(self):
        return None

    def save_model(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.bin").write_text("ok", encoding="utf-8")

    def predict(self, dataset):
        predictions = [[0.0, 1.0] if row["labels"] == 1 else [1.0, 0.0] for row in dataset.rows]
        return type("PredictionOutput", (), {"predictions": predictions})


def _patch_transformers(monkeypatch):
    monkeypatch.setattr(
        finetune_module,
        "_require_transformers",
        lambda: (
            FakeModel,
            FakeTokenizer,
            FakeDataCollatorWithPadding,
            FakeTrainer,
            FakeTrainingArguments,
        ),
    )


def test_train_finetuned_model_smoke(monkeypatch, tmp_path: Path) -> None:
    _patch_transformers(monkeypatch)
    dataset = read_prepared_dataset(PROCESSED_DATASET_PATH)
    result = finetune_module.train_finetuned_model(
        train_examples=dataset.examples[:2],
        validation_examples=dataset.examples[2:3],
        test_examples=dataset.examples[3:4],
        run_id="finetune-run",
        output_dir=tmp_path / "checkpoints",
    )

    assert result["checkpoint_dir"].exists()
    assert len(result["validation_records"]) == 1
    assert len(result["test_records"]) == 1
    assert (tmp_path / "validation_predictions.jsonl").exists()
    assert (tmp_path / "test_predictions.jsonl").exists()


def test_evaluate_finetuned_model_smoke(monkeypatch, tmp_path: Path) -> None:
    _patch_transformers(monkeypatch)
    dataset = read_prepared_dataset(PROCESSED_DATASET_PATH)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    output_path = tmp_path / "predictions.jsonl"

    records = finetune_module.evaluate_finetuned_model(
        examples=dataset.examples[:1],
        run_id="finetune-run",
        split="test",
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
    )

    assert len(records) == 1
    assert read_prediction_records(output_path) == records
