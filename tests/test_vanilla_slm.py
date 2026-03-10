from __future__ import annotations

from pathlib import Path

import torch

from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.evaluation.records import read_prediction_records
from recipe_mpr_qa.slm.vanilla import evaluate_vanilla_slm, mean_pool_embeddings


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"


class FakeTokenizer:
    def __init__(self, vectors):
        self.vectors = vectors

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128, padding=False):
        del return_tensors, truncation, max_length, padding
        texts = [text] if isinstance(text, str) else list(text)
        matrix = torch.tensor([self.vectors[item] for item in texts], dtype=torch.float32)
        return {"input_ids": matrix, "attention_mask": torch.ones((len(texts), 1), dtype=torch.long)}


class FakeModel:
    def to(self, device: str):
        del device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        inputs = kwargs["input_ids"]
        return type("Output", (), {"last_hidden_state": inputs.unsqueeze(1)})


def test_mean_pool_embeddings_respects_attention_mask() -> None:
    pooled = mean_pool_embeddings(
        torch.tensor([[[1.0, 3.0], [5.0, 7.0]]]),
        torch.tensor([[1, 0]]),
    )

    assert pooled.tolist() == [[1.0, 3.0]]


def test_evaluate_vanilla_slm_writes_grouped_predictions(tmp_path: Path) -> None:
    examples = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[:1]
    correct_option = examples[0].answer_option_id
    vectors = {examples[0].query: [1.0, 0.0]}
    for option in examples[0].options:
        vectors[option.text] = [1.0, 0.0] if option.option_id == correct_option else [0.0, 1.0]
    output_path = tmp_path / "predictions.jsonl"

    records = evaluate_vanilla_slm(
        examples=examples,
        run_id="vanilla-run",
        split="test",
        output_path=output_path,
        tokenizer=FakeTokenizer(vectors),
        model=FakeModel(),
    )

    assert len(records) == 1
    assert records[0].predicted_option_id == correct_option
    assert read_prediction_records(output_path) == records
