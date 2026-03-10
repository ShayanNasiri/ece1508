from __future__ import annotations

from pathlib import Path

import torch

from recipe_mpr_qa.data.preparation import read_prepared_dataset
from recipe_mpr_qa.evaluation.records import read_prediction_records
from recipe_mpr_qa.llm.prompts import build_causal_multiple_choice_prompt
from recipe_mpr_qa.slm import causal as causal_module


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"


class FakeCausalTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, path: str):
        del path
        return cls()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize
        body = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            return body + "\nAssistant:"
        return body

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=True,
        max_length=512,
        padding=None,
    ):
        del truncation, max_length, padding
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        token_ids = [1, 2, 3, 4]
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        return "".join(chr(token) for token in tokens if token)

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeCausalGenerateModel:
    def __init__(self, letter: str):
        self.letter = letter

    @classmethod
    def from_pretrained(cls, path: str):
        del path
        return cls("A")

    def to(self, device: str):
        del device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([[ord(self.letter)]], dtype=input_ids.dtype)
        return torch.cat([input_ids, suffix], dim=1)


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


def test_evaluate_causal_slm_writes_predictions(tmp_path: Path) -> None:
    example = read_prepared_dataset(PROCESSED_DATASET_PATH).examples[0]
    _, letter_to_option_id = build_causal_multiple_choice_prompt(
        query=example.query,
        options=example.options,
    )
    correct_letter = next(
        letter for letter, option_id in letter_to_option_id.items() if option_id == example.answer_option_id
    )
    output_path = tmp_path / "predictions.jsonl"

    records = causal_module.evaluate_causal_slm(
        examples=(example,),
        run_id="smollm2-run",
        split="test",
        output_path=output_path,
        tokenizer=FakeCausalTokenizer(),
        model=FakeCausalGenerateModel(correct_letter),
    )

    assert len(records) == 1
    assert records[0].predicted_option_id == example.answer_option_id
    assert read_prediction_records(output_path) == records


def test_train_and_evaluate_causal_slm_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        causal_module,
        "_require_transformers",
        lambda: (
            FakeCausalGenerateModel,
            FakeCausalTokenizer,
            FakeTrainer,
            FakeTrainingArguments,
            object(),
        ),
    )
    dataset = read_prepared_dataset(PROCESSED_DATASET_PATH)
    result = causal_module.train_causal_slm(
        train_examples=dataset.examples[:2],
        validation_examples=dataset.examples[2:3],
        test_examples=dataset.examples[3:4],
        run_id="smollm2-ft",
        output_dir=tmp_path / "checkpoints",
        use_lora=False,
    )

    assert result["checkpoint_dir"].exists()
    assert len(result["validation_records"]) == 1
    assert len(result["test_records"]) == 1
    assert (tmp_path / "validation_predictions.jsonl").exists()
    assert (tmp_path / "test_predictions.jsonl").exists()
