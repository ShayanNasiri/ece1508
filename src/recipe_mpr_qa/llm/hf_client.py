from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] to use Hugging Face inference") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _require_peft():
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] with peft to use LoRA adapters") from exc
    return PeftModel


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install recipe-mpr-qa[train] to use Hugging Face inference") from exc
    return torch


class HFClient:
    """Local Hugging Face inference client with optional adapter support."""

    def __init__(self):
        self._bundles: dict[str, tuple[object, object]] = {}

    def generate(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 16,
    ) -> str:
        return self.query(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    def query(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 16,
    ) -> str:
        torch = _require_torch()
        tokenizer, model = self._load_bundle(model_name)
        rendered_prompt = self._format_prompt(prompt, tokenizer)
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generation_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(temperature > 0),
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        if getattr(tokenizer, "pad_token_id", None) is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        prompt_tokens = encoded["input_ids"].shape[1]
        generated_tokens = outputs[0][prompt_tokens:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def query_loglikelihood(
        self,
        model_name: str,
        prompt: str,
        choices: Sequence[str],
    ) -> str:
        torch = _require_torch()
        tokenizer, model = self._load_bundle(model_name)
        rendered_prompt = self._format_prompt(prompt, tokenizer)
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        best_choice = None
        best_score = float("-inf")
        for choice in choices:
            variants = [choice]
            if len(choice) == 1 and choice.isalpha():
                variants.extend((f" {choice}", f"\n{choice}"))
            score = max(
                (
                    self._sequence_logprob(
                        tokenizer=tokenizer,
                        model=model,
                        prompt_ids=encoded["input_ids"][0],
                        candidate_text=variant,
                    )
                    for variant in variants
                ),
                default=float("-inf"),
            )
            if score == float("-inf"):
                continue
            if score > best_score:
                best_score = score
                best_choice = choice
        if best_choice is None:
            raise RuntimeError("Unable to score any loglikelihood choices")
        return best_choice

    def _sequence_logprob(self, *, tokenizer, model, prompt_ids, candidate_text: str) -> float:
        torch = _require_torch()
        candidate_ids = tokenizer.encode(candidate_text, add_special_tokens=False)
        if not candidate_ids:
            return float("-inf")
        prompt_id_list = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
        full_ids = torch.tensor([prompt_id_list + list(candidate_ids)], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(full_ids, device=model.device)
        with torch.no_grad():
            logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        prompt_length = len(prompt_id_list)
        total = 0.0
        for token_offset, token_id in enumerate(candidate_ids):
            total += float(log_probs[0, prompt_length - 1 + token_offset, token_id].item())
        return total

    def _load_bundle(self, model_name: str):
        if model_name not in self._bundles:
            self._bundles[model_name] = self._load_model_and_tokenizer(model_name)
        return self._bundles[model_name]

    def _load_model_and_tokenizer(self, model_name: str):
        torch = _require_torch()
        AutoModelForCausalLM, AutoTokenizer = _require_transformers()
        model_path = Path(model_name)
        if model_path.is_dir() and (model_path / "adapter_config.json").is_file():
            adapter_config = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
            base_model_name = str(adapter_config["base_model_name_or_path"])
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            PeftModel = _require_peft()
            model = PeftModel.from_pretrained(model, model_path.as_posix())
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model

    def _format_prompt(self, prompt: str, tokenizer) -> str:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except ValueError:
                pass
        return prompt
