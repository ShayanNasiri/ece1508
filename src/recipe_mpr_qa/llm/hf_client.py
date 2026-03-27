from __future__ import annotations

import json
from pathlib import Path


def pipeline(*args, **kwargs):
    """Lazy import wrapper that defers transformers import until first use."""
    from transformers import pipeline as _pipeline

    return _pipeline(*args, **kwargs)


def AutoModelForCausalLM(*args, **kwargs):
    from transformers import AutoModelForCausalLM as _cls

    return _cls(*args, **kwargs)


AutoModelForCausalLM.from_pretrained = None  # placeholder; patched below


def AutoTokenizer(*args, **kwargs):
    from transformers import AutoTokenizer as _cls

    return _cls(*args, **kwargs)


AutoTokenizer.from_pretrained = None  # placeholder; patched below


def PeftModel(*args, **kwargs):
    from peft import PeftModel as _cls

    return _cls(*args, **kwargs)


PeftModel.from_pretrained = None  # placeholder; patched below


class _LazyClass:
    """Proxy that forwards attribute access to the real lazily-imported class."""

    def __init__(self, import_fn):
        self._import_fn = import_fn
        self._cls = None

    def _get(self):
        if self._cls is None:
            self._cls = self._import_fn()
        return self._cls

    def from_pretrained(self, *args, **kwargs):
        return self._get().from_pretrained(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._get()(*args, **kwargs)


AutoModelForCausalLM = _LazyClass(
    lambda: __import__("transformers", fromlist=["AutoModelForCausalLM"]).AutoModelForCausalLM
)
AutoTokenizer = _LazyClass(
    lambda: __import__("transformers", fromlist=["AutoTokenizer"]).AutoTokenizer
)
PeftModel = _LazyClass(lambda: __import__("peft", fromlist=["PeftModel"]).PeftModel)


class HFClient:
    """HuggingFace inference client with the same interface as OllamaClient."""

    def __init__(self):
        self._pipelines: dict = {}
        self._peft_models: dict = {}

    def query(self, model_name: str, prompt: str, temperature: float = 0) -> str:
        if self._is_adapter_path(model_name):
            return self._query_peft(model_name, prompt, temperature)
        return self._query_pipeline(model_name, prompt, temperature)

    def _is_adapter_path(self, model_name: str) -> bool:
        path = Path(model_name)
        return path.is_dir() and (path / "adapter_config.json").is_file()

    def _query_pipeline(self, model_name: str, prompt: str, temperature: float) -> str:
        if model_name not in self._pipelines:
            self._pipelines[model_name] = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                trust_remote_code=True,
            )

        pipe = self._pipelines[model_name]
        gen_kwargs = {
            "max_new_tokens": 256,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        result = pipe(prompt, **gen_kwargs)
        full_text = result[0]["generated_text"]
        return full_text[len(prompt) :]

    def _query_peft(self, adapter_path: str, prompt: str, temperature: float) -> str:
        if adapter_path not in self._peft_models:
            self._peft_models[adapter_path] = self._load_peft_model(adapter_path)

        model, tokenizer = self._peft_models[adapter_path]
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        gen_kwargs: dict = {
            "max_new_tokens": 256,
            "do_sample": temperature > 0,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        output_ids = model.generate(input_ids, **gen_kwargs)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return full_text[len(prompt) :]

    def _load_peft_model(self, adapter_path: str):
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_id = adapter_config["base_model_name_or_path"]

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        return model, tokenizer
