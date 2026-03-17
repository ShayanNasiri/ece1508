def pipeline(*args, **kwargs):
    """Lazy import wrapper — defers transformers import until first use."""
    from transformers import pipeline as _pipeline
    return _pipeline(*args, **kwargs)


class HFClient:
    """HuggingFace inference client — same interface as OllamaClient.

    Loads models via the transformers text-generation pipeline. Models are
    cached in memory after the first load so repeated queries to the same
    model ID don't reload weights.
    """

    def __init__(self):
        self._pipelines = {}

    def query(self, model_name, prompt, temperature=0):
        """Run inference on any HuggingFace model and return the generated continuation.

        Args:
            model_name: HuggingFace model ID (e.g. 'HuggingFaceTB/SmolLM2-135M-Instruct')
            prompt: Input prompt string
            temperature: Sampling temperature. 0 = greedy decoding.

        Returns:
            Generated text continuation (prompt prefix stripped).
        """
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
        return full_text[len(prompt):]
