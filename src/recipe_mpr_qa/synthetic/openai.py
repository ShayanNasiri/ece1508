"""OpenAI Responses client used by the synthetic-data workflow.

This module is intentionally small because it sits on the public boundary
between repo-local artifact code and the hosted generation/review step.
The main behavior maintainers need to remember is environment resolution:
the client will use an explicit key first, then `OPENAI_API_KEY`, then walk
upward through parent directories looking for a repo-local `.env` file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import requests


DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_ENV_FILENAMES = (".env",)


def load_env_file(*, start_dir: Path | None = None) -> str | None:
    """Load ``OPENAI_API_KEY`` from the nearest `.env` found upward from ``start_dir``."""
    search_dir = Path(start_dir or Path.cwd()).resolve()
    for candidate_dir in (search_dir, *search_dir.parents):
        for env_filename in DEFAULT_ENV_FILENAMES:
            env_path = candidate_dir / env_filename
            if not env_path.is_file():
                continue
            api_key = _load_api_key_from_env_path(env_path)
            if api_key is not None:
                return api_key
    return None


def _load_api_key_from_env_path(env_path: Path) -> str | None:
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "OPENAI_API_KEY":
            continue
        normalized_value = value.strip()
        if normalized_value.startswith(("'", '"')) and normalized_value.endswith(("'", '"')):
            normalized_value = normalized_value[1:-1]
        normalized_value = normalized_value.strip()
        if not normalized_value:
            return None
        os.environ.setdefault("OPENAI_API_KEY", normalized_value)
        return normalized_value
    return None


class OpenAIResponsesClient:
    """Thin structured-output wrapper over the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENAI_API_URL,
        session: requests.Session | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or load_env_file()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for synthetic data generation and review")
        self.base_url = base_url
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def create_structured_output(
        self,
        *,
        model: str,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Request one strict JSON-schema response and parse it into a mapping."""
        payload = {
            "model": model,
            "instructions": instructions,
            "input": input_text,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                }
            },
        }
        response = self.session.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"OpenAI Responses API request failed: {response.status_code} {response.text}"
            ) from exc
        response_payload = response.json()
        output_text = self._extract_output_text(response_payload)
        if output_text is None:
            raise RuntimeError("OpenAI Responses API returned no output_text payload")
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI Responses API returned invalid JSON output: {output_text}"
            ) from exc

    @staticmethod
    def _extract_output_text(response_payload: Mapping[str, Any]) -> str | None:
        output_text = response_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        output_items = response_payload.get("output")
        if not isinstance(output_items, list):
            return None
        collected_fragments: list[str] = []
        for item in output_items:
            if not isinstance(item, Mapping):
                continue
            contents = item.get("content")
            if not isinstance(contents, list):
                continue
            for content in contents:
                if not isinstance(content, Mapping):
                    continue
                text_value = content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    collected_fragments.append(text_value)
                    continue
                if isinstance(text_value, Mapping):
                    mapped_text = text_value.get("value")
                    if isinstance(mapped_text, str) and mapped_text.strip():
                        collected_fragments.append(mapped_text)
        if not collected_fragments:
            return None
        return "".join(collected_fragments)
