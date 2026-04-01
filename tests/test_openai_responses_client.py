from __future__ import annotations

import os
from pathlib import Path

from typing import Any

from recipe_mpr_qa.synthetic.openai import OpenAIResponsesClient, load_env_file


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_request: dict[str, Any] | None = None

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float):
        self.last_request = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        }
        return _FakeResponse(self.payload)


def test_openai_responses_client_posts_json_schema_payload() -> None:
    session = _FakeSession(
        {
            "output": [
                {
                    "content": [
                        {"text": '{"answer":"ok"}'}
                    ]
                }
            ]
        }
    )
    client = OpenAIResponsesClient(api_key="test-key", session=session)

    response = client.create_structured_output(
        model="gpt-5.4-mini",
        instructions="Return JSON.",
        input_text="{}",
        schema_name="synthetic_test",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    assert response == {"answer": "ok"}
    assert session.last_request is not None
    request_payload = session.last_request["json"]
    assert request_payload["model"] == "gpt-5.4-mini"
    assert request_payload["text"]["format"]["type"] == "json_schema"
    assert request_payload["text"]["format"]["name"] == "synthetic_test"


def test_load_env_file_reads_openai_key(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test-from-env-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded_key = load_env_file(start_dir=tmp_path)

    assert loaded_key == "test-from-env-file"
    assert os.environ["OPENAI_API_KEY"] == "test-from-env-file"
