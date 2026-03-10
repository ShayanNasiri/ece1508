from __future__ import annotations

from pathlib import Path

import pytest

from recipe_mpr_qa.config import ConfigError, load_llm_experiment_config, load_slm_experiment_config


def test_load_slm_experiment_config_parses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "slm.toml"
    config_path.write_text(
        """
[output]
run_id = "distilbert-vanilla"

[data]
split = "test"

[slm]
mode = "vanilla"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_slm_experiment_config(config_path)

    assert config.mode == "vanilla"
    assert config.vanilla is not None
    assert config.vanilla.model_name == "distilbert-base-uncased"
    assert config.output.artifacts_root.as_posix() == "artifacts/runs"


def test_load_slm_experiment_config_rejects_invalid_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "slm.toml"
    config_path.write_text(
        """
[output]
run_id = "bad"

[slm]
mode = "unsupported"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_slm_experiment_config(config_path)


def test_load_llm_experiment_config_requires_model_name(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.toml"
    config_path.write_text(
        """
[output]
run_id = "llm-run"

[llm]
temperature = 0.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_llm_experiment_config(config_path)
