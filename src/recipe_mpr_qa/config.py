from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from recipe_mpr_qa.data.constants import (
    DEFAULT_PROCESSED_DATASET_PATH,
    DEFAULT_SPLIT_MANIFEST_PATH,
)

VALID_SPLITS = {"train", "validation", "test"}
VALID_SLM_MODES = {"vanilla", "finetune", "causal_baseline", "causal_finetune"}
VALID_VERDICTS = {"correct", "partially_correct", "incorrect"}


class ConfigError(ValueError):
    """Raised when a TOML experiment config is invalid."""


def _require_table(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section {key!r} is required")
    return value


def _coerce_path(value: Any, default: Path | None = None) -> Path:
    if value is None:
        if default is None:
            raise ConfigError("A path value is required")
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Expected path string, got {value!r}")
    return Path(value)


def _coerce_str(value: Any, *, name: str, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise ConfigError(f"{name} is required")
        return default
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _coerce_int(value: Any, *, name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ConfigError(f"{name} is required")
        return default
    if not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _coerce_float(value: Any, *, name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ConfigError(f"{name} is required")
        return default
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    return float(value)


def _coerce_bool(value: Any, *, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _coerce_str_map(value: Any, *, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a table")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_coerce_str(key, name=f"{name} key")] = _coerce_str(
            item, name=f"{name}[{key!r}]"
        )
    return result


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path = DEFAULT_PROCESSED_DATASET_PATH
    split_manifest_path: Path = DEFAULT_SPLIT_MANIFEST_PATH
    split: str = "test"
    augmentation_dataset_path: Path | None = None

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ConfigError(f"split must be one of {sorted(VALID_SPLITS)}")


@dataclass(frozen=True)
class OutputConfig:
    run_id: str
    artifacts_root: Path = Path("artifacts/runs")
    overwrite: bool = False


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    experiment_name: str = "recipe-mpr-qa"
    tracking_uri: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AugmentationConfig:
    teacher_model_name: str
    prompt_version: str = "recipe-mpr-augmentation-v1"
    variants_per_example: int = 2
    temperature: float = 0.2
    max_retries: int = 3
    resume: bool = True

    def __post_init__(self) -> None:
        if self.variants_per_example <= 0:
            raise ConfigError("variants_per_example must be > 0")
        if self.max_retries <= 0:
            raise ConfigError("max_retries must be > 0")


@dataclass(frozen=True)
class VanillaSLMConfig:
    model_name: str = "distilbert-base-uncased"
    prompt_version: str = "embedding-similarity-v1"
    batch_size: int = 8
    max_length: int = 128

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if self.max_length <= 0:
            raise ConfigError("max_length must be > 0")


@dataclass(frozen=True)
class FineTuneConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    learning_rate: float = 5e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: float = 3.0
    use_augmentation: bool = False
    checkpoint_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ConfigError("max_length must be > 0")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ConfigError("batch sizes must be > 0")
        if self.num_train_epochs <= 0:
            raise ConfigError("num_train_epochs must be > 0")


@dataclass(frozen=True)
class CausalBaselineConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    prompt_version: str = "recipe-mpr-chat-mc-v1"
    max_length: int = 512
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ConfigError("max_length must be > 0")
        if self.max_new_tokens <= 0:
            raise ConfigError("max_new_tokens must be > 0")


@dataclass(frozen=True)
class CausalFineTuneConfig:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    prompt_version: str = "recipe-mpr-chat-mc-v1"
    max_length: int = 512
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    learning_rate: float = 2e-4
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_train_epochs: float = 3.0
    use_augmentation: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    checkpoint_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ConfigError("max_length must be > 0")
        if self.max_new_tokens <= 0:
            raise ConfigError("max_new_tokens must be > 0")
        if self.train_batch_size <= 0 or self.eval_batch_size <= 0:
            raise ConfigError("batch sizes must be > 0")
        if self.num_train_epochs <= 0:
            raise ConfigError("num_train_epochs must be > 0")
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ConfigError("lora_r and lora_alpha must be > 0")


@dataclass(frozen=True)
class LLMRunConfig:
    model_name: str
    prompt_version: str = "recipe-mpr-mc-v1"
    temperature: float = 0.0
    max_retries: int = 3
    resume: bool = True

    def __post_init__(self) -> None:
        if self.max_retries <= 0:
            raise ConfigError("max_retries must be > 0")


@dataclass(frozen=True)
class JudgeConfig:
    model_name: str
    prompt_version: str = "recipe-mpr-judge-v1"
    temperature: float = 0.0
    max_retries: int = 3
    verdict_labels: tuple[str, ...] = ("correct", "partially_correct", "incorrect")
    resume: bool = True

    def __post_init__(self) -> None:
        if self.max_retries <= 0:
            raise ConfigError("max_retries must be > 0")
        if set(self.verdict_labels) != VALID_VERDICTS:
            raise ConfigError(f"verdict_labels must equal {sorted(VALID_VERDICTS)}")


@dataclass(frozen=True)
class AugmentationRunConfig:
    data: DataConfig
    output: OutputConfig
    tracking: TrackingConfig
    augmentation: AugmentationConfig


@dataclass(frozen=True)
class SLMExperimentConfig:
    data: DataConfig
    output: OutputConfig
    tracking: TrackingConfig
    mode: str
    vanilla: VanillaSLMConfig | None = None
    finetune: FineTuneConfig | None = None
    causal_baseline: CausalBaselineConfig | None = None
    causal_finetune: CausalFineTuneConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_SLM_MODES:
            raise ConfigError(f"mode must be one of {sorted(VALID_SLM_MODES)}")
        if self.mode == "vanilla" and self.vanilla is None:
            raise ConfigError("vanilla config is required when mode='vanilla'")
        if self.mode == "finetune" and self.finetune is None:
            raise ConfigError("finetune config is required when mode='finetune'")
        if self.mode == "causal_baseline" and self.causal_baseline is None:
            raise ConfigError("causal_baseline config is required when mode='causal_baseline'")
        if self.mode == "causal_finetune" and self.causal_finetune is None:
            raise ConfigError("causal_finetune config is required when mode='causal_finetune'")


@dataclass(frozen=True)
class LLMExperimentConfig:
    data: DataConfig
    output: OutputConfig
    tracking: TrackingConfig
    llm: LLMRunConfig


@dataclass(frozen=True)
class JudgeExperimentConfig:
    data: DataConfig
    output: OutputConfig
    tracking: TrackingConfig
    judge: JudgeConfig
    predictions_path: Path | None = None


def _load_toml(path: Path | str) -> Mapping[str, Any]:
    resolved_path = Path(path)
    with resolved_path.open("rb") as handle:
        return tomllib.load(handle)


def _parse_data_config(payload: Mapping[str, Any]) -> DataConfig:
    data = payload.get("data", {})
    if not isinstance(data, Mapping):
        raise ConfigError("data must be a table")
    split = _coerce_str(data.get("split"), name="data.split", default="test")
    return DataConfig(
        dataset_path=_coerce_path(data.get("dataset_path"), DEFAULT_PROCESSED_DATASET_PATH),
        split_manifest_path=_coerce_path(
            data.get("split_manifest_path"),
            DEFAULT_SPLIT_MANIFEST_PATH,
        ),
        split=split,
        augmentation_dataset_path=(
            _coerce_path(data.get("augmentation_dataset_path"))
            if data.get("augmentation_dataset_path") is not None
            else None
        ),
    )


def _parse_output_config(payload: Mapping[str, Any]) -> OutputConfig:
    output = _require_table(payload, "output")
    return OutputConfig(
        run_id=_coerce_str(output.get("run_id"), name="output.run_id"),
        artifacts_root=_coerce_path(output.get("artifacts_root"), Path("artifacts/runs")),
        overwrite=_coerce_bool(output.get("overwrite"), name="output.overwrite"),
    )


def _parse_tracking_config(payload: Mapping[str, Any]) -> TrackingConfig:
    tracking = payload.get("tracking", {})
    if not isinstance(tracking, Mapping):
        raise ConfigError("tracking must be a table")
    return TrackingConfig(
        enabled=_coerce_bool(tracking.get("enabled"), name="tracking.enabled"),
        experiment_name=_coerce_str(
            tracking.get("experiment_name"),
            name="tracking.experiment_name",
            default="recipe-mpr-qa",
        ),
        tracking_uri=tracking.get("tracking_uri"),
        tags=_coerce_str_map(tracking.get("tags"), name="tracking.tags"),
    )


def load_augmentation_run_config(path: Path | str) -> AugmentationRunConfig:
    payload = _load_toml(path)
    augmentation = _require_table(payload, "augmentation")
    return AugmentationRunConfig(
        data=_parse_data_config(payload),
        output=_parse_output_config(payload),
        tracking=_parse_tracking_config(payload),
        augmentation=AugmentationConfig(
            teacher_model_name=_coerce_str(
                augmentation.get("teacher_model_name"),
                name="augmentation.teacher_model_name",
            ),
            prompt_version=_coerce_str(
                augmentation.get("prompt_version"),
                name="augmentation.prompt_version",
                default="recipe-mpr-augmentation-v1",
            ),
            variants_per_example=_coerce_int(
                augmentation.get("variants_per_example"),
                name="augmentation.variants_per_example",
                default=2,
            ),
            temperature=_coerce_float(
                augmentation.get("temperature"),
                name="augmentation.temperature",
                default=0.2,
            ),
            max_retries=_coerce_int(
                augmentation.get("max_retries"),
                name="augmentation.max_retries",
                default=3,
            ),
            resume=_coerce_bool(augmentation.get("resume"), name="augmentation.resume", default=True),
        ),
    )


def load_slm_experiment_config(path: Path | str) -> SLMExperimentConfig:
    payload = _load_toml(path)
    slm = _require_table(payload, "slm")
    mode = _coerce_str(slm.get("mode"), name="slm.mode")
    vanilla = None
    finetune = None
    causal_baseline = None
    causal_finetune = None
    if mode == "vanilla":
        vanilla = VanillaSLMConfig(
            model_name=_coerce_str(slm.get("model_name"), name="slm.model_name", default="distilbert-base-uncased"),
            prompt_version=_coerce_str(
                slm.get("prompt_version"),
                name="slm.prompt_version",
                default="embedding-similarity-v1",
            ),
            batch_size=_coerce_int(slm.get("batch_size"), name="slm.batch_size", default=8),
            max_length=_coerce_int(slm.get("max_length"), name="slm.max_length", default=128),
        )
    elif mode == "finetune":
        checkpoint_dir = slm.get("checkpoint_dir")
        finetune = FineTuneConfig(
            model_name=_coerce_str(slm.get("model_name"), name="slm.model_name", default="distilbert-base-uncased"),
            max_length=_coerce_int(slm.get("max_length"), name="slm.max_length", default=128),
            learning_rate=_coerce_float(
                slm.get("learning_rate"),
                name="slm.learning_rate",
                default=5e-5,
            ),
            train_batch_size=_coerce_int(
                slm.get("train_batch_size"),
                name="slm.train_batch_size",
                default=8,
            ),
            eval_batch_size=_coerce_int(
                slm.get("eval_batch_size"),
                name="slm.eval_batch_size",
                default=8,
            ),
            num_train_epochs=_coerce_float(
                slm.get("num_train_epochs"),
                name="slm.num_train_epochs",
                default=3.0,
            ),
            use_augmentation=_coerce_bool(
                slm.get("use_augmentation"),
                name="slm.use_augmentation",
            ),
            checkpoint_dir=_coerce_path(checkpoint_dir) if checkpoint_dir is not None else None,
        )
    elif mode == "causal_baseline":
        causal_baseline = CausalBaselineConfig(
            model_name=_coerce_str(
                slm.get("model_name"),
                name="slm.model_name",
                default="HuggingFaceTB/SmolLM2-135M-Instruct",
            ),
            prompt_version=_coerce_str(
                slm.get("prompt_version"),
                name="slm.prompt_version",
                default="recipe-mpr-chat-mc-v1",
            ),
            max_length=_coerce_int(slm.get("max_length"), name="slm.max_length", default=512),
            max_new_tokens=_coerce_int(
                slm.get("max_new_tokens"),
                name="slm.max_new_tokens",
                default=8,
            ),
            temperature=_coerce_float(
                slm.get("temperature"),
                name="slm.temperature",
                default=0.0,
            ),
            top_p=_coerce_float(slm.get("top_p"), name="slm.top_p", default=0.9),
        )
    elif mode == "causal_finetune":
        checkpoint_dir = slm.get("checkpoint_dir")
        causal_finetune = CausalFineTuneConfig(
            model_name=_coerce_str(
                slm.get("model_name"),
                name="slm.model_name",
                default="HuggingFaceTB/SmolLM2-135M-Instruct",
            ),
            prompt_version=_coerce_str(
                slm.get("prompt_version"),
                name="slm.prompt_version",
                default="recipe-mpr-chat-mc-v1",
            ),
            max_length=_coerce_int(slm.get("max_length"), name="slm.max_length", default=512),
            max_new_tokens=_coerce_int(
                slm.get("max_new_tokens"),
                name="slm.max_new_tokens",
                default=8,
            ),
            temperature=_coerce_float(
                slm.get("temperature"),
                name="slm.temperature",
                default=0.0,
            ),
            top_p=_coerce_float(slm.get("top_p"), name="slm.top_p", default=0.9),
            learning_rate=_coerce_float(
                slm.get("learning_rate"),
                name="slm.learning_rate",
                default=2e-4,
            ),
            train_batch_size=_coerce_int(
                slm.get("train_batch_size"),
                name="slm.train_batch_size",
                default=2,
            ),
            eval_batch_size=_coerce_int(
                slm.get("eval_batch_size"),
                name="slm.eval_batch_size",
                default=2,
            ),
            num_train_epochs=_coerce_float(
                slm.get("num_train_epochs"),
                name="slm.num_train_epochs",
                default=3.0,
            ),
            use_augmentation=_coerce_bool(
                slm.get("use_augmentation"),
                name="slm.use_augmentation",
            ),
            use_lora=_coerce_bool(
                slm.get("use_lora"),
                name="slm.use_lora",
                default=True,
            ),
            lora_r=_coerce_int(slm.get("lora_r"), name="slm.lora_r", default=16),
            lora_alpha=_coerce_int(
                slm.get("lora_alpha"),
                name="slm.lora_alpha",
                default=32,
            ),
            lora_dropout=_coerce_float(
                slm.get("lora_dropout"),
                name="slm.lora_dropout",
                default=0.05,
            ),
            checkpoint_dir=_coerce_path(checkpoint_dir) if checkpoint_dir is not None else None,
        )
    return SLMExperimentConfig(
        data=_parse_data_config(payload),
        output=_parse_output_config(payload),
        tracking=_parse_tracking_config(payload),
        mode=mode,
        vanilla=vanilla,
        finetune=finetune,
        causal_baseline=causal_baseline,
        causal_finetune=causal_finetune,
    )


def load_llm_experiment_config(path: Path | str) -> LLMExperimentConfig:
    payload = _load_toml(path)
    llm = _require_table(payload, "llm")
    return LLMExperimentConfig(
        data=_parse_data_config(payload),
        output=_parse_output_config(payload),
        tracking=_parse_tracking_config(payload),
        llm=LLMRunConfig(
            model_name=_coerce_str(llm.get("model_name"), name="llm.model_name"),
            prompt_version=_coerce_str(
                llm.get("prompt_version"),
                name="llm.prompt_version",
                default="recipe-mpr-mc-v1",
            ),
            temperature=_coerce_float(
                llm.get("temperature"),
                name="llm.temperature",
                default=0.0,
            ),
            max_retries=_coerce_int(llm.get("max_retries"), name="llm.max_retries", default=3),
            resume=_coerce_bool(llm.get("resume"), name="llm.resume", default=True),
        ),
    )


def load_judge_experiment_config(path: Path | str) -> JudgeExperimentConfig:
    payload = _load_toml(path)
    judge = _require_table(payload, "judge")
    return JudgeExperimentConfig(
        data=_parse_data_config(payload),
        output=_parse_output_config(payload),
        tracking=_parse_tracking_config(payload),
        judge=JudgeConfig(
            model_name=_coerce_str(judge.get("model_name"), name="judge.model_name"),
            prompt_version=_coerce_str(
                judge.get("prompt_version"),
                name="judge.prompt_version",
                default="recipe-mpr-judge-v1",
            ),
            temperature=_coerce_float(
                judge.get("temperature"),
                name="judge.temperature",
                default=0.0,
            ),
            max_retries=_coerce_int(
                judge.get("max_retries"),
                name="judge.max_retries",
                default=3,
            ),
            resume=_coerce_bool(judge.get("resume"), name="judge.resume", default=True),
        ),
        predictions_path=(
            _coerce_path(payload.get("predictions_path"))
            if payload.get("predictions_path") is not None
            else None
        ),
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if hasattr(value, "__dataclass_fields__"):
            return {key: convert(item) for key, item in asdict(value).items()}
        return value

    return convert(config)
