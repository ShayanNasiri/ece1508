from __future__ import annotations

import json
from pathlib import Path

from recipe_mpr_qa.cli import main
from recipe_mpr_qa.data.preparation import read_prepared_dataset


ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = ROOT / "data" / "500QA.json"
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "data" / "processed" / "primary_split.json"


def test_cli_prepare_data_writes_expected_files(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "prepared.jsonl"
    split_output_path = tmp_path / "split.json"

    exit_code = main(
        [
            "prepare-data",
            "--input",
            str(RAW_DATASET_PATH),
            "--output",
            str(output_path),
            "--split-output",
            str(split_output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()
    assert split_output_path.exists()
    assert "Prepared 500 examples" in captured.out


def test_cli_validate_data_smoke(capsys) -> None:
    exit_code = main(["validate-data", "--input", str(RAW_DATASET_PATH), "--kind", "raw"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Validated 500 raw examples" in captured.out


def test_cli_dataset_stats_smoke(capsys) -> None:
    exit_code = main(
        ["dataset-stats", "--input", str(PROCESSED_DATASET_PATH), "--kind", "prepared"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    stats = json.loads(captured.out)
    assert stats["example_count"] == 500


def test_cli_export_split_smoke(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "test_split.jsonl"

    exit_code = main(
        [
            "export-split",
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--split",
            "test",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()
    assert "Exported 75 test examples" in captured.out


def test_cli_augment_train_writes_augmented_artifact(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "augmented_train.jsonl"

    exit_code = main(
        [
            "augment-train",
            "--dataset",
            str(PROCESSED_DATASET_PATH),
            "--split-manifest",
            str(SPLIT_MANIFEST_PATH),
            "--output",
            str(output_path),
            "--max-variants",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert output_path.exists()
    assert "Augmented 350 parent train examples" in captured.out

    augmented_dataset = read_prepared_dataset(output_path)
    assert len(augmented_dataset.examples) > 0
    train_ids = set(json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))["splits"]["train"])
    assert all(example.example_id.endswith(("-aug-01", "-aug-02")) for example in augmented_dataset.examples)
    assert all(example.example_id not in train_ids for example in augmented_dataset.examples)
    assert all(example.source_metadata["parent_example_id"] in train_ids for example in augmented_dataset.examples)
