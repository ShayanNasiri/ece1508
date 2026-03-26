from __future__ import annotations

import sys
from pathlib import Path

import pytest

from recipe_mpr_qa.data.augmentation import augment_training_examples
from recipe_mpr_qa.data.loaders import get_split_examples, load_dataset, load_split_manifest
from recipe_mpr_qa.data.models import PreparedDataset
from recipe_mpr_qa.data.preparation import write_prepared_dataset

pytest.importorskip("datasets")
pytest.importorskip("trl")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET_PATH = ROOT / "data" / "processed" / "recipe_mpr_qa.jsonl"
SPLIT_MANIFEST_PATH = ROOT / "data" / "processed" / "primary_split.json"
sys.path.insert(0, str(ROOT / "finetuning"))
from finetune import build_hf_datasets


def test_build_hf_datasets_default_sizes_unchanged() -> None:
    train_ds, val_ds, test_ds = build_hf_datasets(
        data_path=str(PROCESSED_DATASET_PATH),
        split_manifest_path=str(SPLIT_MANIFEST_PATH),
    )

    assert len(train_ds) == 350
    assert len(val_ds) == 75
    assert len(test_ds) == 75


def test_build_hf_datasets_appends_augmented_train_examples_only(tmp_path: Path) -> None:
    dataset = load_dataset(PROCESSED_DATASET_PATH)
    manifest = load_split_manifest(SPLIT_MANIFEST_PATH)
    train_examples = get_split_examples(dataset, manifest, "train")
    augmented_examples = augment_training_examples(train_examples[:3], max_variants=2)
    augmented_path = tmp_path / "augmented_train.jsonl"
    write_prepared_dataset(
        PreparedDataset(examples=augmented_examples, metadata={}),
        augmented_path,
    )

    train_ds, val_ds, test_ds = build_hf_datasets(
        data_path=str(PROCESSED_DATASET_PATH),
        split_manifest_path=str(SPLIT_MANIFEST_PATH),
        augmented_train_path=str(augmented_path),
    )

    assert len(train_ds) == 350 + len(augmented_examples)
    assert len(val_ds) == 75
    assert len(test_ds) == 75
    augmented_ids = {example.example_id for example in augmented_examples}
    train_ids = {row["example_id"] for row in train_ds}
    assert augmented_ids.issubset(train_ids)
    assert {row["completion"] for row in train_ds if row["example_id"] in augmented_ids}.issubset(
        {"A", "B", "C", "D", "E"}
    )
