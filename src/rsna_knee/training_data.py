"""Lean training adapters around the canonical knee input pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Sampler

from .model_input import Depth18TrainingDataset
from .input_pipeline import LABELS, KneeInputConfig, KneeStudyDataset, resolve_config_paths


MODEL_INPUT_KEYS = (
    "descriptor_source_ids", "descriptor_positions_mm", "geometry_measured", "geometry_uniform_fallback",
    "series_mask",
    "slice_valid_mask",
    "slice_repeated_mask",
    "pixel_valid_mask",
    "slice_positions_normalized",
    "gap_previous_mm",
    "gap_next_mm",
)


class EpochShuffleSampler(Sampler[int]):
    """Deterministic epoch-addressable shuffle, including exact resume behavior."""

    def __init__(self, size: int, seed: int, *, shuffle: bool = True) -> None:
        if size < 1:
            raise ValueError("sampler size must be positive")
        self.size = int(size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(range(self.size))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def collate_baseline_training(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty baseline batch")
    tensor_keys = (
        "images",
        *MODEL_INPUT_KEYS,
        "labels",
        "label_mask",
        "is_hard",
    )
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "fold": torch.tensor([sample["fold"] for sample in samples], dtype=torch.long),
        **{key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys},
        "descriptor_keep": torch.stack([sample.get("descriptor_keep", torch.tensor(True)) for sample in samples]),
        "available_roles": torch.tensor(
            [sample["available_roles"] for sample in samples], dtype=torch.long
        ),
        "stage_cache_hits": sum(sample["stage_cache_hits"] for sample in samples),
        "stage_cache_roles": sum(sample["stage_cache_roles"] for sample in samples),
    }


@dataclass
class BaselineLoader:
    name: str
    dataset: Depth18TrainingDataset
    sampler: EpochShuffleSampler
    loader: DataLoader

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        self.sampler.set_epoch(epoch)


def build_canonical_dataset(
    *,
    repo_root: str | Path,
    input_config: str | Path,
    include_folds: Sequence[int],
) -> KneeStudyDataset:
    root = Path(repo_root).resolve()
    input_config = Path(input_config)
    if not input_config.is_absolute():
        input_config = root / input_config
    config = KneeInputConfig.from_json(input_config)
    paths = resolve_config_paths(input_config, root)
    return KneeStudyDataset(
        config=config,
        cache_root=paths["cache_root"],
        stage_cache_root=paths["stage_cache_root"],
        selection_manifest=paths["selection_manifest"],
        training_tables=[paths["labelled_csv"], paths["weak_labelled_csv"]],
        training_tables_lock=paths["training_tables_lock"],
        repo_root=root,
        augment=False,
        return_stages=False,
        include_folds=include_folds,
    )


def target_prevalence(dataset: KneeStudyDataset) -> torch.Tensor:
    targets = torch.tensor(
        [[float(row[label]) for label in LABELS] for row in dataset.rows],
        dtype=torch.float32,
    )
    return targets.mean(dim=0)
