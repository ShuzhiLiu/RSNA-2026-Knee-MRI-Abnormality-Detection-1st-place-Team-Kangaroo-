"""Canonical configured-depth model input and training-only GPU augmentation."""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any, Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .input_pipeline import LABELS, _augment_series, _bernoulli, _orientation_targets, _source_positions, stratified_slice_indices

DEPTH = 18
ROLES = 5
TARGETS = len(LABELS)

def central_band_slice_indices(
    positions: np.ndarray,
    slices: int = DEPTH,
    *,
    band: tuple[float, float] = (0.12, 0.88),
    training: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Select deterministic physical strata while retaining global positions.

    Training-loader depth center jitter: ``training=True`` draws a uniform real slice inside each
    stratum from ``generator``; the default is the canonical nearest-to-center rule (validation/inference).
    """

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1 or not positions.size or not np.all(np.isfinite(positions)):
        raise ValueError("slice positions must be a finite non-empty vector")
    if np.any(np.diff(positions) < 0):
        raise ValueError("slice positions must be sorted in ascending physical order")
    if slices < 1 or not (0 <= band[0] < band[1] <= 1):
        raise ValueError("invalid slice count or normalized depth band")

    physical_range = float(positions[-1] - positions[0])
    if positions.size == 1 or physical_range <= 0:
        eligible = np.arange(positions.size, dtype=np.int64)
    else:
        normalized = (positions - positions[0]) / physical_range
        eligible = np.flatnonzero((normalized >= band[0]) & (normalized <= band[1]))
        if not eligible.size:
            eligible = np.asarray([int(np.argmin(np.abs(normalized - 0.5)))], dtype=np.int64)

    if not training:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
    elif generator is None:
        raise ValueError("training-mode slice selection requires an explicit generator")
    selected = stratified_slice_indices(
        positions[eligible], slices, training=bool(training), generator=generator
    )
    indices = eligible[selected["indices"]]
    selected_positions = positions[indices]
    counts = np.bincount(indices, minlength=positions.size)
    selected["indices"] = indices.astype(np.int64)
    selected["repeated"] = (counts[indices] > 1).astype(bool)
    selected["source_multiplicity"] = counts[indices].astype(np.int16)
    selected["positions_mm"] = selected_positions.astype(np.float32)
    selected["positions_normalized"] = (
        ((selected_positions - positions[0]) / physical_range).astype(np.float32)
        if physical_range > 0
        else np.zeros(slices, dtype=np.float32)
    )
    gaps = np.diff(selected_positions).astype(np.float32)
    selected["gap_previous_mm"] = np.zeros(slices, dtype=np.float32)
    selected["gap_next_mm"] = np.zeros(slices, dtype=np.float32)
    if slices > 1:
        selected["gap_previous_mm"][1:] = gaps
        selected["gap_next_mm"][:-1] = gaps
    selected["eligible_source_count"] = np.asarray(eligible.size, dtype=np.int64)
    return selected


def derive_depth18_model_input(
    dataset: Any,
    sample: dict[str, Any],
    *,
    depth: int = DEPTH,
    band: tuple[float, float] = (0.12, 0.88),
    role_bands: list[tuple[float, float]] | None = None,
    role_generators: list[torch.Generator | None] | None = None,
    selection_trace: list[dict[str, Any] | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Derive the model view used identically by training and submission.

    Training loader only: ``role_bands`` overrides the depth band per role (stride jitter),
    ``role_generators`` enables within-stratum center jitter per role. All None (validation,
    inference, default) is the canonical selection rule.
    """

    if depth < 3:
        raise ValueError("model depth must be at least three")
    old = sample["model_input"]
    audits = sample["provenance"]["roles"]
    laterality = sample["provenance"]["laterality"]["laterality"]
    role_count, _, height, width = old["images"].shape
    images = torch.zeros((role_count, depth, height, width), dtype=torch.float32)
    valid = torch.zeros((role_count, depth), dtype=torch.bool)
    repeated = torch.zeros((role_count, depth), dtype=torch.bool)
    positions = torch.zeros((role_count, depth), dtype=torch.float32)
    gap_previous = torch.zeros((role_count, depth), dtype=torch.float32)
    gap_next = torch.zeros((role_count, depth), dtype=torch.float32)
    canonical_indices = torch.full((role_count, depth), -1, dtype=torch.long)
    positions_mm = torch.zeros_like(positions)
    geometry_measured = torch.zeros(role_count, dtype=torch.bool)
    geometry_uniform_fallback = torch.zeros(role_count, dtype=torch.bool)

    for role_index, audit in enumerate(audits):
        if not audit.get("available"):
            continue
        stage = audit.get("deterministic_stage_cache") or {}
        if stage.get("status") != "hit" or not stage.get("directory"):
            raise RuntimeError(f"verified full-depth stage-cache hit required: {audit}")
        stage_dir = Path(stage["directory"])
        stage_meta, _ = dataset._read_metadata(stage_dir / "metadata.json")
        stage_volume = dataset._mmap_volume(stage_dir / "volume.npy")
        source_meta, _ = dataset._read_metadata(dataset.cache_root / audit["metadata_path"])
        _, _, depth_direction = _orientation_targets(audit["role"], laterality)
        source_positions, position_source, _, source_order = _source_positions(
            source_meta, int(stage_volume.shape[0]), depth_direction
        )
        if not np.array_equal(
            source_order, np.asarray(stage_meta["source_order"], dtype=np.int64)
        ):
            raise RuntimeError(f"stage/source canonical order mismatch: {audit['series_uid']}")
        role_band = band if role_bands is None else role_bands[role_index]
        role_generator = None if role_generators is None else role_generators[role_index]
        selected = central_band_slice_indices(
            source_positions, slices=depth, band=role_band,
            training=role_generator is not None, generator=role_generator,
        )
        if selection_trace is not None:
            same_band = central_band_slice_indices(source_positions, slices=depth, band=role_band)
            canonical = central_band_slice_indices(source_positions, slices=depth, band=band)
            selection_trace[role_index] = {
                "eligible": int(selected["eligible_source_count"]),
                "jittered_vs_same_band": int((selected["indices"] != same_band["indices"]).sum()),
                "differs_vs_canonical": int((selected["indices"] != canonical["indices"]).sum()),
            }
        canonical_indices[role_index] = torch.from_numpy(selected["indices"])
        positions_mm[role_index] = torch.from_numpy(selected["positions_mm"])
        geometry_measured[role_index] = position_source in ("canonical_patient_positions", "dicom_spatial_positions")
        geometry_uniform_fallback[role_index] = position_source == "uniform_spacing_fallback"
        values = np.asarray(stage_volume[selected["indices"]], dtype=np.float32).copy()
        images[role_index] = torch.from_numpy(values).div_(65535.0)
        valid[role_index].fill_(True)
        repeated[role_index] = torch.from_numpy(selected["repeated"])
        positions[role_index] = torch.from_numpy(selected["positions_normalized"])
        gap_previous[role_index] = torch.from_numpy(selected["gap_previous_mm"])
        gap_next[role_index] = torch.from_numpy(selected["gap_next_mm"])

    return {
        "images": images,
        "slice_valid_mask": valid,
        "slice_repeated_mask": repeated,
        "slice_positions_normalized": positions,
        "gap_previous_mm": gap_previous,
        "gap_next_mm": gap_next,
        "canonical_indices": canonical_indices,
        "descriptor_source_ids": canonical_indices,
        "descriptor_positions_mm": positions_mm,
        "geometry_measured": geometry_measured,
        "geometry_uniform_fallback": geometry_uniform_fallback,
    }


def descriptor_branch_keep(seed, epoch, index, probability):
    """Separate per-study RNG, no global draw and no inverse probability scaling."""
    if not 0 <= probability <= 1:
        raise ValueError("descriptor dropout probability outside [0,1]")
    keep = True if probability == 0 else bool(np.random.default_rng(
        [int(seed), int(epoch), int(index), 0x44445250]).random() >= probability)
    return keep, dict(seed=int(seed), epoch=int(epoch), index=int(index),
                     probability=float(probability), keep=keep, rng_namespace="0x44445250")

def count_role_dropout(mask: torch.Tensor, seed: int, epoch: int, index: int):
    """Independent stream: uniform count then uniform existing-role subset."""
    existing = torch.flatnonzero(mask.bool()) if hasattr(torch, "flatnonzero") else torch.nonzero(mask.bool(), as_tuple=False).flatten()
    n = int(existing.numel())
    if n == 0:
        raise ValueError("role dropout requires at least one originally available role")
    rng = np.random.default_rng([int(seed), int(epoch), int(index), 0x52444F50])
    k = int(rng.integers(0, min(2, n - 1) + 1))
    dropped = sorted(int(v) for v in rng.choice(existing.numpy(), size=k, replace=False))
    result = mask.bool().clone()
    result[dropped] = False
    return result, {"requested_K": k, "original_mask": mask.bool().tolist(),
                    "dropped_indices": dropped, "resulting_mask": result.tolist(),
                    "rng_namespace": "0x52444F50"}


class Depth18TrainingDataset(Dataset[dict[str, Any]]):
    """Apply the shared model processor, then strip non-training payloads."""

    def __init__(
        self,
        dataset: Any,
        *,
        band: tuple[float, float] = (0.12, 0.88),
        pack_uint16: bool = True,
        depth: int = DEPTH,
        descriptor_dropout_probability: float = 0.0,
        seed: int = 20260823,
        strong_augmentation: dict[str, Any] | None = None,
    ) -> None:
        if dataset.inference_only or dataset.augment:
            raise ValueError("depth adapter requires a supervised identity dataset")
        self.depth = int(depth)
        self.descriptor_dropout_probability = float(descriptor_dropout_probability)
        self.seed = int(seed)
        if self.depth < 3 or not 0 <= self.descriptor_dropout_probability <= 1:
            raise ValueError("invalid depth or descriptor dropout probability")
        self.dataset = dataset
        self.band = tuple(float(value) for value in band)
        self.pack_uint16 = bool(pack_uint16)
        # Training-loader-only augmentation (config train_input_augmentation); None = canonical adapter.
        self.strong_augmentation = strong_augmentation
        if strong_augmentation is not None:
            low, high = (float(v) for v in strong_augmentation["band_scale_range"])
            if not 0 < low <= high or self.band != (0.12, 0.88):
                raise ValueError("band scale jitter requires 0<low<=high and the (0.12,0.88) band")
            if not strong_augmentation["config"].enabled:
                raise ValueError("input-config augmentation block must be enabled")
            self.band_scale_range = (low, high)
            self.augmentation_seed = int(strong_augmentation["seed"])
            log_dir = strong_augmentation.get("trace_dir")
            self.trace_dir = None if log_dir is None else Path(log_dir)
        else:
            self.band_scale_range = None

    def band_scale(self, epoch: int, index: int, role: int) -> float:
        """Deterministic per-(seed, epoch, index, role) depth band-scale draw."""
        low, high = self.band_scale_range
        if low == high:
            return low
        draw = np.random.default_rng([self.augmentation_seed, int(epoch), int(index), int(role)]).random()
        return low + (high - low) * float(draw)

    @staticmethod
    def scaled_band(f: float) -> tuple[float, float]:
        if f == 1.0:
            return (0.12, 0.88)
        return (max(0.0, 0.5 - 0.38 * f), min(1.0, 0.5 + 0.38 * f))

    def role_generator(self, epoch: int, index: int, role: int) -> torch.Generator:
        seed = int(np.random.default_rng([self.augmentation_seed, int(epoch), int(index), int(role), 1]).integers(0, 2**63 - 1))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return generator

    def _strong_getitem(self, index: int, sample: dict[str, Any], model_input: dict[str, Any]) -> dict[str, Any]:
        settings = self.strong_augmentation
        config = settings["config"]
        epoch = int(self.dataset.epoch)
        roles = len(sample["provenance"]["roles"])
        scales = [self.band_scale(epoch, index, role) for role in range(roles)]
        bands = [self.scaled_band(f) for f in scales]
        generators = [self.role_generator(epoch, index, role) for role in range(roles)]
        selection: list[dict[str, Any] | None] = [None] * roles
        derived = derive_depth18_model_input(
            self.dataset, sample, depth=self.depth, band=self.band, role_bands=bands,
            role_generators=generators if settings.get("center_jitter", True) else None,
            selection_trace=selection,
        )
        images = derived["images"]
        valid = derived["slice_valid_mask"].clone()
        series_mask = model_input["series_mask"].clone()
        depth = int(images.shape[1])
        cpu_grids = {}
        trace_rows = []
        role_drop_draws = {}
        for role in range(roles):
            if not bool(series_mask[role]):
                continue
            generator = generators[role]
            full_canvas = torch.ones((1, images.shape[-2], images.shape[-1]), dtype=torch.bool)
            augmented, _, draws = _augment_series(images[role], full_canvas, config, generator)
            if getattr(self, "capture_location", False):
                from .location_data import cpu_affine_grid
                cpu_grids[role] = cpu_affine_grid(images.shape[-2:], draws)
            dropped = torch.zeros(depth, dtype=torch.bool)
            if config.slice_dropout_probability > 0:
                dropped |= torch.rand(depth, generator=generator) < config.slice_dropout_probability
            if config.block_dropout_max_slices > 0 and _bernoulli(generator, config.block_dropout_probability):
                length = int(torch.randint(1, config.block_dropout_max_slices + 1, (), generator=generator).item())
                start = int(torch.randint(0, depth - min(length, depth) + 1, (), generator=generator).item())
                dropped[start:start + min(length, depth)] = True
            if dropped.all():
                dropped[depth // 2] = False
            augmented[dropped] = 0
            images[role] = augmented
            valid[role] &= ~dropped
            if settings.get("role_dropout_policy") is None:
                role_drop_draws[role] = _bernoulli(generator, config.role_dropout_probability)
            trace_rows.append([role, scales[role], selection[role]["eligible"], selection[role]["jittered_vs_same_band"],
                               selection[role]["differs_vs_canonical"], draws["angle_degrees"], draws["translate_x_pixels"],
                               draws["translate_y_pixels"], draws["scale"], draws["contrast"], draws["gamma"], draws["noise_std"],
                               int(dropped.sum())])
        if settings.get("role_dropout_policy") is not None:
            assert settings["role_dropout_policy"] == "uniform_count_0_1_2_v1"
            series_mask, dropout_trace = count_role_dropout(series_mask, self.augmentation_seed, epoch, index)
            role_drop_draws = {role: role in dropout_trace["dropped_indices"] for role in range(roles)
                               if dropout_trace["original_mask"][role]}
            # Apply the native guard below to the original mask, preserving exact masking semantics.
            series_mask = torch.tensor(dropout_trace["original_mask"], dtype=torch.bool)
        else:
            dropout_trace = None
        role_dropped = {}
        for role in sorted(role_drop_draws):
            drop = role_drop_draws[role] and int(series_mask.sum()) > 1
            if drop:
                images[role].zero_()
                valid[role].fill_(False)
                series_mask[role] = False
            role_dropped[role] = bool(drop)
        if self.trace_dir is not None:
            with (self.trace_dir / f"worker_{os.getpid()}.tsv").open("a") as log:
                for row in trace_rows:
                    log.write("\t".join([str(epoch), str(index), str(sample["sample_id"]), *map(repr, row),
                                          str(int(role_dropped[row[0]])), str(int(role_drop_draws[row[0]]))]) + "\n")
        if dropout_trace is not None and self.trace_dir is not None:
            import json
            with (self.trace_dir / f"role_dropout_worker_{os.getpid()}.jsonl").open("a") as log:
                log.write(json.dumps({"epoch": epoch, "index": index, "sample_id": sample["sample_id"], **dropout_trace}) + "\n")
        derived["cpu_grids"] = cpu_grids
        derived["slice_valid_mask"] = valid
        return derived, series_mask

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        model_input = dict(sample["model_input"])
        if self.strong_augmentation is None:
            model_input.update(
                derive_depth18_model_input(self.dataset, sample, depth=self.depth, band=self.band)
            )
        else:
            derived, series_mask = self._strong_getitem(index, sample, model_input)
            model_input.update(derived)
            model_input["series_mask"] = series_mask
        descriptor_keep, descriptor_trace = descriptor_branch_keep(
            self.seed, int(self.dataset.epoch), index,
            self.descriptor_dropout_probability if self.strong_augmentation is not None else 0.0)
        if self.strong_augmentation is not None and self.trace_dir is not None:
            import json
            with (self.trace_dir / f"descriptor_dropout_worker_{os.getpid()}.jsonl").open("a") as log:
                log.write(json.dumps({"sample_id":sample["sample_id"], **descriptor_trace}) + "\n")
        images = model_input["images"]
        if self.pack_uint16:
            images = torch.round(images.clamp(0, 1) * 65535).to(torch.uint16)
        available = [role for role in sample["provenance"]["roles"] if role.get("available")]
        result = {
            "sample_id": sample["sample_id"],
            "fold": sample["fold"],
            "images": images,
            "series_mask": model_input["series_mask"],
            "slice_valid_mask": model_input["slice_valid_mask"],
            "slice_repeated_mask": model_input["slice_repeated_mask"],
            "pixel_valid_mask": model_input["pixel_valid_mask"],
            "slice_positions_normalized": model_input["slice_positions_normalized"],
            "gap_previous_mm": model_input["gap_previous_mm"],
            "gap_next_mm": model_input["gap_next_mm"],
            **{key: model_input[key] for key in ("descriptor_source_ids", "descriptor_positions_mm", "geometry_measured", "geometry_uniform_fallback")},
            "descriptor_keep": torch.tensor(descriptor_keep, dtype=torch.bool),
            "labels": sample["targets"]["labels"],
            "label_mask": sample["targets"]["label_mask"],
            "is_hard": sample["targets"]["is_hard"],
            "available_roles": int(model_input["series_mask"].sum()),
            "stage_cache_hits": len(available),
            "stage_cache_roles": len(available),
        }

        if getattr(self, "capture_location", False):
            return self.attach_location(result, sample, model_input)
        return result


def make_safe_augment(profile: dict[str, float]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the selected no-flip, no-zoom-out GPU augmentation."""

    rotation_deg = float(profile["rotation_deg"])
    zoom_in = float(profile["zoom_in"])
    shift = float(profile["shift"])
    intensity_jitter = float(profile["intensity"])

    def augment(values: torch.Tensor, *, return_grid: bool = False):
        count, device = values.shape[0], values.device
        rotation = (torch.rand(count, device=device) - 0.5) * 2
        rotation = rotation * (rotation_deg * math.pi / 180.0)
        scale = 1.0 + torch.rand(count, device=device) * zoom_in
        shift_x = (torch.rand(count, device=device) - 0.5) * (2 * shift)
        shift_y = (torch.rand(count, device=device) - 0.5) * (2 * shift)
        cosine = torch.cos(rotation) / scale
        sine = torch.sin(rotation) / scale
        theta = torch.zeros(count, 2, 3, device=device, dtype=torch.float32)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cosine, -sine, shift_x
        theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sine, cosine, shift_y
        grid = F.affine_grid(theta, values.shape, align_corners=False)
        values = F.grid_sample(
            values, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        intensity = 1.0 + (
            torch.rand(count, 1, 1, 1, device=device) - 0.5
        ) * (2 * intensity_jitter)
        augment.last_theta = theta.detach()
        augment.last_intensity = intensity.detach()
        output = (values * intensity).clamp(0, 1)
        return (output, grid) if return_grid else output

    return augment
