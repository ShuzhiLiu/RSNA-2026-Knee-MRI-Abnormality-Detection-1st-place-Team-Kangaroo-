"""Canonical study-level MRI input pipeline for knee input design v2.

The module deliberately keeps deployable image inputs separate from training-only
targets. It consumes the locked wide CSV tables for supervision and uses only the
``selected_series`` and ``series_candidates`` image-linkage sections of the legacy
selection manifest. Stale labels, folds and text from that manifest are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


PIPELINE_VERSION = "knee-input-pipeline-v2"
DEFAULT_VIEW_VERSION = "knee-view-default-5x16x384-v1"
BEST_VIEW_VERSION = "knee-view-information-preserving-v1"
ROLES = (
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_STRUCT",
    "COR_STRUCT",
)
LABELS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
SEQUENCE_FAMILY_IDS = {"unknown": 0, "other": 1, "pd": 2, "t1": 3, "t2": 4, "stir": 5}
PLANE_IDS = {"SAG": 0, "COR": 1, "AX": 2}
LATERALITY_IDS = {None: 0, "L": 1, "R": 2}


def clinfusion_text_column(label: str) -> str:
    return f"ClinFusion Dense - {label}"


def hulu_text_column(label: str) -> str:
    return f"Hulu Dense - {label}"


def clinfusion_mask_column(label: str) -> str:
    return f"ClinFusion Embedding Eligible - {label}"


def hulu_mask_column(label: str) -> str:
    return f"Hulu Embedding Eligible - {label}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    rotation_degrees: float
    translation_fraction: float
    scale_min: float
    scale_max: float
    gamma_min: float
    gamma_max: float
    contrast_min: float
    contrast_max: float
    noise_std_max: float
    slice_dropout_probability: float
    block_dropout_probability: float
    block_dropout_max_slices: int
    role_dropout_probability: float
    horizontal_flip_probability: float
    vertical_flip_probability: float

    def validate(self) -> None:
        if self.horizontal_flip_probability != 0 or self.vertical_flip_probability != 0:
            raise ValueError("knee input v2 forbids horizontal and vertical flip augmentation")
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("invalid augmentation scale range")
        if not 0 < self.gamma_min <= self.gamma_max:
            raise ValueError("invalid augmentation gamma range")
        if not 0 < self.contrast_min <= self.contrast_max:
            raise ValueError("invalid augmentation contrast range")
        for value in (
            self.slice_dropout_probability,
            self.block_dropout_probability,
            self.role_dropout_probability,
        ):
            if not 0 <= value < 1:
                raise ValueError("dropout probabilities must be in [0,1)")
        if self.block_dropout_max_slices < 0:
            raise ValueError("block_dropout_max_slices must be non-negative")


@dataclass(frozen=True)
class KneeInputConfig:
    version: str
    roles: tuple[str, ...]
    slices_per_series: int
    canvas_mm: float
    image_size: int
    cache_dtype: str
    cache_config_fingerprint: str
    selection_manifest_sha256: str
    selection_manifest_bytes: int
    cache_intensity_contract: str
    sampling: dict[str, Any]
    physical_crop: dict[str, Any]
    orientation: dict[str, Any]
    deterministic_stage_cache: dict[str, Any]
    seed: int
    augmentation: AugmentationConfig

    @classmethod
    def from_json(cls, path: str | Path) -> "KneeInputConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(
            version=str(value["version"]),
            roles=tuple(value["roles"]),
            slices_per_series=int(value["slices_per_series"]),
            canvas_mm=float(value["canvas_mm"]),
            image_size=int(value["image_size"]),
            cache_dtype=str(value["cache_dtype"]),
            cache_config_fingerprint=str(value["cache_config_fingerprint"]),
            selection_manifest_sha256=str(value["selection_manifest_sha256"]),
            selection_manifest_bytes=int(value["selection_manifest_bytes"]),
            cache_intensity_contract=str(value["cache_intensity_contract"]),
            sampling=dict(value["sampling"]),
            physical_crop=dict(value["physical_crop"]),
            orientation=dict(value["orientation"]),
            deterministic_stage_cache=dict(value["deterministic_stage_cache"]),
            seed=int(value["seed"]),
            augmentation=AugmentationConfig(**value["augmentation"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.version != PIPELINE_VERSION:
            raise ValueError(f"unsupported input pipeline version: {self.version}")
        if self.roles != ROLES:
            raise ValueError(f"role order must be exactly {ROLES}")
        if self.slices_per_series < 1 or self.canvas_mm <= 0 or self.image_size < 1:
            raise ValueError("invalid spatial/depth input dimensions")
        if self.cache_dtype != "uint16":
            raise ValueError("input v2 currently requires canonical uint16 cache")
        if len(self.selection_manifest_sha256) != 64 or self.selection_manifest_bytes < 1:
            raise ValueError("selection manifest artifact fingerprint is invalid")
        if self.sampling.get("allow_depth_reversal") is not False:
            raise ValueError("stochastic depth reversal must be disabled")
        if self.physical_crop.get("keep_physical_aspect_ratio") is not True:
            raise ValueError("physical aspect ratio must be preserved")
        if self.orientation.get("appearance_inference") is not False:
            raise ValueError("appearance-based laterality inference is forbidden")
        if self.deterministic_stage_cache.get("dtype") != "uint16":
            raise ValueError("deterministic stage cache must preserve uint16 precision")
        if self.deterministic_stage_cache.get("write_policy") != "offline-atomic-only":
            raise ValueError("workers must never write the deterministic stage cache")
        self.augmentation.validate()

    @property
    def fingerprint(self) -> str:
        value = {
            "version": self.version,
            "roles": self.roles,
            "slices_per_series": self.slices_per_series,
            "canvas_mm": self.canvas_mm,
            "image_size": self.image_size,
            "cache_dtype": self.cache_dtype,
            "cache_config_fingerprint": self.cache_config_fingerprint,
            "cache_intensity_contract": self.cache_intensity_contract,
            "sampling": self.sampling,
            "physical_crop": self.physical_crop,
            "orientation": self.orientation,
            "deterministic_stage_cache": self.deterministic_stage_cache,
            "seed": self.seed,
            "augmentation": self.augmentation.__dict__,
        }
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class StudyViewConfig:
    """A materialized view derived from the same lazy :class:`StudyInput`.

    ``stratified_fixed`` is the backwards-compatible, batchable view. ``all_valid``
    is intentionally ragged: it preserves every real source slice and must not be
    silently padded or truncated by this preprocessing layer.
    """

    version: str
    depth_mode: str
    slices_per_series: int | None
    fov_mode: str
    canvas_mm: float | None
    image_size: int
    include_complementary_series: bool

    @classmethod
    def from_json(cls, path: str | Path) -> "StudyViewConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        view = cls(
            version=str(value["version"]),
            depth_mode=str(value["depth_mode"]),
            slices_per_series=(
                None
                if value.get("slices_per_series") is None
                else int(value["slices_per_series"])
            ),
            fov_mode=str(value["fov_mode"]),
            canvas_mm=None if value.get("canvas_mm") is None else float(value["canvas_mm"]),
            image_size=int(value["image_size"]),
            include_complementary_series=bool(value["include_complementary_series"]),
        )
        view.validate()
        return view

    def validate(self) -> None:
        if self.depth_mode not in {"stratified_fixed", "all_valid"}:
            raise ValueError(f"unsupported depth mode: {self.depth_mode}")
        if self.depth_mode == "stratified_fixed":
            if self.slices_per_series is None or self.slices_per_series < 1:
                raise ValueError("stratified_fixed view requires a positive slice count")
        elif self.slices_per_series is not None:
            raise ValueError("all_valid view must not declare a fixed slice count")
        if self.fov_mode not in {"fixed_physical_canvas", "full_source_fov"}:
            raise ValueError(f"unsupported FOV mode: {self.fov_mode}")
        if self.fov_mode == "fixed_physical_canvas":
            if self.canvas_mm is None or self.canvas_mm <= 0:
                raise ValueError("fixed physical canvas requires canvas_mm")
        elif self.canvas_mm is not None:
            raise ValueError("full source FOV derives its canvas from each series")
        if self.image_size < 1:
            raise ValueError("image_size must be positive")

    def is_default_for(self, config: KneeInputConfig) -> bool:
        return bool(
            self.version == default_view_version(config)
            and self.depth_mode == "stratified_fixed"
            and self.slices_per_series == config.slices_per_series
            and self.fov_mode == "fixed_physical_canvas"
            and self.canvas_mm == config.canvas_mm
            and self.image_size == config.image_size
            and not self.include_complementary_series
        )

    def is_stage_cacheable_for(self, config: KneeInputConfig) -> bool:
        """Whether this published view has a deterministic full-depth cache."""

        if self.image_size != config.image_size:
            return False
        if self.version == default_view_version(config):
            return self.is_default_for(config)
        return bool(
            self.version == BEST_VIEW_VERSION
            and self.depth_mode == "all_valid"
            and self.slices_per_series is None
            and self.fov_mode == "full_source_fov"
            and self.canvas_mm is None
            and self.include_complementary_series
        )


def default_view_version(config: KneeInputConfig) -> str:
    """Preserve the published cache identity for each configured resolution."""
    return f"knee-view-default-5x{config.slices_per_series}x{config.image_size}-v1"


def default_study_view(config: KneeInputConfig) -> StudyViewConfig:
    """Return the locked, batchable configured ``[5,16,H,W]`` view."""

    view = StudyViewConfig(
        version=default_view_version(config),
        depth_mode="stratified_fixed",
        slices_per_series=config.slices_per_series,
        fov_mode="fixed_physical_canvas",
        canvas_mm=config.canvas_mm,
        image_size=config.image_size,
        include_complementary_series=False,
    )
    view.validate()
    return view


def best_study_view(config: KneeInputConfig) -> StudyViewConfig:
    """Return the information-preserving candidate view at fixed 320 resolution."""

    view = StudyViewConfig(
        version=BEST_VIEW_VERSION,
        depth_mode="all_valid",
        slices_per_series=None,
        fov_mode="full_source_fov",
        canvas_mm=None,
        image_size=config.image_size,
        include_complementary_series=True,
    )
    view.validate()
    return view


@dataclass(frozen=True)
class SeriesInput:
    """Lazy reference to one cached series; no pixel array is loaded here."""

    role: str
    slot: str
    kind: str
    record: dict[str, Any]


@dataclass(frozen=True)
class StudyInput:
    """Complete image-side study description shared by Train and Test."""

    sample_id: str
    primary_series: tuple[SeriesInput | None, ...]
    complementary_series: tuple[SeriesInput, ...]
    laterality: dict[str, Any]
    candidate_series_count: int


def _uniform(generator: torch.Generator, lower: float, upper: float) -> float:
    if lower == upper:
        return float(lower)
    return float(torch.empty((), dtype=torch.float32).uniform_(lower, upper, generator=generator))


def _bernoulli(generator: torch.Generator, probability: float) -> bool:
    return bool(probability > 0 and torch.rand((), generator=generator).item() < probability)


def _stable_seed(base_seed: int, epoch: int, sample_id: str) -> int:
    payload = f"{base_seed}:{epoch}:{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _patient_center_x(metadata: dict[str, Any]) -> float | None:
    """Return the series center in DICOM LPS x, without guessing from pixels."""

    orientation = metadata.get("image_orientation_patient")
    positions = metadata.get("image_positions_patient")
    spacing = metadata.get("pixel_spacing")
    rows = metadata.get("rows")
    columns = metadata.get("columns")
    try:
        orientation_array = np.asarray(orientation, dtype=np.float64)
        positions_array = np.asarray(positions, dtype=np.float64)
        spacing_array = np.asarray(spacing, dtype=np.float64)
        if (
            orientation_array.shape != (6,)
            or positions_array.ndim != 2
            or positions_array.shape[1] != 3
            or spacing_array.shape != (2,)
            or not rows
            or not columns
            or not np.all(np.isfinite(positions_array))
        ):
            return None
        center = np.median(positions_array, axis=0)
        center += orientation_array[:3] * spacing_array[1] * (int(columns) - 1) / 2
        center += orientation_array[3:] * spacing_array[0] * (int(rows) - 1) / 2
        return float(center[0])
    except (TypeError, ValueError):
        return None


def determine_study_laterality(
    metadata_values: Sequence[dict[str, Any]],
    *,
    coordinate_threshold_mm: float = 30.0,
) -> dict[str, Any]:
    """Conservatively combine DICOM tags and patient-coordinate geometry.

    Strong tag/geometry disagreement, conflicting tags and centerline geometry are
    represented as unknown instead of being resolved from image appearance.
    """

    tags: set[str] = set()
    center_x_values: list[float] = []
    for metadata in metadata_values:
        raw = str(metadata.get("image_laterality") or metadata.get("laterality") or "").upper()
        if raw in {"L", "LEFT"}:
            tags.add("L")
        elif raw in {"R", "RIGHT"}:
            tags.add("R")
        center_x = _patient_center_x(metadata)
        if center_x is not None:
            center_x_values.append(center_x)
    median_x = float(np.median(center_x_values)) if center_x_values else None
    geometry = None
    if median_x is not None and median_x > coordinate_threshold_mm:
        geometry = "L"
    elif median_x is not None and median_x < -coordinate_threshold_mm:
        geometry = "R"

    if len(tags) > 1:
        laterality, source = None, "unknown_conflicting_dicom_tags"
    elif tags:
        tag = next(iter(tags))
        if geometry is not None and tag != geometry:
            laterality, source = None, "unknown_tag_geometry_conflict"
        elif geometry is None:
            laterality, source = tag, "dicom_laterality"
        else:
            laterality, source = tag, "dicom_laterality_geometry_agree"
    elif geometry is not None:
        laterality, source = geometry, "patient_coordinate_geometry"
    else:
        laterality, source = None, "unknown_centerline_or_missing_geometry"
    return {
        "laterality": laterality,
        "source": source,
        "quality": laterality is not None,
        "median_center_x_mm": median_x,
        "coordinate_threshold_mm": coordinate_threshold_mm,
        "dicom_tags": sorted(tags),
    }


def _orientation_targets(role: str, laterality: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Canonical output horizontal, vertical and depth directions in DICOM LPS."""

    inferior = np.array([0.0, 0.0, -1.0])
    anterior = np.array([0.0, -1.0, 0.0])
    posterior = np.array([0.0, 1.0, 0.0])
    lateral = None
    if laterality == "L":
        lateral = np.array([1.0, 0.0, 0.0])
    elif laterality == "R":
        lateral = np.array([-1.0, 0.0, 0.0])
    patient_left = np.array([1.0, 0.0, 0.0])
    if role.startswith("SAG"):
        return anterior, inferior, lateral
    if role.startswith("COR"):
        return lateral if lateral is not None else patient_left, inferior, posterior
    if role.startswith("AX"):
        return lateral if lateral is not None else patient_left, posterior, inferior
    raise ValueError(f"unsupported role: {role}")


def canonicalize_inplane_orientation(
    images: torch.Tensor,
    pixel_spacing: Sequence[float],
    metadata: dict[str, Any],
    role: str,
    laterality: str | None,
) -> tuple[torch.Tensor, list[float], dict[str, Any]]:
    """Apply only geometry-defined transpose/flips; never inspect anatomy."""

    orientation = metadata.get("image_orientation_patient")
    trace: dict[str, Any] = {
        "method": "identity_missing_geometry",
        "transpose": False,
        "flip_horizontal": False,
        "flip_vertical": False,
        "minimum_axis_alignment": 0.0,
        "quality": False,
    }
    spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    try:
        source = np.asarray(orientation, dtype=np.float64)
        if source.shape != (6,) or not np.all(np.isfinite(source)):
            return images, spacing, trace
        source_horizontal = source[:3]
        source_vertical = source[3:]
        target_horizontal, target_vertical, _ = _orientation_targets(role, laterality)
        direct = abs(float(source_horizontal @ target_horizontal)) + abs(float(source_vertical @ target_vertical))
        swapped = abs(float(source_vertical @ target_horizontal)) + abs(float(source_horizontal @ target_vertical))
        transpose = swapped > direct
        aligned_horizontal = source_vertical if transpose else source_horizontal
        aligned_vertical = source_horizontal if transpose else source_vertical
        horizontal_alignment = float(aligned_horizontal @ target_horizontal)
        vertical_alignment = float(aligned_vertical @ target_vertical)
        if transpose:
            images = images.transpose(-2, -1)
            spacing = [spacing[1], spacing[0]]
        flip_horizontal = horizontal_alignment < 0
        flip_vertical = vertical_alignment < 0
        if flip_horizontal:
            images = torch.flip(images, dims=(-1,))
        if flip_vertical:
            images = torch.flip(images, dims=(-2,))
        minimum_alignment = min(abs(horizontal_alignment), abs(vertical_alignment))
        trace = {
            "method": "dicom_patient_coordinates",
            "transpose": transpose,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "horizontal_alignment": horizontal_alignment,
            "vertical_alignment": vertical_alignment,
            "minimum_axis_alignment": minimum_alignment,
            "quality": bool(metadata.get("orientation_consistent")) and minimum_alignment >= 0.7,
        }
        return images, spacing, trace
    except (TypeError, ValueError):
        return images, spacing, trace


def _source_positions(
    metadata: dict[str, Any],
    count: int,
    depth_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, str, float, np.ndarray]:
    order = np.arange(count, dtype=np.int64)
    if depth_direction is not None:
        image_positions = metadata.get("image_positions_patient")
        try:
            positions_array = np.asarray(image_positions, dtype=np.float64)
            if positions_array.shape == (count, 3) and np.all(np.isfinite(positions_array)):
                values = positions_array @ depth_direction
                order = np.argsort(values, kind="stable")
                ordered = values[order]
                spacing = float(np.median(np.diff(ordered))) if count > 1 else 0.0
                return ordered, "canonical_patient_positions", spacing, order
        except (TypeError, ValueError):
            pass
    explicit = metadata.get("slice_positions_mm")
    if isinstance(explicit, list) and len(explicit) == count:
        values = np.asarray(explicit, dtype=np.float64)
        if np.all(np.isfinite(values)) and np.all(np.diff(values) >= 0):
            spacing = float(np.median(np.diff(values))) if count > 1 else 0.0
            return values, "dicom_spatial_positions", spacing, order

    spacing = metadata.get("spacing_between_slices")
    if spacing is None or not math.isfinite(float(spacing)) or float(spacing) <= 0:
        spacing = metadata.get("slice_thickness")
    if spacing is None or not math.isfinite(float(spacing)) or float(spacing) <= 0:
        spacing = 1.0
        source = "unit_spacing_fallback"
    else:
        spacing = abs(float(spacing))
        source = "uniform_spacing_fallback"
    return np.arange(count, dtype=np.float64) * spacing, source, spacing, order


def stratified_slice_indices(
    positions: np.ndarray,
    slices: int,
    *,
    training: bool,
    generator: torch.Generator,
) -> dict[str, np.ndarray]:
    """Sample the complete physical range without reversing depth order."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1 or not positions.size or not np.all(np.isfinite(positions)):
        raise ValueError("slice positions must be a finite non-empty vector")
    if np.any(np.diff(positions) < 0):
        raise ValueError("slice positions must be sorted in ascending physical order")

    if positions.size == 1 or positions[-1] == positions[0]:
        indices = np.zeros(slices, dtype=np.int64)
        centers = np.full(slices, positions[0], dtype=np.float64)
    else:
        edges = np.linspace(positions[0], positions[-1], slices + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        chosen: list[int] = []
        for index, center in enumerate(centers):
            if index == slices - 1:
                candidates = np.flatnonzero((positions >= edges[index]) & (positions <= edges[index + 1]))
            else:
                candidates = np.flatnonzero((positions >= edges[index]) & (positions < edges[index + 1]))
            if candidates.size:
                if training:
                    offset = int(torch.randint(candidates.size, (), generator=generator).item())
                    selected = int(candidates[offset])
                else:
                    selected = int(candidates[np.argmin(np.abs(positions[candidates] - center))])
            else:
                selected = int(np.argmin(np.abs(positions - center)))
            chosen.append(selected)
        indices = np.asarray(chosen, dtype=np.int64)

    counts = np.bincount(indices, minlength=positions.size)
    repeated = counts[indices] > 1
    selected_positions = positions[indices]
    normalized = np.zeros(slices, dtype=np.float32)
    physical_range = float(positions[-1] - positions[0])
    if physical_range > 0:
        normalized = ((selected_positions - positions[0]) / physical_range).astype(np.float32)
    gap_previous = np.zeros(slices, dtype=np.float32)
    gap_next = np.zeros(slices, dtype=np.float32)
    if slices > 1:
        gaps = np.diff(selected_positions).astype(np.float32)
        gap_previous[1:] = gaps
        gap_next[:-1] = gaps
    return {
        "indices": indices,
        "repeated": repeated.astype(bool),
        "source_multiplicity": counts[indices].astype(np.int16),
        "positions_mm": selected_positions.astype(np.float32),
        "positions_normalized": normalized,
        "gap_previous_mm": gap_previous,
        "gap_next_mm": gap_next,
        "stratum_centers_mm": centers.astype(np.float32),
        "distance_to_center_mm": np.abs(selected_positions - centers).astype(np.float32),
    }


def all_valid_slice_indices(positions: np.ndarray) -> dict[str, np.ndarray]:
    """Select every real slice once while retaining the sampling metadata schema."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1 or not positions.size or not np.all(np.isfinite(positions)):
        raise ValueError("slice positions must be a finite non-empty vector")
    if np.any(np.diff(positions) < 0):
        raise ValueError("slice positions must be sorted in ascending physical order")
    count = positions.size
    indices = np.arange(count, dtype=np.int64)
    normalized = np.zeros(count, dtype=np.float32)
    physical_range = float(positions[-1] - positions[0])
    if physical_range > 0:
        normalized = ((positions - positions[0]) / physical_range).astype(np.float32)
    gap_previous = np.zeros(count, dtype=np.float32)
    gap_next = np.zeros(count, dtype=np.float32)
    if count > 1:
        gaps = np.diff(positions).astype(np.float32)
        gap_previous[1:] = gaps
        gap_next[:-1] = gaps
    return {
        "indices": indices,
        "repeated": np.zeros(count, dtype=bool),
        "source_multiplicity": np.ones(count, dtype=np.int16),
        "positions_mm": positions.astype(np.float32),
        "positions_normalized": normalized,
        "gap_previous_mm": gap_previous,
        "gap_next_mm": gap_next,
        "stratum_centers_mm": positions.astype(np.float32),
        "distance_to_center_mm": np.zeros(count, dtype=np.float32),
    }


def physical_center_crop_resize(
    images: torch.Tensor,
    pixel_spacing: Sequence[float],
    canvas_mm: float,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Center a square physical canvas, padding instead of stretching anatomy."""

    if images.ndim != 3:
        raise ValueError(f"expected [D,H,W], found {tuple(images.shape)}")
    row_spacing, column_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    if row_spacing <= 0 or column_spacing <= 0:
        raise ValueError("pixel spacing must be positive")
    depth, height, width = images.shape
    crop_height = max(1, int(round(canvas_mm / row_spacing)))
    crop_width = max(1, int(round(canvas_mm / column_spacing)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    bottom = top + crop_height
    right = left + crop_width
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - height)
    pad_right = max(0, right - width)

    padded = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    valid = F.pad(
        torch.ones((1, height, width), dtype=torch.float32),
        (pad_left, pad_right, pad_top, pad_bottom),
        value=0.0,
    )
    top += pad_top
    left += pad_left
    cropped = padded[:, top:top + crop_height, left:left + crop_width]
    cropped_valid = valid[:, top:top + crop_height, left:left + crop_width]
    output = F.interpolate(
        cropped.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    output_valid = F.interpolate(
        cropped_valid.unsqueeze(0),
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0) > 0.5
    # Antialiased interpolation can otherwise bleed a small non-zero value into
    # the region that the nearest-neighbor validity mask calls padding.
    output = torch.where(output_valid.expand_as(output), output, torch.zeros_like(output))
    trace = {
        "source_shape": [int(depth), int(height), int(width)],
        "source_fov_mm": [height * row_spacing, width * column_spacing],
        "requested_canvas_mm": canvas_mm,
        "crop_shape_pixels": [crop_height, crop_width],
        "crop_box_source_pixels": [
            (height - crop_height) // 2,
            (width - crop_width) // 2,
            (height - crop_height) // 2 + crop_height,
            (width - crop_width) // 2 + crop_width,
        ],
        "padding_pixels": [pad_top, pad_left, pad_bottom, pad_right],
        "padding_fraction": float(1.0 - output_valid.float().mean().item()),
        "output_shape": [int(depth), image_size, image_size],
        "center_method": "source_image_center",
    }
    return output.clamp_(0.0, 1.0), output_valid, trace


def full_source_fov_resize(
    images: torch.Tensor,
    pixel_spacing: Sequence[float],
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit the complete physical source FOV into a square, padding the short axis."""

    if images.ndim != 3:
        raise ValueError(f"expected [D,H,W], found {tuple(images.shape)}")
    row_spacing, column_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
    if row_spacing <= 0 or column_spacing <= 0:
        raise ValueError("pixel spacing must be positive")
    canvas_mm = max(images.shape[-2] * row_spacing, images.shape[-1] * column_spacing)
    output, valid, trace = physical_center_crop_resize(
        images,
        pixel_spacing,
        canvas_mm,
        image_size,
    )
    trace["fov_mode"] = "full_source_fov"
    trace["source_coverage_fraction"] = 1.0
    trace["derived_canvas_mm"] = canvas_mm
    return output, valid, trace


def quantize_deterministic_stage(images: torch.Tensor) -> torch.Tensor:
    """Make online cache misses numerically identical to the uint16 stage cache."""

    return torch.round(images.clamp(0.0, 1.0) * 65535.0).div_(65535.0)


def deterministic_stage_fingerprint(
    config: KneeInputConfig,
    source_metadata_sha256: str,
    role: str,
    laterality: str | None,
    view: StudyViewConfig | None = None,
) -> str:
    view = view or default_study_view(config)
    view.validate()
    payload = {
        "pipeline": config.fingerprint,
        "source_metadata_sha256": source_metadata_sha256,
        "role": role,
        "laterality": laterality,
        "stage": config.deterministic_stage_cache,
        "view": {
            "version": view.version,
            "fov_mode": view.fov_mode,
            "canvas_mm": view.canvas_mm,
            "image_size": view.image_size,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def build_deterministic_stage_cache(
    *,
    config: KneeInputConfig,
    source_volume_path: str | Path,
    source_metadata_path: str | Path,
    destination: str | Path,
    role: str,
    laterality: str | None,
    view: StudyViewConfig | None = None,
    chunk_depth: int = 16,
) -> dict[str, Any]:
    """Atomically cache only deterministic spatial work for multi-epoch reuse."""

    source_volume_path = Path(source_volume_path)
    source_metadata_path = Path(source_metadata_path)
    destination = Path(destination)
    view = view or default_study_view(config)
    view.validate()
    if not view.is_stage_cacheable_for(config):
        raise ValueError(f"view is not eligible for stage cache: {view.version}")
    if chunk_depth < 1:
        raise ValueError("chunk_depth must be positive")
    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_fingerprint") != config.cache_config_fingerprint:
        raise RuntimeError(f"source cache fingerprint mismatch: {source_metadata_path}")
    volume = np.load(source_volume_path, mmap_mode="r", allow_pickle=False)
    if volume.dtype != np.uint16 or volume.ndim != 3:
        raise RuntimeError(f"invalid source cache volume: {source_volume_path}")
    _, _, depth_direction = _orientation_targets(role, laterality)
    _, position_source, _, source_order = _source_positions(metadata, volume.shape[0], depth_direction)
    source_sha = file_sha256(source_metadata_path)
    fingerprint = deterministic_stage_fingerprint(
        config, source_sha, role, laterality, view
    )
    destination.mkdir(parents=True, exist_ok=True)
    volume_path = destination / "volume.npy"
    valid_path = destination / "pixel_valid.npy"
    metadata_path = destination / "metadata.json"
    existing = [path.is_file() for path in (volume_path, valid_path, metadata_path)]
    if any(existing):
        if not all(existing):
            raise RuntimeError(f"refusing incomplete deterministic stage cache: {destination}")
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
        if current.get("fingerprint") != fingerprint:
            raise RuntimeError(f"refusing incompatible deterministic stage cache: {destination}")
        current["cache_reused"] = True
        return current
    temporary_volume = destination / f".volume.{os.getpid()}.npy"
    temporary_valid = destination / f".pixel_valid.{os.getpid()}.npy"
    temporary_metadata = destination / f".metadata.{os.getpid()}.json"
    output = np.lib.format.open_memmap(
        temporary_volume,
        mode="w+",
        dtype=np.uint16,
        shape=(volume.shape[0], config.image_size, config.image_size),
    )
    crop_trace: dict[str, Any] | None = None
    orientation_trace: dict[str, Any] | None = None
    oriented_spacing: list[float] | None = None
    pixel_valid: torch.Tensor | None = None
    try:
        for start in range(0, volume.shape[0], chunk_depth):
            ordered_indices = source_order[start:start + chunk_depth]
            source_images = torch.from_numpy(
                np.asarray(volume[ordered_indices], dtype=np.float32).copy()
            ).div_(65535.0)
            oriented, oriented_spacing, orientation_trace = canonicalize_inplane_orientation(
                source_images,
                metadata["pixel_spacing"],
                metadata,
                role,
                laterality,
            )
            if view.fov_mode == "fixed_physical_canvas":
                assert view.canvas_mm is not None
                resized, pixel_valid, crop_trace = physical_center_crop_resize(
                    oriented,
                    oriented_spacing,
                    view.canvas_mm,
                    view.image_size,
                )
                crop_trace["fov_mode"] = "fixed_physical_canvas"
                source_area = float(np.prod(crop_trace["source_fov_mm"]))
                retained_area = min(
                    float(crop_trace["source_fov_mm"][0]), view.canvas_mm
                ) * min(float(crop_trace["source_fov_mm"][1]), view.canvas_mm)
                crop_trace["source_coverage_fraction"] = (
                    retained_area / source_area if source_area > 0 else 0.0
                )
            else:
                resized, pixel_valid, crop_trace = full_source_fov_resize(
                    oriented,
                    oriented_spacing,
                    view.image_size,
                )
            output[start:start + len(ordered_indices)] = np.rint(
                resized.numpy() * 65535.0
            ).astype(np.uint16)
        output.flush()
        assert pixel_valid is not None and crop_trace is not None and orientation_trace is not None
        crop_trace["source_shape"][0] = int(volume.shape[0])
        crop_trace["output_shape"][0] = int(volume.shape[0])
        np.save(temporary_valid, pixel_valid.numpy(), allow_pickle=False)
        stage_metadata = {
            "schema_version": "knee-deterministic-stage-cache-v2",
            "fingerprint": fingerprint,
            "input_config_fingerprint": config.fingerprint,
            "view_version": view.version,
            "fov_mode": view.fov_mode,
            "source_metadata_sha256": source_sha,
            "source_volume_bytes": source_volume_path.stat().st_size,
            "source_shape": list(volume.shape),
            "output_shape": list(output.shape),
            "dtype": "uint16",
            "role": role,
            "laterality": laterality,
            "position_source": position_source,
            "source_order": source_order.tolist(),
            "oriented_pixel_spacing": oriented_spacing,
            "orientation": orientation_trace,
            "crop": crop_trace,
        }
        temporary_metadata.write_text(
            json.dumps(stage_metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        del output
        os.replace(temporary_volume, volume_path)
        os.replace(temporary_valid, valid_path)
        os.replace(temporary_metadata, metadata_path)
        return stage_metadata
    finally:
        try:
            del output
        except UnboundLocalError:
            pass
        temporary_volume.unlink(missing_ok=True)
        temporary_valid.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def _augment_series(
    images: torch.Tensor,
    pixel_valid: torch.Tensor,
    config: AugmentationConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    angle = _uniform(generator, -config.rotation_degrees, config.rotation_degrees)
    translate_x = int(round(_uniform(generator, -config.translation_fraction, config.translation_fraction) * images.shape[-1]))
    translate_y = int(round(_uniform(generator, -config.translation_fraction, config.translation_fraction) * images.shape[-2]))
    scale = _uniform(generator, config.scale_min, config.scale_max)
    contrast = _uniform(generator, config.contrast_min, config.contrast_max)
    gamma = _uniform(generator, config.gamma_min, config.gamma_max)
    noise_std = _uniform(generator, 0.0, config.noise_std_max)

    images = TVF.affine(
        images,
        angle=angle,
        translate=[translate_x, translate_y],
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    pixel_valid = TVF.affine(
        pixel_valid.float(),
        angle=angle,
        translate=[translate_x, translate_y],
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    ) > 0.5
    expanded_valid = pixel_valid.expand_as(images)
    if expanded_valid.any():
        mean = images[expanded_valid].mean()
        images = ((images - mean) * contrast + mean).clamp(0.0, 1.0)
        images = images.clamp_min(0.0).pow(gamma)
        if noise_std > 0:
            noise = torch.randn(images.shape, generator=generator, dtype=images.dtype) * noise_std
            images = (images + noise).clamp(0.0, 1.0)
    images = torch.where(expanded_valid, images, torch.zeros_like(images))
    return images, pixel_valid, {
        "angle_degrees": angle,
        "translate_x_pixels": float(translate_x),
        "translate_y_pixels": float(translate_y),
        "scale": scale,
        "contrast": contrast,
        "gamma": gamma,
        "noise_std": noise_std,
        "horizontal_flip": 0.0,
        "vertical_flip": 0.0,
    }


def verify_training_table_artifact(table_path: Path, lock_path: Path, repo_root: Path) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    resolved = table_path.resolve()
    matches = [
        item
        for item in lock.get("artifacts") or []
        if (repo_root / item.get("path", "")).resolve() == resolved
    ]
    if len(matches) != 1:
        raise RuntimeError(f"training table is not covered by release lock: {table_path}")
    item = matches[0]
    if table_path.stat().st_size != item["bytes"] or file_sha256(table_path) != item["sha256"]:
        raise RuntimeError(f"training table fingerprint mismatch: {table_path}")


def _validate_training_row(
    row: dict[str, str], *, path: Path, line_number: int
) -> None:
    """Reject malformed or incomplete supervision before the first epoch starts."""

    location = f"{path}:{line_number}"
    uid = row["Sample ID"]
    if not uid or not uid.strip():
        raise RuntimeError(f"empty Sample ID at {location}")
    try:
        fold = int(row["fold"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid fold for {uid} at {location}: {row['fold']!r}") from exc
    if str(fold) != row["fold"].strip() or fold not in {-1, 0, 1, 2, 3, 4}:
        raise RuntimeError(f"invalid fold for {uid} at {location}: {row['fold']!r}")
    for label in LABELS:
        try:
            value = float(row[label])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid target for {uid}/{label} at {location}: {row[label]!r}"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"target outside [0,1] for {uid}/{label} at {location}: {row[label]!r}"
            )
        if fold >= 0 and value not in {0.0, 1.0}:
            raise RuntimeError(
                f"non-binary hard target for {uid}/{label} at {location}: {row[label]!r}"
            )
        for source, text_column, mask_column in (
            ("ClinFusion", clinfusion_text_column(label), clinfusion_mask_column(label)),
            ("Hulu", hulu_text_column(label), hulu_mask_column(label)),
        ):
            if not row[text_column] or not row[text_column].strip():
                raise RuntimeError(
                    f"empty {source} Dense text for {uid}/{label} at {location}"
                )
            if row[mask_column].strip() not in {"0", "1"}:
                raise RuntimeError(
                    f"invalid {source} embedding mask for {uid}/{label} at {location}: "
                    f"{row[mask_column]!r}"
                )
    if not row["Report"] or not row["Report"].strip():
        raise RuntimeError(f"empty Report for {uid} at {location}")


def _read_training_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    required = {
        "Sample ID",
        "Report",
        "fold",
        *LABELS,
        *(clinfusion_text_column(label) for label in LABELS),
        *(hulu_text_column(label) for label in LABELS),
        *(clinfusion_mask_column(label) for label in LABELS),
        *(hulu_mask_column(label) for label in LABELS),
    }
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(set(reader.fieldnames or [])):
                raise RuntimeError(f"training CSV schema mismatch: {path}")
            for line_number, row in enumerate(reader, 2):
                _validate_training_row(row, path=path, line_number=line_number)
                uid = row["Sample ID"]
                if uid in seen:
                    raise RuntimeError(f"duplicate Sample ID: {uid!r}")
                seen.add(uid)
                rows.append(row)
    return rows


def _selection_map(path: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            uid = str(value.get("study_uid") or "")
            if uid not in wanted_ids:
                continue
            selected = value.get("selected_series")
            candidates = value.get("series_candidates") or []
            if (
                uid in result
                or not isinstance(selected, dict)
                or set(selected) != set(ROLES)
                or not isinstance(candidates, list)
            ):
                raise RuntimeError(f"invalid selected-series record at {path}:{number}")
            # Keep only image linkage. Legacy labels, folds, report text and weak
            # supervision remain intentionally unreachable from model processing.
            result[uid] = {
                "study_uid": uid,
                "selected_series": selected,
                "series_candidates": candidates,
            }
    missing = wanted_ids - set(result)
    if missing:
        raise RuntimeError(f"selection manifest is missing {len(missing)} requested studies")
    return result


class KneeStudyDataset(Dataset[dict[str, Any]]):
    """Build a lazy study object, then derive the same canonical view for Train/Test."""

    def __init__(
        self,
        *,
        config: KneeInputConfig,
        cache_root: str | Path,
        selection_manifest: str | Path,
        stage_cache_root: str | Path | None = None,
        training_tables: Sequence[str | Path] | None = None,
        training_tables_lock: str | Path | None = None,
        selection_manifest_sha256: str | None = None,
        selection_manifest_bytes: int | None = None,
        repo_root: str | Path | None = None,
        augment: bool = False,
        return_stages: bool = False,
        verify_table_lock: bool = True,
        lru_cache_entries: int = 32,
        include_folds: Sequence[int] | None = None,
    ) -> None:
        self.config = config
        self.cache_root = Path(cache_root)
        self.stage_cache_root = Path(stage_cache_root) if stage_cache_root is not None else None
        self.selection_manifest = Path(selection_manifest)
        self.training_tables = tuple(Path(path) for path in training_tables or ())
        self.inference_only = not self.training_tables
        self.augment = bool(augment)
        self.return_stages = bool(return_stages)
        self._shared_epoch = multiprocessing.Value("q", 0)
        if lru_cache_entries < 0:
            raise ValueError("lru_cache_entries must be non-negative")
        self.lru_cache_entries = int(lru_cache_entries)
        self._metadata_cache: OrderedDict[Path, tuple[dict[str, Any], str]] = OrderedDict()
        self._volume_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
        if self.augment and not self.config.augmentation.enabled:
            raise ValueError("augment=True but augmentation is disabled in config")
        expected_selection_sha256 = selection_manifest_sha256 or self.config.selection_manifest_sha256
        expected_selection_bytes = selection_manifest_bytes or self.config.selection_manifest_bytes
        if (
            self.selection_manifest.stat().st_size != expected_selection_bytes
            or file_sha256(self.selection_manifest) != expected_selection_sha256
        ):
            raise RuntimeError(f"selection manifest fingerprint mismatch: {self.selection_manifest}")
        if verify_table_lock and not self.inference_only:
            if training_tables_lock is None:
                raise ValueError("training_tables_lock is required when verification is enabled")
            for path in self.training_tables:
                verify_training_table_artifact(path, Path(training_tables_lock), self.repo_root)
        if self.inference_only:
            if include_folds is not None:
                raise ValueError("include_folds is training-only")
            inference_ids: list[str] = []
            with self.selection_manifest.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        inference_ids.append(str(json.loads(line).get("study_uid") or ""))
            if not inference_ids or len(set(inference_ids)) != len(inference_ids) or "" in inference_ids:
                raise RuntimeError("inference selection manifest has empty or duplicate study IDs")
            self.rows = [{"Sample ID": uid} for uid in inference_ids]
        else:
            self.rows = _read_training_rows(self.training_tables)
            if include_folds is not None:
                wanted_folds = {int(value) for value in include_folds}
                if not wanted_folds.issubset({-1, 0, 1, 2, 3, 4}):
                    raise ValueError("include_folds must contain only -1 or 0..4")
                self.rows = [row for row in self.rows if int(row["fold"]) in wanted_folds]
                if not self.rows:
                    raise RuntimeError("fold filter produced an empty dataset")
        wanted_ids = {row["Sample ID"] for row in self.rows}
        self.selections = _selection_map(self.selection_manifest, wanted_ids)
        self.default_view = default_study_view(self.config)
        self.best_view = best_study_view(self.config)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # DataLoader worker processes open their own mmap handles safely.
        state["_metadata_cache"] = OrderedDict()
        state["_volume_cache"] = OrderedDict()
        return state

    def _remember(self, cache: OrderedDict, key: Path, value: Any) -> Any:
        if self.lru_cache_entries == 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.lru_cache_entries:
            cache.popitem(last=False)
        return value

    def _read_metadata(self, path: Path) -> tuple[dict[str, Any], str]:
        cached = self._metadata_cache.get(path)
        if cached is not None:
            self._metadata_cache.move_to_end(path)
            return cached
        raw = path.read_bytes()
        value = (json.loads(raw), hashlib.sha256(raw).hexdigest())
        return self._remember(self._metadata_cache, path, value)

    def _mmap_volume(self, path: Path) -> np.ndarray:
        cached = self._volume_cache.get(path)
        if cached is not None:
            self._volume_cache.move_to_end(path)
            return cached
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._remember(self._volume_cache, path, value)

    def set_epoch(self, epoch: int) -> None:
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._shared_epoch.value)

    def __len__(self) -> int:
        return len(self.rows)

    def _generator(self, sample_id: str) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(self.config.seed, self.epoch, sample_id))
        return generator

    def _series_cache_available(self, series: dict[str, Any]) -> bool:
        volume_path = series.get("volume_path")
        metadata_path = series.get("metadata_path")
        return bool(
            volume_path
            and metadata_path
            and series.get("cache_available", True)
            and (self.cache_root / str(volume_path)).is_file()
            and (self.cache_root / str(metadata_path)).is_file()
        )

    @staticmethod
    def _is_localizer(series: dict[str, Any]) -> bool:
        description = str(series.get("series_description") or "").lower()
        tokens = ("localizer", "localiser", "scout", "survey", "3-plane", "3 plane")
        return any(token in description for token in tokens)

    def _select_complementary_series(
        self,
        selected: dict[str, dict[str, Any] | None],
        candidates: Sequence[dict[str, Any]],
    ) -> tuple[SeriesInput, ...]:
        """Choose conservative optional evidence without changing the five primary roles."""

        primary_uids = {
            str(series.get("series_uid"))
            for series in selected.values()
            if series is not None
        }
        eligible = [
            series
            for series in candidates
            if str(series.get("series_uid")) not in primary_uids
            and self._series_cache_available(series)
            and not self._is_localizer(series)
            and bool(series.get("orientation_quality", False))
            and float(series.get("decode_quality") or 0.0) >= 0.99
        ]
        chosen: list[SeriesInput] = []
        used: set[str] = set()

        axial_structural = [
            series
            for series in eligible
            if str(series.get("plane") or "").lower().startswith("ax")
            and not bool(series.get("fluid_sensitive"))
        ]
        if axial_structural:
            series = max(
                axial_structural,
                key=lambda value: (
                    int(value.get("slice_count") or 0),
                    -sum(float(x) for x in (value.get("pixel_spacing") or [99.0, 99.0])),
                    str(value.get("series_uid") or ""),
                ),
            )
            uid = str(series.get("series_uid"))
            used.add(uid)
            chosen.append(SeriesInput("AX_STRUCT", "AX_STRUCT", "axial_structural", series))

        for role in ROLES:
            primary = selected.get(role)
            if primary is None:
                continue
            primary_family = str(primary.get("sequence_family") or "unknown").lower()
            alternatives = []
            for series in eligible:
                uid = str(series.get("series_uid"))
                role_score = (series.get("role_scores") or {}).get(role)
                family = str(series.get("sequence_family") or "unknown").lower()
                if uid in used or role_score is None or family == primary_family:
                    continue
                alternatives.append((float(role_score), series))
            if not alternatives:
                continue
            _, series = max(
                alternatives,
                key=lambda item: (item[0], int(item[1].get("slice_count") or 0), str(item[1].get("series_uid") or "")),
            )
            uid = str(series.get("series_uid"))
            used.add(uid)
            chosen.append(
                SeriesInput(role, f"{role}_COMPLEMENT", "distinct_sequence_family", series)
            )
        return tuple(chosen)

    def build_study_input(self, index: int) -> StudyInput:
        """Resolve image references and geometry without materializing pixel arrays."""

        row = self.rows[index]
        uid = row["Sample ID"]
        selection = self.selections[uid]
        selected = selection["selected_series"]
        primary = tuple(
            None
            if selected[role] is None
            else SeriesInput(role, role, "primary", selected[role])
            for role in self.config.roles
        )
        selected_metadata = [
            self._read_metadata(self.cache_root / item.record["metadata_path"])[0]
            for item in primary
            if item is not None
        ]
        laterality = determine_study_laterality(
            selected_metadata,
            coordinate_threshold_mm=float(
                self.config.orientation.get("coordinate_threshold_mm", 30.0)
            ),
        )
        complementary = self._select_complementary_series(
            selected,
            selection["series_candidates"],
        )
        return StudyInput(
            sample_id=uid,
            primary_series=primary,
            complementary_series=complementary,
            laterality=laterality,
            candidate_series_count=len(selection["series_candidates"]),
        )

    def _load_role(
        self,
        role: str,
        series: dict[str, Any],
        generator: torch.Generator,
        laterality: str | None,
        view: StudyViewConfig,
    ) -> dict[str, Any]:
        volume_path = self.cache_root / series["volume_path"]
        metadata_path = self.cache_root / series["metadata_path"]
        metadata, source_metadata_sha = self._read_metadata(metadata_path)
        if metadata.get("config_fingerprint") != self.config.cache_config_fingerprint:
            raise RuntimeError(f"cache fingerprint mismatch: {metadata_path}")
        source_shape_value = metadata.get("shape")
        if (
            not isinstance(source_shape_value, list)
            or len(source_shape_value) != 3
            or any(not isinstance(value, int) or value <= 0 for value in source_shape_value)
            or metadata.get("dtype") != "uint16"
        ):
            raise RuntimeError(f"invalid cached volume metadata: {metadata_path}")
        source_shape = tuple(source_shape_value)
        _, _, depth_direction = _orientation_targets(role, laterality)
        positions, position_source, nominal_spacing, source_order = _source_positions(
            metadata,
            source_shape[0],
            depth_direction,
        )
        if view.depth_mode == "stratified_fixed":
            assert view.slices_per_series is not None
            sampled = stratified_slice_indices(
                positions,
                view.slices_per_series,
                training=self.augment,
                generator=generator,
            )
        else:
            sampled = all_valid_slice_indices(positions)
        volume_indices = source_order[sampled["indices"]]
        pixel_spacing = metadata.get("pixel_spacing")
        if not isinstance(pixel_spacing, list) or len(pixel_spacing) != 2:
            raise RuntimeError(f"missing pixel spacing: {metadata_path}")
        source_images: torch.Tensor | None = None
        oriented_images: torch.Tensor | None = None
        stage_cache_status = "disabled"
        stage_cache_directory: Path | None = None
        stage_hit = False
        chunked = False
        stage_eligible = view.is_stage_cacheable_for(self.config)
        if not stage_eligible:
            stage_cache_status = "disabled_for_nondefault_view"
        if (
            stage_eligible
            and self.stage_cache_root is not None
            and self.config.deterministic_stage_cache.get("enabled")
        ):
            stage_cache_directory = (
                self.stage_cache_root
                / view.version
                / Path(series["volume_path"]).parent
            )
            stage_metadata_path = stage_cache_directory / "metadata.json"
            stage_volume_path = stage_cache_directory / "volume.npy"
            stage_valid_path = stage_cache_directory / "pixel_valid.npy"
            expected = deterministic_stage_fingerprint(
                self.config,
                source_metadata_sha,
                role,
                laterality,
                view,
            )
            if stage_metadata_path.is_file() and stage_volume_path.is_file() and stage_valid_path.is_file():
                try:
                    stage_metadata, _ = self._read_metadata(stage_metadata_path)
                    stage_volume = self._mmap_volume(stage_volume_path)
                    stage_valid = self._mmap_volume(stage_valid_path)
                    stage_hit = bool(
                        stage_metadata.get("fingerprint") == expected
                        and stage_volume.dtype == np.uint16
                        and tuple(stage_volume.shape)
                        == (source_shape[0], view.image_size, view.image_size)
                        and tuple(stage_valid.shape) == (1, view.image_size, view.image_size)
                    )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    stage_hit = False
            if stage_hit and not self.return_stages:
                images = torch.from_numpy(
                    np.asarray(stage_volume[sampled["indices"]], dtype=np.float32).copy()
                ).div_(65535.0)
                pixel_valid = torch.from_numpy(np.asarray(stage_valid, dtype=bool).copy())
                oriented_spacing = [float(value) for value in stage_metadata["oriented_pixel_spacing"]]
                orientation_trace = dict(stage_metadata["orientation"])
                crop_trace = dict(stage_metadata["crop"])
                stage_cache_status = "hit"
            elif stage_hit:
                stage_cache_status = "bypassed_for_visual_review"
            else:
                stage_cache_status = "miss_online_equivalent"
        if not stage_hit or self.return_stages:
            volume = self._mmap_volume(volume_path)
            if (
                volume.dtype != np.uint16
                or volume.ndim != 3
                or tuple(volume.shape) != source_shape
            ):
                raise RuntimeError(
                    f"invalid cached volume: {volume_path} {volume.shape} {volume.dtype}"
                )

            def process_source(
                source: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, list[float], dict[str, Any], dict[str, Any], torch.Tensor]:
                oriented, spacing, orientation_value = canonicalize_inplane_orientation(
                    source,
                    pixel_spacing,
                    metadata,
                    role,
                    laterality,
                )
                if view.fov_mode == "fixed_physical_canvas":
                    assert view.canvas_mm is not None
                    output, valid, crop_value = physical_center_crop_resize(
                        oriented,
                        spacing,
                        view.canvas_mm,
                        view.image_size,
                    )
                    crop_value["fov_mode"] = "fixed_physical_canvas"
                    source_area = float(np.prod(crop_value["source_fov_mm"]))
                    retained_area = min(
                        float(crop_value["source_fov_mm"][0]), view.canvas_mm
                    ) * min(float(crop_value["source_fov_mm"][1]), view.canvas_mm)
                    crop_value["source_coverage_fraction"] = (
                        retained_area / source_area if source_area > 0 else 0.0
                    )
                else:
                    output, valid, crop_value = full_source_fov_resize(
                        oriented,
                        spacing,
                        view.image_size,
                    )
                return output, valid, spacing, orientation_value, crop_value, oriented

            chunked = bool(
                view.depth_mode == "all_valid"
                and not self.return_stages
                and len(volume_indices) > 16
            )
            if chunked:
                outputs = []
                for start in range(0, len(volume_indices), 16):
                    chunk = torch.from_numpy(
                        np.asarray(
                            volume[volume_indices[start:start + 16]],
                            dtype=np.float32,
                        ).copy()
                    ).div_(65535.0)
                    output, pixel_valid, oriented_spacing, orientation_trace, crop_trace, _ = (
                        process_source(chunk)
                    )
                    outputs.append(output)
                images = torch.cat(outputs, dim=0)
                crop_trace["source_shape"][0] = int(len(volume_indices))
                crop_trace["output_shape"][0] = int(len(volume_indices))
            else:
                source_images = torch.from_numpy(
                    np.asarray(volume[volume_indices], dtype=np.float32).copy()
                ).div_(65535.0)
                (
                    images,
                    pixel_valid,
                    oriented_spacing,
                    orientation_trace,
                    crop_trace,
                    oriented_images,
                ) = process_source(source_images)
            images = quantize_deterministic_stage(images)
        if "fov_mode" not in crop_trace:
            crop_trace["fov_mode"] = "fixed_physical_canvas"
            source_area = float(np.prod(crop_trace["source_fov_mm"]))
            retained_area = min(
                float(crop_trace["source_fov_mm"][0]),
                float(crop_trace["requested_canvas_mm"]),
            ) * min(
                float(crop_trace["source_fov_mm"][1]),
                float(crop_trace["requested_canvas_mm"]),
            )
            crop_trace["source_coverage_fraction"] = (
                retained_area / source_area if source_area > 0 else 0.0
            )
        pre_augmentation = images.clone() if self.return_stages else None
        pre_pixel_valid = pixel_valid.clone() if self.return_stages else None
        augmentation_trace = {
            "angle_degrees": 0.0,
            "translate_x_pixels": 0.0,
            "translate_y_pixels": 0.0,
            "scale": 1.0,
            "contrast": 1.0,
            "gamma": 1.0,
            "noise_std": 0.0,
            "horizontal_flip": 0.0,
            "vertical_flip": 0.0,
        }
        if self.augment:
            images, pixel_valid, augmentation_trace = _augment_series(
                images,
                pixel_valid,
                self.config.augmentation,
                generator,
            )

        output_depth = int(images.shape[0])
        slice_source_mask = torch.ones(output_depth, dtype=torch.bool)
        slice_dropout_mask = torch.zeros_like(slice_source_mask)
        if self.augment:
            probability = self.config.augmentation.slice_dropout_probability
            if probability > 0:
                slice_dropout_mask |= torch.rand(
                    output_depth,
                    generator=generator,
                ) < probability
            if (
                self.config.augmentation.block_dropout_max_slices > 0
                and _bernoulli(generator, self.config.augmentation.block_dropout_probability)
            ):
                length = int(
                    torch.randint(
                        1,
                        self.config.augmentation.block_dropout_max_slices + 1,
                        (),
                        generator=generator,
                    ).item()
                )
                start = int(
                    torch.randint(
                        0,
                        output_depth - min(length, output_depth) + 1,
                        (),
                        generator=generator,
                    ).item()
                )
                slice_dropout_mask[start:start + min(length, output_depth)] = True
            if slice_dropout_mask.all():
                slice_dropout_mask[output_depth // 2] = False
        slice_valid_mask = slice_source_mask & ~slice_dropout_mask
        images[slice_dropout_mask] = 0
        role_dropout = self.augment and _bernoulli(
            generator,
            self.config.augmentation.role_dropout_probability,
        )
        if role_dropout:
            images.zero_()
            pixel_valid.zero_()
            slice_valid_mask.zero_()
            slice_dropout_mask.fill_(True)

        sequence_family = str(series.get("sequence_family") or "unknown").lower()
        audit = {
            "role": role,
            "view_version": view.version,
            "view_depth_mode": view.depth_mode,
            "view_fov_mode": view.fov_mode,
            "spatial_processing_chunk_depth": 16 if chunked else None,
            "series_uid": series.get("series_uid"),
            "volume_path": str(series["volume_path"]),
            "metadata_path": str(series["metadata_path"]),
            "series_description": series.get("series_description"),
            "plane": series.get("plane"),
            "fluid_sensitive": bool(series.get("fluid_sensitive")),
            "fat_suppression": bool(series.get("fat_suppression")),
            "sequence_family": sequence_family,
            "source_slice_count": int(source_shape[0]),
            "source_dtype": str(metadata["dtype"]),
            "position_source": position_source,
            "position_quality": position_source in {"dicom_spatial_positions", "canonical_patient_positions"},
            "nominal_slice_spacing_mm": nominal_spacing,
            "sampled_indices": volume_indices.tolist(),
            "canonical_depth_order_indices": sampled["indices"].tolist(),
            "sampled_positions_mm": sampled["positions_mm"].tolist(),
            "stratum_centers_mm": sampled["stratum_centers_mm"].tolist(),
            "distance_to_center_mm": sampled["distance_to_center_mm"].tolist(),
            "repeated_slices": int(sampled["repeated"].sum()),
            "max_source_multiplicity": int(sampled["source_multiplicity"].max()),
            "max_sampled_gap_mm": float(max(sampled["gap_previous_mm"].max(), sampled["gap_next_mm"].max())),
            "orientation_quality": bool(orientation_trace["quality"]),
            "orientation": orientation_trace,
            "deterministic_stage_cache": {
                "status": stage_cache_status,
                "directory": str(stage_cache_directory) if stage_cache_directory else None,
            },
            "crop": crop_trace,
            "augmentation": augmentation_trace,
            "slice_dropout_indices": torch.nonzero(
                slice_dropout_mask,
                as_tuple=False,
            ).flatten().tolist(),
            "role_dropout": bool(role_dropout),
            "output_min": float(images.min()),
            "output_max": float(images.max()),
            "output_mean": float(images.mean()),
        }
        return {
            "images": images,
            "pixel_valid_mask": pixel_valid,
            "slice_source_mask": slice_source_mask,
            "slice_valid_mask": slice_valid_mask,
            "slice_dropout_mask": slice_dropout_mask,
            "slice_repeated_mask": torch.from_numpy(sampled["repeated"]),
            "slice_source_multiplicity": torch.from_numpy(sampled["source_multiplicity"]),
            "positions_mm": torch.from_numpy(sampled["positions_mm"]),
            "positions_normalized": torch.from_numpy(sampled["positions_normalized"]),
            "gap_previous_mm": torch.from_numpy(sampled["gap_previous_mm"]),
            "gap_next_mm": torch.from_numpy(sampled["gap_next_mm"]),
            "pixel_spacing_mm": torch.tensor(oriented_spacing, dtype=torch.float32),
            "source_fov_mm": torch.tensor(crop_trace["source_fov_mm"], dtype=torch.float32),
            "slice_spacing_mm": torch.tensor(nominal_spacing, dtype=torch.float32),
            "crop_padding_fraction": torch.tensor(crop_trace["padding_fraction"], dtype=torch.float32),
            "crop_quality": torch.tensor(False),
            "selection_quality": torch.tensor(bool(series.get("orientation_quality"))),
            "decode_quality": torch.tensor(int(metadata.get("skipped_slice_count") or 0) == 0),
            "fluid_sensitive": torch.tensor(bool(series.get("fluid_sensitive"))),
            "fat_suppression": torch.tensor(bool(series.get("fat_suppression"))),
            "orientation_quality": torch.tensor(bool(orientation_trace["quality"])),
            "position_quality": torch.tensor(position_source in {"dicom_spatial_positions", "canonical_patient_positions"}),
            "sequence_family_id": torch.tensor(SEQUENCE_FAMILY_IDS.get(sequence_family, 0)),
            "series_dropout": torch.tensor(role_dropout),
            "audit": audit,
            "source_images": source_images if self.return_stages else None,
            "oriented_images": oriented_images if self.return_stages else None,
            "pre_augmentation_images": pre_augmentation,
            "pre_augmentation_pixel_valid_mask": pre_pixel_valid,
        }

    def _assemble_fixed_view(
        self,
        study_input: StudyInput,
        role_results: Sequence[dict[str, Any] | None],
        view: StudyViewConfig,
    ) -> dict[str, Any]:
        assert view.slices_per_series is not None
        role_count = len(self.config.roles)
        depth = view.slices_per_series
        size = view.image_size
        images = torch.zeros((role_count, depth, size, size), dtype=torch.float32)
        pixel_valid_mask = torch.zeros((role_count, 1, size, size), dtype=torch.bool)
        series_source_mask = torch.zeros(role_count, dtype=torch.bool)
        series_mask = torch.zeros(role_count, dtype=torch.bool)
        series_dropout_mask = torch.zeros(role_count, dtype=torch.bool)
        depth_fields = {
            "slice_source_mask": torch.zeros((role_count, depth), dtype=torch.bool),
            "slice_valid_mask": torch.zeros((role_count, depth), dtype=torch.bool),
            "slice_dropout_mask": torch.zeros((role_count, depth), dtype=torch.bool),
            "slice_repeated_mask": torch.zeros((role_count, depth), dtype=torch.bool),
            "slice_source_multiplicity": torch.zeros((role_count, depth), dtype=torch.int16),
            "positions_mm": torch.zeros((role_count, depth), dtype=torch.float32),
            "positions_normalized": torch.zeros((role_count, depth), dtype=torch.float32),
            "gap_previous_mm": torch.zeros((role_count, depth), dtype=torch.float32),
            "gap_next_mm": torch.zeros((role_count, depth), dtype=torch.float32),
        }
        role_fields = {
            "pixel_spacing_mm": torch.zeros((role_count, 2), dtype=torch.float32),
            "source_fov_mm": torch.zeros((role_count, 2), dtype=torch.float32),
            "slice_spacing_mm": torch.zeros(role_count, dtype=torch.float32),
            "crop_padding_fraction": torch.zeros(role_count, dtype=torch.float32),
            "crop_quality": torch.zeros(role_count, dtype=torch.bool),
            "selection_quality": torch.zeros(role_count, dtype=torch.bool),
            "decode_quality": torch.zeros(role_count, dtype=torch.bool),
            "fluid_sensitive": torch.zeros(role_count, dtype=torch.bool),
            "fat_suppression": torch.zeros(role_count, dtype=torch.bool),
            "orientation_quality": torch.zeros(role_count, dtype=torch.bool),
            "position_quality": torch.zeros(role_count, dtype=torch.bool),
            "sequence_family_id": torch.zeros(role_count, dtype=torch.long),
        }
        audit_roles: list[dict[str, Any]] = []
        review_source: list[torch.Tensor | None] = []
        review_oriented: list[torch.Tensor | None] = []
        review_pre: list[torch.Tensor | None] = []
        for role_index, (role, result) in enumerate(
            zip(self.config.roles, role_results, strict=True)
        ):
            if result is None:
                audit_roles.append({"role": role, "available": False, "reason": "role_not_selected"})
                review_source.append(None)
                review_oriented.append(None)
                review_pre.append(None)
                continue
            images[role_index] = result["images"]
            pixel_valid_mask[role_index] = result["pixel_valid_mask"]
            series_source_mask[role_index] = True
            series_dropout_mask[role_index] = result["series_dropout"]
            series_mask[role_index] = not bool(result["series_dropout"])
            for name, target in depth_fields.items():
                target[role_index] = result[name]
            for name, target in role_fields.items():
                target[role_index] = result[name]
            audit_roles.append({"available": True, **result["audit"]})
            review_source.append(result["source_images"])
            review_oriented.append(result["oriented_images"])
            review_pre.append(result["pre_augmentation_images"])

        model_input = {
            "images": images,
            "series_mask": series_mask,
            "series_source_mask": series_source_mask,
            "series_dropout_mask": series_dropout_mask,
            "slice_valid_mask": depth_fields["slice_valid_mask"],
            "slice_source_mask": depth_fields["slice_source_mask"],
            "slice_dropout_mask": depth_fields["slice_dropout_mask"],
            "slice_repeated_mask": depth_fields["slice_repeated_mask"],
            "slice_source_multiplicity": depth_fields["slice_source_multiplicity"],
            "pixel_valid_mask": pixel_valid_mask,
            "slice_positions_mm": depth_fields["positions_mm"],
            "slice_positions_normalized": depth_fields["positions_normalized"],
            "gap_previous_mm": depth_fields["gap_previous_mm"],
            "gap_next_mm": depth_fields["gap_next_mm"],
            "pixel_spacing_mm": role_fields["pixel_spacing_mm"],
            "source_fov_mm": role_fields["source_fov_mm"],
            "slice_spacing_mm": role_fields["slice_spacing_mm"],
            "crop_padding_fraction": role_fields["crop_padding_fraction"],
            "crop_quality": role_fields["crop_quality"],
            "selection_quality": role_fields["selection_quality"],
            "decode_quality": role_fields["decode_quality"],
            "role_ids": torch.arange(role_count, dtype=torch.long),
            "plane_ids": torch.tensor(
                [PLANE_IDS[role.split("_")[0]] for role in self.config.roles]
            ),
            "laterality_id": torch.tensor(
                LATERALITY_IDS[study_input.laterality["laterality"]], dtype=torch.long
            ),
            "laterality_quality": torch.tensor(bool(study_input.laterality["quality"])),
            "sequence_family_ids": role_fields["sequence_family_id"],
            "fluid_sensitive": role_fields["fluid_sensitive"],
            "fat_suppression": role_fields["fat_suppression"],
            "orientation_quality": role_fields["orientation_quality"],
            "position_quality": role_fields["position_quality"],
        }
        return {
            "view_version": view.version,
            "fixed_shape": True,
            "model_input": model_input,
            "provenance": {
                "input_pipeline_version": self.config.version,
                "input_config_fingerprint": self.config.fingerprint,
                "view_version": view.version,
                "augmentation_enabled": self.augment,
                "laterality": study_input.laterality,
                "study_input": {
                    "candidate_series_count": study_input.candidate_series_count,
                    "complementary_series_count": len(study_input.complementary_series),
                    "pixels_materialized_during_build": False,
                },
                "roles": audit_roles,
            },
            "review_stages": {
                "source_selected_images": review_source,
                "oriented_selected_images": review_oriented,
                "pre_augmentation_images": review_pre,
            },
        }

    def derive_model_view(
        self,
        study_input: StudyInput,
        view: StudyViewConfig | None = None,
    ) -> dict[str, Any]:
        """Materialize one model view through the canonical Train/Test processor."""

        view = view or self.default_view
        view.validate()
        generator = self._generator(study_input.sample_id)
        primary_results = tuple(
            None
            if item is None
            else self._load_role(
                item.role,
                item.record,
                generator,
                study_input.laterality["laterality"],
                view,
            )
            for item in study_input.primary_series
        )
        if view.depth_mode == "stratified_fixed":
            if view.include_complementary_series:
                raise ValueError("fixed primary view cannot silently add complementary series")
            return self._assemble_fixed_view(study_input, primary_results, view)

        complements = []
        if view.include_complementary_series:
            for item in study_input.complementary_series:
                complements.append(
                    {
                        "slot": item.slot,
                        "role": item.role,
                        "kind": item.kind,
                        "series_uid": item.record.get("series_uid"),
                        "series": self._load_role(
                            item.role,
                            item.record,
                            generator,
                            study_input.laterality["laterality"],
                            view,
                        ),
                    }
                )
        return {
            "view_version": view.version,
            "fixed_shape": False,
            "image_size": view.image_size,
            "depth_contract": "one entry per real source slice; no repeat, interpolation, truncation, or depth padding",
            "fov_contract": "complete physical source FOV fit into 320x320; short axis padded",
            "primary_series": {
                role: result
                for role, result in zip(self.config.roles, primary_results, strict=True)
            },
            "complementary_series": complements,
            "laterality": study_input.laterality,
            "provenance": {
                "input_pipeline_version": self.config.version,
                "input_config_fingerprint": self.config.fingerprint,
                "view_version": view.version,
                "augmentation_enabled": self.augment,
                "study_input": {
                    "candidate_series_count": study_input.candidate_series_count,
                    "complementary_series_count": len(study_input.complementary_series),
                    "pixels_materialized_during_build": False,
                },
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        uid = row["Sample ID"]
        study_input = self.build_study_input(index)
        derived = self.derive_model_view(study_input, self.default_view)
        model_input = derived["model_input"]
        provenance = derived["provenance"]
        review_stages = derived["review_stages"]
        if self.inference_only:
            output = {
                "sample_id": uid,
                "model_input": model_input,
                "provenance": provenance,
            }
            if self.return_stages:
                output["review_stages"] = review_stages
            return output

        fold = int(row["fold"])
        is_hard = fold >= 0
        labels = torch.tensor([float(row[label]) for label in LABELS], dtype=torch.float32)
        if is_hard and not torch.all((labels == 0) | (labels == 1)):
            raise RuntimeError(f"non-binary hard target for {uid}")
        if getattr(self, 'numeric_targets', False):
            return {
                'sample_id': uid, 'fold': fold, 'model_input': model_input,
                'targets': {'labels': labels, 'label_mask': torch.ones(len(LABELS), dtype=torch.bool),
                            'is_hard': torch.tensor(is_hard)},
                'provenance': provenance,
                **({'review_stages': review_stages} if self.return_stages else {}),
            }
        targets = {
            "labels": labels,
            "label_mask": torch.ones(len(LABELS), dtype=torch.bool),
            "is_hard": torch.tensor(is_hard),
            "clinfusion_text": [row[clinfusion_text_column(label)] for label in LABELS],
            "clinfusion_embedding_eligible": torch.tensor(
                [int(row[clinfusion_mask_column(label)]) for label in LABELS],
                dtype=torch.bool,
            ),
            "hulu_text": [row[hulu_text_column(label)] for label in LABELS],
            "hulu_embedding_eligible": torch.tensor(
                [int(row[hulu_mask_column(label)]) for label in LABELS],
                dtype=torch.bool,
            ),
        }
        output = {
            "sample_id": uid,
            "fold": fold,
            "model_input": model_input,
            "targets": targets,
            "provenance": {
                **provenance,
                "report": row["Report"],
                "label_kind": "hard" if is_hard else "soft",
            },
        }
        if self.return_stages:
            output["review_stages"] = review_stages
        return output


def collate_knee_studies(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    model_keys = samples[0]["model_input"].keys()
    model_input = {
        key: torch.stack([sample["model_input"][key] for sample in samples])
        for key in model_keys
    }
    output = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "model_input": model_input,
        "provenance": [sample["provenance"] for sample in samples],
    }
    if "targets" not in samples[0]:
        if any("targets" in sample or "fold" in sample for sample in samples):
            raise ValueError("cannot mix training and inference samples in one batch")
        return output
    if any("targets" not in sample or "fold" not in sample for sample in samples):
        raise ValueError("cannot mix training and inference samples in one batch")
    tensor_target_keys = (
        "labels",
        "label_mask",
        "is_hard",
        "clinfusion_embedding_eligible",
        "hulu_embedding_eligible",
    )
    targets = {
        key: torch.stack([sample["targets"][key] for sample in samples])
        for key in tensor_target_keys
    }
    targets["clinfusion_text"] = [sample["targets"]["clinfusion_text"] for sample in samples]
    targets["hulu_text"] = [sample["targets"]["hulu_text"] for sample in samples]
    output["fold"] = torch.tensor([sample["fold"] for sample in samples], dtype=torch.long)
    output["targets"] = targets
    return output


def resolve_config_paths(config_path: str | Path, repo_root: str | Path) -> dict[str, Path]:
    value = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root = Path(repo_root)
    return {name: root / relative for name, relative in value["paths"].items()}
