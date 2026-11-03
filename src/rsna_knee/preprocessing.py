"""Canonical DICOM preprocessing used by both training cache and inference.

This module deliberately contains no training-only behavior. The public
``preprocess_series`` entry point must also be used on hidden test series.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PIPELINE_VERSION = "dicom-series-v1"
METADATA_SCHEMA_VERSION = "dicom-series-metadata-v2"


@dataclass(frozen=True)
class PreprocessConfig:
    """All pixel-changing choices; its hash identifies cache compatibility."""

    version: str = PIPELINE_VERSION
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    percentiles_on_nonzero: bool = True
    output_dtype: str = "uint16"

    def __post_init__(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
        if self.output_dtype != "uint16":
            raise ValueError("v1 supports only uint16 output")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    dataset: Any
    spatial_position: float | None
    instance_number: int | None


@dataclass
class ProcessedSeries:
    volume: np.ndarray
    metadata: dict[str, Any]


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Iterable):
        return [_json_value(item) for item in value]
    return str(value)


def _spatial_position(dataset: Any) -> float | None:
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    position = getattr(dataset, "ImagePositionPatient", None)
    if orientation is None or position is None:
        return None
    try:
        orientation_array = np.asarray(orientation, dtype=np.float64)
        position_array = np.asarray(position, dtype=np.float64)
        if orientation_array.shape != (6,) or position_array.shape != (3,):
            return None
        normal = np.cross(orientation_array[:3], orientation_array[3:])
        if np.linalg.norm(normal) < 1e-8:
            return None
        return float(np.dot(position_array, normal))
    except (TypeError, ValueError):
        return None


def sort_slice_records(records: Sequence[SliceRecord]) -> tuple[list[SliceRecord], str]:
    """Sort one series spatially, with deterministic fallbacks."""

    if not records:
        raise ValueError("a DICOM series cannot be empty")
    if all(record.spatial_position is not None for record in records):
        return sorted(records, key=lambda r: (r.spatial_position, r.path.name)), "spatial"
    if all(record.instance_number is not None for record in records):
        return sorted(records, key=lambda r: (r.instance_number, r.path.name)), "instance_number"
    return sorted(records, key=lambda r: r.path.name), "filename"


def normalize_to_uint16(
    volume: np.ndarray, config: PreprocessConfig
) -> tuple[np.ndarray, float, float]:
    """Robust series-level normalization while retaining 16-bit precision."""

    values = volume[np.isfinite(volume)]
    if config.percentiles_on_nonzero:
        nonzero = values[values != 0]
        if nonzero.size:
            values = nonzero
    if not values.size:
        return np.zeros(volume.shape, dtype=np.uint16), 0.0, 0.0

    lower, upper = np.percentile(
        values, [config.lower_percentile, config.upper_percentile]
    ).astype(float)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros(volume.shape, dtype=np.uint16), lower, upper

    clean = np.nan_to_num(volume, nan=lower, posinf=upper, neginf=lower)
    scaled = (np.clip(clean, lower, upper) - lower) / (upper - lower)
    output = np.rint(scaled * np.iinfo(np.uint16).max).astype(np.uint16)
    return output, lower, upper


def _read_datasets(
    paths: Sequence[Path], *, stop_before_pixels: bool = False
) -> list[SliceRecord]:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("pydicom is required to read DICOM series") from exc

    records: list[SliceRecord] = []
    for path in paths:
        dataset = pydicom.dcmread(path, stop_before_pixels=stop_before_pixels)
        records.append(
            SliceRecord(
                path=path,
                dataset=dataset,
                spatial_position=_spatial_position(dataset),
                instance_number=_optional_int(getattr(dataset, "InstanceNumber", None)),
            )
        )
    return records


def _geometry_metadata(records: Sequence[SliceRecord]) -> dict[str, Any]:
    """Return slice-aligned geometry used by every downstream input view."""

    if not records:
        raise ValueError("cannot extract geometry from an empty series")
    first = records[0].dataset
    orientations = [
        _json_value(getattr(record.dataset, "ImageOrientationPatient", None))
        for record in records
    ]
    first_orientation = orientations[0]
    orientation_consistent = bool(
        first_orientation is not None
        and all(
            orientation is not None
            and np.allclose(
                np.asarray(orientation, dtype=np.float64),
                np.asarray(first_orientation, dtype=np.float64),
                atol=1e-5,
                rtol=0,
            )
            for orientation in orientations
        )
    )
    spatial_positions = [record.spatial_position for record in records]
    complete_positions = all(value is not None for value in spatial_positions)
    return {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "slice_positions_mm": (
            [float(value) for value in spatial_positions if value is not None]
            if complete_positions
            else None
        ),
        "image_positions_patient": [
            _json_value(getattr(record.dataset, "ImagePositionPatient", None))
            for record in records
        ],
        "instance_numbers": [record.instance_number for record in records],
        "image_orientation_patient": first_orientation,
        "orientation_consistent": orientation_consistent,
        "patient_position": str(getattr(first, "PatientPosition", "")) or None,
        "laterality": str(getattr(first, "Laterality", "")) or None,
        "image_laterality": str(getattr(first, "ImageLaterality", "")) or None,
        "patient_orientation": _json_value(
            getattr(first, "PatientOrientation", None)
        ),
        "rows": _optional_int(getattr(first, "Rows", None)),
        "columns": _optional_int(getattr(first, "Columns", None)),
    }


def _records_in_cached_order(
    records: Sequence[SliceRecord], source_files: Sequence[str]
) -> list[SliceRecord]:
    by_name = {record.path.name: record for record in records}
    if len(by_name) != len(records) or any(name not in by_name for name in source_files):
        raise RuntimeError("cached source-file order cannot be reproduced from DICOM headers")
    return [by_name[name] for name in source_files]


def _decode_pixels(record: SliceRecord) -> np.ndarray:
    dataset = record.dataset
    try:
        pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    except Exception as exc:
        transfer_syntax = getattr(
            getattr(dataset, "file_meta", None), "TransferSyntaxUID", "unknown"
        )
        if bool(getattr(transfer_syntax, "is_compressed", False)):
            hint = "install the matching pydicom pixel decoder or check pixel data integrity"
        else:
            hint = "pixel data may be truncated/corrupt or inconsistent with its DICOM tags"
        raise RuntimeError(
            f"cannot decode {record.path} (TransferSyntaxUID={transfer_syntax}); "
            f"{hint}"
        ) from exc
    if pixels.ndim != 2:
        raise ValueError(f"expected a 2-D MR slice, got {pixels.shape} in {record.path}")
    slope = _optional_float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = _optional_float(getattr(dataset, "RescaleIntercept", 0.0))
    return pixels * (1.0 if slope is None else slope) + (0.0 if intercept is None else intercept)


def _decode_series_records(
    records: Sequence[SliceRecord],
) -> tuple[list[SliceRecord], list[np.ndarray], list[dict[str, str]]]:
    """Decode a series, tolerating only a small, fully audited corrupt minority."""

    decoded_records: list[SliceRecord] = []
    slices: list[np.ndarray] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        try:
            pixels = _decode_pixels(record)
        except RuntimeError as exc:
            skipped.append({"file": record.path.name, "error": str(exc)})
            continue
        decoded_records.append(record)
        slices.append(pixels)

    max_skipped = max(1, int(len(records) * 0.10))
    if skipped and (not slices or len(skipped) > max_skipped):
        raise RuntimeError(
            f"refusing series with {len(skipped)}/{len(records)} undecodable slices; "
            f"maximum allowed is {max_skipped}; first={skipped[0]['error']}"
        )
    return decoded_records, slices, skipped


def preprocess_series(
    dicom_paths: Sequence[str | Path], config: PreprocessConfig | None = None
) -> ProcessedSeries:
    """Decode and canonicalize one complete MRI series.

    Call this exact function both while building caches and during submission.
    """

    config = config or PreprocessConfig()
    paths = [Path(path) for path in dicom_paths]
    if not paths:
        raise ValueError("no DICOM paths supplied")
    source_records, sort_method = sort_slice_records(_read_datasets(paths))
    records, slices, skipped = _decode_series_records(source_records)
    shapes = {array.shape for array in slices}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent slice shapes in one series: {sorted(shapes)}")

    float_volume = np.stack(slices)
    photometric = str(getattr(records[0].dataset, "PhotometricInterpretation", ""))
    if photometric.upper() == "MONOCHROME1":
        finite = float_volume[np.isfinite(float_volume)]
        if finite.size:
            float_volume = float(finite.min() + finite.max()) - float_volume

    volume, lower, upper = normalize_to_uint16(float_volume, config)
    first = records[0].dataset
    metadata = {
        "pipeline_version": config.version,
        "config_fingerprint": config.fingerprint,
        "config": asdict(config),
        "sort_method": sort_method,
        "slice_count": len(records),
        "source_file_count": len(source_records),
        "skipped_slice_count": len(skipped),
        "skipped_source_files": skipped,
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "normalization_lower": lower,
        "normalization_upper": upper,
        "photometric_interpretation": photometric or None,
        "study_instance_uid": str(getattr(first, "StudyInstanceUID", "")) or None,
        "series_instance_uid": str(getattr(first, "SeriesInstanceUID", "")) or None,
        "series_description": str(getattr(first, "SeriesDescription", "")) or None,
        "pixel_spacing": _json_value(getattr(first, "PixelSpacing", None)),
        "slice_thickness": _optional_float(getattr(first, "SliceThickness", None)),
        "spacing_between_slices": _optional_float(
            getattr(first, "SpacingBetweenSlices", None)
        ),
        **_geometry_metadata(records),
        "source_files": [record.path.name for record in records],
    }
    return ProcessedSeries(volume=volume, metadata=metadata)


def save_cached_series(processed: ProcessedSeries, cache_dir: str | Path) -> None:
    """Atomically save one series as volume.npy plus metadata.json."""

    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    volume_path = destination / "volume.npy"
    metadata_path = destination / "metadata.json"
    volume_tmp = destination / f".volume.{os.getpid()}.npy"
    metadata_tmp = destination / f".metadata.{os.getpid()}.json"
    try:
        np.save(volume_tmp, processed.volume, allow_pickle=False)
        metadata_tmp.write_text(
            json.dumps(processed.metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(volume_tmp, volume_path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        volume_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)


def load_cached_series(cache_dir: str | Path) -> ProcessedSeries:
    source = Path(cache_dir)
    volume = np.load(source / "volume.npy", allow_pickle=False)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    return ProcessedSeries(volume=volume, metadata=metadata)


def refresh_cached_series_metadata(
    dicom_paths: Sequence[str | Path], cache_dir: str | Path
) -> dict[str, Any]:
    """Atomically add current geometry metadata without rewriting cached pixels."""

    source = Path(cache_dir)
    volume = np.load(source / "volume.npy", mmap_mode="r", allow_pickle=False)
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records, sort_method = sort_slice_records(
        _read_datasets(
            [Path(path) for path in dicom_paths], stop_before_pixels=True
        )
    )
    if metadata.get("sort_method") != sort_method:
        raise RuntimeError(
            f"sort method drift: cached={metadata.get('sort_method')} "
            f"headers={sort_method}"
        )
    source_files = metadata.get("source_files")
    if isinstance(source_files, list):
        records = _records_in_cached_order(records, source_files)
    if len(records) != int(volume.shape[0]):
        raise RuntimeError(
            f"geometry slice count {len(records)} does not match "
            f"cached volume {volume.shape[0]}"
        )
    updated = {**metadata, **_geometry_metadata(records)}
    temporary = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(updated, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
    return updated


def verify_cached_series(
    dicom_paths: Sequence[str | Path],
    cache_dir: str | Path,
    config: PreprocessConfig | None = None,
) -> None:
    """Recompute through the online path and require byte-identical pixels."""

    config = config or PreprocessConfig()
    cached = load_cached_series(cache_dir)
    if cached.metadata.get("config_fingerprint") != config.fingerprint:
        raise AssertionError("cache config fingerprint does not match requested preprocessing")
    online = preprocess_series(dicom_paths, config)
    if not np.array_equal(cached.volume, online.volume):
        raise AssertionError("cached volume differs from online preprocessing output")
