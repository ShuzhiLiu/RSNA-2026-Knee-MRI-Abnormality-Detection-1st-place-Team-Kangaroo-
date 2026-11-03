"""Deployable five-role selection and cache materialization for hidden test MRI."""

from __future__ import annotations

import csv
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .input_pipeline import ROLES, file_sha256
from .preprocessing import (
    METADATA_SCHEMA_VERSION,
    PreprocessConfig,
    preprocess_series,
    refresh_cached_series_metadata,
    save_cached_series,
)


def infer_sequence_family(description: str | None, fluid_sensitive: bool) -> str:
    text = (description or "").casefold()
    if "stir" in text:
        return "stir"
    if re.search(r"(^|[^a-z])t1([^a-z]|$)", text):
        return "t1"
    if re.search(r"(^|[^a-z])t2([^a-z]|$)", text):
        return "t2"
    if "pd" in text or "proton" in text:
        return "pd"
    return "other" if text else ("pd" if fluid_sensitive else "unknown")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _header_value(dataset: Any, name: str, default: Any = None) -> Any:
    return getattr(dataset, name, default)


def _relative_paths(split: str, study_uid: str, series_uid: str) -> tuple[str, str]:
    directory = Path(split) / study_uid / series_uid
    return str(directory / "volume.npy"), str(directory / "metadata.json")


def build_series_candidate(
    row: Mapping[str, str],
    *,
    data_root: Path,
    cache_root: Path,
    split: str,
) -> dict[str, Any]:
    study_uid = str(row["StudyInstanceUID"])
    series_uid = str(row["SeriesInstanceUID"])
    volume_relative, metadata_relative = _relative_paths(split, study_uid, series_uid)
    volume_path = cache_root / volume_relative
    metadata_path = cache_root / metadata_relative
    raw_directory = data_root / f"{split}_series" / study_uid / series_uid
    raw_files = sorted(raw_directory.glob("*.dcm"))
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    if not metadata and raw_files:
        try:
            import pydicom
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pydicom is required for raw inference selection") from error
        header = pydicom.dcmread(raw_files[len(raw_files) // 2], stop_before_pixels=True)
        spacing = _header_value(header, "PixelSpacing")
        rows = int(_header_value(header, "Rows", 0) or 0)
        columns = int(_header_value(header, "Columns", 0) or 0)
        metadata = {
            "series_description": str(_header_value(header, "SeriesDescription", "") or "") or None,
            "shape": [len(raw_files), rows, columns],
            "slice_count": len(raw_files),
            "pixel_spacing": [float(spacing[0]), float(spacing[1])] if spacing is not None else None,
            "spacing_between_slices": float(
                _header_value(header, "SpacingBetweenSlices", None)
                or _header_value(header, "SliceThickness", 1.0)
                or 1.0
            ),
            "image_orientation_patient": _header_value(header, "ImageOrientationPatient", None),
            "orientation_consistent": _header_value(header, "ImageOrientationPatient", None) is not None,
        }
    shape = metadata.get("shape") or [0, 0, 0]
    spacing = metadata.get("pixel_spacing")
    geometry_usable = bool(
        isinstance(shape, list)
        and len(shape) == 3
        and int(shape[0]) > 0
        and int(shape[1]) > 0
        and int(shape[2]) > 0
        and isinstance(spacing, list)
        and len(spacing) == 2
        and all(math.isfinite(float(value)) and float(value) > 0 for value in spacing)
    )
    spacing_values = [float(spacing[0]), float(spacing[1])] if geometry_usable else [1.0, 1.0]
    fluid = row.get("Fluid_Sensitive") == "1"
    fat = row.get("Fat_Suppression") == "1"
    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "plane": row.get("Anatomical_Plane") or "Unknown",
        "fluid_sensitive": fluid,
        "fat_suppression": fat,
        "sequence_family": infer_sequence_family(metadata.get("series_description"), fluid),
        "series_description": metadata.get("series_description"),
        "slice_count": int(metadata.get("slice_count") or shape[0]),
        "shape": [int(value) for value in shape],
        "pixel_spacing": spacing_values,
        "slice_spacing": float(
            metadata.get("spacing_between_slices") or metadata.get("slice_thickness") or 1.0
        ),
        "field_of_view": [
            float(shape[1]) * spacing_values[0],
            float(shape[2]) * spacing_values[1],
        ],
        "orientation_quality": bool(metadata.get("orientation_consistent") or metadata.get("image_orientation_patient")),
        "decode_quality": 1.0 if metadata.get("skipped_slice_count", 0) == 0 else 0.5,
        "geometry_usable": geometry_usable,
        "cache_available": bool(
            volume_path.is_file()
            and metadata_path.is_file()
            and metadata.get("config_fingerprint") == PreprocessConfig().fingerprint
        ),
        "raw_available": bool(raw_files),
        "raw_directory": str(raw_directory),
        "volume_path": volume_relative,
        "metadata_path": metadata_relative,
        "config_fingerprint": metadata.get("config_fingerprint"),
    }


def score_role(candidate: Mapping[str, Any], role: str) -> float | None:
    expected_plane = {"SAG": "Sagittal", "COR": "Coronal", "AX": "Axial"}[role.split("_")[0]]
    if candidate["plane"] != expected_plane or not candidate["geometry_usable"]:
        return None
    description = str(candidate.get("series_description") or "").casefold()
    if any(token in description for token in ("localizer", "scout", "survey", "locator")):
        return None
    fluid = bool(candidate["fluid_sensitive"])
    fat = bool(candidate["fat_suppression"])
    if role.endswith("FLUID_FS"):
        if not fluid:
            return None
        protocol = 120.0 if fat else 90.0
    else:
        if fluid and fat:
            return None
        protocol = 115.0 if not fluid else 75.0
        if infer_sequence_family(candidate.get("series_description"), fluid) == "t1":
            protocol += 8.0
    mean_spacing = sum(float(value) for value in candidate["pixel_spacing"]) / 2
    resolution = 4.0 / max(mean_spacing, 0.1)
    coverage = 6.0 * min(min(float(value) for value in candidate["field_of_view"]) / 180.0, 1.0)
    depth = 2.0 * min(int(candidate["slice_count"]) / 64.0, 1.0)
    orientation = 1.0 if candidate["orientation_quality"] else 0.0
    decode = float(candidate["decode_quality"])
    return float(protocol + resolution + coverage + depth + orientation + decode)


def select_protocol_roles(
    candidates: Sequence[Mapping[str, Any]],
    *,
    blacklist: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any] | None], list[dict[str, Any]]]:
    blacklist = blacklist or set()
    enriched = [dict(candidate) for candidate in candidates]
    selected: dict[str, dict[str, Any] | None] = {}
    used: set[str] = set()
    for role in ROLES:
        scored = []
        for candidate in enriched:
            score = score_role(candidate, role)
            candidate.setdefault("role_scores", {})[role] = score
            uid = str(candidate["series_uid"])
            available = candidate.get("cache_available") or candidate.get("raw_available")
            if score is not None and available and uid not in used and uid not in blacklist:
                scored.append((score, int(candidate["slice_count"]), uid, candidate))
        if not scored:
            selected[role] = None
            continue
        _, _, _, winner = max(scored, key=lambda item: (item[0], item[1], item[2]))
        record = {key: value for key, value in winner.items() if key not in {"role_scores", "raw_directory", "raw_available", "geometry_usable", "study_uid"}}
        record["role"] = role
        record["selection_score"] = winner["role_scores"][role]
        selected[role] = record
        used.add(str(winner["series_uid"]))
    return selected, enriched


def _selected_cache_is_deployable(metadata: Mapping[str, Any], config: PreprocessConfig) -> bool:
    positions = metadata.get("slice_positions_mm")
    shape = metadata.get("shape")
    spacing = metadata.get("pixel_spacing")
    return bool(
        metadata.get("config_fingerprint") == config.fingerprint
        and metadata.get("metadata_schema_version") == METADATA_SCHEMA_VERSION
        and isinstance(shape, list)
        and isinstance(positions, list)
        and len(positions) == int(shape[0])
        and isinstance(spacing, list)
        and len(spacing) == 2
    )


def prepare_inference_manifest(
    *,
    data_root: str | Path,
    cache_root: str | Path,
    output_path: str | Path,
    sample_ids: set[str] | None = None,
    max_studies: int | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    """Select, materialize and validate hidden-test roles without reading reports."""

    data_root, cache_root, output_path = Path(data_root), Path(cache_root), Path(output_path)
    config = preprocess_config or PreprocessConfig()
    studies = _read_rows(data_root / "test.csv")
    if sample_ids:
        studies = [row for row in studies if row["StudyInstanceUID"] in sample_ids]
        missing = sample_ids - {row["StudyInstanceUID"] for row in studies}
        if missing:
            raise RuntimeError(f"test study IDs not found: {sorted(missing)}")
    if max_studies is not None:
        studies = studies[:max_studies]
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_rows(data_root / "test_series.csv"):
        grouped[row["StudyInstanceUID"]].append(row)
    records = []
    failures = []
    decoded = refreshed = reused = 0
    for study in studies:
        study_uid = study["StudyInstanceUID"]
        rows = grouped.get(study_uid, [])
        blacklist: set[str] = set()
        while True:
            candidates = [
                build_series_candidate(row, data_root=data_root, cache_root=cache_root, split="test")
                for row in rows
            ]
            selected, enriched = select_protocol_roles(candidates, blacklist=blacklist)
            pending = {
                str(value["series_uid"]): value
                for value in selected.values()
                if value is not None
            }
            failed = False
            for series_uid, value in pending.items():
                raw_directory = data_root / "test_series" / study_uid / series_uid
                source_paths = sorted(raw_directory.glob("*.dcm"))
                cache_directory = cache_root / "test" / study_uid / series_uid
                try:
                    metadata_path = cache_directory / "metadata.json"
                    volume_path = cache_directory / "volume.npy"
                    metadata = (
                        json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata_path.is_file() and volume_path.is_file()
                        else None
                    )
                    if metadata is None or metadata.get("config_fingerprint") != config.fingerprint:
                        if metadata is not None:
                            raise RuntimeError("refusing incompatible existing pixel cache")
                        processed = preprocess_series(source_paths, config)
                        save_cached_series(processed, cache_directory)
                        metadata = processed.metadata
                        decoded += 1
                    elif not _selected_cache_is_deployable(metadata, config):
                        metadata = refresh_cached_series_metadata(source_paths, cache_directory)
                        refreshed += 1
                    else:
                        reused += 1
                    if not _selected_cache_is_deployable(metadata, config):
                        raise RuntimeError("selected series remains non-deployable after preprocessing")
                except Exception as error:  # noqa: BLE001
                    if "install the matching pydicom pixel decoder" in str(error):
                        raise RuntimeError("required DICOM decoder is missing; refusing silent role loss") from error
                    blacklist.add(series_uid)
                    failures.append(
                        {
                            "study_uid": study_uid,
                            "series_uid": series_uid,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    failed = True
            if not failed:
                final_candidates = [
                    build_series_candidate(row, data_root=data_root, cache_root=cache_root, split="test")
                    for row in rows
                ]
                final_selected, final_candidates = select_protocol_roles(
                    final_candidates,
                    blacklist=blacklist,
                )
                records.append(
                    {
                        "schema_version": "rsna-knee-inference-selection-v2",
                        "study_uid": study_uid,
                        "selected_series": final_selected,
                        "series_candidates": final_candidates,
                        "decode_failures": [
                            failure for failure in failures if failure["study_uid"] == study_uid
                        ],
                    }
                )
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output_path)
    return {
        "pipeline": "rsna-knee-inference-selection-v2",
        "studies": len(records),
        "decoded_series": decoded,
        "refreshed_metadata_series": refreshed,
        "reused_series": reused,
        "decode_failures": failures,
        "preprocess_config_fingerprint": config.fingerprint,
        "report_or_training_csv_read": False,
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": file_sha256(output_path),
    }
