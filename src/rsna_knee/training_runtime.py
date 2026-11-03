"""Runtime, checkpoint I/O and tensor transfer shared by final training/inference."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any
import numpy as np
import torch

MODEL_KEYS = (
    "images",
    "series_mask",
    "slice_valid_mask",
    "slice_repeated_mask",
    "pixel_valid_mask",
    "slice_positions_normalized",
    "gap_previous_mm",
    "gap_next_mm",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def environment_record(root: Path) -> dict[str, Any]:
    try:
        import matplotlib
        import pandas
        import timm
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("pinned training dependencies are missing") from exc
    gpu = {}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "executable": os.path.realpath(os.sys.executable),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "timm": timm.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "gpu": gpu,
        "controlled_backend": assert_controlled_backend(),
        "git_commit": _git_output(root, "rev-parse", "HEAD"),
        "git_status": _git_output(root, "status", "--short"),
    }


def _gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    values = completed.stdout.strip().split(", ") if completed.returncode == 0 else []
    names = ("utilization_percent", "power_watts", "temperature_c", "memory_used_mib", "memory_total_mib")
    if len(values) != len(names):
        return {"query_error": completed.stderr.strip() or completed.stdout.strip()}
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def to_device(batch: dict[str, Any], device: torch.device, *, evidence_descriptors: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if not batch["series_mask"].any(dim=1).all():
        raise ValueError("every study must contain at least one valid role")
    valid = batch["slice_valid_mask"] & batch["series_mask"].unsqueeze(-1)
    if not (valid.any(dim=-1) == batch["series_mask"]).all():
        raise ValueError("valid roles must contain at least one valid slice")
    inputs: dict[str, torch.Tensor] = {}
    images = batch["images"].to(device, non_blocking=True)
    if images.dtype == torch.uint16:
        images = images.to(torch.float32).div_(65535.0)
    else:
        images = images.to(torch.float32)
    inputs["images"] = images
    for key in MODEL_KEYS[1:]:
        inputs[key] = batch[key].to(device, non_blocking=True)
    if "box_data" in batch:
        inputs["box_data"] = batch["box_data"]
    if evidence_descriptors:
        for key in ("descriptor_source_ids", "descriptor_positions_mm", "geometry_measured", "geometry_uniform_fallback", "descriptor_keep"):
            inputs[key] = batch[key].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True, dtype=torch.float32)
    label_mask = batch["label_mask"].to(device, non_blocking=True)
    return inputs, labels, label_mask


def controlled_backend_flags():
    return {'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def assert_controlled_backend():
    flags = controlled_backend_flags()
    assert flags == {'cudnn_enabled': True, 'cudnn_benchmark': False,
                     'cudnn_deterministic': True, 'deterministic_algorithms': True,
                     'deterministic_warn_only': False, 'matmul_tf32': True,
                     'cudnn_tf32': True, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}, flags
    return flags


@contextmanager
def evaluation_backend():
    """Canonical evaluation flags, restoring the registered controlled training flags."""
    before = assert_controlled_backend()
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=True):
        yield
    assert assert_controlled_backend() == before

