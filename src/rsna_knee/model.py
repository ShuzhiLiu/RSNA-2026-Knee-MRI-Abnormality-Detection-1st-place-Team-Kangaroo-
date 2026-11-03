"""Small shared model provenance utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BackboneLoadReport:
    architecture: str
    checkpoint: str
    checkpoint_sha256: str
    checkpoint_keys: int
    model_keys: int
    parameters: int
    all_keys_matched: bool
