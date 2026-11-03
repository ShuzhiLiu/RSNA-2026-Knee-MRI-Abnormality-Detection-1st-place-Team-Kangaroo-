"""Local-first structured logging for reproducible training runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_value(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class RunLogger:
    """Write readable text plus lossless JSONL events with immediate flushing."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"rsna-knee-training-{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(self.run_dir / "train.log", mode="a", encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self._events = (self.run_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self.logger.warning(message, *args)

    def event(self, kind: str, **values: Any) -> None:
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **values,
        }
        self._events.write(json.dumps(_json_value(record), ensure_ascii=False, sort_keys=True) + "\n")

    def save_json(self, name: str, value: Any) -> Path:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def save_text(self, name: str, value: str) -> Path:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
        return path

    def close(self) -> None:
        if not self._events.closed:
            self._events.flush()
            self._events.close()
        for handler in self.logger.handlers:
            handler.flush()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
