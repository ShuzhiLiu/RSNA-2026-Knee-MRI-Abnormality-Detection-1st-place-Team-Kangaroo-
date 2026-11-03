"""Native PyTorch exponential moving average with explicit checkpoint state."""

from __future__ import annotations

from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def build_ema(model: nn.Module, decay: float = 0.999) -> AveragedModel:
    if not 0 < decay < 1:
        raise ValueError("EMA decay must be in (0,1)")
    return AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(decay),
        use_buffers=True,
    )
