"""Native MaxxViT model functions; no module wrapping or task-model ownership."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import types

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..model import BackboneLoadReport, file_sha256

@contextmanager
def _preserve_bn_on_recompute(module):
    saved = [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
             for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    try:
        yield
    finally:
        with torch.no_grad():
            for m, mean, var, count in saved:
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(count)


def _checkpointed_stage_forward(self, x):
    """MaxViT stage forward with per-block activation checkpointing that keeps BatchNorm statistics unchanged on recompute."""
    if self.grad_checkpointing and self.training and torch.is_grad_enabled():
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False,
                           context_fn=lambda b=block: (nullcontext(), _preserve_bn_on_recompute(b)))
        return x
    return self.blocks(x)


def load_pretrained_backbone(
    checkpoint: str | Path,
    *,
    architecture: str,
    image_size: int,
    expected_sha256: str | None = None,
    gradient_checkpointing: bool = True,
) -> tuple[nn.Module, BackboneLoadReport]:
    """Strictly load the pinned local timm checkpoint for ``architecture`` (``timm:<name>``)."""

    try:
        import timm
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("training requires pinned timm and safetensors") from exc

    if not architecture.startswith("timm:"):
        raise ValueError(f"backbone must be a timm architecture: {architecture}")
    name = architecture.removeprefix("timm:")
    checkpoint = Path(checkpoint).resolve()
    actual_sha256 = file_sha256(checkpoint)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"backbone checkpoint SHA mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    backbone = timm.create_model(
        name,
        pretrained=False,
        num_classes=0,
        img_size=image_size,
    )
    state = load_file(str(checkpoint), device="cpu")
    model_keys = set(backbone.state_dict())
    extra = set(state) - model_keys
    expected_extra = {"head.fc.weight", "head.fc.bias"}
    if extra != expected_extra:
        raise RuntimeError(f"unexpected pretrained-only keys: {sorted(extra)}")
    missing = model_keys - set(state)
    if missing:
        raise RuntimeError(f"missing backbone keys: {sorted(missing)}")
    backbone.load_state_dict({key: state[key] for key in model_keys}, strict=True)
    for stage in backbone.stages:
        stage.forward = types.MethodType(_checkpointed_stage_forward, stage)
    if gradient_checkpointing:
        backbone.set_grad_checkpointing(True)
    return backbone, BackboneLoadReport(
        architecture=architecture,
        checkpoint=str(checkpoint),
        checkpoint_sha256=actual_sha256,
        checkpoint_keys=len(state),
        model_keys=len(model_keys),
        parameters=sum(parameter.numel() for parameter in backbone.parameters()),
        all_keys_matched=True,
    )


# The task model constructs these buffers in the original order and dtype.
INPUT_MEAN = (0.5, 0.5, 0.5)
INPUT_STD = (0.5, 0.5, 0.5)
DEEPEST_LAYER = 4


def pretrained_layer_id(name: str) -> int | None:
    if name.startswith("backbone.stem."):
        return 0
    if name.startswith("backbone.stages."):
        return int(name.split(".")[2]) + 1
    if name.startswith("backbone.norm.") or name.startswith("backbone.head."):
        return 4
    if name.startswith("backbone."):
        return 0
    return None


def validate_location(backbone: nn.Module, image_size: int) -> int:
    assert int(backbone.num_features) in (768, 1024) and isinstance(backbone.head.fc, nn.Identity)
    assert backbone.head.drop.p == 0
    return image_size // 32


def feature_map(backbone: nn.Module, values: torch.Tensor) -> torch.Tensor:
    return backbone.forward_features(values)


def pool(backbone: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return backbone.forward_head(features)
