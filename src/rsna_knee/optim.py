"""Optimizer and learning-rate utilities for the knee baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

def _uses_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim > 1 and not name.endswith(
        ("cls_token", "reg_token", "gamma_1", "gamma_2")
    )


def build_adamw(
    model: nn.Module,
    *,
    backbone_lr: float = 1e-5,
    new_lr: float = 1e-4,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    fused: bool = True,
    stage_lr_decay: float = 1.0,
) -> tuple[torch.optim.AdamW, list[dict[str, Any]]]:
    if not 0 < backbone_lr <= new_lr:
        raise ValueError("expected 0 < backbone_lr <= new_lr")
    if not 0 < stage_lr_decay <= 1:
        raise ValueError("stage_lr_decay must be in (0,1]")
    deepest_layer = int(getattr(model, "pretrained_deepest_layer", 4))
    if deepest_layer < 0:
        raise ValueError("pretrained_deepest_layer must be nonnegative")
    grouped: dict[tuple[str, bool], dict[str, Any]] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise RuntimeError(f"parameter appears more than once: {name}")
        seen.add(identity)
        layer = model.pretrained_layer_id(name)
        if layer is None:
            family = "new"
            lr = new_lr
        else:
            family = f"backbone_stage_{layer:02d}"
            # Native MaxxViT retains the historical accepted layer-5 alias.
            if layer not in range(deepest_layer + 2):
                raise ValueError(f"unexpected backbone stage: {layer}")
            lr = backbone_lr * stage_lr_decay ** (deepest_layer - min(layer, deepest_layer))
        decay = _uses_weight_decay(name, parameter)
        key = (family, decay)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": weight_decay if decay else 0.0,
                "group_name": f"{family}/{'decay' if decay else 'no_decay'}",
                "parameter_names": [],
            }
        grouped[key]["params"].append(parameter)
        grouped[key]["parameter_names"].append(name)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError("optimizer parameter coverage mismatch")
    groups = sorted(grouped.values(), key=lambda item: item["group_name"])
    optimizer_groups = [
        {key: value for key, value in group.items() if key != "parameter_names"}
        for group in groups
    ]
    use_fused = bool(fused and torch.cuda.is_available())
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        betas=betas,
        fused=use_fused,
    )
    summary = [
        {
            "group_name": group["group_name"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameter_tensors": len(group["params"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "parameter_names": group["parameter_names"],
            "fused": use_fused,
        }
        for group in groups
    ]
    return optimizer, summary


def group_gradient_norms(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    names = []
    squared_values = []
    for group in optimizer.param_groups:
        squared = torch.zeros((), device=group["params"][0].device)
        for parameter in group["params"]:
            if parameter.grad is not None:
                squared += parameter.grad.detach().float().pow(2).sum()
        names.append(str(group["group_name"]))
        squared_values.append(squared)
    # One device synchronization for all groups instead of one .item() per group.
    norms = torch.stack(squared_values).sqrt().cpu().tolist()
    return dict(zip(names, norms, strict=True))


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group["group_name"]): float(group["lr"])
        for group in optimizer.param_groups
    }


def build_onecycle(optimizer, *, total_steps: int, schedule: dict):
    """B1's complete LR AND beta1 trajectory, including PyTorch defaults."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["initial_lr"] for g in optimizer.param_groups],
        total_steps=total_steps, **schedule,
    )
