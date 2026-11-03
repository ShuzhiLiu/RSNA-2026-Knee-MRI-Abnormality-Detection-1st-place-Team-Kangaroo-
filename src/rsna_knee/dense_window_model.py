"""Selected dense-window 2.5D model for the twelve knee targets (Baseline backbones: MaxxViTv2 Base 384 / Nano 256, see configs/baseline_*.json)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .model_input import DEPTH, ROLES, TARGETS, make_safe_augment
from .model import BackboneLoadReport
from .model_backbones import get_backbone_module, load_backbone
# Keep private historical imports available for external migration tools.
from .model_backbones.maxxvit import _preserve_bn_on_recompute, _checkpointed_stage_forward


WINDOW_STRIDE = 1
WINDOWS = DEPTH - 2


def load_pretrained_backbone(
    checkpoint: str | Path,
    *,
    architecture: str,
    image_size: int,
    expected_sha256: str | None = None,
    gradient_checkpointing: bool = True,
    backbone_module: str | None = None,
) -> tuple[nn.Module, BackboneLoadReport]:
    """Compatibility entry point for strict per-model local checkpoint loading."""
    return load_backbone(
        checkpoint, architecture=architecture, image_size=image_size,
        expected_sha256=expected_sha256, gradient_checkpointing=gradient_checkpointing,
        backbone_module=backbone_module,
    )


class DenseWindowModel(nn.Module):
    """Disease-specific attention over all adjacent triplets within each role."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        augment: Callable[[torch.Tensor], torch.Tensor],
        train_windows_per_role: int = 6,
        depth: int = DEPTH,
        targets: int = TARGETS,
        hidden: int = 256,
        head_hidden: int = 256,
        dropout: float = 0.2,
        backbone_module: str | None = None,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.windows = self.depth - 2
        if self.depth < 3:
            raise ValueError("model depth must be at least three")
        if not 1 <= train_windows_per_role <= self.windows:
            raise ValueError(f"train_windows_per_role must be in [1, {self.windows}]")
        self.backbone = backbone
        self.backbone_module = backbone_module or getattr(backbone, "_rsna_backbone_module", "maxxvit")
        backbone_functions = get_backbone_module(self.backbone_module)
        self.pretrained_deepest_layer = backbone_functions.DEEPEST_LAYER
        self.augment = augment
        self.roles = ROLES
        self.targets = int(targets)
        self.window_stride = WINDOW_STRIDE
        self.window_counts = (self.windows,) * ROLES
        self.train_windows_per_role = int(train_windows_per_role)
        self.feature_dim = int(backbone.num_features)
        self.hidden = int(hidden)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Linear(self.feature_dim, hidden), nn.GELU()
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.role_embedding = nn.Parameter(torch.randn(ROLES, hidden) * 0.02)
        self.window_embedding = nn.Parameter(torch.randn(self.windows, hidden) * 0.02)
        self.queries = nn.Parameter(torch.randn(targets, hidden) * 0.02)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(ROLES * hidden),
                nn.Linear(ROLES * hidden, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(targets)
        )
        self.register_buffer(
            "mean", torch.tensor(backbone_functions.INPUT_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(backbone_functions.INPUT_STD).view(1, 3, 1, 1)
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def initialize_head_bias(self, prevalence: torch.Tensor) -> None:
        if prevalence.shape != (self.targets,):
            raise ValueError("target prevalence must have one value per target")
        with torch.no_grad():
            for index, head in enumerate(self.heads):
                head[-1].bias.copy_(
                    torch.logit(prevalence[index].float().clamp(0.01, 0.99)).reshape(1)
                )

    def pretrained_layer_id(self, name: str) -> int | None:
        return get_backbone_module(self.backbone_module).pretrained_layer_id(name)

    def _selected_window_indices(self, device: torch.device) -> torch.Tensor:
        if not self.training or self.train_windows_per_role >= self.windows:
            return torch.arange(self.windows, device=device)
        return torch.randperm(self.windows, device=device)[
            : self.train_windows_per_role
        ].sort().values

    @staticmethod
    def _triplets(values: torch.Tensor, selected: list[int] | torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [values[:, :, int(index) : int(index) + 3] for index in selected],
            dim=2,
        )

    def forward(
        self,
        images: torch.Tensor,
        series_mask: torch.Tensor,
        slice_valid_mask: torch.Tensor,
        slice_repeated_mask: torch.Tensor,
        pixel_valid_mask: torch.Tensor | None = None,
        slice_positions_normalized: torch.Tensor | None = None,
        gap_previous_mm: torch.Tensor | None = None,
        gap_next_mm: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del pixel_valid_mask, gap_previous_mm, gap_next_mm
        if images.ndim != 5 or images.shape[1:3] != (ROLES, self.depth):
            raise ValueError(f"expected [B,{ROLES},{self.depth},H,W], got {tuple(images.shape)}")
        batch, _, _, height, width = images.shape
        selected_windows = [
            self._selected_window_indices(images.device) for _ in range(ROLES)
        ]
        count = int(selected_windows[0].numel())
        if any(int(selected.numel()) != count for selected in selected_windows):
            raise RuntimeError("all roles must use the same number of windows")

        # Transfer each role's indices once, preserving CUDA sampling/RNG order.
        selected_host = [selected.tolist() for selected in selected_windows]

        image_triplets = [
            self._triplets(images[:, role : role + 1], selected).squeeze(1)
            for role, selected in enumerate(selected_host)
        ]
        flat_values = torch.cat(image_triplets, dim=1).reshape(-1, 3, height, width)
        if self.training:
            flat_values = self.augment(flat_values)
        flat_values = (flat_values - self.mean) / self.std
        flat_features = self.backbone(flat_values)
        if flat_features.ndim != 2 or flat_features.shape[-1] != self.feature_dim:
            raise RuntimeError(f"unexpected backbone feature shape: {tuple(flat_features.shape)}")
        feature_norm = flat_features.float().norm(dim=-1).mean()
        flat_features = flat_features.reshape(batch, ROLES * count, self.feature_dim)

        if slice_positions_normalized is None:
            positions = torch.linspace(
                0, 1, self.depth, device=images.device, dtype=flat_features.dtype
            ).view(1, 1, self.depth).expand(batch, ROLES, self.depth)
        else:
            positions = slice_positions_normalized.to(flat_features.dtype)

        contexts = []
        entropies = []
        for role, selected in enumerate(selected_windows):
            features = flat_features[:, role * count : (role + 1) * count]
            valid = self._triplets(
                slice_valid_mask[:, role : role + 1], selected_host[role]
            ).squeeze(1)
            repeated = self._triplets(
                slice_repeated_mask[:, role : role + 1], selected_host[role]
            ).squeeze(1)
            grouped_positions = self._triplets(
                positions[:, role : role + 1], selected_host[role]
            ).squeeze(1)
            window_mask = valid.any(-1) & series_mask[:, role].unsqueeze(-1)
            valid_count = valid.sum(-1).clamp_min(1)
            unique_ratio = (valid & ~repeated).sum(-1).float() / valid_count.float()
            valid_ratio = valid.float().mean(-1)
            geometry = torch.stack(
                (
                    grouped_positions.mean(-1),
                    (grouped_positions[..., -1] - grouped_positions[..., 0]).abs(),
                    valid_ratio.to(features.dtype),
                    unique_ratio.to(features.dtype),
                ),
                dim=-1,
            )
            hidden = self.feature_projection(features) + self.geometry_projection(geometry)
            hidden = hidden + self.role_embedding[role].view(1, 1, self.hidden)
            hidden = hidden + self.window_embedding[selected].view(1, count, self.hidden)
            scores = torch.einsum("bwh,th->btw", hidden, self.queries) / math.sqrt(
                self.hidden
            )
            attention = scores.masked_fill(
                ~window_mask.unsqueeze(1), -1e4
            ).softmax(-1)
            context = torch.einsum("btw,bwh->bth", attention, hidden)
            context = context * series_mask[:, role, None].unsqueeze(1).to(context.dtype)
            contexts.append(context)
            entropies.append(
                -(attention.clamp_min(1e-8).log() * attention).sum(-1).mean()
            )

        stacked = torch.stack(contexts, dim=2)
        concatenated = stacked.reshape(batch, self.targets, ROLES * self.hidden)
        logits = torch.cat(
            [
                head(concatenated[:, target]).reshape(batch, 1)
                for target, head in enumerate(self.heads)
            ],
            dim=1,
        )
        if not return_diagnostics:
            return logits
        return logits, {
            "tta_windows": torch.tensor(float(ROLES * count), device=images.device),
            "role_feature_norm": stacked.norm(dim=-1).mean(),
            "window_attention_entropy": torch.stack(entropies).mean(),
            "backbone_feature_norm": feature_norm,
            "logit_mean": logits.mean(),
            "logit_std": logits.std(unbiased=False),
        }


def build_model(root: Path, config: dict[str, Any], *, model_override=None) -> tuple[DenseWindowModel, BackboneLoadReport]:
    model_config = config["model"]
    manifest = json.loads(
        (root / model_config["pretrained_manifest"]).read_text(encoding="utf-8")
    )
    checkpoint = root / manifest["target_dir"] / "model.safetensors"
    expected = next(
        item["sha256"]
        for item in manifest["files"]
        if item["name"] == "model.safetensors"
    )
    backbone, report = load_pretrained_backbone(
        checkpoint,
        architecture=model_config["backbone"],
        image_size=int(model_config["input_shape"][-1]),
        expected_sha256=expected,
        gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
        backbone_module=model_config.get("backbone_module"),
    )
    if model_config.get("backbone_image_chunk") is not None:
        if model_config.get("backbone_module") != "coatnet" or model_config["backbone_image_chunk"] != 32:
            raise ValueError("only final CoAtNet chunk32 is supported")
        backbone._rsna_backbone_image_chunk = 32
    model_class = DenseWindowModel
    extra = {}
    if model_config.get("location_pooling", False):
        from .location_pooling import LocationDenseWindowModel
        model_class = LocationDenseWindowModel
        if config.get("evidence_descriptors", {}).get("enabled", False):
            raise ValueError("descriptor research must use the explicit research model factory")
        extra = {"image_size": int(model_config["input_shape"][-1]), "location_seed": int(config["seed"])}
    if config.get("boxcombo"):
        if not model_config.get("location_pooling"):
            raise ValueError("lesion evidence requires complete location pooling")
        from .box_model import BoxModel
        model_class = BoxModel
        extra.update(residual_coefficient=config["boxcombo"]["residual_coefficient"],
                     lesion_weight=config["boxcombo"]["lesion_weight"])
    if model_override is not None and not config.get("boxcombo"):
        model_class = model_override
    return model_class(
        backbone,
        **extra,
        depth=int(model_config["input_shape"][1]),
        augment=make_safe_augment(config["augmentation"]),
        train_windows_per_role=int(model_config["train_windows_per_role"]),
        targets=int(model_config["targets"]),
        hidden=int(model_config["fusion_hidden"]),
        head_hidden=int(model_config["head_hidden"]),
        dropout=float(model_config["dropout"]),
    ), report
