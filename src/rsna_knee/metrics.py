"""Dependency-light multilabel losses and audit metrics."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets and mask must have identical shape")
    element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    active = mask.to(element.dtype)
    active_per_target = active.sum(dim=0)
    per_target = (element * active).sum(dim=0) / active_per_target.clamp_min(1)
    supervised_targets = active_per_target > 0
    if not supervised_targets.any():
        raise ValueError("loss mask contains no supervised elements")
    return per_target[supervised_targets].mean(), per_target


def binary_auc(truth: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    if truth.ndim != 1 or not np.isfinite(truth).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("binary_auc requires explicit binary truth; use soft_auc for continuous truth")
    truth = truth.astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    if score.shape != truth.shape or not np.isfinite(score).all():
        raise ValueError("invalid binary AUC score shape/values")
    positive = int(truth.sum())
    negative = int(truth.size - positive)
    if positive == 0 or negative == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(truth.size, dtype=np.float64)
    start = 0
    while start < truth.size:
        end = start + 1
        while end < truth.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    rank_sum = float(ranks[truth == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: Sequence[str],
) -> dict[str, float]:
    if logits.shape != targets.shape or logits.shape[1] != len(labels):
        raise ValueError("metric shapes do not match label schema")
    probabilities = torch.sigmoid(logits.float()).cpu().numpy()
    truth = targets.float().cpu().numpy()
    output: dict[str, float] = {}
    aucs = []
    for index, label in enumerate(labels):
        auc = binary_auc(truth[:, index], probabilities[:, index])
        output[f"auc/{label}"] = auc
        output[f"brier/{label}"] = float(np.mean((probabilities[:, index] - truth[:, index]) ** 2))
        clipped = np.clip(probabilities[:, index], 1e-7, 1 - 1e-7)
        output[f"logloss/{label}"] = float(
            -np.mean(truth[:, index] * np.log(clipped) + (1 - truth[:, index]) * np.log(1 - clipped))
        )
        if math.isfinite(auc):
            aucs.append(auc)
    output["macro_auc"] = float(np.mean(aucs)) if aucs else math.nan
    output["macro_brier"] = float(np.mean([output[f"brier/{label}"] for label in labels]))
    output["macro_logloss"] = float(np.mean([output[f"logloss/{label}"] for label in labels]))
    return output
