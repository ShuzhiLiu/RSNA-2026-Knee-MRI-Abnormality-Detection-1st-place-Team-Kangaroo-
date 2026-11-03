"""Shared optimizer steps and loss accounting; no validation or selection policy."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import torch
import torch.nn.functional as F
from .input_pipeline import LABELS
from .metrics import masked_bce_with_logits
from .model import file_sha256
from .optim import group_gradient_norms, learning_rates
from .training_runtime import (assert_controlled_backend, to_device, _rng_state, _gpu_snapshot)

class TrainingEngine:
    def resolved_config(self) -> dict[str, Any]:
        value = json.loads(json.dumps(self.config))
        for key, item in self.overrides.items():
            value.setdefault("runtime_overrides", {})[key] = item
        return value


    @property
    def batch_size(self) -> int:
        return int(self.overrides.get("batch_size", self.config["data"]["batch_size"]))


    @property
    def workers(self) -> int:
        return int(self.overrides.get("workers", self.config["data"]["workers"]))


    def _configure_runtime(self) -> None:
        runtime = self.config["runtime"]
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(runtime["allow_tf32"])
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == runtime['cublas_workspace_config'] == ':4096:8'
        torch.backends.cudnn.enabled = bool(runtime['cudnn_enabled'])
        torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
        torch.backends.cudnn.deterministic = bool(runtime['cudnn_deterministic'])
        torch.use_deterministic_algorithms(bool(runtime['deterministic_algorithms']))
        assert_controlled_backend()
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


    def _checkpoint(self, epoch: int) -> dict[str, Any]:
        return {
            "schema_version": "local-onecycle-checkpoint-v1",
            "config_sha256": file_sha256(self.config_path),
            "run_contract_sha256": self.run_contract_sha256,
            "model": self.model.state_dict(), "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step, "epoch": epoch,
            "best_ema_auc": self.best_ema_auc, "best_epoch": self.best_epoch,
            "rng": _rng_state(),
        }


    def _amp_context(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == "cuda")


    def _ranking_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config["loss"]["soft_ranking"]
        target_delta = targets[:, None, :] - targets[None, :, :]
        logit_delta = logits[:, None, :] - logits[None, :, :]
        absolute_gap = target_delta.abs()
        upper = torch.triu(
            torch.ones(logits.shape[0], logits.shape[0], dtype=torch.bool, device=logits.device),
            diagonal=1,
        ).unsqueeze(-1)
        eligible = (
            upper
            & mask[:, None, :].bool()
            & mask[None, :, :].bool()
            & (absolute_gap >= float(config["minimum_target_gap"]))
        )
        weights = absolute_gap * eligible.to(absolute_gap.dtype)
        pair_loss = F.softplus(
            -target_delta.sign() * logit_delta / float(config["temperature"])
        )
        denominator = weights.sum(dim=(0, 1))
        per_target = (pair_loss * weights).sum(dim=(0, 1)) / denominator.clamp_min(1e-8)
        active = denominator > 0
        self._rank_per_target = per_target.detach()
        self._rank_active = active.detach()
        self._rank_denominator = denominator.detach()
        if not active.any():
            return logits.sum() * 0.0, eligible.sum(dim=(0, 1))
        return per_target[active].mean(), eligible.sum(dim=(0, 1))


    def training_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bce, per_target = masked_bce_with_logits(logits, targets, mask)
        ranking, pair_counts = self._ranking_loss(logits, targets, mask)
        batch = int(targets.shape[0])
        self._ranking_loss_sum += float(ranking.detach()) * batch
        self._bce_loss_sum += float(bce.detach()) * batch
        self._ranking_samples += batch
        counts = pair_counts.detach().to(torch.float64).cpu()
        if self._ranking_pairs is None:
            self._ranking_pairs = torch.zeros_like(counts)
        self._ranking_pairs += counts
        weight = float(self.config["loss"]["soft_ranking"]["weight"])
        main_loss = bce + weight * ranking
        if self.config["model"].get("location_pooling", False):
            auxiliary, terms = self.model.auxiliary_loss(targets, self.global_step)
            # Per-target rank is normalized on eligible pairs; the global rank
            # averages active targets only. Record exact contributions as well.
            active_count = self._rank_active.sum().clamp_min(1)
            rank_contribution = weight * self._rank_per_target / active_count
            bce_active = mask.sum(0) > 0
            bce_contribution = per_target.detach() / bce_active.sum().clamp_min(1)
            row = {"step": self.global_step, "epoch": self.global_step // self.model.steps_per_epoch,
                   "main_with_rank": float(main_loss.detach()), "weighted_auxiliary": float(auxiliary.detach()),
                   "weak_bce_all12": per_target.detach().cpu().tolist(),
                   "rank_all12": self._rank_per_target.cpu().tolist(),
                   "bce_target_count_all12": mask.sum(0).detach().cpu().tolist(),
                   "rank_pair_count_all12": pair_counts.detach().cpu().tolist(),
                   "rank_pair_weight_sum_all12": self._rank_denominator.cpu().tolist(),
                   "rank_active_all12": self._rank_active.cpu().tolist(),
                   "main_contribution_all12": (bce_contribution + rank_contribution).cpu().tolist(),
                   "bce_plus_weighted_rank_all12": (per_target.detach()+weight*self._rank_per_target).cpu().tolist(),
                   **{k: float(v) for k,v in terms.items()}, **self.model.aux_target_diagnostics,
                   **self._location_diagnostics,
                   "positive_windows": int(self.model.aux["eligible_positive_count"]),
                   "mil_empty_study_target_pairs": int(self.model.aux["mil_empty_pairs"]),
                   "negative_eligible_groups": int(self.model.aux["negative_active_groups"])}
            self._location_rows.append(row)
            with (self.run_dir / "location_step_diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            return main_loss + auxiliary, per_target
        return main_loss, per_target


    def train_epoch(
        self,
        epoch: int,
        *,
        max_steps: int | None = None,
        profiler: Any | None = None,
    ) -> dict[str, Any]:
        self._ranking_loss_sum = self._bce_loss_sum = 0.0
        self._ranking_samples = 0
        self._ranking_pairs = None
        loader = self.loaders["train"]
        loader.set_epoch(epoch)
        self.model.train()
        self.logger.event("controlled_backend", epoch=epoch, stage="train_epoch_start", flags=assert_controlled_backend())
        per_target_sum = torch.zeros(len(LABELS), dtype=torch.float64)
        samples = 0
        loss_sum = 0.0
        cache_hits = 0
        cache_roles = 0
        data_wait_total = 0.0
        epoch_start = time.perf_counter()
        last_end = epoch_start
        log_every = int(self.config["runtime"]["log_every_steps"])
        optimizer_config = self.config["optimizer"]
        actual_steps = min(len(loader.loader), max_steps) if max_steps else len(loader.loader)
        for local_step, batch in enumerate(loader.loader):
            if max_steps is not None and local_step >= max_steps:
                break
            data_ready = time.perf_counter()
            data_wait = data_ready - last_end
            data_wait_total += data_wait
            should_log = local_step % log_every == 0 or local_step + 1 == actual_steps
            if should_log and self.device.type == "cuda":
                torch.cuda.synchronize()
            timed_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("host_to_device"):
                inputs, targets, mask = to_device(batch, self.device, evidence_descriptors=self.config.get("evidence_descriptors", {}).get("enabled", False))
            if self.config["model"].get("location_pooling", False):
                self.model.annotation_batch = batch
                self.model.locpool.replacement_step.fill_(self.global_step)
            with torch.profiler.record_function("forward_and_loss"):
                with self._amp_context():
                    logits, diagnostics = self.model(
                        **inputs, return_diagnostics=True
                    )
                    if self.config["model"].get("location_pooling", False):
                        self._location_diagnostics = {k: float(v.detach()) for k,v in diagnostics.items()}
                    loss, per_target = self.training_loss(logits, targets, mask)
            torch._assert_async(
                torch.isfinite(loss),
                f"non-finite loss at global step {self.global_step}",
            )
            with torch.profiler.record_function("backward"):
                loss.backward()
            grad_groups = group_gradient_norms(self.optimizer) if should_log else {}
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(optimizer_config["gradient_clip_norm"])
            )
            with torch.profiler.record_function("optimizer"):
                assert_controlled_backend()
                self.optimizer.step()
                self.scheduler.step()
            with torch.profiler.record_function("ema"):
                self.ema.update_parameters(self.model)
            if profiler is not None:
                profiler.step()
            if should_log and self.device.type == "cuda":
                torch.cuda.synchronize()
            step_wall = time.perf_counter() - timed_start
            batch_size = targets.shape[0]
            samples += batch_size
            loss_sum += float(loss.detach()) * batch_size
            per_target_sum += per_target.detach().double().cpu() * batch_size
            cache_hits += int(batch["stage_cache_hits"])
            cache_roles += int(batch["stage_cache_roles"])
            self.global_step += 1

            if should_log:
                per_target_values = per_target.detach().float().cpu().tolist()
                probability_values = torch.sigmoid(logits.detach().float()).mean(dim=0).cpu().tolist()
                target_values = targets.detach().mean(dim=0).cpu().tolist()
                memory = {
                    "allocated_gib": torch.cuda.memory_allocated() / 2**30 if self.device.type == "cuda" else 0,
                    "reserved_gib": torch.cuda.memory_reserved() / 2**30 if self.device.type == "cuda" else 0,
                    "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else 0,
                }
                event = {
                    "epoch": epoch,
                    "local_step": local_step,
                    "global_step": self.global_step,
                    "loss": float(loss.detach()),
                    "batch_size": batch_size,
                    "samples_per_second": batch_size / max(step_wall, 1e-9),
                    "data_wait_seconds": data_wait,
                    "step_seconds": step_wall,
                    "hard_fraction": float(batch["is_hard"].float().mean()),
                    "available_roles_mean": float(batch["available_roles"].float().mean()),
                    "stage_cache_hit_fraction": (
                        int(batch["stage_cache_hits"]) / max(1, int(batch["stage_cache_roles"]))
                    ),
                    "gradient_norm_total": float(total_grad_norm),
                    "gradient_norm_groups": grad_groups,
                    "learning_rates": learning_rates(self.optimizer),
                    "memory": memory,
                    "diagnostics": diagnostics,
                    "per_target_loss": {
                        label: per_target_values[index] for index, label in enumerate(LABELS)
                    },
                    "probability_mean": {
                        label: probability_values[index]
                        for index, label in enumerate(LABELS)
                    },
                    "target_mean": {
                        label: target_values[index] for index, label in enumerate(LABELS)
                    },
                }
                self.logger.event("train_step", **event)
                self.logger.info(
                    "%s e%02d %d/%d loss=%.5f %.2f study/s data=%.3fs step=%.3fs VRAM=%.2fGiB",
                    "train",
                    epoch + 1,
                    local_step + 1,
                    actual_steps,
                    event["loss"],
                    event["samples_per_second"],
                    data_wait,
                    step_wall,
                    memory["max_allocated_gib"],
                )
            last_end = time.perf_counter()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - epoch_start
        result = {
            "epoch": epoch,
            "samples": samples,
            "steps": min(actual_steps, local_step + 1),
            "loss": loss_sum / max(1, samples),
            "per_target_loss": {
                label: float(per_target_sum[index] / max(1, samples))
                for index, label in enumerate(LABELS)
            },
            "duration_seconds": duration,
            "samples_per_second": samples / max(duration, 1e-9),
            "data_wait_fraction": data_wait_total / max(duration, 1e-9),
            "stage_cache_hit_fraction": cache_hits / max(1, cache_roles),
            "gpu": _gpu_snapshot(),
        }
        count = max(1, self._ranking_samples)
        result.update(bce_loss=self._bce_loss_sum / count,
                      ranking_loss=self._ranking_loss_sum / count,
                      ranking_pair_counts=self._ranking_pairs.tolist())
        if self._location_rows:
            rows = self._location_rows
            scalar_keys = [k for k,v in rows[0].items() if isinstance(v,(int,float))]
            result['location_diagnostics'] = {k: sum(r[k] for r in rows)/len(rows) for k in scalar_keys}
            detail = {'steps': len(rows), **result['location_diagnostics']}
            for k in ('weak_bce_all12','rank_all12','main_contribution_all12','bce_plus_weighted_rank_all12'):
                detail[k] = [sum(r[k][t] for r in rows)/len(rows) for t in range(12)]
            self.logger.save_json(f'location_epoch_{epoch:02d}.json', detail)
            self._location_rows = []
        self.logger.event("train_epoch", **result)
        return result


    def close(self) -> None:
        self.logger.close()

