"""CoAtNet-rMLP-2 / 384 backbone module used by the final CoAtNet recipe.

Strict loading (approved architecture, 384 px, pinned pretrained SHA256), stage-wise LLRD layer ids,
pooling, input normalisation and per-block activation checkpointing with recompute-only BatchNorm
buffer clones. feature_map is a single forward_features call unless `_rsna_backbone_image_chunk`
is set; the final recipe sets chunk 32 (images per backbone call) to bound activation memory.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from . import maxxvit as native

INPUT_MEAN = native.INPUT_MEAN
INPUT_STD = native.INPUT_STD
DEEPEST_LAYER = native.DEEPEST_LAYER
pretrained_layer_id = native.pretrained_layer_id
pool = native.pool
_RECOMPUTE_DEPTH = ContextVar('coatnet_recompute_depth', default=0)
ARCHITECTURE = 'timm:coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
WEIGHT_SHA256 = '91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e'


@contextmanager
def _recompute_buffers(module):
    """Clone BN buffers at recompute entry, restore original references afterwards.

    Real forwards keep native BN statistics/update rules; only recomputation writes temporary buffers.
    """
    token = _RECOMPUTE_DEPTH.set(_RECOMPUTE_DEPTH.get() + 1)
    originals = []
    try:
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm) and child.track_running_stats:
                saved = {key: child._buffers[key] for key in ('running_mean', 'running_var', 'num_batches_tracked')}
                originals.append((child, saved))
                for key, value in saved.items():
                    child._buffers[key] = None if value is None else value.detach().clone()
        yield
    finally:
        for child, saved in reversed(originals):
            child._buffers.update(saved)
        _RECOMPUTE_DEPTH.reset(token)


def _checkpointed_stage_forward(self, values):
    if self.grad_checkpointing and self.training and torch.is_grad_enabled():
        for block in self.blocks:
            values = checkpoint(block, values, use_reentrant=False, preserve_rng_state=True,
                                context_fn=lambda block=block: (nullcontext(), _recompute_buffers(block)))
        return values
    return self.blocks(values)


def feature_map(backbone, values):
    chunk = getattr(backbone, '_rsna_backbone_image_chunk', None)
    if getattr(backbone, '_rsna_track_calls', False):
        backbone._rsna_chunk_calls.append(dict(shape=list(values.shape), chunk=chunk))
    if not chunk:
        return backbone.forward_features(values)
    training = backbone.training and torch.is_grad_enabled()
    maps = []
    for part in values.split(int(chunk), dim=0):
        if training:
            maps.append(checkpoint(backbone.forward_features, part, use_reentrant=False, preserve_rng_state=True,
                                   context_fn=lambda: (nullcontext(), _recompute_buffers(backbone))))
        else:
            maps.append(backbone.forward_features(part))
    return torch.cat(maps, dim=0)


def validate_location(backbone, image_size):
    if image_size != 384 or int(backbone.num_features) != 1024 or len(backbone.stages) != 4:
        raise ValueError('CoAtNet requires its native 1024-channel 12x12 map at 384')
    return native.validate_location(backbone, image_size)


def load_pretrained_backbone(checkpoint, *, architecture, image_size, expected_sha256=None, gradient_checkpointing=True):
    if architecture != ARCHITECTURE or image_size != 384 or expected_sha256 != WEIGHT_SHA256:
        raise ValueError('CoAtNet requires the approved architecture/384/pretrained SHA')
    if not gradient_checkpointing:
        raise ValueError('recipe requires native activation checkpointing ON')
    net, report = native.load_pretrained_backbone(checkpoint, architecture=architecture, image_size=image_size,
                                                  expected_sha256=WEIGHT_SHA256, gradient_checkpointing=gradient_checkpointing)
    for stage in net.stages:
        stage.forward = types.MethodType(_checkpointed_stage_forward, stage)
    validate_location(net, image_size)
    return net, report
