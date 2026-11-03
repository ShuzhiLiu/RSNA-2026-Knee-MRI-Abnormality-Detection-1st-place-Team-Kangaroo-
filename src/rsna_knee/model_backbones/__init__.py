"""Explicit per-model functions shared by the task model and trainer.

Add each reviewed backbone as its own file and register it below. Modules expose
load_pretrained_backbone, INPUT_MEAN/STD, DEEPEST_LAYER, pretrained_layer_id,
validate_location (returns square grid size), feature_map and pool. They return
the native backbone, never a wrapping nn.Module, preserving checkpoint key names.
Future tipsv2/dinov3_vit/dinov3_convnext/mambavision/transnext support must
be implemented and reviewed before registration; no fallback silently runs it.
"""
from importlib import import_module

_MODULES = {"maxxvit": ".maxxvit", "coatnet": ".coatnet"}
_NATIVE_ARCHITECTURES = {
    "timm:coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k": "coatnet",
    "timm:maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k": "maxxvit",
    "timm:maxxvitv2_nano_rw_256.sw_in1k": "maxxvit",
}


def get_backbone_module(name: str = "maxxvit"):
    try:
        module_path = _MODULES[name]
    except KeyError:
        raise ValueError(f"unimplemented backbone module: {name!r}") from None
    return import_module(module_path, __name__)


def load_backbone(checkpoint, *, architecture, backbone_module=None, **kwargs):
    if backbone_module is None:
        try:
            backbone_module = _NATIVE_ARCHITECTURES[architecture]
        except KeyError:
            raise ValueError(f"architecture requires an implemented explicit backbone_module: {architecture!r}") from None
    module = get_backbone_module(backbone_module)
    backbone, report = module.load_pretrained_backbone(checkpoint, architecture=architecture, **kwargs)
    backbone._rsna_backbone_module = backbone_module
    return backbone, report
