"""Final three-model recipe validation, independent of research/deployment locks."""
from __future__ import annotations
import json
from pathlib import Path
from .final_preparation import sha

FULL_PROTOCOL = 'full-data-s4349-h58-no-heldout-v1'
BACKBONES = {
    'timm:maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k': 'base384',
    'timm:coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k': 'coat384',
}


def validate_recipe(config, root):
    if config.get('evaluation_protocol') != FULL_PROTOCOL:
        raise ValueError('final trainer accepts only fixed full-data fitting')
    model = config['model']
    if model['backbone'] not in BACKBONES:
        raise ValueError('not one of the final two backbones')
    expected = dict(input_shape=[5,32,384,384], role_slices=[32]*5, targets=12,
                    roles=5, triplet_count=30, train_windows_per_role=8,
                    triplet_depth=3, window_stride=1, location_pooling=True,
                    replacement_probability=0.0, gradient_checkpointing=True,
                    fusion_hidden=256, head_hidden=256, depth_band=[.12,.88])
    if any(model.get(k) != v for k,v in expected.items()):
        raise ValueError('final model/input/auxiliary architecture changed')
    if 'evidence_descriptors' in config or 'role_dropout_policy' in config.get('train_input_augmentation',{}):
        raise ValueError('descriptor and alternate role-drop experiments are not final recipes')
    pipeline = json.loads((Path(root)/config['input_config']).read_text())
    if pipeline['image_size'] != 384 or pipeline['slices_per_series'] != 16 or pipeline['canvas_mm'] != 140:
        raise ValueError('final canonical spatial input changed')
    view = pipeline['derived_model_view']
    if any(view.get(k) != v for k,v in dict(depth=32, band=[.12,.88], triplet_depth=3,
           stride=1, train_windows=8, eval_windows=30).items()):
        raise ValueError('final depth/window view changed')
    if pipeline['augmentation']['role_dropout_probability'] != .04:
        raise ValueError('final recipe retains original CPU role dropout .04')
    if config.get('release_format') != 'final-solution-v1' or pipeline.get('supervision_schema') != 'final-numeric-v1':
        raise ValueError('use a workspace created by `final_solution.py prepare` (configs/ holds templates only)')
    location = config['location_supervision']
    expected = dict(mil_backbone_gradient='connected', spatial_hinge_tau=.7,
        negative_mass_tau=.1, aux_weights=dict(mil=.1,location=.1,window=.1), annotation_inference_required=False)
    if any(location.get(k) != v for k,v in expected.items()):
        raise ValueError('final anatomical loss semantics changed')
    expected = dict(residual_coefficient=.25, lesion_weight=.1, global_only=True,
                    annotation_inference_required=False, case_positive_threshold=.5, target_count=12)
    if any(config['boxcombo'].get(k) != v for k,v in expected.items()):
        raise ValueError('final lesion evidence semantics changed')
    for key in ('index','index_manifest','index_sha256','index_manifest_sha256'):
        if not config['boxcombo'].get(key):
            raise ValueError('lesion supervision requires explicit provenance')
    return BACKBONES[model['backbone']]


def verify_location_assets(root, config):
    location = config['location_supervision']
    for path, expected in ((Path(root)/location['lookup_run']/'lookup_manifest.json',location['lookup_manifest_sha256']),
                           (Path(root)/location['usage_map'],location['usage_map_sha256'])):
        if sha(path) != expected:
            raise ValueError(f'changed spatial annotation manifest: {path}')
