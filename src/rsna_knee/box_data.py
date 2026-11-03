"""Train-only lesion labels and annotation-free continuous inference geometry."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .location_data import LocationDataset, collate_location
from .model_input import Depth18TrainingDataset
from .training_data import collate_baseline_training
from .box_geometry import source_to_canvas_affine

ROLES = ['SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS', 'SAG_STRUCT', 'COR_STRUCT']


def norm_matrix(h, w):
    return np.array([[2/w, 0, -1], [0, 2/h, -1], [0, 0, 1]], np.float64)


def grid_matrix(grid):
    g = grid.detach().cpu().double().numpy(); h, w = g.shape[:2]
    dx = (g[h//2, -1]-g[h//2, 0])/(2*(w-1)/w)
    dy = (g[-1, w//2]-g[0, w//2])/(2*(h-1)/h)
    xy = np.array([2*(w//2+.5)/w-1, 2*(h//2+.5)/h-1])
    c = g[h//2, w//2]-dx*xy[0]-dy*xy[1]
    return np.array([[dx[0], dy[0], c[0]], [dx[1], dy[1], c[1]], [0, 0, 1]])


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_train_index(index, manifest, rows):
    """Require exactly the loader's weak train complement, never infer it from boxes."""
    weak = {str(r['Sample ID']) for r in rows if int(r['fold']) == -1}
    ids = [str(r['Sample ID']) for r in rows]
    if len(ids) != len(set(ids)) or weak != set(manifest['weak_train_ids']):
        raise ValueError('lesion index train complement differs from loader')
    if not set(index).issubset(weak):
        raise ValueError('non-training or hard study entered lesion supervision')
    if manifest['target_order'] != list(__import__('rsna_knee.input_pipeline', fromlist=['LABELS']).LABELS):
        raise ValueError('lesion index target order mismatch')


class BoxDataset(LocationDataset):
    def __init__(self, *args, lookup_run=None, box_index=None, index_manifest=None,
                 index_sha256=None, manifest_sha256=None, **kwargs):
        training = kwargs.get('strong_augmentation') is not None
        if training:
            super().__init__(*args, lookup_run=lookup_run, **kwargs)
            if not index_manifest or not index_sha256 or not manifest_sha256:
                raise ValueError('training lesion index requires immutable provenance')
            if file_sha(box_index) != index_sha256 or file_sha(index_manifest) != manifest_sha256:
                raise ValueError('lesion index/provenance hash mismatch')
            self.box_index = json.loads(Path(box_index).read_text())
            manifest = json.loads(Path(index_manifest).read_text())
            if manifest['index_sha256'] != index_sha256:
                raise ValueError('lesion manifest does not bind the index')
            validate_train_index(self.box_index, manifest, self.dataset.rows)
        else:
            # Do not open anatomical sidecars, lesion indices or their manifests at evaluation.
            Depth18TrainingDataset.__init__(self, *args, **kwargs)
            self.capture_location = True
            self.box_index = {}
        self.box_training = training

    def attach_location(self, result, sample, model_input):
        if self.box_training:
            result = super().attach_location(result, sample, model_input)
        uid = result['sample_id']; items = []; size = result['images'].shape[-1]
        for ri, audit in enumerate(sample['provenance']['roles']):
            if not audit.get('available'):
                items.append(None); continue
            role = ROLES[ri]
            sm, _ = self.dataset._read_metadata(self.dataset.cache_root / audit['metadata_path'])
            stage, _ = self.dataset._read_metadata(Path(audit['deterministic_stage_cache']['directory']) / 'metadata.json')
            if sm['series_instance_uid'] != audit['series_uid'] or sm['study_instance_uid'] != uid:
                raise RuntimeError('lesion input study/series identity mismatch')
            if file_sha(self.dataset.cache_root / audit['metadata_path']) != stage['source_metadata_sha256']:
                raise RuntimeError('lesion geometry source hash mismatch')
            ci = model_input['canonical_indices'][ri].numpy()
            ix = np.asarray(stage['source_order'], dtype=np.int64)[ci]
            if np.any(ci < 0) or np.any(ix < 0):
                raise RuntimeError('lesion geometry has invalid source indices')
            if self.box_training and not np.array_equal(ix, result['source_indices'][ri].numpy()):
                raise RuntimeError('anatomical and lesion source ordering differ')
            cpu = grid_matrix(model_input['cpu_grids'][ri]) if ri in model_input.get('cpu_grids', {}) else np.eye(3)
            ann = self.box_index.get(uid, {}).get(role, {})
            if ann and (ann['series_uid'] != audit['series_uid'] or ann['source_metadata_sha256'] != stage['source_metadata_sha256']):
                raise RuntimeError('lesion annotation source identity mismatch')
            items.append({'M': source_to_canvas_affine(stage, size), 'cpu': cpu,
                          'annotations': ann.get('annotations', {}), 'source_indices': ix.tolist(),
                          'canonical_indices': ci.tolist(), 'sops': [Path(sm['source_files'][i]).stem for i in ix],
                          'source_shape': list(sm['shape'][-2:])})
        result['box_data'] = items
        return result


def collate_box(samples):
    collate = collate_location if 'anatomy_states' in samples[0] else collate_baseline_training
    out = collate(samples)
    out['box_data'] = [s['box_data'] for s in samples]
    return out
