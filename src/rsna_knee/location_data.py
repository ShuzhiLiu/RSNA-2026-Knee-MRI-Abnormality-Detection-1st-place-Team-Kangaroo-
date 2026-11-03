"""Training-only anatomical lookup with exact depth selection and affine grids."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF
from torchvision.transforms import _functional_tensor as TVFT
from .model_input import Depth18TrainingDataset
from .training_data import collate_baseline_training

def cpu_affine_grid(shape, draws):
    """Recreate torchvision's tensor affine grid from its actual sampled parameters.

    This performs no random draw and uses the same pinned torchvision primitive
    as affine(), without intercepting any process-global function.
    """
    height, width = shape
    matrix = TVF._get_inverse_affine_matrix([0.0, 0.0], draws['angle_degrees'], [int(draws['translate_x_pixels']), int(draws['translate_y_pixels'])], draws['scale'], [0.0, 0.0])
    theta = torch.tensor(matrix, dtype=torch.float32).reshape(1, 2, 3)
    return TVFT._gen_affine_grid(theta, w=width, h=height, ow=width, oh=height)[0]

class LocationDataset(Depth18TrainingDataset):

    def __init__(self, *args, lookup_run, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_location = True
        self.lookup_run = Path(lookup_run)
        self.lookup = json.loads((self.lookup_run / 'lookup_manifest.json').read_text())['index']
        self.cache = {}

    def attach_location(self, result, sample, model_input):
        height, width = result['images'].shape[-2:]
        identity = F.affine_grid(torch.eye(2, 3)[None], (1, 1, height, width), align_corners=False)[0]
        grids = identity[None].repeat(5, 1, 1, 1)
        indices = model_input['canonical_indices']
        source = torch.full_like(indices, -1)
        image_ids = torch.full_like(indices, -1)
        depth = int(indices.shape[1])
        states = torch.full((5, depth, 6), -1, dtype=torch.int8)
        boxes = torch.full((5, depth, 6, 4), float('nan'))
        for role_index, audit in enumerate(sample['provenance']['roles']):
            if not audit.get('available'):
                continue
            entry = self.lookup[result['sample_id']][audit['role']]
            if audit['series_uid'] != entry['series_uid']:
                raise RuntimeError('Annotation and input series identity differ')
            key = (result['sample_id'], audit['role'])
            if key not in self.cache:
                path = self.lookup_run / entry['path']
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != entry['sha256']:
                    raise RuntimeError('Anatomical sidecar hash differs from immutable manifest')
                with np.load(path) as values:
                    self.cache[key] = {k: values[k] for k in values.files}
                if len(self.cache) > 32:
                    self.cache.pop(next(iter(self.cache)))
            values = self.cache[key]
            selected = indices[role_index].numpy()
            source[role_index] = torch.from_numpy(values['source_indices'][selected])
            image_ids[role_index] = torch.from_numpy(values['image_ids'][selected])
            states[role_index] = torch.from_numpy(values['states'][selected])
            boxes[role_index] = torch.from_numpy(values['boxes'][selected])
            repeated = np.bincount(selected, minlength=entry['slices'])[selected] > 1
            if not np.array_equal(repeated, result['slice_repeated_mask'][role_index].numpy()):
                raise RuntimeError('Annotation and repeated-slice identity differ')
            if role_index in model_input.get('cpu_grids', {}):
                grids[role_index] = model_input['cpu_grids'][role_index]
        result.update(canonical_indices=indices, source_indices=source, image_ids=image_ids, anatomy_states=states, anatomy_boxes=boxes, cpu_grids=grids)
        return result

def collate_location(samples):
    result = collate_baseline_training(samples)
    for key in ('canonical_indices', 'source_indices', 'image_ids', 'anatomy_states', 'anatomy_boxes', 'cpu_grids'):
        result[key] = torch.stack([sample[key] for sample in samples])
    return result
