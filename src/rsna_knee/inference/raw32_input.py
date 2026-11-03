"""Frozen FastSub decode with native32 central selection and canonical geometry."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TIMM_USE_SCIPY_INTERP']='0'
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset
from . import raw_decode as P
from rsna_knee.model_input import central_band_slice_indices
from rsna_knee.box_geometry import source_to_canvas_affine
from rsna_knee.input_pipeline import _source_positions,_orientation_targets,canonicalize_inplane_orientation,physical_center_crop_resize
P.DEPTH=32;P.WINDOWS=30
LABELS=P.LABELS
MODEL_KEYS=('images','series_mask','slice_valid_mask','slice_repeated_mask','pixel_valid_mask','slice_positions_normalized','gap_previous_mm','gap_next_mm')
class Studies(P.Studies):
 def __init__(self,study_ids,series_rows,size=384):super().__init__(study_ids,series_rows);self.size=size
 def __getitem__(self,i):
  candidates,blobs=self._io(i);loaded={};failed=set()
  while True:
   chosen=P.select([c for c in candidates if c['uid'] not in failed]);pending=[c for c in chosen.values() if c['uid'] not in loaded]
   if not pending:break
   blobs={**blobs,**P.prefetch([f for c in pending for f in c['files'] if f not in blobs])}
   for c in pending:
    try:loaded[c['uid']]=P.load_series(c['files'],blobs);c['n']=len(loaded[c['uid']]['pixels'])
    except Exception as e:failed.add(c['uid']);print('SERIES_FALLBACK',self.study_ids[i],c['uid'],repr(e),flush=True)
  series={role:loaded[c['uid']] for role,c in chosen.items()};side=P.laterality(series.values());size=self.size
  out=dict(images=torch.zeros(5,32,size,size,dtype=torch.uint16),series_mask=torch.tensor([r in series for r in P.ROLES]),slice_valid_mask=torch.zeros(5,32,dtype=torch.bool),slice_repeated_mask=torch.zeros(5,32,dtype=torch.bool),pixel_valid_mask=torch.zeros(5,1,size,size,dtype=torch.bool),slice_positions_normalized=torch.zeros(5,32),gap_previous_mm=torch.zeros(5,32),gap_next_mm=torch.zeros(5,32))
  out['box_data']=[None]*5
  for k,role in enumerate(P.ROLES):
   if role not in series:continue
   s=series[role];_,_,direction=_orientation_targets(role,side);positions,source,_,order=_source_positions(s['source_metadata'],len(s['pixels']),direction);selected=central_band_slice_indices(positions,slices=32)
   raw=s['pixels'][order[selected['indices']]]
   if s['lut'] is not None:
    dtype,flip=P._SMALL_INT[raw.dtype];x=torch.from_numpy(s['lut'][raw.view(dtype)^dtype(flip) if flip else raw])
   else:x=torch.from_numpy(np.rint((np.clip(raw,s['lo'],s['hi'])-s['lo'])/(s['hi']-s['lo'])*65535).astype(np.float32)/65535)
   x,spacing,orientation=canonicalize_inplane_orientation(x,s['spacing'],dict(image_orientation_patient=None if s['iop'] is None else s['iop'].tolist(),orientation_consistent=True),role,side)
   x,pixel,crop=physical_center_crop_resize(x,spacing,140.,size)
   out['box_data'][k]={'M':source_to_canvas_affine({'orientation':orientation,'crop':crop},size),'cpu':np.eye(3),'source_shape':list(s['pixels'].shape[-2:]),'annotations':{}}
   out['images'][k]=torch.from_numpy(np.rint(x.numpy()*65535).astype(np.uint16));out['pixel_valid_mask'][k]=pixel;out['slice_valid_mask'][k]=True
   for key,skey in [('slice_positions_normalized','positions_normalized'),('slice_repeated_mask','repeated'),('gap_previous_mm','gap_previous_mm'),('gap_next_mm','gap_next_mm')]:out[key][k]=torch.from_numpy(selected[skey])
  assert out['series_mask'].any(),'no valid role for '+self.study_ids[i]
  return out

def inputs_on_device(batch,device):
 out={k:batch[k].to(device,non_blocking=True) for k in MODEL_KEYS};out['images']=out['images'].float().div_(65535.);out['box_data']=batch['box_data'];return out

def collate(samples):
 return {**{k:torch.stack([s[k] for s in samples]) for k in MODEL_KEYS},'box_data':[s['box_data'] for s in samples]}
