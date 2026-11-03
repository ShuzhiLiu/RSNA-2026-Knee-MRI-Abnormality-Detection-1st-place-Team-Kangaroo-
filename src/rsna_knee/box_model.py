"""Global-only main-connected12-map head and center-SOP lesion MIL."""
import math,time
from .polygon_fast import overlap as fast_overlap
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .location_pooling import LocationDenseWindowModel
from .model_backbones import get_backbone_module
from .box_data import norm_matrix

def rect(x0,y0,x1,y1):return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],np.float64)
def transform(p,M):return np.c_[p,np.ones(len(p))].dot(M.T)[:,:2]
def area(p):return abs(float(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1))))*.5 if len(p)>2 else 0.
def clip(poly,bound):
 """Sutherland-Hodgman; either polygon orientation accepted."""
 p=np.asarray(poly,dtype=np.float64)
 sign=np.sign(np.dot(bound[:,0],np.roll(bound[:,1],-1))-np.dot(bound[:,1],np.roll(bound[:,0],-1)))
 for a,b in zip(bound,np.roll(bound,-1,axis=0)):
  if not len(p):break
  delta=b-a;cross=lambda q:sign*(delta[0]*(q[1]-a[1])-delta[1]*(q[0]-a[0]))
  out=[]
  for u,v in zip(p,np.roll(p,-1,axis=0)):
   cu,cv=cross(u),cross(v);inu,inv=cu>=-1e-9,cv>=-1e-9
   if inu:out.append(u)
   if inu!=inv:out.append(u+(v-u)*(cu/(cu-cv)))
  p=np.asarray(out,dtype=np.float64).reshape(-1,2)
 return p

def union_rectangles(boxes):
 """Exact disjoint rectangular decomposition of a union in source coordinates."""
 if not boxes:return []
 xx=sorted({float(b[j]) for b in boxes for j in (0,2)});out=[]
 for x0,x1 in zip(xx,xx[1:]):
  yy=sorted((b[1],b[3]) for b in boxes if b[0]<x1 and b[2]>x0)
  merged=[]
  for lo,hi in yy:
   if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
   else:merged.append([lo,hi])
  out.extend(rect(x0,lo,x1,hi) for lo,hi in merged)
 return out

def oracle_overlap(polys,valid,G,size=256):
 """Exact continuous geometric overlap; no dilation or minimum-area threshold."""
 out=np.zeros((G,G),np.float64);s=size/G
 for p in polys:
  p=clip(p,valid)
  if len(p)<3:continue
  x0,y0=np.floor(p.min(0)/s).astype(int);x1,y1=np.floor(p.max(0)/s).astype(int)
  for y in range(max(0,y0),min(G-1,y1)+1):
   for x in range(max(0,x0),min(G-1,x1)+1):out[y,x]+=area(clip(p,rect(x*s,y*s,(x+1)*s,(y+1)*s)))/(s*s)
 return np.minimum(out,1).astype(np.float32)

overlap=fast_overlap

def lse(z,weights,dim=-1):
 w=weights.to(z.dtype);total=w.sum(dim);return torch.logsumexp(z+torch.where(w>0,w.log(),torch.full_like(w,-torch.inf)),dim)-total.clamp_min(1e-30).log()

def center_entry(item, target, j, valid, repeated, role_valid):
 if not role_valid or not bool(valid[j:j+3].all()) or bool(repeated[j:j+3].any()):return None
 indices=item['source_indices'][j:j+3]
 if len(set(indices))!=3 or any(i<0 for i in indices):return None
 a=item['annotations'].get(str(target))
 if a is None:return None
 if not a['present']:return (False,[])
 center=str(item['canonical_indices'][j+1]);rows=a['slices'].get(center,[])
 if not rows:return None
 sop=item['sops'][j+1]
 if not sop:return None
 if not all(v['sop']==sop and v['source_index']==indices[1] for v in rows):raise RuntimeError('center SOP/geometry mismatch')
 return (True,rows)

class BoxModel(LocationDenseWindowModel):
 def __init__(self,*args,residual_coefficient=.25,lesion_weight=.1,**kwargs):
  self.residual_coefficient=float(residual_coefficient);self.lesion_weight=float(lesion_weight)
  super().__init__(*args,**kwargs)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(20260823)
   self.box_norm=nn.LayerNorm(self.feature_dim);self.box_projection=nn.Conv2d(self.feature_dim,12,1)
  self.register_buffer('box_prior',torch.zeros(12));self.box_intervention='native';self.box_aux_enabled=True
 def initialize_head_bias(self,prevalence):
  super().initialize_head_bias(prevalence)
  with torch.no_grad():
   b=torch.logit(prevalence.clamp(1e-4,1-1e-4));self.box_prior.copy_(b);self.box_projection.bias.copy_(b)
 def evidence(self,x):return self.box_projection(self.box_norm(x.permute(0,2,3,1)).permute(0,3,1,2)).float()
 def box_forward(self,base,features,selected,grid,wv,images,series_mask,slice_valid,slice_repeated,data):
  host_started=time.perf_counter();assert data is not None
  overlap=oracle_overlap if getattr(self,'box_original_host',False) else fast_overlap
  wv=wv.detach().cpu();slice_valid=slice_valid.detach().cpu();slice_repeated=slice_repeated.detach().cpu();series_mask=series_mask.detach().cpu()
  selected=[v.detach().cpu() if isinstance(v,torch.Tensor) else v for v in selected]
  B=len(data);K=len(selected[0]);G=self.grid_size;dev=images.device;size=images.shape[-1]
  maps=self.evidence(features).reshape(B,5,K,12,G,G)
  thetas=self.augment.last_theta.detach().cpu().double().numpy() if self.training else np.tile(np.eye(3)[:2],(B*5*K,1,1))
  N=norm_matrix(size,size);invN=np.linalg.inv(N)
  global_weights=np.zeros((B,5,K,G,G),np.float32)
  lesion_weights=[];lesion_indices=[];lesion_labels=[];coverage=np.zeros(12,np.int64)
  terms=[[{} for _ in range(12)] for _ in range(B)]
  for i in range(B):
   for r,item in enumerate(data[i]):
    if item is None:continue
    assert not ({'native','centers','span'} & item.keys())
    M=item['M'];CPU=invN@item['cpu']@N;forward_cpu=np.linalg.inv(CPU)
    H,W=item['source_shape'];source_poly=rect(0,0,W,H)
    global_before=clip(transform(source_poly,M),rect(0,0,size,size))
    global_after=clip(transform(global_before,forward_cpu),rect(0,0,size,size))
    for k in range(K):
     if not bool(wv[i,r,k]):continue
     f=(i*5+r)*K+k;GPU=invN@np.vstack([thetas[f],[0,0,1]])@N;forward_gpu=np.linalg.inv(GPU)
     valid=clip(transform(global_after,forward_gpu),rect(0,0,size,size))
     weights=overlap([valid],rect(0,0,size,size),G,size=size)
     if weights.sum()<=0:continue
     global_weights[i,r,k]=weights
     if not self.box_aux_enabled:continue
     j=int(selected[r][k]);idx=item['source_indices'][j:j+3]
     for ts,a in item['annotations'].items():
      t=int(ts);entry=center_entry(item,t,j,slice_valid[i,r],slice_repeated[i,r],bool(series_mask[i,r]))
      if entry is None:continue
      present,annotations=entry
      polys=[transform(p,forward_gpu@forward_cpu@M) for p in union_rectangles([v['box'] for v in annotations])] if present else [valid]
      lw=overlap(polys,valid,G,size=size)
      if lw.sum()<=0:continue
      terms[i][t].setdefault((r,idx[1]),[]).append(len(lesion_weights))
      lesion_weights.append(lw);lesion_indices.append((i,r,k,t));lesion_labels.append(float(present));coverage[t]+=1
  # One geometry transfer, batched spatial/window reductions using continuous fractions.
  valid_windows=global_weights.sum((-1,-2))>0
  safe_weights=np.where(valid_windows[...,None,None],global_weights,1.)
  weights=torch.as_tensor(safe_weights,device=dev,dtype=torch.float32)
  scores=lse(maps.flatten(-2),weights.flatten(-2).unsqueeze(-2))
  role_i,role_r=np.nonzero(valid_windows.any(-1));assert len(role_i)>0
  role_window_mask=torch.as_tensor(valid_windows[role_i,role_r],device=dev)
  ri=torch.as_tensor(role_i,device=dev);rr=torch.as_tensor(role_r,device=dev)
  role_scores=torch.logsumexp(scores[ri,rr].masked_fill(~role_window_mask[...,None],-torch.inf),1)-role_window_mask.sum(1).float().log()[:,None]
  evidence=torch.stack([role_scores[torch.as_tensor(np.flatnonzero(role_i==i),device=dev)].mean(0) for i in range(B)])
  assert all((role_i==i).any() for i in range(B))
  residual=self.residual_coefficient*(evidence-self.box_prior)
  if self.box_intervention=='all_off':residual=residual*0
  target_losses=[]
  if lesion_weights:
   indices=torch.as_tensor(lesion_indices,device=dev).T
   logits=lse(maps[tuple(indices)].flatten(-2),torch.as_tensor(np.stack(lesion_weights),device=dev).flatten(-2))
   losses=F.binary_cross_entropy_with_logits(logits,torch.as_tensor(lesion_labels,device=dev),reduction='none')
   for t in range(12):
    study_losses=[]
    for i in range(B):
     if terms[i][t]:study_losses.append(torch.stack([losses[v].mean() for v in terms[i][t].values()]).mean())
    if study_losses:target_losses.append(torch.stack(study_losses).mean())
  self.box_loss=torch.stack(target_losses).mean() if target_losses else base.sum()*0
  coverage_tensor=torch.as_tensor(coverage,device=dev)
  self.box_last={'base_logits':base.detach(),'lesion_loss':self.box_loss.detach(),'counts':coverage_tensor,'global':evidence.detach(),'residual':residual.detach(),'backbone_local_inputs':0,'host_and_local_forward_seconds':time.perf_counter()-host_started}
  diag={'box_lesion':self.box_loss.detach(),'box_residual_abs':residual.detach().abs().mean(),'box_local_images':base.new_tensor(0)}
  diag.update({f'box_t{t}_entries':coverage_tensor[t] for t in range(12)})
  return base+residual,diag
 def auxiliary_loss(self,labels,step):
  old,diag=super().auxiliary_loss(labels,step)
  if not hasattr(self,'box_history'): self.box_history=[]
  self.box_history.append({'step':step,'lesion_loss':float(self.box_loss.detach()),'counts':self.box_last['counts'].cpu().tolist()})
  lesion=self.box_loss;diag['lesion']=lesion.detach()
  self.aux_target_diagnostics['weighted_lesion']=float(diag['lambda'])*self.lesion_weight*float(lesion.detach())
  return old+self.lesion_weight*float(diag['lambda'])*lesion,diag
