"""v002 source qualification; does not change original renderer or viewer."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jsonschema
from PIL import Image
from . import geometry
from . import viewer
from . import coordinates
from .common import dump, read, sha

def qualify(records,meta,uid,sid):
 assert len(records)==meta['slice_count']==meta['shape'][0]
 names=[r.path.name for r in records]
 assert len(set(names))==len(names) and names==meta['source_files'],'duplicate/misordered source names'
 evidence=[];sops=[]
 for i,r in enumerate(records):
  ds=r.dataset
  assert str(ds.StudyInstanceUID)==uid and str(ds.SeriesInstanceUID)==sid,'source identity'
  sops.append(str(ds.SOPInstanceUID));assert sops[-1],'empty SOP identity'
  assert [int(ds.Rows),int(ds.Columns)]==meta['shape'][1:],'dimensions'
  ori=np.asarray(ds.ImageOrientationPatient,float);pos=np.asarray(ds.ImagePositionPatient,float);sp=np.asarray(ds.PixelSpacing,float)
  assert ori.shape==(6,) and np.isfinite(ori).all(),'invalid IOP'
  assert pos.shape==(3,) and np.isfinite(pos).all(),'invalid IPP'
  assert sp.shape==(2,) and np.isfinite(sp).all() and np.all(sp>0),'invalid spacing'
  assert abs(np.linalg.norm(ori[:3])-1)<1e-3 and abs(np.linalg.norm(ori[3:])-1)<1e-3 and abs(ori[:3]@ori[3:])<1e-3,'non-unit/nonorthogonal IOP'
  assert np.allclose(sp,meta['pixel_spacing'],atol=1e-8,rtol=0),'renderer-inconsistent spacing'
  evidence.append({'native_index':i,'source_file':r.path.name,'sop_instance_uid':sops[-1],'study_uid':uid,'series_uid':sid,'rows':int(ds.Rows),'columns':int(ds.Columns),'iop':ori.tolist(),'ipp':pos.tolist(),'pixel_spacing':sp.tolist()})
 assert len(set(sops))==len(sops),'duplicate SOP identity'
 consistent=all(np.allclose(e['iop'],evidence[0]['iop'],atol=1e-5,rtol=0) for e in evidence)
 assert bool(meta['orientation_consistent'])==consistent,'common orientation flag mismatch'
 if consistent:
  row,col,positions,spacing,normal=geometry.frame(meta)
  assert np.allclose(positions,[e['ipp'] for e in evidence],atol=0,rtol=0),'frame position mismatch'
  assert np.allclose(np.r_[row,col],evidence[0]['iop'],atol=0,rtol=0),'frame orientation mismatch'
  gaps=np.diff(positions@normal);assert np.all(gaps>0),'non-increasing common-frame positions'
  gap=float(gaps.min()) if len(gaps) else None
 else:
  try:geometry.frame(meta)
  except AssertionError:pass
  else:raise AssertionError('varying-direction frame accepted')
  gap=None
 return {'native2D_integrity':True,'geometry_eligible':consistent,'capability':'common_frame' if consistent else 'native_independent_only','reason':'consistent complete geometry' if consistent else 'valid per-slice direction variation; no physical continuity claim','projected_slice_gap_mm_min':gap},evidence

def permissions(case,caps):
 eligible=viewer.eligible_keys(case);view,compare=viewer.tool_schemas(case)
 assert len(eligible)>=2,'fewer than two comparison sources'
 for s in case['series']:
  key=s['key'];flag=caps[key]['geometry_eligible']
  assert key in view['properties']['series']['enum']
  assert (key in eligible)==flag
  assert (key in compare['properties']['reference']['enum'])==flag
  assert (key in compare['properties']['targets']['items']['enum'])==flag
 return {'independent_view_keys':view['properties']['series']['enum'],'comparison_reference_keys':eligible,'comparison_target_keys':compare['properties']['targets']['items']['enum']}

def synthetic(run):
 folder=run/'synthetic-contract';folder.mkdir()
 def fixture(key,var):
  v=np.arange(3*40*60,dtype=np.uint16).reshape(3,40,60)*9
  path=folder/(key+'.npy');np.save(path,v)
  oris=[[1.,0.,0.,0.,1.,0.] for _ in range(3)]
  if var:oris[1]=[0.,1.,0.,-1.,0.,0.]
  records=[SimpleNamespace(path=Path(f'{i}.dcm'),dataset=SimpleNamespace(StudyInstanceUID='SYN',SeriesInstanceUID=key,SOPInstanceUID=f'{key}.{i}',Rows=40,Columns=60,ImageOrientationPatient=o,ImagePositionPatient=[0.,0.,float(i)*2],PixelSpacing=[.6,.4])) for i,o in enumerate(oris)]
  meta={'shape':[3,40,60],'slice_count':3,'source_files':[r.path.name for r in records],'orientation_consistent':not var,'image_orientation_patient':oris[0],'image_positions_patient':[[0.,0.,float(i)*2] for i in range(3)],'pixel_spacing':[.6,.4]}
  mp=folder/(key+'.json');dump(mp,meta)
  series={'key':key,'series_uid':key,'plane':'Synthetic','description':'synthetic','slice_count':3,'shape':[3,40,60],'pixel_spacing':[.6,.4],'volume_path':str(path),'volume_sha256':sha(path),'metadata_path':str(mp),'metadata_sha256':sha(mp)}
  return records,meta,series
 fixtures=[fixture('VAR',True),fixture('COMMON1',False),fixture('COMMON2',False)];caps={};errors=[]
 for rec,m,s in fixtures:caps[s['key']],_=qualify(rec,m,'SYN',s['key'])
 perms=permissions({'series':[s for _,_,s in fixtures]},caps)
 rec,m,s=fixtures[0]
 negatives={
 'missing_iop':lambda r,m:delattr(r[0].dataset,'ImageOrientationPatient'),
 'missing_ipp':lambda r,m:delattr(r[0].dataset,'ImagePositionPatient'),
 'nonfinite_iop':lambda r,m:setattr(r[0].dataset,'ImageOrientationPatient',[float('nan'),0,0,0,1,0]),
 'nonfinite_ipp':lambda r,m:setattr(r[0].dataset,'ImagePositionPatient',[0,float('inf'),0]),
 'nonorthogonal':lambda r,m:setattr(r[0].dataset,'ImageOrientationPatient',[1,0,0,1,0,0]),
 'nonnormalized':lambda r,m:setattr(r[0].dataset,'ImageOrientationPatient',[2,0,0,0,1,0]),
 'illegal_spacing':lambda r,m:setattr(r[0].dataset,'PixelSpacing',[0,.4]),
 'renderer_spacing':lambda r,m:setattr(r[0].dataset,'PixelSpacing',[.7,.4]),
 'wrong_dimensions':lambda r,m:setattr(r[0].dataset,'Rows',41),
 'wrong_identity':lambda r,m:setattr(r[0].dataset,'SeriesInstanceUID','OTHER'),
 'duplicate_file':lambda r,m:setattr(r[1],'path',r[0].path),
 'duplicate_sop':lambda r,m:setattr(r[1].dataset,'SOPInstanceUID',r[0].dataset.SOPInstanceUID),
 }
 for name,mutate in negatives.items():
  rr,mm=copy.deepcopy(rec),copy.deepcopy(m);mutate(rr,mm)
  try:qualify(rr,mm,'SYN','VAR')
  except (AssertionError,AttributeError,TypeError,ValueError):errors.append(name)
  else:raise AssertionError('negative accepted: '+name)
 vs,cs=viewer.tool_schemas({'series':[q for _,_,q in fixtures]})
 for role,args in [('reference',{'reference':'VAR','targets':['COMMON1']}),('target',{'reference':'COMMON1','targets':['VAR']})]:
  args.update(index=1,point_xy=[.5,.5],diameter_mm=40,question='synthetic')
  try:jsonschema.validate(args,cs)
  except jsonschema.ValidationError:errors.append('compare_'+role+'_rejected')
  else:raise AssertionError('independent-only source admitted to compare')
 # Original renderer, isolated single slice; independently reconstruct image interior.
 maxerr=0
 for i in range(3):
  im,info=viewer.render(s,[i],[0,0,1,1]);im.save(folder/f'VAR_slice{i}.png')
  meta=coordinates.manifest(s,[i],[0,0,1,1]);cell=meta['cells'][0];l,t,r,b=cell['image_rect_xyxy']
  pixels=np.rint(np.load(s['volume_path'])[i].astype(np.float32)/257).astype(np.uint8)
  expected=Image.fromarray(pixels).convert('RGB').resize((r-l,b-t),Image.Resampling.LANCZOS)
  assert np.array_equal(np.asarray(im.crop((l,t,r,b))),np.asarray(expected)),'renderer pixel replay'
  native=[7.25,9.5,37.5,29.25];xy=coordinates.to_sheet(native,cell,meta['sheet_wh'])
  back=coordinates.to_native({'xyxy':xy,'series_key':'VAR','native_index':i,'channel':'original'},meta)
  maxerr=max(maxerr,float(np.abs(np.array(native)-back).max()))
 assert maxerr<=1
 out={'status':'PASS','qualification':caps,'permissions':perms,'negative_controls_rejected':errors,'pixel_replay_exact':True,'max_native_roundtrip_error_pixels':maxerr,'model_calls':0,'synthetic_only':True}
 dump(run/'native-view-contract-tests.json',out);return out
