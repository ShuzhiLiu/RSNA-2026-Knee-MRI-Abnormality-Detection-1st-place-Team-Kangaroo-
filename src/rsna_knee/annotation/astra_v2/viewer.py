"""Read-only per-case MRI MCP viewer. Only manifest-selected arrays are accessible."""
import argparse,base64,hashlib,io,json,math,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def intensity(v,window):
    lo,hi=window
    return np.rint(np.clip((v.astype(np.float32)/65535-lo)/(hi-lo),0,1)*255).astype(np.uint8)

def render(series,indices,box,tile=640,window=None):
    v=np.load(series['volume_path'],mmap_mode='r');assert v.dtype==np.uint16
    h,w=v.shape[1:];x0,y0,x1,y1=box
    left,top=max(0,math.floor(x0*w)),max(0,math.floor(y0*h));right,bottom=min(w,math.ceil(x1*w)),min(h,math.ceil(y1*h))
    assert right>left and bottom>top
    spacing=series['pixel_spacing']or[1,1];paired=window is not None
    cols=2 if paired or len(indices)>1 else 1;rows=len(indices)if paired else math.ceil(len(indices)/cols)
    sheet=Image.new('RGB',(cols*tile,rows*(tile+36)),'black');draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=20)
    windows=[[0,1],window]if paired else [[0,1]]
    for j,idx in enumerate(indices):
      for k,win in enumerate(windows):
        raw=v[idx,top:bottom,left:right]
        # Preserve the exact S3/S4 default rounding, separately window the uint16 source.
        pixels=np.rint(raw.astype(np.float32)/257).astype(np.uint8)if k==0 else intensity(raw,win)
        im=Image.fromarray(pixels).convert('RGB')
        ph,pw=(bottom-top)*spacing[0],(right-left)*spacing[1];scale=min(tile/pw,tile/ph);nw,nh=max(1,round(pw*scale)),max(1,round(ph*scale));im=im.resize((nw,nh),Image.Resampling.LANCZOS)
        xx,yy=(k*tile,j*(tile+36))if paired else((j%cols)*tile,(j//cols)*(tile+36))
        sheet.paste(im,(xx+(tile-nw)//2,yy+36+(tile-nh)//2));label='original'if k==0 else f'window {win[0]:.2f}..{win[1]:.2f}'
        title=f'{series["key"]} {idx} | {label}'if paired else f'{series["key"]} slice {idx}'
        draw.text((xx+6,yy+4),title,font=font,fill='white')
    return sheet,{'native_box_pixels':[left,top,right,bottom],'normalized_box':box,'indices':indices,'physical_aspect_preserved':True,'source_volume_sha256':series['volume_sha256'],'window':window,'original_always_included':True,'display_layout':'one slice per row: original left, adjusted right'if paired else'original only','window_domain':'canonical uint16 / 65535; already clipped at series preprocessing percentiles'}

from . import geometry
from . import coordinates

def metadata(s):
 assert sha(s['metadata_path'])==s['metadata_sha256'],'metadata source changed'
 return json.loads(Path(s['metadata_path']).read_text())

def eligible_keys(case):
 keys=[]
 for s in case['series']:
  try:geometry.frame(metadata(s));keys.append(s['key'])
  except (AssertionError,KeyError,TypeError,ValueError):pass
 return keys

def compare_native_specs(case,args):
 series={s['key']:s for s in case['series']};ref=args['reference'];targets=args['targets'];assert ref in series and 1<=len(targets)<=3 and len(set(targets))==len(targets)and ref not in targets and all(k in series for k in targets)
 assert 40<=args['diameter_mm']<=100
 point=geometry.anchor(metadata(series[ref]),args['index'],args['point_xy']);specs=[];unavailable=[]
 for key in [ref,*targets]:
  try:
   info=geometry.neighborhood(metadata(series[key]),point,args['diameter_mm'])
   native={'series':key,'indices':info['indices'],'box':info['normalized_box'],'question':args['question']}
   assert native['box'][2]-native['box'][0]>=.2-1e-9 and native['box'][3]-native['box'][1]>=.2-1e-9
   specs.append({'arguments':native,'correspondence':info})
  except (AssertionError,KeyError,TypeError,ValueError)as e:unavailable.append({'series':key,'reason':str(e)})
 assert any(s['arguments']['series']==ref for s in specs)and len(specs)>=2,'fewer than two covered native series at this point; choose another point/pair or use independent views'
 return specs,unavailable,point.tolist()

def tool_schemas(case):
 keys=[s['key']for s in case['series']];eligible=eligible_keys(case)
 view={'type':'object','properties':{'series':{'type':'string','enum':keys},'indices':{'type':'array','items':{'type':'integer'},'minItems':1,'maxItems':4},'box':{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4},'question':{'type':'string'}},'required':['series','indices','box','question'],'additionalProperties':False}
 cmp={'type':'object','properties':{'reference':{'type':'string','enum':eligible},'index':{'type':'integer','minimum':0},'point_xy':{'type':'array','items':{'type':'number','minimum':0,'maximum':1},'minItems':2,'maxItems':2},'targets':{'type':'array','items':{'type':'string','enum':eligible},'minItems':1,'maxItems':3,'uniqueItems':True},'diameter_mm':{'type':'number','minimum':40,'maximum':100},'question':{'type':'string'}},'required':['reference','index','point_xy','targets','diameter_mm','question'],'additionalProperties':False}
 view['properties']['window']={'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}
 return view,cmp

def main():
 import jsonschema
 p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--log',type=Path,required=True);a=p.parse_args();case=json.loads(a.case.read_text());series={s['key']:s for s in case['series']};a.log.mkdir(parents=True,exist_ok=True);count=0;call_count=0;vs,cs=tool_schemas(case)
 annotation={'readOnlyHint':True,'destructiveHint':False,'openWorldHint':False}
 descriptions={'view_slices':'Optionally set window=[low,high] within0..1, minimum width0.05. Original is always shown left beside adjusted right, one slice per row; omit window for unchanged original only. Windowing cannot recover preprocessing-clipped intensity or establish enhancement. Inspect1–4 exact native slices with rectangular zoom; normalized box uses full image edges, minimum span0.20. All selected series remain independently accessible. Budget10 rendered VIEW sheets total and10 tool calls. Each sheet has one series.','compare_location':'Show candidate corresponding native neighborhoods in a reference series and1–3 target series. Choose a point on a REFERENCE native slice using normalized point_xy=[x/(width-1),y/(height-1)], x right/y down. Returns each series as its own VIEW with central and neighboring native slices. diameter_mm40–100 sets local physical context (expanded if necessary to retain minimum0.20 native frame span). Uses metadata only, NOT verified registration: examine matching anatomy and plane distances, do not treat a coordinate match as diagnostic proof. Frame identifiers differ. Uncovered targets are reported; use independent view_slices where geometry is unreliable. Each returned series consumes one of10 VIEW sheets; at most10 tool calls total. Compare fluid and structural tissue when available. Cite each returned VIEW/series/native indices.'}
 tools=[{'name':name,'description':descriptions[name],'inputSchema':schema,'annotations':annotation}for name,schema in [('view_slices',vs),('compare_location',cs)]]
 for line in sys.stdin:
  try:
   req=json.loads(line);method=req.get('method');rid=req.get('id')
   if rid is None:continue
   if method=='initialize':result={'protocolVersion':req['params'].get('protocolVersion','2024-11-05'),'capabilities':{'tools':{'listChanged':False}},'serverInfo':{'name':'candidate-correspondence-mri-viewer','version':'3.0.0'}}
   elif method=='tools/list':result={'tools':tools}
   elif method in ['ping','resources/list','prompts/list']:result={'resources':[]}if method=='resources/list'else({'prompts':[]}if method=='prompts/list'else{})
   elif method=='tools/call':
    params=req['params'];name=params.get('name');args=params.get('arguments',{});call_count+=1;call={'call':call_count,'tool':name,'arguments':args,'views':[],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
     assert call_count<=10,'max10 logical tool calls'
     if name=='view_slices':jsonschema.validate(args,vs);specs=[{'arguments':args}];unavailable=[];point=None
     elif name=='compare_location':jsonschema.validate(args,cs);specs,unavailable,point=compare_native_specs(case,args)
     else:raise AssertionError('only bounded MRI tools are available')
     assert count+len(specs)<=10,f'insufficient VIEW-sheet budget; remaining{10-count}'
     for spec in specs:
      b=spec['arguments'];s=series[b['series']];indices=b['indices'];box=b['box'];assert len(set(indices))==len(indices)and all(type(i)is int and 0<=i<s['slice_count']for i in indices);assert all(math.isfinite(x)and 0<=x<=1 for x in box)and box[2]-box[0]>=.2-1e-9 and box[3]-box[1]>=.2-1e-9;assert sha(s['volume_path'])==s['volume_sha256'],'volume source changed'
     for spec in specs:
      win=spec['arguments'].get('window');assert win is None or(all(math.isfinite(x)and 0<=x<=1 for x in win)and win[1]-win[0]>=.05-1e-9),'invalid display window'
     content=[]
     if name=='compare_location':content.append({'type':'text','text':json.dumps({'call':call_count,'candidate_anchor_lps_mm':point,'unavailable':unavailable,'registration_verified':False,'notice':'Metadata-only candidate correspondence; inspect matching anatomy and neighboring slices. Source pixels are native, unregistered and unresampled. Each following series has its own VIEW number.'})})
     for spec in specs:
      count+=1;b=spec['arguments'];s=series[b['series']];im,info=render(s,b['indices'],b['box'],window=b.get('window'));path=a.log/f'view_{count:02d}.png';im.save(path);payload=path.read_bytes();entry={'view':count,'call':call_count,'origin_tool':name,'arguments':b,'status':'ok','image_path':str(path),'image_sha256':hashlib.sha256(payload).hexdigest(),**info}
      entry['coordinate_manifest']=coordinates.manifest(s,b['indices'],b['box'],b.get('window'))
      if 'correspondence'in spec:entry['correspondence']=spec['correspondence']
      (a.log/f'view_{count:02d}.json').write_text(json.dumps(entry,indent=2)+'\n');call['views'].append(count);content.extend([{'type':'text','text':json.dumps({'view':count,'series':s['key'],'plane':s['plane'],'sequence':s['description'],**info,**({'correspondence':spec['correspondence']}if 'correspondence'in spec else{})})},{'type':'image','mimeType':'image/png','data':base64.b64encode(payload).decode(),'_meta':{'codex/imageDetail':'original'}}])
     call['emitted_metadata']=[c['text'] for c in content if c['type']=='text'];call['emitted_image_sha256']=[hashlib.sha256(base64.b64decode(c['data'])).hexdigest() for c in content if c['type']=='image']
     call.update({'status':'ok','unavailable':unavailable,'candidate_anchor_lps_mm':point});result={'content':content,'isError':False}
    except Exception as e:call.update({'status':'error','error':str(e)});result={'content':[{'type':'text','text':f'MRI request rejected: {e}. Remaining VIEW sheets:{10-count}; remaining tool calls:{max(0,10-call_count)}. Independent view_slices remains available.'}],'isError':True}
    (a.log/f'call_{call_count:02d}.json').write_text(json.dumps(call,indent=2)+'\n')
   else:
    print(json.dumps({'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Method not found'}}),flush=True);continue
   print(json.dumps({'jsonrpc':'2.0','id':rid,'result':result}),flush=True)
  except Exception as e:print(json.dumps({'jsonrpc':'2.0','id':req.get('id')if'req'in locals()else None,'error':{'code':-32603,'message':str(e)}}),flush=True)
if __name__=='__main__':main()
