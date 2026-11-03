import json,math,io,hashlib
import jsonschema
from . import acquire_audit
from . import viewer
from . import coordinates as co
from . import transport
from .common import read, dump, sha
from .definitions import LABELS

def audit(study,d,source,sentinel=False):
 raw=read(d/'output.json');events=[json.loads(l) for l in (d/'events.jsonl').read_text().splitlines() if l.strip()]
 assert set(raw['probabilities'])==set(LABELS)
 for v in raw['probabilities'].values():assert type(v) in (int,float) and math.isfinite(v) and 0<=v<=1
 assert list(raw)[-1]=='probabilities'
 assert 'localisations' not in raw,'BBox schema forbidden'
 transport.summaries(raw,LABELS)
 threads=[e for e in events if e.get('type')=='thread.started'];assert len(threads)==1,'not exactly one conversation'
 acquire_audit.audit_events(events)
 views=[read(p) for p in sorted((d/'views').glob('view_*.json'))];calls=[read(p) for p in sorted((d/'views').glob('call_*.json'))]
 acquire_audit.audit_native_calls(events,{'series':study['series']},views,calls)
 tool_events=[e['item'] for e in events if e.get('type')=='item.completed' and e.get('item',{}).get('type')=='mcp_tool_call']
 transport_rows=[transport.inspect(e,c) for e,c in zip(tool_events,calls)]
 dump(d/'transport-evidence.json',{'events':transport_rows,'provider_receipt':'UNKNOWN'})
 seen={v['arguments']['series'] for v in views};assert {'SAG_FLUID','AX_FLUID','COR_FLUID'}<=seen
 extra={s['key'] for s in study['series'] if '_ALT' in s['key']};struct={s['key'] for s in study['series'] if s['kind']=='structural'}
 assert not extra or extra&seen;assert not struct or struct&seen
 assert any(c['status']=='ok' and c['tool']=='view_slices' for c in calls)
 assert sum(v['arguments'].get('window') not in [None,[0,1]] for v in views)>=2
 bykey={s['key']:s for s in study['series']};byview={v['view']:v for v in views}
 for v in views:
  a=v['arguments'];s=bykey[a['series']];im,info=viewer.render(s,a['indices'],a['box'],window=a.get('window'));buf=io.BytesIO();im.save(buf,format='PNG')
  assert hashlib.sha256(buf.getvalue()).hexdigest()==v['image_sha256']==sha(v['image_path'])
  assert all(v[k]==x for k,x in info.items());assert v['coordinate_manifest']==co.manifest(s,a['indices'],a['box'],a.get('window'))
  c=next(c for c in calls if c['call']==v['call']);meta=next(json.loads(z) for z in c['emitted_metadata'] if json.loads(z).get('view')==v['view'])
  assert 'coordinate_manifest' not in meta;assert v['image_sha256'] in c['emitted_image_sha256']
 violations=[];boxes=[]
 try:jsonschema.validate(raw,read(source/'acquire_schema.json'))
 except jsonschema.ValidationError as e:
  violations.append({'kind':'text_or_quote_schema','error':str(e)})
 report=' '.join(study['report_projection']['projected_report'].split())
 for t in LABELS:
  for quote in raw.get('findings',{}).get(t,{}).get('report_evidence',[]):
   if not quote.strip() or ' '.join(quote.split()) not in report:violations.append({'kind':'report_quote','target':t,'quote':quote})
 v={'status':'valid' if not violations else 'invalid','classification_valid':True,'study_uid':study['study_uid'],'ordinal':study['ordinal'],'thread_id':threads[0].get('thread_id'),'thread_started_count':1,'views':len(views),'source_render_exact':True,'provider_receipt':'UNKNOWN','transport_events':transport_rows,'violations':violations,'boxes':boxes}
 dump(d/'validation.json',v);dump(d/'accepted.json',{'study_uid':study['study_uid'],'prediction':raw,'views':views,'usage':next(e['usage'] for e in events if e.get('type')=='turn.completed'),'files':{str(p):sha(p) for p in d.rglob('*') if p.is_file() and p.name not in ['accepted.json','accepted.receipt.json']}})
 return v
