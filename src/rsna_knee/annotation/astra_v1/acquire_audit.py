import math
from . import viewer
def audit_events(events):
 for e in events:
  if e.get('type')not in ['item.started','item.completed']:continue
  it=e['item'];typ=it['type'];assert typ in ['agent_message','error','reasoning','mcp_tool_call'],f'unexpected item type {typ}'
  if typ=='mcp_tool_call':assert it['server']=='mri'and it['tool']in ['view_slices','compare_location'],'unexpected tool'
 assert sum(e.get('type')=='turn.completed'for e in events)==1

def audit_native_calls(events,case,logs,call_logs):
 calls=[e['item']for e in events if e.get('type')=='item.completed'and e.get('item',{}).get('type')=='mcp_tool_call']
 assert len(calls)==len(call_logs)and 1<=len(calls)<=10
 byview={x['view']:x for x in logs};used=[];comparisons=[]
 for event,call in zip(calls,call_logs):
  assert event['server']=='mri'and event['tool']==call['tool']and event['arguments']==call['arguments']
  if call['status']!='ok':assert not call['views'];continue
  if call['tool']=='view_slices':specs=[{'arguments':call['arguments']}]
  else:
   specs,unavailable,point=viewer.compare_native_specs(case,call['arguments']);assert unavailable==call['unavailable']and point==call['candidate_anchor_lps_mm'];comparisons.append(call)
  assert len(specs)==len(call['views'])
  for spec,idx in zip(specs,call['views']):
   log=byview[idx];assert log['status']=='ok'and log['call']==call['call']and log['origin_tool']==call['tool']and log['arguments']==spec['arguments']
   if 'correspondence'in spec:assert log['correspondence']==spec['correspondence']
   used.append(idx)
 assert len(used)==len(set(used))==len(logs)and set(used)==set(byview)and 1<=len(logs)<=10
 assert comparisons,'at least one successful multi-series comparison required'
 eligible_struct={x['key']for x in case['series']if x['kind']=='structural'}&set(viewer.eligible_keys(case))
 if eligible_struct:
  assert any(any(byview[i]['arguments']['series']in eligible_struct for i in c['views'])and any(byview[i]['arguments']['series']not in {x['key']for x in case['series']if x['kind']=='structural'}for i in c['views'])for c in comparisons),'compare a fluid and an eligible structural sequence'

def normalize_output(raw):
 label=raw['binary_label'];confidence=raw['confidence_correct'];assert type(label)is int and label in [0,1]
 assert type(confidence)in [int,float]and math.isfinite(confidence)and .5<=confidence<=1
 return {**raw,'probability_present':confidence if label else 1-confidence}
