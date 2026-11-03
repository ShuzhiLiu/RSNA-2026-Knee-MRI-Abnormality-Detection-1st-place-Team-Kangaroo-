"""Saved event completeness is distinct from emitted bytes and provider receipt."""
import base64,hashlib,json,re

def inspect(event,call):
 result=event.get('result');representation='direct';content=result.get('content') if isinstance(result,dict) else None
 if isinstance(content,list) and len(content)==1 and content[0].get('type')=='text':
  try:
   nested=json.loads(content[0]['text'])
   if isinstance(nested,dict) and isinstance(nested.get('content'),list):content=nested['content'];representation='nested'
  except (ValueError,TypeError):pass
 row={'call':call['call'],'representation':representation,'provider_receipt':'UNKNOWN','status':'UNKNOWN','emitted_images':len(call.get('emitted_image_sha256',[]))}
 if not isinstance(content,list):return row
 encoded=json.dumps(content,ensure_ascii=False)
 if re.search(r'…\d+ chars truncated…|\.\.\.\d+ chars truncated\.\.\.',encoded):row['status']='PARTIAL_TRUNCATED';return row
 texts=[x.get('text') for x in content if x.get('type')=='text'];images=[x for x in content if x.get('type')=='image']
 if any(x.get('type') not in ['text','image'] for x in content):return row
 # Successful logged MCP result with conventional complete content is comparable.
 if call['status']!='ok':return row
 expected=call['emitted_metadata'];hashes=[]
 for x in images:
  try:raw=base64.b64decode(x['data'],validate=True)
  except (ValueError,KeyError,TypeError):row['status']='UNKNOWN';row['reason']='undecodable image without explicit truncation marker';return row
  hashes.append(hashlib.sha256(raw).hexdigest())
 assert texts==expected,'Complete event metadata mismatch'
 assert hashes==call['emitted_image_sha256'],'Complete event image hash mismatch'
 row.update(status='COMPLETE_EXACT',logged_images=len(images));return row

def summaries(raw,labels):
 assert isinstance(raw.get('findings'),dict) and set(raw['findings'])==set(labels),'all12 findings required'
 for t in labels:
  f=raw['findings'][t];assert isinstance(f,dict),'finding must be object'
  text=f.get('summary_and_limitations');assert isinstance(text,str) and text.strip(),'nonempty summary required'

def synthetic_tests():
 h=hashlib.sha256(b'image').hexdigest();call={'call':1,'status':'ok','emitted_metadata':['{"view":1}'],'emitted_image_sha256':[h]}
 content=[{'type':'text','text':call['emitted_metadata'][0]},{'type':'image','data':base64.b64encode(b'image').decode()}]
 assert inspect({'result':{'content':content}},call)['status']=='COMPLETE_EXACT'
 assert inspect({'result':{'content':[{'type':'text','text':json.dumps({'content':content,'isError':False})}]}},call)['status']=='COMPLETE_EXACT'
 partial=[content[0],{'type':'image','data':'abc…500 chars truncated…'}]
 assert inspect({'result':{'content':[{'type':'text','text':json.dumps({'content':partial})}]}},call)['status']=='PARTIAL_TRUNCATED'
 assert inspect({'result':{}},call)['status']=='UNKNOWN'
 for bad in [[{'type':'text','text':'wrong'},content[1]],[content[0],{'type':'image','data':base64.b64encode(b'wrong').decode()}]]:
  try:inspect({'result':{'content':bad}},call)
  except AssertionError:pass
  else:raise AssertionError('complete mismatch accepted')
 summaries({'findings':{'A':{'summary_and_limitations':'No focal defect.'}}},['A'])
 for raw in [{'findings':{}},{'findings':{'A':{'summary_and_limitations':''}}},{'findings':{'A':None}}]:
  try:summaries(raw,['A'])
  except AssertionError:pass
  else:raise AssertionError('missing summary accepted')
 return {'status':'PASS','complete_direct_and_nested':2,'partial':1,'unknown':1,'complete_mismatch_rejected':2,'summary_negative_controls':3,'model_calls':0}
