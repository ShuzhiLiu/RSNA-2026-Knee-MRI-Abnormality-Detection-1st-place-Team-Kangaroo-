"""Truth-free joint12 Stage1 method shared by weak screen and full evaluation."""
import hashlib
import json
import math
import os
import time

from . import vlm_hulumed as h

MODEL='Qwen/Qwen3.8-Flash-Next-FP8'
PROTOCOL='qwen38-native-single-target-probability-v6'
ASSESSMENTS=('present','absent','equivocal','not_mentioned')


def prompt(report, labels):
    units=h.report_units(report)
    definitions='\n\n'.join(f'TARGET: {label}\n{h.shared_guidance(label)}' for label in labels)
    displayed='\n'.join(f'[{u.id}] {u.text}' for u in units)
    return f'''Read this knee MRI and its original radiology report. Assess each target
under its supplied definition, considering target-specific evidence and uncertainty.

TARGET DEFINITIONS
{definitions}

ORIGINAL REPORT (may be multilingual)
{displayed}

For every target, separate what the report actually states from what is visible
in these sparse MRI slices. Report silence and sparse non-visualization are not
negative findings. Preserve negation, anatomic compartment and uncertainty. Do
not infer one target just from another associated abnormality. If the sources
disagree, state the disagreement. Do not invent unseen morphology or patient context.

After reviewing the evidence, estimate the probability that the defined target
is truly present in this study, given the supplied evidence and its limitations.
This is a continuous clinical probability, not confidence in spelling a yes/no
token and not merely a binary decision. Use any number between0 and1 supported
by the evidence; do not force a small set of score bins. Do not use cohort-level
prevalence, other patients, example answers or results from other models.

Return ONLY a JSON object with exactly these target names as keys:
{json.dumps(list(labels))}
Each value must have the following fields, in this order:
1. report_assessment: exactly present, absent, equivocal or not_mentioned.
2. report_unit_ids: a list of up to3 distinct exact R### IDs for report units
   that directly support this target's assessment. The program will retrieve the
   original text from these IDs; do not rewrite or translate source quotations.
   Use an empty list when there is no target-specific report evidence.
3. image_evidence: a short description of independently visible target findings
   or the limitation preventing assessment. If no images were supplied, say so.
4. evidence_balance: one short sentence summarizing evidence and uncertainty,
   keeping report and image claims separate. Do not provide a detailed reasoning trace.
5. probability: a JSON number between0 and1 inclusive, not a string.
Keep each text field concise. Do not add fields, markdown or text outside JSON.'''


def parse(content, report, label):
    data=json.loads(content)
    assert isinstance(data,dict) and set(data)=={label},'target keys mismatch'
    units={u.id:u.text for u in h.report_units(report)}
    for label,value in data.items():
        assert set(value)=={'report_assessment','report_unit_ids','image_evidence','evidence_balance','probability'},label
        assert value['report_assessment'] in ASSESSMENTS,label
        p=value['probability'];assert isinstance(p,(int,float)) and not isinstance(p,bool)
        assert math.isfinite(p) and 0<=p<=1,label
        for field in ('image_evidence','evidence_balance'):
            assert isinstance(value[field],str) and 0<len(value[field].strip())<=2500,(label,field)
        ids=value['report_unit_ids'];assert isinstance(ids,list) and len(ids)<=len(units)
        assert all(isinstance(uid,str) and uid in units for uid in ids)
        assert len(ids)==len(set(ids))
        if value['report_assessment'] in ('present','absent','equivocal'):
            assert ids,(label,'asserted report assessment without evidence')
        value['report_evidence']=[{'unit_id':uid,'quote':units[uid]} for uid in ids]
    return data


def invocation():
    return {'model':MODEL,'temperature':0,'max_tokens':12288,
            'chat_template_kwargs':{'enable_thinking':True},
            'response_format':{'type':'json_object'}}


def request(base_url,row,label,images):
    assert set(row)=={'StudyInstanceUID','Report'},'truth or other model data in row'
    labels=(label,)
    text=prompt(row['Report'],labels)
    if images:
        messages,archived,provenance=h.image_messages(text,images)
    else:
        messages=[{'role':'system','content':'You are a careful musculoskeletal radiologist. Use only the supplied report. No images are available.'},
                  {'role':'user','content':text}]
        archived=messages;provenance=[]
    common={'study_uid':row['StudyInstanceUID'],'label':label,'model':MODEL,'protocol':PROTOCOL,
            'prompt':text,'messages':archived,'image_inputs':provenance,
            'request_sha256':h.canonical_sha256(messages),'invocation':invocation(),
            'report_sha256':hashlib.sha256(row['Report'].encode()).hexdigest(),
            'target_order':list(labels)}
    started=time.monotonic();failures=[]
    for attempt in range(3):
        raw=None
        try:
            raw=h.http_post_json(base_url.rstrip('/')+'/chat/completions',
                                 {'messages':messages,**invocation()},os.environ.get('QWEN38_API_KEY','EMPTY'),600)
            assert raw['model']==MODEL
            choice=raw['choices'][0];assert choice['finish_reason']=='stop',choice['finish_reason']
            msg=choice['message'];reasoning=msg.get('reasoning',msg.get('reasoning_content'))
            assert isinstance(reasoning,str) and reasoning.strip(),'reasoning not active'
            parsed=parse(msg['content'],row['Report'],label)
            return {**common,'status':'ok','raw_response':raw,'parsed':parsed,'usage':raw.get('usage'),
                    'failed_attempts':failures,'attempts':attempt+1,'elapsed_seconds':time.monotonic()-started}
        except Exception as exc:
            failures.append({'attempt':attempt+1,'error':f'{type(exc).__name__}: {exc}','raw_response':raw})
            if attempt<2:time.sleep(2**attempt)
    return {**common,'status':'error','failed_attempts':failures,'attempts':3,
            'elapsed_seconds':time.monotonic()-started}
