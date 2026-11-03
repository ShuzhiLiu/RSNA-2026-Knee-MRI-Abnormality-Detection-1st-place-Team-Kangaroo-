"""J12-SOL61-PROD1 case builder: IND24 v006 initial input for one study from an Astra-format prepared input.json.

Reuses from the prepared case only: series selection/order, report projection, catalog lines (Attachment lines
removed, exactly as v004/v006), native volume/metadata references and identity. Every image is rendered fresh
with the v006 code path (24 rounded-unique full-span indices per fluid series, individual 384x384 body, LANCZOS,
physical-aspect letterbox). No mosaic or overview sheet is ever produced or sent. No model calls.
"""
import hashlib
import json
import os
import pathlib

import numpy as np
from PIL import Image

TILE = 384
MAX_SLICES = 24


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def shab(b):
    return hashlib.sha256(b).hexdigest()


def dump(p, v):
    """Same serialization as the v006 prepare.py dump (indent=2, ensure_ascii=False, trailing newline), atomic."""
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(v, indent=2, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def indices24(slice_count):
    return np.unique(np.rint(np.linspace(0, slice_count - 1, min(MAX_SLICES, slice_count)))).astype(int).tolist()


def render_body(v, idx, spacing):
    """Byte-for-byte the v006 individual body renderer."""
    arr = np.rint(v[idx].astype(np.float32) / 257).astype(np.uint8)
    h, w = arr.shape
    sp = spacing or [1, 1]
    scale = min(TILE / (w * sp[1]), TILE / (h * sp[0]))
    nw, nh = max(1, round(w * sp[1] * scale)), max(1, round(h * sp[0] * scale))
    body = Image.new('RGB', (TILE, TILE), 'black')
    body.paste(Image.fromarray(arr).convert('RGB').resize((nw, nh), Image.Resampling.LANCZOS), ((TILE - nw) // 2, (TILE - nh) // 2))
    return body


def prompt_for(prompt_template, s):
    catalog = '\n'.join(line for line in s['input_map'].splitlines() if not line.startswith('Attachment '))
    return (prompt_template.replace('Overviews are navigation images', 'The grouped independent images are navigation images')
            .replace('actual overviews', 'the grouped independent images')
            .replace('OVERVIEW AND SERIES CATALOG', 'INDIVIDUAL IMAGE GROUPS AND SERIES CATALOG')
            .format(input_map=catalog, report=s['report_projection']['projected_report']))


def build_case(s, case_dir, prompt_template, verify_all_volumes=True):
    """Build one case directory. Returns the case record (hashes, counts). Raises on any source inconsistency."""
    case_dir = pathlib.Path(case_dir)
    (case_dir / 'individual').mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(s['report'].encode()).hexdigest() == s['report_sha256'], 'report hash'
    deps = {}
    for q in s['series']:
        if verify_all_volumes or q['kind'] == 'fluid':
            for pk, hk in [('volume_path', 'volume_sha256'), ('metadata_path', 'metadata_sha256'),
                           ('original_metadata_path', 'original_metadata_sha256')]:
                assert sha(q[pk]) == q[hk], (s['study_uid'], q['key'], pk)
                deps[q[pk]] = q[hk]
    fluid = [q for q in s['series'] if q['kind'] == 'fluid']
    assert fluid, 'no fluid series'
    assert [im['series_key'] for im in s['images']] == [q['key'] for q in fluid], 'prepared image order != fluid series order'
    series = []
    for q0 in s['series']:
        q = dict(q0)
        q['indices'] = indices24(q['slice_count'])
        series.append(q)
    groups, images, pix = [], [], []
    for q in [x for x in series if x['kind'] == 'fluid']:
        v = np.load(q['volume_path'], mmap_mode='r')
        assert v.dtype == np.uint16 and list(v.shape) == list(q['shape']) and v.shape[0] == q['slice_count']
        g = {'series_key': q['key'], 'plane': q['plane'], 'description': q['description'], 'fluid_sensitive': q['fluid_sensitive'],
             'fat_suppression': q['fat_suppression'], 'native_indices': q['indices'], 'images': []}
        for idx in q['indices']:
            body = render_body(v, idx, q['pixel_spacing'])
            assert body.size == (TILE, TILE) and body.mode == 'RGB'
            ip = case_dir / 'individual' / f'{q["key"]}_{idx:03d}.png'
            body.save(ip)
            back = Image.open(ip)
            assert back.size == (TILE, TILE) and back.convert('RGB').tobytes() == body.tobytes()
            e = {'path': str(ip), 'sha256': sha(ip), 'series_key': q['key'], 'native_index': idx, 'pixel_sha256': shab(body.tobytes()),
                 'width': TILE, 'height': TILE}
            g['images'].append(e)
            images.append(e)
        groups.append(g)
    prompt = prompt_for(prompt_template, s)
    initial = [{'type': 'text', 'text': prompt}]
    for g in groups:
        initial.append({'type': 'text', 'text': 'IMAGE GROUP ' + json.dumps({k: v for k, v in g.items() if k != 'images'}, ensure_ascii=False)})
        for e in g['images']:
            initial.extend([{'type': 'text', 'text': f"{e['series_key']} native index {e['native_index']}"}, {'type': 'localImage', 'path': e['path']}])
    assert all(initial[i - 1]['type'] == 'text' for i, x in enumerate(initial) if x['type'] == 'localImage')
    assert sum(x['type'] == 'localImage' for x in initial) == len(images)
    dump(case_dir / 'initial-input.json', initial)
    (case_dir / 'clinical-prompt.txt').write_text(prompt)
    dump(case_dir / 'report-projection.json', s['report_projection'])
    dump(case_dir / 'acquire_case.json', {'series': series})
    rec = {'ordinal': s['ordinal'], 'study_uid': s['study_uid'], 'case_dir': str(case_dir),
           'acquire_case_sha256': sha(case_dir / 'acquire_case.json'), 'initial_input_sha256': sha(case_dir / 'initial-input.json'),
           'clinical_prompt_sha256': sha(case_dir / 'clinical-prompt.txt'), 'report_projection_sha256': sha(case_dir / 'report-projection.json'),
           'initial_items': len(initial), 'initial_images': len(images), 'initial_local_images_384': sum(1 for e in images if e['width'] == e['height'] == TILE),
           'mosaic_images': 0, 'groups': [{k: v for k, v in g.items() if k != 'images'} for g in groups],
           'image_order': [{'series_key': e['series_key'], 'native_index': e['native_index'], 'sha256': e['sha256'], 'pixel_sha256': e['pixel_sha256']} for e in images],
           'image_bytes': sum(pathlib.Path(e['path']).stat().st_size for e in images), 'report_sha256': s['report_sha256'],
           'series_count': len(series), 'fluid_series': len(groups), 'dependencies': deps}
    dump(case_dir / 'case-manifest.json', rec)
    return rec
