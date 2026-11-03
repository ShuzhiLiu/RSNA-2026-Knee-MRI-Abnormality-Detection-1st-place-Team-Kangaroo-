"""Prepare final numeric supervision and portable recipes, without model/API work."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

RECIPES = ('final_mix5_base384', 'final_focus_base384', 'final_mix5_coat384')
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def read_targets(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        key = next((k for k in ('StudyInstanceUID', 'Sample ID', 'study_uid') if k in (reader.fieldnames or [])), None)
        if key is None or not set(TARGETS).issubset(reader.fieldnames):
            raise ValueError(f'incomplete target schema: {path}')
        rows = {}
        for row in reader:
            uid = row[key]
            values = [float(row[t]) for t in TARGETS]
            if not uid or uid in rows or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError(f'invalid or duplicate study: {uid}')
            rows[uid] = values
    return rows


def mix_sources(labels):
    """Exact global 3:2:3:1:1 formula; join by identifier, never row position."""
    sources = [read_targets(Path(labels) / (name + '_weak4349.csv')) for name in
               ('astra_v1', 'astra_v2', 'teammate_astra', 'hulu', 'qwen_next')]
    ids = set(sources[0])
    if len(ids) != 4349 or any(set(s) != ids for s in sources):
        raise ValueError('Mix5 requires the same 4349 IDs in every source')
    return {u: [sum(w*s[u][t] for w, s in zip((3, 2, 3, 1, 1), sources))/10
                for t in range(12)] for u in sources[0]}


def prepare(*, code_root, supervision, hard_labels, selection, cache_root,
            stage_cache_root, base_pretrained, coat_pretrained, output):
    """Create a NEW workspace; preserve supplied annotations and historical records."""
    code_root, supervision, output = map(lambda p: Path(p).resolve(), (code_root, supervision, output))
    if output.exists():
        raise FileExistsError(f'refusing to overwrite workspace: {output}')
    labels = supervision / 'labels'
    if load(labels / 'target_order.json') != TARGETS:
        raise ValueError('released target order differs from classifier')
    hard = read_targets(hard_labels)
    with (labels / 'training_membership.csv').open(newline='') as stream:
        members = list(csv.DictReader(stream))
    ids = [m['StudyInstanceUID'] for m in members]
    if len(ids) != 4407 or len(set(ids)) != 4407:
        raise ValueError('expected 4407 unique training IDs')
    if [int(m['training_order']) for m in members] != list(range(4407)):
        raise ValueError('membership is not in frozen training order')
    if any(m['label_source'] not in ('derived', 'official_human') for m in members):
        raise ValueError('unknown label source')
    hids = {m['StudyInstanceUID'] for m in members if m['label_source'] == 'official_human'}
    sids = set(ids) - hids
    if len(hids) != 58 or not hids <= hard.keys() or any(v not in (0, 1) for u in hids for v in hard[u]):
        raise ValueError('expected original 58 binary human labels')
    tables = {name: read_targets(labels / f'{name}_weak4349.csv') for name in ('mix5', 'focus_guided')}
    if any(set(t) != sids for t in tables.values()):
        raise ValueError('final soft-label membership mismatch')
    mixed = mix_sources(labels)
    if any(abs(a-b) > 1e-12 for u in sids for a,b in zip(mixed[u], tables['mix5'][u])):
        raise ValueError('released Mix5 differs from component sources')
    selection = Path(selection).resolve(strict=True)
    with selection.open() as stream:
        selected = [json.loads(line)['study_uid'] for line in stream if line.strip()]
    if len(selected) != 4407 or set(selected) != set(ids):
        raise ValueError('selection manifest must cover exactly final training studies')
    cache_root = Path(cache_root).resolve(strict=True)
    stage_cache_root = Path(stage_cache_root).resolve(strict=True)
    anatomy = supervision / 'anatomy'
    lookup = load(anatomy / 'lookup_manifest.json')
    if set(lookup['index']) != set(ids):
        raise ValueError('anatomy study membership mismatch')
    lesion = supervision / 'lesions/lesion_train_index.json'
    lesion_manifest = supervision / 'lesions/manifest.json'
    lm = load(lesion_manifest)
    if set(lm['weak_train_ids']) != sids or not set(load(lesion)) <= sids or lm['index_sha256'] != sha(lesion):
        raise ValueError('lesion membership/index hash mismatch')
    manifests = {}
    for name, directory in [('pretrained_maxxvitv2_base384.json', base_pretrained), ('pretrained_coatnet_rmlp2.json', coat_pretrained)]:
        manifest = load(code_root / 'configs' / name)
        directory = Path(directory).resolve(strict=True)
        for item in manifest['files']:
            file = directory / item['name']
            if file.stat().st_size != item['size'] or sha(file) != item['sha256']:
                raise ValueError(f'pretrained integrity failure: {file}')
        manifest['target_dir'] = str(directory)
        manifests[name] = manifest
    # Only after input validation do we create the new workspace.
    output.mkdir(parents=True)
    pipeline = load(code_root / 'configs/knee_input_pipeline_384_astra111.json')
    pipeline['supervision_schema'] = 'final-numeric-v1'
    pipeline.pop('training_tables_lock_sha256', None)
    pipeline['paths'] = dict(cache_root=str(cache_root), stage_cache_root=str(stage_cache_root), selection_manifest=str(selection))
    pipeline['selection_manifest_sha256'] = sha(selection)
    pipeline['selection_manifest_bytes'] = selection.stat().st_size
    save(output / 'configs/knee_input_pipeline_384_astra111.json', pipeline)
    for name, manifest in manifests.items():
        save(output / 'configs' / name, manifest)
    for recipe in RECIPES:
        kind = 'focus_guided' if recipe == 'final_focus_base384' else 'mix5'
        records = []
        for u in ids:
            h = u in hids
            records.append(dict(sample_id=u, source_group='H' if h else 'S', is_hard=h,
                                split_role='train', original_oof_fold=0 if h else -1,
                                diagnostic_fold=-1, train_labels=hard[u] if h else tables[kind][u]))
        source_files = {'soft': labels/f'{kind}_weak4349.csv', 'hard': Path(hard_labels).resolve(),
                        'membership': labels/'training_membership.csv', 'selection': selection}
        payload = dict(protocol_id='full-data-s4349-h58-no-heldout-v1', targets=TARGETS, records=records,
                       sources={k: dict(path=str(p), sha256=sha(p)) for k,p in source_files.items()},
                       release_format='final-solution-v1',
                       fold_note='H fold=0 is a hard-label flag only; final fitting uses no CV folds.')
        rp = output / 'data' / f'{kind}_records.json'
        if not rp.exists():
            save(rp, payload)
        config = load(code_root / 'configs' / f'{recipe}.json')
        config['release_format'] = 'final-solution-v1'
        config['fulldata'] = dict(records_path=str(rp), records_sha256=sha(rp))
        location = config['location_supervision']
        location.update(lookup_run=str(anatomy), usage_map=str(anatomy/'usage_map.json'),
                        lookup_manifest_sha256=sha(anatomy/'lookup_manifest.json'), usage_map_sha256=sha(anatomy/'usage_map.json'))
        config['boxcombo'].update(index=str(lesion), index_manifest=str(lesion_manifest),
                                  index_sha256=sha(lesion), index_manifest_sha256=sha(lesion_manifest))
        save(output / 'configs' / f'{recipe}.json', config)
    save(output/'preparation.json', dict(release_format='final-solution-v1', recipes=list(RECIPES),
                                       studies=4407, weak=4349, hard=58, fresh_training_verified=False,
                                       configs_sha256={str(p.relative_to(output)): sha(p) for p in (output/'configs').glob('*.json')}))
    return output


def extract_hard_labels(train_csv, output):
    """Write the 58 official human-labelled rows (all 12 targets present) of train.csv."""
    with Path(train_csv).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if 'StudyInstanceUID' not in (reader.fieldnames or []) or not set(TARGETS).issubset(reader.fieldnames):
            raise ValueError('expected the official train.csv schema')
        rows = []
        for row in reader:
            values = [row[t].strip() for t in TARGETS]
            if not any(values):
                continue
            if not all(values) or any(v not in ('0', '1', '0.0', '1.0') for v in values):
                raise ValueError(f'partial or non-binary human label row: {row["StudyInstanceUID"]}')
            rows.append([row['StudyInstanceUID'], *(int(float(v)) for v in values)])
    if len(rows) != 58 or len({r[0] for r in rows}) != 58:
        raise ValueError(f'expected 58 unique human-labelled studies, found {len(rows)}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID', *TARGETS])
        writer.writerows(rows)
    return output


def download_pretrained(manifest_path, output_root):
    """Download the pinned Hugging Face revision of one backbone and verify size/SHA256."""
    import shutil
    import urllib.parse
    import urllib.request
    manifest = load(manifest_path)
    target = Path(output_root) / manifest['repo_id'].split('/')[-1]
    target.mkdir(parents=True, exist_ok=True)
    for item in manifest['files']:
        path = target / item['name']
        if not path.exists():
            url = (f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/"
                   + urllib.parse.quote(item['name']))
            partial = path.with_name(path.name + '.download')
            with urllib.request.urlopen(url) as response, partial.open('wb') as stream:
                shutil.copyfileobj(response, stream, length=1 << 20)
            partial.replace(path)
        if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError(f'pretrained integrity failure: {path}')
    return target
