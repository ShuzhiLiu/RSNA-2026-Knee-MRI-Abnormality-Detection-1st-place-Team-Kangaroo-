"""Image-only manifests and final classifier cache construction."""
from __future__ import annotations
import json
import gzip
from pathlib import Path
from .final_preparation import sha


def export_selection(source, output):
    """Strip labels/folds/report metadata, preserving consumed image linkage."""
    opener = gzip.open if str(source).endswith('.gz') else open
    with opener(source, 'rt') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [r['study_uid'] for r in rows]
    if len(ids) != 4407 or len(set(ids)) != 4407:
        raise ValueError('expected final 4407-study image manifest')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open('x') as stream:
        for row in rows:
            stream.write(json.dumps({key: row[key] for key in
                ('study_uid', 'selected_series', 'series_candidates')}, separators=(',', ':'))+'\n')


def build_inputs(*, raw_root, selection, cache_root, stage_cache_root, input_config):
    """Decode only selected series; build the unchanged deterministic spatial stage."""
    from .preprocessing import PreprocessConfig, preprocess_series, save_cached_series
    from .input_pipeline import KneeInputConfig, KneeStudyDataset, default_study_view, build_deterministic_stage_cache
    raw_root, selection, cache_root, stage_cache_root = map(Path, (raw_root, selection, cache_root, stage_cache_root))
    config = KneeInputConfig.from_json(input_config)
    pixel_config = PreprocessConfig()
    if pixel_config.fingerprint != config.cache_config_fingerprint:
        raise ValueError('pixel preprocessing does not match final input config')
    with selection.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    for row in records:
        uid = row['study_uid']
        for series in row['selected_series'].values():
            if series is None:
                continue
            relative = Path(series['volume_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe cache path in selection')
            destination = cache_root/relative.parent
            volume, metadata = destination/'volume.npy', destination/'metadata.json'
            if volume.exists() or metadata.exists():
                if not volume.is_file() or not metadata.is_file() or json.loads(metadata.read_text()).get('config_fingerprint') != pixel_config.fingerprint:
                    raise ValueError(f'incomplete or incompatible cache: {destination}')
                continue
            files = sorted((raw_root/'train_series'/uid/series['series_uid']).glob('*.dcm'))
            if not files:
                raise FileNotFoundError(f'no DICOM files for {uid}/{series["series_uid"]}')
            save_cached_series(preprocess_series(files, pixel_config), destination)
    dataset = KneeStudyDataset(config=config, cache_root=cache_root,
        stage_cache_root=stage_cache_root, selection_manifest=selection,
        selection_manifest_sha256=sha(selection), selection_manifest_bytes=selection.stat().st_size,
        training_tables=None, augment=False)
    view = default_study_view(config)
    for i in range(len(dataset)):
        study = dataset.build_study_input(i)
        for item in study.primary_series:
            if item is None:
                continue
            series = item.record
            build_deterministic_stage_cache(config=config,
                source_volume_path=cache_root/series['volume_path'],
                source_metadata_path=cache_root/series['metadata_path'],
                destination=stage_cache_root/view.version/Path(series['volume_path']).parent,
                role=item.role, laterality=study.laterality['laterality'], view=view)
        if i % 100 == 0:
            print(f'Prepared spatial inputs: {i+1}/{len(dataset)}', flush=True)


def blend_predictions(members, output, teammate=None):
    """Mean the three member probability tables, then optionally blend 1:1."""
    import csv
    from .final_preparation import read_targets, TARGETS
    if len(members) != 3 or len({Path(p).resolve() for p in members}) != 3:
        raise ValueError('expected three distinct final member tables')
    tables = [read_targets(p) for p in members]
    ids = list(tables[0])
    if any(set(t) != set(ids) for t in tables):
        raise ValueError('member prediction ID mismatch')
    other = read_targets(teammate) if teammate else None
    if other is not None and set(other) != set(ids):
        raise ValueError('teammate prediction ID mismatch')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open('x',newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['StudyInstanceUID',*TARGETS])
        for uid in ids:
            values = [sum(t[uid][i] for t in tables)/3 for i in range(12)]
            if other is not None:
                values = [(a+b)/2 for a,b in zip(values,other[uid])]
            writer.writerow([uid,*values])
