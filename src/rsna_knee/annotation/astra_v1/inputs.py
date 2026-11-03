"""Source-only deployment of the S14 catalog and S15 fixed evidence packet."""
import csv
import math
from pathlib import Path

import numpy as np

from . import geometry
from . import native_preprocessing as pp
from . import project
from . import viewer
from .common import completed, dump, read, sealed, sha, text_sha
from .overview import overview

PLANES = [('Sagittal', 'SAG'), ('Axial', 'AX'), ('Coronal', 'COR')]


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def eligible(candidate):
    return (candidate['cache_available'] and candidate['orientation_quality']
            and candidate['decode_quality'] >= .99
            and candidate['plane'] in dict(PLANES)
            and not any(word in str(candidate.get('series_description') or '').lower()
                        for word in ['localizer', 'localiser', 'scout', 'survey', 'locator', '3-plane']))


def selection(source, cache):
    """Same primary tuple as vlm_data.select_series; explicit missing-contrast fallback."""
    candidates = [q for q in source['series_candidates'] if eligible(q)]
    metadata = {q['series_uid']: read(cache / q['metadata_path']) for q in candidates}

    def score(q):
        m = metadata[q['series_uid']]
        spacing = m.get('pixel_spacing') or []
        return (-int(q['fluid_sensitive']), -int(q['fat_suppression']),
                spacing[0] * spacing[1] if len(spacing) == 2 else math.inf,
                m.get('slice_thickness') if m.get('slice_thickness') is not None else math.inf,
                -int(m.get('slice_count') or 0), q['series_uid'])

    selected = []
    for plane, prefix in PLANES:
        available = [q for q in candidates if q['plane'] == plane]
        if not available:
            raise RuntimeError(f'Missing native plane: {source["study_uid"]} {plane}')
        q = min(available, key=score)
        selected.append((prefix + '_FLUID', 'fluid', q))
    used = {q['series_uid'] for _, _, q in selected}
    for key in ['SAG_STRUCT', 'COR_STRUCT']:
        q = source['selected_series'].get(key)
        if q and q['series_uid'] not in used and eligible(q):
            selected.append((key, 'structural', q))
            used.add(q['series_uid'])
    counters = {}
    extra = sorted([q for q in candidates if q['series_uid'] not in used],
                   key=lambda q: (q['plane'], not q['fluid_sensitive'],
                                  q.get('sequence_family') or '', q['series_uid']))
    for q in extra:
        kind = 'fluid' if q['fluid_sensitive'] else 'structural'
        base = dict(PLANES)[q['plane']] + '_' + ('FLUID' if kind == 'fluid' else 'STRUCT')
        counters[base] = counters.get(base, 0) + 1
        selected.append((base + '_ALT' + str(counters[base]), kind, q))
    assert len(selected) == len(candidates)
    return selected


def prepare_case(spec, run, contract):
    directory = run / 'cases' / spec['study_uid']
    ready = directory / 'input.json'
    existing = completed(ready)
    if existing:
        assert existing['spec_sha256'] == text_sha(__import__('json').dumps(spec, sort_keys=True))
        return existing
    cache = Path(contract['cache_root'])
    raw_root = Path(contract['raw_root'])
    series, images, lines, proofs, files = [], [], [], [], {}
    for selected in spec['selected']:
        key, kind, candidate = selected
        vp, original = cache / candidate['volume_path'], cache / candidate['metadata_path']
        m = read(original)
        volume = np.load(vp, mmap_mode='r')
        assert volume.dtype == np.uint16 and list(volume.shape) == m['shape'] == candidate['shape']
        assert m['slice_count'] == volume.shape[0]
        original_hash, volume_hash = sha(original), sha(vp)
        proof = {'series_uid': candidate['series_uid'], 'original_metadata_sha256': original_hash,
                 'volume_sha256': volume_hash, 'enriched': False}
        try:
            geometry.frame(m)
            mp = original
        except (AssertionError, KeyError, TypeError, ValueError):
            # Read headers in the ACTUAL cached source-file order, never re-sort pixels.
            paths = sorted((raw_root / 'train_series' / spec['study_uid'] /
                            candidate['series_uid']).glob('*.dcm'))
            assert paths, 'Raw headers required for missing geometry'
            records, method = pp.sort_slice_records(pp._read_datasets(paths, stop_before_pixels=True))
            assert method == m['sort_method']
            records = pp._records_in_cached_order(records, m['source_files'])
            assert len(records) == volume.shape[0]
            m = {**m, **pp._geometry_metadata(records)}
            assert [m['rows'], m['columns']] == list(volume.shape[1:])
            mp = directory / 'metadata' / (candidate['series_uid'] + '.json')
            dump(mp, m)
            proof.update(enriched=True, cached_source_order_verified=True,
                         raw_files_sha256={str(r.path): sha(r.path) for r in records})
        try:
            geometry.frame(m)
            proof['geometry_eligible'] = True
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            proof.update(geometry_eligible=False, limitation=str(error))
        q = dict(key=key, kind=kind, plane=candidate['plane'],
                 series_uid=candidate['series_uid'], description=candidate['series_description'],
                 volume_path=str(vp), volume_sha256=volume_hash,
                 metadata_path=str(mp), metadata_sha256=sha(mp),
                 original_metadata_path=str(original), original_metadata_sha256=original_hash,
                 slice_count=m['slice_count'], shape=m['shape'], pixel_spacing=m['pixel_spacing'],
                 slice_thickness=m.get('slice_thickness'),
                 image_orientation_patient=m.get('image_orientation_patient'),
                 fluid_sensitive=bool(candidate['fluid_sensitive']),
                 fat_suppression=bool(candidate['fat_suppression']),
                 indices=np.unique(np.rint(np.linspace(0, m['slice_count'] - 1,
                                min(48, m['slice_count']))).astype(int)).tolist())
        series.append(q)
        lines.append(f"{key}: {q['plane']}, {kind}, description={q['description']!r}, "
                     f"native indices0..{q['slice_count']-1}, native shape={q['shape']}, "
                     f"pixel spacing={q['pixel_spacing']}mm, slice thickness={q['slice_thickness']}mm. "
                     'Tool box coordinates refer to original image with x rightwards,y downwards; planes are not registered.')
        if key in ['SAG_FLUID', 'AX_FLUID', 'COR_FLUID'] and not (q['fluid_sensitive'] and q['fat_suppression']):
            lines.append(f"{key} SOURCE LIMITATION: requested fluid-plane role uses the best available "
                         f"native sequence; observed fluid_sensitive={q['fluid_sensitive']}, "
                         f"fat_suppression={q['fat_suppression']}. The role name does not establish contrast; "
                         'interpret its actual description and pixels, and retain visibility uncertainty.')
        if kind == 'fluid':
            path = directory / 'overviews' / (key + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            overview(q).save(path)
            images.append(dict(path=str(path), sha256=sha(path), plane=q['plane'],
                               series_key=key, indices=q['indices']))
            lines.append(f"Attachment {len(images)} is {key} overview; {len(q['indices'])}/{q['slice_count']} "
                         'native slices shown in row-major order. Use tool for detailed views.')
            files[str(path)] = sha(path)
        files.update({str(vp): volume_hash, str(original): original_hash, str(mp): sha(mp)})
        proofs.append(proof)
    eligible_keys = viewer.eligible_keys({'series': series})
    # No silent relaxation of the historical S14 comparison requirement.
    assert len(eligible_keys) >= 2, 'Fewer than two geometry-eligible series; needs source review'
    lines.append('CANDIDATE CORRESPONDENCE: geometry-eligible series=' + ','.join(eligible_keys) +
                 '. Other series remain available for independent views. Coordinate matches are not verified registration.')
    if any('_ALT' in q['key'] for q in series):
        lines.append('Navigation requirement for the additional sources listed above: inspect at least one _ALT series '
                     'in a detailed VIEW, choosing the source/location from its contrast and anatomical coverage. '
                     'Preserve detailed views of all three original SAG_FLUID, AX_FLUID, COR_FLUID series and available '
                     'structural coverage, a successful comparison, an independent view, and two adjusted-window views '
                     'within the SAME total10 VIEW sheets/10 calls. Do not assume every extra fluid series requires its own view.')
    projected = project.project(spec['report'])
    assert projected['projected_report']
    case = directory / 'acquire_case.json'
    dump(case, {'series': series})
    files[str(case)] = sha(case)
    result = dict(study_uid=spec['study_uid'], ordinal=spec['ordinal'], report=spec['report'],
                  report_sha256=text_sha(spec['report']), report_projection=projected,
                  series=series, images=images, input_map='\n'.join(lines),
                  case_path=str(case), source_proofs=proofs, files=files,
                  spec_sha256=text_sha(__import__('json').dumps(spec, sort_keys=True)))
    sealed(ready, result)
    return result


def fixed_packet(study, acquisition, directory):
    by_key = {q['key']: q for q in study['series']}
    packet = []
    for view in acquisition['views']:
        q = by_key[view['arguments']['series']]
        metadata = dict(view=view['view'], series=q['key'], plane=q['plane'], sequence=q['description'])
        for key in ['native_box_pixels', 'normalized_box', 'indices', 'physical_aspect_preserved',
                    'source_volume_sha256', 'window', 'original_always_included', 'display_layout',
                    'window_domain', 'correspondence']:
            if key in view:
                metadata[key] = view[key]
        packet.append(dict(image_path=view['image_path'], image_sha256=view['image_sha256'], metadata=metadata))
    mapping = '\n'.join(line for line in study['input_map'].splitlines()
                        if line.startswith(('SAG_', 'AX_', 'COR_', 'Attachment ')))
    mapping = mapping.replace(' Use tool for detailed views.', ' Detailed images follow through review_evidence.')
    mapping += (f'\nFixed packet contains{len(packet)} detailed VIEW sheets. VIEW numbering follows '
                'the packet metadata; overview attachment numbers are separate.')
    path = directory / 'readout_case.json'
    if path.exists():
        assert read(path) == {'views': packet}, 'Fixed packet changed'
    else:
        dump(path, {'views': packet})
    return {**study, 'fixed_views': packet, 'input_map': mapping, 'case_path': str(path)}
