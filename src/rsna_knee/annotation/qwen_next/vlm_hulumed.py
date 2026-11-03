"""Final annotation request helpers extracted from the consumed source."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from .vlm_data import DISEASE_GUIDES, LABELS, PLANES, REPORT_GUIDES, auc_score, binary_choice_probability, http_post_json, image_data_url, load_study_rows, read_env

@dataclass(frozen=True)
class ReportUnit:
    id: str
    start: int
    end: int
    text: str

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()

def report_units(report: str) -> list[ReportUnit]:
    """Split a report into addressable units while preserving exact offsets."""
    boundaries = [match.end() for match in re.finditer('(?:\\r?\\n+|(?<=[.!?。！？；;])\\s*)', report)]
    if not boundaries or boundaries[-1] != len(report):
        boundaries.append(len(report))
    units: list[ReportUnit] = []
    cursor = 0
    for boundary in boundaries:
        raw = report[cursor:boundary]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            start, end = (cursor + left, cursor + right)
            units.append(ReportUnit(f'R{len(units) + 1:03d}', start, end, report[start:end]))
        cursor = boundary
    if not units:
        raise ValueError('report produced no evidence units')
    return units

def shared_guidance(label: str) -> str:
    return f"REPORT DEFINITION: {REPORT_GUIDES[label]}\nMRI DIAGNOSTIC FOCUS: {' '.join(DISEASE_GUIDES[label].split())}"

def image_messages(prompt: str, image_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    intro = 'Review ordered knee MRI input: 7 sagittal, 7 coronal, then 7 axial slices.'
    request_content: list[dict[str, Any]] = [{'type': 'text', 'text': intro}]
    archived_content: list[dict[str, Any]] = [{'type': 'text', 'text': intro}]
    provenance: list[dict[str, Any]] = []
    active_plane = None
    for item in image_items:
        if item['plane'] != active_plane:
            active_plane = item['plane']
            heading = f'{active_plane.upper()} INPUT'
            request_content.append({'type': 'text', 'text': heading})
            archived_content.append({'type': 'text', 'text': heading})
        path = item['path']
        sha = file_sha256(path)
        request_content.append({'type': 'image_url', 'image_url': {'url': image_data_url(str(path))}})
        archived_content.append({'type': 'image_url', 'image_url': {'url': f'sha256:{sha}'}})
        provenance.append({'plane': item['plane'], 'ordinal': item['ordinal'], 'source_index': item['source_index'], 'path': str(path), 'sha256': sha})
    request_content.append({'type': 'text', 'text': prompt})
    archived_content.append({'type': 'text', 'text': prompt})
    system = {'role': 'system', 'content': 'You are a careful musculoskeletal radiologist. Use only the supplied MRI images and original radiology report, keep their evidence separate, and never invent clinical context.'}
    return ([system, {'role': 'user', 'content': request_content}], [system, {'role': 'user', 'content': archived_content}], provenance)
