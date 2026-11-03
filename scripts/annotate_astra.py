#!/usr/bin/env python3
"""Astra v1 study-level annotation (reference runner).

  prepare  Build one study's annotation case from the preprocessed series cache: series selection,
           fluid-series overview sheets, report projection and the viewer manifest. No model call.
  run      Print (default) or execute (--execute) the `codex exec` command in which the model reads the
           case through the bounded MRI viewer (MCP server, <=10 calls / <=10 sheets) and returns the
           JSON defined by acquire_schema.json: twelve findings and twelve continuous probabilities.
  collect  Gather output.json files into a StudyInstanceUID x 12-target probability CSV.

The production run additionally had durable retries, budget accounting and resumable bookkeeping;
those are intentionally omitted here. Re-running annotation is a paid, non-deterministic API
operation and will not reproduce the released probabilities exactly. The released numeric labels
are what the classifiers were trained on.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
SOURCE = ROOT/'src/rsna_knee/annotation/astra_v1'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
           'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def selection_row(path, uid):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for ordinal, line in enumerate(stream):
            row = json.loads(line)
            if row['study_uid'] == uid:
                return ordinal, row
    raise KeyError(f'study not in selection manifest: {uid}')


def report_for(train_csv, uid):
    with Path(train_csv).open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['StudyInstanceUID'] == uid:
                return row['Report']
    raise KeyError(f'study not in train.csv: {uid}')


def ensure_cache(raw_root, cache_root, uid, candidate):
    """Decode a candidate series that is not cached yet (identical preprocessing to training)."""
    from rsna_knee.preprocessing import PreprocessConfig, preprocess_series, save_cached_series
    destination = Path(cache_root)/Path(candidate['volume_path']).parent
    if (destination/'volume.npy').is_file() and (destination/'metadata.json').is_file():
        return
    files = sorted((Path(raw_root)/'train_series'/uid/candidate['series_uid']).glob('*.dcm'))
    if not files:
        raise FileNotFoundError(f'no DICOM files for {uid}/{candidate["series_uid"]}')
    save_cached_series(preprocess_series(files, PreprocessConfig()), destination)


def prepare(args):
    from rsna_knee.annotation.astra_v1 import inputs
    ordinal, source = selection_row(args.selection, args.study)
    for candidate in source['series_candidates']:
        if inputs.eligible(candidate):
            ensure_cache(args.raw_root, args.cache_root, args.study, candidate)
    spec = dict(study_uid=args.study, ordinal=ordinal, report=report_for(args.train_csv, args.study),
                selected=inputs.selection(source, Path(args.cache_root)))
    result = inputs.prepare_case(spec, Path(args.run_dir),
                                 dict(cache_root=str(Path(args.cache_root).resolve()), raw_root=str(Path(args.raw_root).resolve())))
    print(Path(args.run_dir)/'cases'/args.study/'input.json')
    return result


def command(case, attempt, model, reasoning):
    """The production `codex exec` argv: no shell/web/project docs, read-only sandbox, MRI viewer only."""
    settings = {
        'model_reasoning_effort': reasoning, 'project_doc_max_bytes': 0,
        'model_instructions_file': str(SOURCE/'acquire_system.txt'),
        'features.shell_tool': False, 'web_search': 'disabled',
        'mcp_servers.mri.command': sys.executable,
        'mcp_servers.mri.args': [str(ROOT/'scripts/prepare_final_annotation.py'), 'astra-viewer', '--version', 'v1',
                                 '--case', case['case_path'], '--log', str(attempt/'views')],
        'mcp_servers.mri.required': True, 'mcp_servers.mri.enabled_tools': ['view_slices', 'compare_location'],
        'mcp_servers.mri.startup_timeout_sec': 30, 'mcp_servers.mri.tool_timeout_sec': 60}
    overrides = []
    for key, value in settings.items():
        overrides.extend(['-c', key+'='+json.dumps(value)])
    return ['codex', 'exec', '--strict-config', '--ignore-user-config', '--ignore-rules', '--skip-git-repo-check',
            '--sandbox', 'read-only', '-C', str(attempt/'empty_cwd'), '-m', model, *overrides, '--json',
            '--output-schema', str(SOURCE/'acquire_schema.json'), '-o', str(attempt/'output.json'),
            *[f'--image={image["path"]}' for image in case['images']], '-']


def run(args):
    case = json.loads(args.case.read_text())
    prompt = (SOURCE/'acquire_prompt.txt').read_text().format(
        input_map=case['input_map'], report=case['report_projection']['projected_report'])
    attempt = args.attempt.resolve()
    argv = command(case, attempt, args.model, args.reasoning)
    if not args.execute:
        print(json.dumps(dict(argv=argv, stdin_prompt_characters=len(prompt)), indent=2))
        return
    import jsonschema
    (attempt/'empty_cwd').mkdir(parents=True, exist_ok=False)
    (attempt/'prompt.txt').write_text(prompt)
    with (attempt/'events.jsonl').open('wb') as events, (attempt/'stderr.log').open('wb') as errors:
        subprocess.run(argv, input=prompt.encode(), stdout=events, stderr=errors, timeout=args.timeout, check=True)
    output = json.loads((attempt/'output.json').read_text())
    jsonschema.validate(output, json.loads((SOURCE/'acquire_schema.json').read_text()))
    (attempt/'result.json').write_text(json.dumps(dict(study_uid=case['study_uid'], **output), indent=2)+'\n')
    print(json.dumps(output['probabilities'], indent=2))


def collect(args):
    rows = {}
    for path in sorted(Path(args.attempts).rglob('result.json')):
        value = json.loads(path.read_text())
        uid = value['study_uid']
        if uid in rows:
            raise ValueError(f'duplicate result for {uid}: {path}')
        rows[uid] = [float(value['probabilities'][t]) for t in TARGETS]
    with Path(args.output).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID', *TARGETS])
        for uid, values in rows.items():
            writer.writerow([uid, *values])
    print(f'{len(rows)} studies -> {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--selection', type=Path, default=ROOT/'assets/final_solution/training_selection.jsonl.gz')
    p.add_argument('--train-csv', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True, help='official competition data root')
    p.add_argument('--cache-root', type=Path, required=True, help='series cache (same as final_solution.py inputs)')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--study', required=True)
    p = sub.add_parser('run')
    p.add_argument('--case', type=Path, required=True, help='<run-dir>/cases/<study>/input.json')
    p.add_argument('--attempt', type=Path, required=True, help='new output directory for this attempt')
    p.add_argument('--model', default='gpt-6-astra')
    p.add_argument('--reasoning', default='medium')
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--execute', action='store_true', help='actually call the model (paid API usage)')
    p = sub.add_parser('collect')
    p.add_argument('--attempts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'collect': collect}[args.command](args)


if __name__ == '__main__':
    main()
