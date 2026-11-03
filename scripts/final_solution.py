#!/usr/bin/env python3
"""RSNA knee MRI final three-model solution.

Typical order (see README.md):

  hard-labels -> pretrained -> inputs -> prepare -> smoke (optional) -> train -> export -> predict -> blend
  released weights: import-weights -> predict
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rsna_knee.final_preparation import RECIPES, prepare


def verify_workspace(workspace, recipe):
    """Check the prepared configuration binding and annotation identities before using a GPU."""
    from rsna_knee.final_preparation import load, sha
    binding = load(workspace/'preparation.json')
    if not binding.get('configs_sha256'):
        raise ValueError('workspace has no prepared configuration binding')
    for name, expected in binding['configs_sha256'].items():
        if sha(workspace/name) != expected:
            raise ValueError(f'prepared configuration changed: {name}')
    from rsna_knee.full_data import read_full_config
    from rsna_knee.final_config import verify_location_assets
    config = read_full_config(workspace/'configs'/f'{recipe}.json', workspace)
    verify_location_assets(workspace, config)
    for path_key, hash_key in [('index', 'index_sha256'), ('index_manifest', 'index_manifest_sha256')]:
        if sha(config['boxcombo'][path_key]) != config['boxcombo'][hash_key]:
            raise ValueError('lesion annotation binding changed')
    return config


def require_free_gpu():
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True)
    if gpu.stdout.strip():
        raise RuntimeError('GPU occupied; refusing concurrent training')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('hard-labels', help='extract the 58 human-labelled studies from the official train.csv')
    p.add_argument('--train-csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('pretrained', help='download and verify the two pinned ImageNet backbones from Hugging Face')
    p.add_argument('--output-root', type=Path, required=True)
    p = sub.add_parser('inputs', help='decode selected training DICOMs and build spatial caches')
    for name in ('raw-root', 'selection', 'cache-root', 'stage-cache-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--input-config', type=Path, default=ROOT/'configs/knee_input_pipeline_384_astra111.json')
    p = sub.add_parser('prepare', help='bind released annotations and numeric targets to a new workspace')
    for name in ('supervision', 'hard-labels', 'selection', 'cache-root', 'stage-cache-root', 'base-pretrained', 'coat-pretrained', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('smoke', help='run a few optimizer steps of one recipe and package the EMA state (pipeline check only)')
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--recipe', choices=RECIPES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--workers', type=int, help='data-loader workers (default: recipe value)')
    p = sub.add_parser('train', help='train one fixed final recipe (19 epochs), then average EMA epochs 15-19')
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--recipe', choices=RECIPES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    p = sub.add_parser('export', help='package newly trained EMA SWA for the final predictor')
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--recipe', choices=RECIPES, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('import-weights', help='convert a downloaded trained-weight dataset into a predictor package')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('predict', help='MRI-only inference from one or three member packages')
    p.add_argument('--packages', nargs='+', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=('fp16', 'bf16', 'fp32'), default='fp16')
    p.add_argument('--workers', type=int, default=2)
    p = sub.add_parser('blend', help='align and average the three members; optionally blend 1:1 with the teammate group')
    p.add_argument('--members', type=Path, nargs=3, required=True)
    p.add_argument('--teammate', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.command == 'hard-labels':
        from rsna_knee.final_preparation import extract_hard_labels
        print(extract_hard_labels(args.train_csv, args.output))
        return
    if args.command == 'pretrained':
        from rsna_knee.final_preparation import download_pretrained
        for name in ('pretrained_maxxvitv2_base384.json', 'pretrained_coatnet_rmlp2.json'):
            print(download_pretrained(ROOT/'configs'/name, args.output_root))
        return
    if args.command == 'import-weights':
        from rsna_knee.final_inference import import_weights
        print(import_weights(source=args.source, output=args.output))
        return
    if args.command == 'blend':
        from rsna_knee.final_data import blend_predictions
        blend_predictions(args.members, args.output, args.teammate)
        return
    if args.command == 'inputs':
        from rsna_knee.final_data import build_inputs
        kwargs = vars(args).copy();kwargs.pop('command')
        build_inputs(**kwargs)
        return
    if args.command == 'prepare':
        kwargs = vars(args).copy();kwargs.pop('command')
        print(prepare(code_root=ROOT, **kwargs))
        return
    # Deterministic CUDA backend required by the trainer and predictor.
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', TIMM_USE_SCIPY_INTERP='0')
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
    if args.command == 'predict':
        from rsna_knee.final_inference import predict
        predict(packages=args.packages, data=args.data, output=args.output,
                device=args.device, precision=args.precision, workers=args.workers)
        return
    if args.command == 'export':
        from rsna_knee.final_inference import export_package
        export_package(workspace=args.workspace, recipe=args.recipe, run=args.run, output=args.output)
        return
    workspace = args.workspace.resolve(strict=True)
    output = args.output.resolve()
    if args.command == 'smoke':
        from rsna_knee.trainer import FinalTrainer
        from rsna_knee.final_inference import smoke_package
        if args.steps < 1:
            raise ValueError('--steps must be positive')
        verify_workspace(workspace, args.recipe)
        require_free_gpu()
        output.mkdir(parents=True, exist_ok=False)
        trainer = FinalTrainer(repo_root=workspace, config_path=workspace/'configs'/f'{args.recipe}.json',
                               run_dir=output/'train', workers=args.workers)
        try:
            result = trainer.train_epoch(0, max_steps=args.steps)
            smoke_package(workspace=workspace, recipe=args.recipe, ema_state=trainer.ema.state_dict(),
                          output=output/'package', steps=trainer.global_step, loss=result['loss'])
        finally:
            trainer.close()
        print(json.dumps(dict(recipe=args.recipe, global_steps=trainer.global_step, mean_loss=result['loss'],
                              package=str(output/'package')), indent=2))
        return
    # train
    if not args.frozen:
        verify_workspace(workspace, args.recipe)
        require_free_gpu()
        output.mkdir(parents=True, exist_ok=False)
        # Freeze the exact code and prepared configuration used by this run.
        snapshot = output/'source'
        shutil.copytree(ROOT/'src', snapshot/'src', ignore=shutil.ignore_patterns('__pycache__'))
        (snapshot/'scripts').mkdir()
        shutil.copy2(__file__, snapshot/'scripts/final_solution.py')
        shutil.copytree(ROOT/'configs', snapshot/'configs')
        files = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in snapshot.rglob('*') if p.is_file()}
        (output/'source_manifest.json').write_text(json.dumps(files, indent=2)+'\n')
        config_files = {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (workspace/'configs').glob('*.json')}
        (output/'workspace_config_manifest.json').write_text(json.dumps(config_files, indent=2)+'\n')
        frozen_workspace = output/'workspace'
        shutil.copytree(workspace/'configs', frozen_workspace/'configs')
        subprocess.run([sys.executable, str(snapshot/'scripts/final_solution.py'), 'train',
                        '--workspace', str(frozen_workspace), '--recipe', args.recipe,
                        '--output', str(output), '--frozen'], check=True)
        return
    from rsna_knee.trainer import FinalTrainer
    from rsna_knee.full_data import build_swa
    trainer = FinalTrainer(repo_root=workspace, config_path=workspace/'configs'/f'{args.recipe}.json', run_dir=output/'train')
    try:
        trainer.run()
    finally:
        trainer.close()
    build_swa(output/'train', output/'swa')


if __name__ == '__main__':
    main()
