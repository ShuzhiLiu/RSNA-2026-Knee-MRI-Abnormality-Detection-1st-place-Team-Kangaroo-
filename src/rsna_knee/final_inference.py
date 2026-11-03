"""Final MRI-only inference and portable model packaging."""
from __future__ import annotations
import json
from pathlib import Path
import shutil

from .final_preparation import load, save, sha, TARGETS


def export_package(*, workspace, recipe, run, output):
    """Package raw model_swa tensors; keep all inference dependencies local."""
    workspace, run, output = map(Path, (workspace, run, output))
    config = load(workspace/'configs'/f'{recipe}.json')
    if config.get('release_format') != 'final-solution-v1':
        raise ValueError('expected a prepared final workspace')
    # Bind export to the recipe that actually produced this run.
    trained = load(run/'train/resolved_config.json')
    if config != trained:
        raise ValueError('workspace recipe differs from the trained resolved config')
    if load(run/'train/summary.json')['epochs_run'] != 19:
        raise ValueError('final training is incomplete')
    _write_package(workspace=workspace, config=config, output=output, recipe=recipe,
                   write_state=lambda path: shutil.copy2(run/'swa/model_swa.pt', path),
                   state_format='raw_model')


def smoke_package(*, workspace, recipe, ema_state, output, steps, loss):
    """Package a few-step EMA state so the MRI-only predictor can be exercised end to end.

    The weights are NOT a usable model; this only checks the train -> package -> predict path.
    """
    import torch
    workspace, output = Path(workspace), Path(output)
    config = load(workspace/'configs'/f'{recipe}.json')
    _write_package(workspace=workspace, config=config, output=output, recipe=recipe,
                   write_state=lambda path: torch.save(ema_state, path), state_format='ema_module_prefixed',
                   smoke=dict(optimizer_steps=int(steps), mean_loss=float(loss), usable_model=False))


def _write_package(*, workspace, config, output, recipe, write_state, state_format, **extra):
    output.mkdir(parents=True, exist_ok=False)
    manifest = load(workspace/config['model']['pretrained_manifest'])
    pretrained = workspace/manifest['target_dir']
    for entry in manifest['files']:
        source = pretrained/entry['name']
        if sha(source) != entry['sha256']:
            raise ValueError(f'pretrained file changed: {source}')
        target = output/'pretrained'/entry['name'];target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest['target_dir'] = 'pretrained'
    config['model']['pretrained_manifest'] = 'configs/pretrained.json'
    save(output/'configs/pretrained.json', manifest)
    save(output/'configs/run.json', config)
    write_state(output/'state.pt')
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    save(output/'PACKAGE.json', dict(format='final-model-package-v1', recipe=recipe,
                                     state_format=state_format, files_sha256=files,
                                     prediction_reload_verified=False, **extra))


def build_member(package, device):
    import torch
    from .dense_window_model import build_model
    package = Path(package).resolve(strict=True)
    if (package/'PACKAGE.json').exists():
        for name, expected in load(package/'PACKAGE.json')['files_sha256'].items():
            path = (package/name).resolve()
            if not path.is_relative_to(package) or sha(path) != expected:
                raise ValueError(f'package integrity failure: {name}')
    config = load(package/'configs/run.json')
    model, _ = build_model(package, config)
    if config['model'].get('backbone_image_chunk') is not None:
        model.backbone._rsna_backbone_image_chunk = config['model']['backbone_image_chunk']
        model.backbone._rsna_track_calls = False
        model.backbone._rsna_chunk_calls = []
    if model.depth != 32 or model.windows != 30 or model.grid_size != 12:
        raise ValueError('expected final native32 / 384px model')
    # Only these known train-only keys may be absent from a predictor state.
    del model.locpool.mil
    for name in ('replacement_step', 'replacement_seed'):
        delattr(model.locpool, name)
    state = torch.load(package/'state.pt', map_location='cpu', weights_only=True)
    if 'n_averaged' in state or any(k.startswith('module.') for k in state):
        if not all(k.startswith('module.') for k in state if k != 'n_averaged'):
            raise ValueError('mixed raw and EMA-prefixed state keys')
        state = {k.removeprefix('module.'): v for k,v in state.items() if k != 'n_averaged'}
    wanted = model.state_dict()
    extra = set(state)-set(wanted)
    if set(wanted)-set(state) or any(not (k.startswith('locpool.mil.') or k in
        ('locpool.replacement_step', 'locpool.replacement_seed')) for k in extra):
        raise ValueError('checkpoint does not match final prediction modules')
    model.load_state_dict({k: state[k] for k in wanted}, strict=True)
    model.eval();model.annotation_batch = None;model.box_aux_enabled = False
    return model.to(device)


def read_series_rows(path):
    """Group test_series.csv by study.

    Only the identifiers are strings; Fluid_Sensitive / Fat_Suppression must stay numeric because
    series selection tests them with `== 1` (identical to the deployed competition predictor).
    """
    import pandas as pd
    series = pd.read_csv(path, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    return {u: g.to_dict('records') for u, g in series.groupby('StudyInstanceUID')}


def predict(*, packages, data, output, device='cuda:0', precision='fp16', workers=2):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from .inference.raw32_input import Studies, P, collate, inputs_on_device
    from .training_runtime import evaluation_backend
    packages = [Path(p).resolve() for p in packages]
    if len(packages) not in (1, 3) or len(set(packages)) != len(packages):
        raise ValueError('supply one member or three distinct final members')
    if device == 'cpu' and precision != 'fp32':
        raise ValueError('CPU inference requires --precision fp32')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.backends.cudnn.enabled = True;torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True;torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    P.DATA = Path(data)
    ids = pd.read_csv(P.DATA/'test.csv', dtype=str).StudyInstanceUID.tolist()
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('test study identifiers must be nonempty and unique')
    rows = read_series_rows(P.DATA/'test_series.csv')
    models = [build_member(p, device) for p in packages]
    dataset = Studies(ids, rows, size=384)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=collate, pin_memory=device.startswith('cuda'))
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': None}[precision]
    values = [[] for _ in models]
    with torch.inference_mode(), evaluation_backend():
        for batch in loader:
            inputs = inputs_on_device(batch, device)
            for j, model in enumerate(models):
                if dtype is None:
                    logits = model(**inputs)
                else:
                    with torch.autocast('cuda', dtype=dtype):
                        logits = model(**inputs)
                values[j].extend(logits.float().sigmoid().cpu().tolist())
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.shape != (len(models), len(ids), 12) or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('invalid final probabilities')
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probabilities.mean(axis=0), index=pd.Index(ids, name='StudyInstanceUID'), columns=TARGETS).to_csv(output, mode='x')
    save(output.with_suffix('.members.json'), dict(sample_ids=ids, targets=TARGETS,
         packages=[str(p) for p in packages], probabilities=probabilities.tolist(), precision=precision))


def import_weights(*, source, output):
    """Convert a downloaded trained-weight dataset into a predictor package.

    Accepted layout (as published on Kaggle): state.pt, pretrained.safetensors, PROVENANCE.json and
    code/configs/{run,pretrained}.json (+ code/pretrained/README.md, config.json); `code.zip` instead of
    an expanded `code/` directory is also accepted.
    """
    import zipfile
    source, output = Path(source).resolve(strict=True), Path(output)
    provenance = load(source/'PROVENANCE.json')
    expected = {w['filename']: w['sha256'] for w in provenance['weights']}
    for name in ('state.pt', 'pretrained.safetensors'):
        if sha(source/name) != expected.get(name):
            raise ValueError(f'weight file does not match PROVENANCE.json: {name}')

    def read_code(relative):
        if (source/'code').is_dir():
            return (source/'code'/relative).read_bytes()
        with zipfile.ZipFile(source/'code.zip') as archive:
            return archive.read(relative)

    run = json.loads(read_code('configs/run.json'))
    manifest = json.loads(read_code('configs/pretrained.json'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'pretrained').mkdir()
    for entry in manifest['files']:
        target = output/'pretrained'/entry['name']
        if entry['name'] == 'model.safetensors':
            shutil.copy2(source/'pretrained.safetensors', target)
        else:
            target.write_bytes(read_code('pretrained/'+entry['name']))
        if sha(target) != entry['sha256']:
            raise ValueError(f'pretrained file mismatch: {entry["name"]}')
    manifest['target_dir'] = 'pretrained'
    save(output/'configs/run.json', run)
    save(output/'configs/pretrained.json', manifest)
    shutil.copy2(source/'state.pt', output/'state.pt')
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    package = provenance.get('package', {})
    save(output/'PACKAGE.json', dict(format='final-model-package-v1', recipe=package.get('label'),
                                     state_format='raw_model', files_sha256=files,
                                     source_state_sha256=expected['state.pt'],
                                     swa_epochs_1indexed=package.get('swa_epochs_1indexed')))
    return output
