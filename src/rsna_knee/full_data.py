"""Final S4349+H58 protocol; FinalTrainer owns model/loaders and TrainingEngine owns optimizer steps.

Ported from the frozen FULLMIX5-SWA implementation. No validation or selection;
30-epoch OneCycle, fixed 19 epochs, uniform EMA epochs 15--19.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .final_config import validate_recipe
from .input_pipeline import LABELS
from .final_preparation import sha
from .training_data import build_canonical_dataset

FULL_PROTOCOL = 'full-data-s4349-h58-no-heldout-v1'
FULL_VERSION = 'full-data-fixed-ema-swa-training-v1'
HORIZON_EPOCHS = 30
STOP_AFTER_EPOCHS = 19
SWA_EPOCHS_0IDX = (14, 15, 16, 17, 18)
SWA_SPEC = {'members_epochs_0indexed': list(SWA_EPOCHS_0IDX), 'members_epochs_1indexed': [15, 16, 17, 18, 19],
            'weighting': 'uniform', 'source': 'ema', 'bn_recalibration': False,
            'nonfloat_buffers': 'copied from last member (epoch index 18)'}
CHECKPOINT_RULE = 'fixed_uniform_ema_swa_epochs_15_19_1indexed_no_selection'


def full_audit(records):
    ids = [r['sample_id'] for r in records]
    weak = [r for r in records if r['source_group'] == 'S']
    hard = [r for r in records if r['source_group'] == 'H']
    if len(ids) != 4407 or len(set(ids)) != 4407 or len(weak) != 4349 or len(hard) != 58:
        raise ValueError('full-data records require unique S4349+H58')
    if (any(r['is_hard'] for r in weak) or any(not r['is_hard'] for r in hard)
            or any(r['split_role'] != 'train' for r in records)):
        raise ValueError('full-data source flags or training partition mismatch')
    labels = np.asarray([r['train_labels'] for r in records], dtype=float)
    if labels.shape != (4407, 12) or not np.isfinite(labels).all() or not ((labels >= 0) & (labels <= 1)).all():
        raise ValueError('full-data labels must be finite 12-target probabilities')
    if not all(np.isin(r['train_labels'], [0, 1]).all() for r in hard):
        raise ValueError('H must retain original binary labels')
    return dict(passed=True, train_count=4407, train_S=4349, train_H=58, validation_count=0,
                ids_unique=True, S_H_disjoint=True, hard_prevalence_initialization=False)


def verify_full(config, root):
    spec = config['fulldata']; path = Path(root) / spec['records_path']
    if sha(path) != spec['records_sha256']:
        raise ValueError('full-data records hash mismatch')
    payload = json.loads(path.read_text())
    if payload['protocol_id'] != FULL_PROTOCOL or payload['targets'] != list(LABELS):
        raise ValueError('full-data records protocol/target order mismatch')
    for item in payload['sources'].values():
        if sha(Path(root) / item['path']) != item['sha256']:
            raise ValueError(f"full-data source hash mismatch: {item['path']}")
    full_audit(payload['records'])
    return payload


def read_full_config(path, root):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = {'version': FULL_VERSION, 'evaluation_protocol': FULL_PROTOCOL,
                'onecycle_horizon_epochs': HORIZON_EPOCHS, 'stop_after_epochs': STOP_AFTER_EPOCHS,
                'swa': SWA_SPEC, 'checkpoint_rule': CHECKPOINT_RULE}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('full-data protocol, horizon, stop or SWA rule differs from final training')
    if any(key in value for key in ('early_stopping', 'simlb', 'joint_oof')):
        raise ValueError('full-data training cannot select checkpoints on a holdout')
    if value['data']['batch_size'] != 8 or not value['data']['drop_last']:
        raise ValueError('final training requires batch8/drop_last and 550 steps per epoch')
    coat = value['model']['backbone'] == 'timm:coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
    if (coat and (value['model'].get('backbone_module') != 'coatnet' or value['model'].get('backbone_image_chunk') != 32)
            or not coat and value['model'].get('backbone_image_chunk') is not None):
        raise ValueError('final CoAtNet requires chunk32; final Base uses no chunking')
    validate_recipe(value, Path(root))
    verify_full(value, root)
    if value['runtime']['amp_dtype'] != 'bfloat16':
        raise ValueError('the selected baseline uses bf16')
    return value


def build_full_dataset(*, repo_root, config):
    payload = verify_full(config, repo_root)
    if config.get('release_format') == 'final-solution-v1':
        # Final classifiers use numeric targets only. Build the image dataset
        # without requiring unused reports, diagnostic text or historical folds.
        from .input_pipeline import KneeInputConfig, KneeStudyDataset, resolve_config_paths
        path = Path(repo_root) / config['input_config']
        paths = resolve_config_paths(path, repo_root)
        source = KneeStudyDataset(
            config=KneeInputConfig.from_json(path), cache_root=paths['cache_root'],
            stage_cache_root=paths['stage_cache_root'],
            selection_manifest=paths['selection_manifest'], repo_root=repo_root,
            training_tables=None, augment=False)
    else:
        source = build_canonical_dataset(repo_root=repo_root, input_config=config['input_config'], include_folds=(-1, 0, 1, 2, 3, 4))
    base = {r['Sample ID']: r for r in source.rows}
    if len(source.rows) != len(base) or set(base) != {r['sample_id'] for r in payload['records']}:
        raise ValueError('canonical study IDs differ from pinned full-data records')
    rows = []
    for record in payload['records']:
        # Same row routing as simlb.build_simlb_dataset(role='train').
        row = dict(base[record['sample_id']]); row.update(zip(LABELS, record['train_labels']))
        row.update(source_group=record['source_group'], is_hard=record['is_hard'], original_oof_fold=record['original_oof_fold'],
                   diagnostic_fold=record['diagnostic_fold'], split_role='train')
        row['fold'] = record['original_oof_fold'] if record['is_hard'] else -1
        rows.append(row)
    source.rows = rows
    if config.get('release_format') == 'final-solution-v1':
        source.inference_only = False
        source.numeric_targets = True
    return source


def run_full_data(self) -> dict[str, Any]:
    """Train 19 of 30 OneCycle epochs; save EMA at the end of epochs 14..18 (0-indexed). No evaluation."""
    from .training_runtime import atomic_torch_save
    self.total_steps = self.scheduler.total_steps
    self.stop_step = STOP_AFTER_EPOCHS * len(self.loaders["train"].loader)
    started = time.perf_counter(); epochs = []
    members = self.run_dir / 'swa_members'; members.mkdir(exist_ok=True); records = []
    steps = len(self.loaders['train'].loader)
    assert self.scheduler.total_steps == self.total_steps and self.stop_step == STOP_AFTER_EPOCHS * steps
    for epoch in range(STOP_AFTER_EPOCHS):
        train = self.train_epoch(epoch)
        if not np.isfinite(train['loss']):
            raise RuntimeError('non-finite training loss')
        assert self.global_step == (epoch + 1) * steps
        if epoch in SWA_EPOCHS_0IDX:
            path = members / f'epoch_{epoch:02d}_ema.pt'
            atomic_torch_save(self.ema.state_dict(), path)
            records.append(dict(epoch_0indexed=epoch, epoch_1indexed=epoch + 1, global_step=self.global_step,
                                ema_n_averaged=int(self.ema.n_averaged), sha256=sha(path),
                                learning_rates=[float(g['lr']) for g in self.optimizer.param_groups]))
            self.logger.save_json('swa_members/manifest.json', dict(members=records))
        if epoch == STOP_AFTER_EPOCHS - 1:
            atomic_torch_save(self.model.state_dict(), members / f'epoch_{epoch:02d}_raw.pt')
        checkpoint = self._checkpoint(epoch)
        atomic_torch_save(checkpoint, self.run_dir / 'checkpoint_latest.pt')
        epochs.append(dict(epoch=epoch, train=train, global_step=self.global_step,
                           learning_rates=[float(g['lr']) for g in self.optimizer.param_groups]))
        self.logger.save_json('epoch_summary.json', epochs)
    assert self.global_step == self.stop_step, (self.global_step, self.stop_step)
    summary = dict(status='trained_pending_swa', protocol_id=FULL_PROTOCOL, epochs_run=len(epochs), global_steps=self.global_step,
                   total_schedule_steps=self.total_steps, stop_step=self.stop_step, stop_reason='fixed_stop_after_epoch_19',
                   swa_members=records, training_seconds=time.perf_counter() - started, onecycle_horizon_epochs=HORIZON_EPOCHS,
                   raw_epoch18_sha256=sha(members / 'epoch_18_raw.pt'), validation=None)
    self.logger.save_json('summary.json', summary)
    return summary

def uniform_average(states, int_source):
    """Identical semantics to scripts/build_swa.py uniform_average."""
    keys = list(int_source)
    if not states or any(list(s) != keys for s in states):
        raise ValueError('SWA state keys differ')
    return {k: (sum(s[k].double() for s in states) / len(states)).to(int_source[k].dtype) if int_source[k].is_floating_point() else int_source[k].clone()
            for k in keys}


def build_swa(train_dir: Path, output_dir: Path) -> dict:
    """Verify the five saved EMA members and export their exact uniform mean on CPU.

    Loading the saved tensors back verifies serialization, not prediction parity.
    """
    from .training_runtime import atomic_torch_save
    summary = json.loads((train_dir / 'summary.json').read_text())
    members = json.loads((train_dir / 'swa_members/manifest.json').read_text())['members']
    if (summary['epochs_run'] != STOP_AFTER_EPOCHS or
            summary['global_steps'] != 10450 or summary['stop_step'] != 10450 or
            summary['total_schedule_steps'] != 16500 or
            [m['epoch_0indexed'] for m in members] != list(SWA_EPOCHS_0IDX)):
        raise ValueError('incomplete final training or wrong SWA window')
    states = []
    for member in members:
        path = train_dir / 'swa_members' / f"epoch_{member['epoch_0indexed']:02d}_ema.pt"
        if sha(path) != member['sha256']:
            raise ValueError(f'SWA member changed: {path}')
        states.append(torch.load(path, map_location='cpu', weights_only=True))
    latest = torch.load(train_dir / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    if latest['epoch'] != 18 or latest['global_step'] != 10450:
        raise ValueError('latest checkpoint is not the completed final epoch')
    if list(latest['ema']) != list(states[-1]) or any(not torch.equal(latest['ema'][k], v) for k, v in states[-1].items()):
        raise ValueError('last SWA member differs from latest checkpoint')
    raw = torch.load(train_dir / 'swa_members/epoch_18_raw.pt', map_location='cpu', weights_only=True)
    if list(raw) != list(latest['model']) or any(not torch.equal(raw[k], v) for k, v in latest['model'].items()):
        raise ValueError('last raw state differs from latest checkpoint')
    for state in states:
        if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
            raise ValueError('non-finite SWA member')
    averaged = uniform_average(states, states[-1])
    if any(not torch.isfinite(v).all() for v in averaged.values() if v.is_floating_point()):
        raise ValueError('non-finite SWA result')
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_torch_save(dict(latest, ema=averaged, swa=SWA_SPEC), output_dir / 'checkpoint_swa_ema.pt')
    model = {k.removeprefix('module.'): v for k, v in averaged.items() if k != 'n_averaged'}
    atomic_torch_save(model, output_dir / 'model_swa.pt')
    reloaded = torch.load(output_dir / 'model_swa.pt', map_location='cpu', weights_only=True)
    if list(reloaded) != list(model) or any(not torch.equal(reloaded[k], v) for k, v in model.items()):
        raise RuntimeError('SWA serialization verification failed')
    result = dict(passed=True, members=members, specification=SWA_SPEC,
                  config_sha256=latest['config_sha256'], run_contract_sha256=latest['run_contract_sha256'],
                  checkpoint_swa_ema_sha256=sha(output_dir / 'checkpoint_swa_ema.pt'),
                  model_swa_sha256=sha(output_dir / 'model_swa.pt'), tensor_reload_exact=True,
                  prediction_reload_verified=False)
    (output_dir / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
