"""Final full-data trainer: 19 epochs, EMA15–19 averaging, no validation selection."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .box_data import BoxDataset, collate_box
from .dense_window_model import build_model
from .ema import build_ema
from .final_config import verify_location_assets
from .full_data import read_full_config, build_full_dataset, full_audit, verify_full, run_full_data
from .input_pipeline import LABELS
from .logger import RunLogger
from .optim import build_adamw, build_onecycle
from .training_data import BaselineLoader, EpochShuffleSampler
from .training_engine import TrainingEngine
from .training_runtime import seed_everything, environment_record


def weak_prevalence(dataset):
    """Initialize from the 4,349 S targets only; retain H binary targets in loss."""
    rows = [r for r in dataset.rows if r['source_group'] == 'S']
    if len(rows) != 4349:
        raise ValueError('final fitting requires S4349 prevalence')
    return torch.tensor([[float(r[t]) for t in LABELS] for r in rows],dtype=torch.float32).mean(0)


class FinalTrainer(TrainingEngine):
    def __init__(self, *, repo_root, config_path, run_dir, workers=None):
        self.root = Path(repo_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.config = read_full_config(self.config_path,self.root)
        # Optional loader-worker override (smoke runs only); recorded in resolved_config.json.
        self.overrides = {} if workers is None else {'workers': int(workers)}
        self.run_dir = Path(run_dir).resolve()
        self.logger = RunLogger(self.run_dir)
        self.device = torch.device(self.config['runtime']['device'])
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('final fitting requires CUDA')
        self.seed = int(self.config['seed'])
        seed_everything(self.seed)
        self._configure_runtime()
        resolved = self.resolved_config()
        self.run_contract_sha256 = hashlib.sha256(json.dumps(resolved,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.logger.save_json('resolved_config.json',resolved)
        self.logger.save_json('environment.json',environment_record(self.root))
        verify_location_assets(self.root,self.config)
        self.loaders = self._build_loaders()
        self.model, report = build_model(self.root,self.config)
        self.model.initialize_head_bias(weak_prevalence(self.loaders['train'].dataset.dataset))
        self.model.to(self.device)
        self._location_rows = []
        usage = json.loads((self.root/self.config['location_supervision']['usage_map']).read_text())['combinations']
        roles = self.loaders['train'].dataset.dataset.config.roles
        classes = ['MCL','MM','LM','MedOA','LatOA','PFOA']
        self.model.dense = torch.tensor([[usage.get(c+'|'+r,{}).get('usage')=='DENSE' for c in classes] for r in roles],device=self.device)
        self.model.steps_per_epoch = len(self.loaders['train'].loader)
        self.logger.save_json('backbone_load_report.json',report.__dict__)
        self.logger.save_json('model_summary.json',dict(parameters=sum(p.numel() for p in self.model.parameters()),
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),input_shape=self.config['model']['input_shape']))
        opt = self.config['optimizer']
        self.optimizer, groups = build_adamw(self.model,backbone_lr=float(opt['backbone_lr']),new_lr=float(opt['new_lr']),
            weight_decay=float(opt['weight_decay']),betas=tuple(float(v) for v in opt['betas']),fused=bool(opt['fused']),stage_lr_decay=float(opt['stage_lr_decay']))
        self.logger.save_json('optimizer_groups.json',groups)
        self.scheduler = build_onecycle(self.optimizer,total_steps=30*len(self.loaders['train'].loader),schedule=self.config['schedule'])
        self.ema = build_ema(self.model,float(self.config['ema']['decay'])).to(self.device)
        self.global_step = 0
        # Kept only as checkpoint-format fields, never used for model selection.
        self.best_ema_auc = -float('inf')
        self.best_epoch = None

    def _build_loaders(self):
        source = build_full_dataset(repo_root=self.root,config=self.config)
        data = self.config['data'];lesion = self.config['boxcombo']
        settings = self.config['train_input_augmentation']
        trace = self.run_dir/'augmentation_trace';trace.mkdir(parents=True,exist_ok=True)
        strong = dict(config=source.config.augmentation,seed=self.seed,
            band_scale_range=tuple(float(v) for v in settings['depth_band_scale']),center_jitter=bool(settings['depth_center_jitter']),
            role_dropout_policy=None,trace_dir=trace if settings.get('trace',True) else None)
        dataset = BoxDataset(source,lookup_run=self.root/self.config['location_supervision']['lookup_run'],
            box_index=self.root/lesion['index'],index_manifest=self.root/lesion['index_manifest'],
            index_sha256=lesion['index_sha256'],manifest_sha256=lesion['index_manifest_sha256'],
            depth=32,descriptor_dropout_probability=0.0,seed=self.seed,band=(.12,.88),
            pack_uint16=data['host_image_dtype']=='uint16',strong_augmentation=strong)
        sampler = EpochShuffleSampler(len(dataset),self.seed,shuffle=True)
        kwargs = dict(dataset=dataset,batch_size=self.batch_size,sampler=sampler,num_workers=self.workers,
            pin_memory=bool(data['pin_memory']),drop_last=True,collate_fn=collate_box,
            persistent_workers=bool(data['persistent_workers'] and self.workers>0))
        if self.workers>0:kwargs['prefetch_factor']=int(data['prefetch_factor'])
        loader = BaselineLoader('train',dataset,sampler,DataLoader(**kwargs))
        if len(loader.loader)!=550:
            raise ValueError('final recipe requires 550 optimizer steps per epoch')
        self.logger.save_json('data_contract.json',dict(train=dict(rows=len(dataset),hard=58,weak=4349,
            steps=550,derived_input_shape=self.config['model']['input_shape'],depth_band=[.12,.88],
            strong_train_augmentation=True,cpu_block=vars(strong['config']),depth_center_jitter=strong['center_jitter'],
            train_depth_band_scale=list(strong['band_scale_range']))))
        self.logger.save_json('split_audit.json',full_audit(verify_full(self.config,self.root)['records']))
        return {'train':loader}

    def run(self):
        return run_full_data(self)
