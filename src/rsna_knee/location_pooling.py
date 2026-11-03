"""Adopted location pooling: stage-4 residual spatial attention and train-only MIL.

State names and operation order match the sealed LOCBASE1 candidate. A single
DenseWindowModel subclass shares initialization; no global function patching.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from rsna_knee.dense_window_model import DenseWindowModel
from .location_masks import annotation_views
from .model_backbones import get_backbone_module
LOCAL = (1, 2, 3, 4, 5, 6)

class TokenHead(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.projection = nn.Conv2d(channels, 6, 1)

    def forward(self, x):
        return self.projection(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

class LocationModules(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.spatial = TokenHead(channels)
        self.mil = TokenHead(channels)
        self.gamma = nn.Parameter(torch.full((6,), 0.1))
        self.register_buffer('replacement_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('replacement_seed', torch.tensor(20260823, dtype=torch.long))

class LocationDenseWindowModel(DenseWindowModel):

    def __init__(self, *args, image_size=384, location_seed=20260823, **kwargs):
        super().__init__(*args, **kwargs)
        backbone_functions = get_backbone_module(self.backbone_module)
        grid_size = backbone_functions.validate_location(self.backbone, image_size)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(location_seed)
            self.locpool = LocationModules(self.feature_dim)
        self.steps_per_epoch = 543
        self.grid_size = grid_size
        self.token_count = self.grid_size ** 2
        self.disabled = False
        self.annotation_batch = None
        self.dense = None
        self.aux = {}

    def forward(self, images, series_mask, slice_valid_mask, slice_repeated_mask, pixel_valid_mask=None, slice_positions_normalized=None, gap_previous_mm=None, gap_next_mm=None, *, return_diagnostics=False, box_data=None):
        if self.disabled:
            return super().forward(images, series_mask, slice_valid_mask, slice_repeated_mask, pixel_valid_mask, slice_positions_normalized, gap_previous_mm, gap_next_mm, return_diagnostics=return_diagnostics)
        if images.ndim != 5 or images.shape[1:3] != (5, self.depth):
            raise ValueError(f'Expected B,5,{self.depth},H,W')
        b, _, _, height, width = images.shape
        selected = [self._selected_window_indices(images.device) for _ in range(5)]
        ss = [v.tolist() for v in selected]
        count = len(ss[0])
        flat = torch.cat([self._triplets(images[:, role:role + 1], v).squeeze(1) for role, v in enumerate(ss)], 1).reshape(-1, 3, height, width)
        grid = None
        if self.training:
            flat, grid = self.augment(flat, return_grid=True)
        flat = (flat - self.mean) / self.std
        backbone_functions = get_backbone_module(self.backbone_module)
        features = backbone_functions.feature_map(self.backbone, flat)
        if features.shape[1:] != (self.feature_dim, self.grid_size, self.grid_size):
            raise RuntimeError(f'Unexpected {self.backbone_module} location map shape: {tuple(features.shape)}')
        g = backbone_functions.pool(self.backbone, features)
        spatial_logits = self.locpool.spatial(features).float()
        logatt = spatial_logits.flatten(-2).log_softmax(-1)
        attention = logatt.exp()
        pooled = torch.einsum('nts,ncs->ntc', attention, features.flatten(-2))
        h = backbone_functions.pool(self.backbone, pooled.reshape(-1, self.feature_dim, 1, 1)).reshape(-1, 6, self.feature_dim)
        gamma = self.locpool.gamma
        mixed = g[:, None] + gamma[None, :, None] * (h - g[:, None])
        replacement = torch.zeros((b, 6), device=images.device, dtype=torch.bool)
        if self.training:
            generator = torch.Generator(device=images.device)
            generator.manual_seed((int(self.locpool.replacement_seed) + 1000003 * int(self.locpool.replacement_step)) % (2 ** 63 - 1))
            replacement = torch.rand((b, 6), device=images.device, generator=generator) < 0.0
            mixed = torch.where(replacement[:, None, None, :, None].expand(b, 5, count, 6, self.feature_dim).reshape(-1, 6, self.feature_dim), h, mixed)
        gg = g.reshape(b, 5, count, self.feature_dim)
        ff = mixed.reshape(b, 5, count, 6, self.feature_dim)
        positions = torch.linspace(0, 1, self.depth, device=images.device, dtype=g.dtype).view(1, 1, self.depth).expand(b, 5, self.depth) if slice_positions_normalized is None else slice_positions_normalized.to(g.dtype)
        contexts = []
        window_att = []
        window_valid = []
        entropy = []
        for role, chosen in enumerate(selected):
            valid = self._triplets(slice_valid_mask[:, role:role + 1], ss[role]).squeeze(1)
            repeated = self._triplets(slice_repeated_mask[:, role:role + 1], ss[role]).squeeze(1)
            pos = self._triplets(positions[:, role:role + 1], ss[role]).squeeze(1)
            wm = valid.any(-1) & series_mask[:, role, None]
            ratio = (valid & ~repeated).sum(-1).float() / valid.sum(-1).clamp_min(1).float()
            geometry = torch.stack([pos.mean(-1), (pos[..., -1] - pos[..., 0]).abs(), valid.float().mean(-1).to(g.dtype), ratio.to(g.dtype)], -1)
            geom = self.geometry_projection(geometry)
            base = self.feature_projection(gg[:, role])
            hidden = []
            for target in range(12):
                projected = self.feature_projection(ff[:, role, :, LOCAL.index(target)]) if target in LOCAL else base
                v = projected + geom
                v = v + self.role_embedding[role].view(1, 1, self.hidden)
                v = v + self.window_embedding[chosen].view(1, count, self.hidden)
                hidden.append(v)
            hid = torch.stack(hidden, 2)
            scores = torch.einsum('bwth,th->btw', hid, self.queries) / math.sqrt(self.hidden)
            wa = scores.masked_fill(~wm[:, None], -10000.0).softmax(-1)
            context = torch.einsum('btw,bwth->bth', wa, hid)
            context = context * series_mask[:, role, None].unsqueeze(1).to(context.dtype)
            contexts.append(context)
            window_att.append(wa)
            window_valid.append(wm)
            entropy.append(-(wa.clamp_min(1e-08).log() * wa).sum(-1).mean())
        stacked = torch.stack(contexts, 2)
        emb = stacked.reshape(b, 12, 5 * self.hidden)
        logits = torch.cat([head(emb[:, target]).reshape(b, 1) for target, head in enumerate(self.heads)], 1)
        wa = torch.stack(window_att, 1)
        wv = torch.stack(window_valid, 1)
        self.aux = {'zero': logits.sum() * 0}
        views = None
        if self.annotation_batch is not None:
            if grid is None:
                grid = F.affine_grid(torch.eye(2, 3, device=images.device)[None].expand(b * 5 * count, -1, -1), (b * 5 * count, 1, height, width), align_corners=False)
            views = annotation_views(self.annotation_batch, ss, grid, self.dense)
            self.aux = self._auxiliary(features, logatt, wa, wv, views, b, count, logits)
        spatial = attention.reshape(b, 5, count, 6, self.token_count)
        diag = {'tta_windows': logits.new_tensor(float(5 * count)), 'role_feature_norm': stacked.norm(dim=-1).mean(), 'window_attention_entropy': torch.stack(entropy).mean(), 'backbone_feature_norm': g.float().norm(dim=-1).mean(), 'logit_mean': logits.mean(), 'logit_std': logits.std(unbiased=False)}
        for c, target in enumerate(LOCAL):
            gp = g.float()
            hp = h[:, c].float()
            diag[f'loc_gamma_{target}'] = self.locpool.gamma[c]
            diag[f'loc_entropy_{target}'] = -(spatial[:, :, :, c].clamp_min(1e-12).log() * spatial[:, :, :, c]).sum(-1).mean() / math.log(self.token_count)
            diag[f'loc_gnorm_{target}'] = gp.norm(dim=-1).mean()
            diag[f'loc_hnorm_{target}'] = hp.norm(dim=-1).mean()
            diag[f'loc_cosine_{target}'] = F.cosine_similarity(gp, hp, dim=-1).mean()
            diag[f'loc_residual_ratio_{target}'] = (self.locpool.gamma[c].abs() * (hp - gp).norm(dim=-1) / (gp.norm(dim=-1) + 1e-08)).mean()
            diag[f'loc_replacement_{target}'] = replacement[:, c].float().mean()
            if views is not None:
                eligible = views['positive'][..., c]
                mass = (spatial[:, :, :, c] * views['box_mask'][..., c, :, :].flatten(-2)).sum(-1)
                diag[f'loc_box_count_{target}'] = eligible.sum().float()
                diag[f'loc_box_mass_{target}'] = mass[eligible].mean() if eligible.any() else mass.sum() * 0
        if views is not None:
            with torch.no_grad():
                valid_flat = wv.flatten()
                masks = views['box_mask'].flatten(-2)
                mass = (spatial * masks).sum(-1)
                area = masks.sum(-1).float()
                for c, target in enumerate(LOCAL):
                    normg = g.float().norm(dim=-1)
                    normh = h[:, c].float().norm(dim=-1)
                    cos = F.cosine_similarity(g.float(), h[:, c].float(), dim=-1)
                    ratio = self.locpool.gamma[c].abs() * (h[:, c].float() - g.float()).norm(dim=-1) / (normg + 1e-08)
                    diag[f'loc_valid_count_{target}'] = valid_flat.sum()
                    diag[f'loc_valid_gnorm_sum_{target}'] = normg[valid_flat].sum()
                    diag[f'loc_valid_hnorm_sum_{target}'] = normh[valid_flat].sum()
                    diag[f'loc_valid_cosine_sum_{target}'] = cos[valid_flat].sum()
                    diag[f'loc_inference_residual_ratio_sum_{target}'] = ratio[valid_flat].sum()
                    diag[f'loc_replacement_count_{target}'] = replacement[:, c].float().sum()
                    for role in range(5):
                        eligible = views['positive'][:, role, :, c]
                        prefix = f'loc_t{target}_r{role}'
                        diag[prefix + '_positive_count'] = eligible.sum()
                        diag[prefix + '_mass_sum'] = mass[:, role, :, c][eligible].sum()
                        diag[prefix + '_area_sum'] = area[:, role, :, c][eligible].sum()
                        diag[prefix + '_uniform_mass_sum'] = (area[:, role, :, c][eligible] / self.token_count).sum()
                        diag[prefix + '_hinge_active_count'] = (mass[:, role, :, c][eligible] < 0.7).sum()
                        diag[prefix + '_reliable_negative_count'] = views['negative'][:, role, :, c].sum()
                        if role in [1, 4] and c in [1, 2, 3, 4]:
                            reflected = (spatial[:, role, :, c].reshape(b, count, self.grid_size, self.grid_size).flip(-1).flatten(-2) * masks[:, role, :, c]).sum(-1)
                            diag[prefix + '_own_minus_reflected_sum'] = (mass[:, role, :, c] - reflected)[eligible].sum()
        self.last_views = views
        self.last_window_attention = wa.detach()
        self.last_spatial_attention = spatial.detach()
        if hasattr(self, "box_forward"):
            logits, extra = self.box_forward(logits, features, selected, grid, wv, images, series_mask, slice_valid_mask, slice_repeated_mask, box_data)
            diag.update(extra)
        return (logits, diag) if return_diagnostics else logits

    def _auxiliary(self, features, logatt, wa, wv, views, b, count, logits):
        masks = views['box_mask'].flatten(-2)
        eligible = views['positive']
        zero = logits.sum() * 0
        flatmask = masks.reshape(-1, 6, self.token_count)
        pos = eligible.reshape(-1, 6)
        terms = {}
        if pos.any():
            lp = logatt[pos]
            mp = flatmask[pos]
            mass = lp.masked_fill(~mp, -torch.inf).logsumexp(-1).exp()
            loc = F.relu(0.7 - mass).mean()
        else:
            loc = zero
        ml = self.locpool.mil(features).float().flatten(-2)
        region = ml.new_zeros(ml.shape[:2])
        if pos.any():
            region[pos] = ml[pos].masked_fill(~flatmask[pos], -torch.inf).logsumexp(-1) - flatmask[pos].sum(-1).float().log()
        region = region.reshape(b, 5, count, 6)
        bags = []
        bag_indices = []
        for study in range(b):
            for c in range(6):
                roles = []
                for role in range(5):
                    ix = eligible[study, role, :, c]
                    if ix.any():
                        roles.append(torch.logsumexp(region[study, role, ix, c], 0) - ix.sum().float().log())
                if roles:
                    bags.append(torch.logsumexp(torch.stack(roles), 0) - math.log(len(roles)))
                    bag_indices.append((study, LOCAL[c]))
        neg = views['negative'].permute(0, 1, 3, 2)
        countneg = neg.sum(-1)
        validcount = wv.sum(-1)[:, :, None]
        active = (countneg > 0) & (countneg < validcount)
        negative_mass = (wa[:, :, list(LOCAL)].float() * neg).sum(-1)
        win = F.relu(negative_mass[active] - 0.1).mean() if active.any() else zero
        self.aux_observed = {'location_hinge': F.relu(0.7 - logatt.detach().exp().mul(flatmask).sum(-1)).reshape(b, 5, count, 6), 'positive': eligible.detach(), 'negative_hinge': F.relu(negative_mass.detach() - 0.1), 'negative_active': active.detach(), 'negative_mass': negative_mass.detach(), 'negative_count': countneg.detach(), 'valid_count': validcount.detach()}
        return {'zero': zero, 'location': loc, 'window': win, 'mil_bag_logits': torch.stack(bags) if bags else None, 'mil_bag_indices': bag_indices, 'eligible_positive_count': eligible.sum(), 'mil_empty_pairs': b * 6 - len(bags), 'negative_active_groups': active.sum()}

    def auxiliary_loss(self, labels, step):
        if self.disabled:
            return labels.sum() * 0
        aux = self.aux
        zero = aux['zero']
        pairs = aux['mil_bag_indices']
        mil = F.binary_cross_entropy_with_logits(aux['mil_bag_logits'], torch.stack([labels[i, t] for i, t in pairs])) if pairs else zero
        epoch = step // self.steps_per_epoch
        progress = step / self.steps_per_epoch
        weight = progress if epoch == 0 else 1.0 if epoch <= 9 else 1 - 0.7 * (epoch - 10) / 19
        with torch.no_grad():
            obs = self.aux_observed
            self.aux_target_diagnostics = {}
            losses = F.binary_cross_entropy_with_logits(aux['mil_bag_logits'].detach(), torch.stack([labels[i, t] for i, t in pairs]), reduction='none') if pairs else None
            for c, target in enumerate(LOCAL):
                ix = [j for j, (_, t) in enumerate(pairs) if t == target]
                d = self.aux_target_diagnostics
                d[f'mil_t{target}_count'] = len(ix)
                d[f'mil_t{target}_sum'] = float(losses[ix].sum()) if ix else 0.0
                for role in range(5):
                    pos = obs['positive'][:, role, :, c]
                    neg = obs['negative_active'][:, role, c]
                    pre = f'aux_t{target}_r{role}'
                    d[pre + '_location_count'] = int(pos.sum())
                    d[pre + '_location_sum'] = float(obs['location_hinge'][:, role, :, c][pos].sum())
                    d[pre + '_negative_count'] = int(neg.sum())
                    d[pre + '_negative_sum'] = float(obs['negative_hinge'][:, role, c][neg].sum())
                    d[pre + '_negative_mass_sum'] = float(obs['negative_mass'][:, role, c][neg].sum())
            d['weighted_mil'] = float(weight) * 0.1 * float(mil)
            d['weighted_location'] = float(weight) * 0.1 * float(aux['location'])
            d['weighted_window'] = float(weight) * 0.1 * float(aux['window'])
        return (float(weight) * (0.1 * mil + 0.1 * aux['location'] + 0.1 * aux['window']), {'mil': mil.detach(), 'location': aux['location'].detach(), 'window': aux['window'].detach(), 'lambda': weight})
