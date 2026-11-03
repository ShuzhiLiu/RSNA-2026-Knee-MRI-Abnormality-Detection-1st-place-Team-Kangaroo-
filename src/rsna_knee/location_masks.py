"""Use observed native sampling grids and actual derived indices for annotation masks."""
import torch
import torch.nn.functional as F

def annotation_views(batch, selected, gpu_grid, dense):
    device = gpu_grid.device
    b = batch['anatomy_states'].shape[0]
    count = len(selected[0])
    h, w = gpu_grid.shape[1:3]
    states = batch['anatomy_states'].to(device)
    boxes = batch['anatomy_boxes'].to(device)
    valid = batch['slice_valid_mask'].to(device)
    repeated = batch['slice_repeated_mask'].to(device)
    rolevalid = batch['series_mask'].to(device)
    chosen = []
    centres = []
    negative = []
    eligible = []
    cpu = []
    for role, ss in enumerate(selected):
        ix = torch.as_tensor(ss, device=device, dtype=torch.long)
        valid_trip = torch.stack([valid[:, role, j:j + 3].all(-1) & ~repeated[:, role, j:j + 3].any(-1) for j in ss], 1) & rolevalid[:, role, None]
        st = torch.stack([states[:, role, j:j + 3] for j in ss], 1)
        centres.append(st[:, :, 1])
        chosen.append(boxes[:, role, ix + 1])
        negative.append((st == 0).all(2) & valid_trip[:, :, None] & dense[role][None, None])
        eligible.append(valid_trip)
        cpu.append(batch['cpu_grids'][:, role, None].expand(b, count, h, w, 2))
    centre = torch.stack(centres, 1)
    box = torch.stack(chosen, 1)
    tripvalid = torch.stack(eligible, 1)
    neg = torch.stack(negative, 1)
    positive = (centre == 1) & tripvalid[:, :, :, None] & dense[None, :, None]
    x = torch.arange(w, device=device).view(1, 1, 1, 1, 1, w)
    y = torch.arange(h, device=device).view(1, 1, 1, 1, h, 1)
    masks = (x < box[..., 2, None, None]) & (x + 1 > box[..., 0, None, None]) & (y < box[..., 3, None, None]) & (y + 1 > box[..., 1, None, None]) & positive[..., None, None]
    masks = masks.reshape(b * 5 * count, 6, h, w).float()
    cpu_grid = torch.stack(cpu, 1).reshape(b * 5 * count, h, w, 2).to(device)
    cpu_masks = F.grid_sample(masks, cpu_grid, mode='nearest', padding_mode='zeros', align_corners=False)
    transformed = F.grid_sample(cpu_masks, gpu_grid, mode='nearest', padding_mode='zeros', align_corners=False)
    tokens = F.max_pool2d(transformed, 32, 32)
    tokens = F.max_pool2d(tokens, 3, 1, 1) > 0
    assert h == w and h in (256, 384) and (tokens.shape[-2:] == (h // 32, w // 32))
    tokens = tokens.reshape(b, 5, count, 6, h // 32, w // 32)
    positive = positive & tokens.flatten(-2).any(-1)
    return {'box_mask': tokens, 'positive': positive, 'negative': neg, 'centre_state': centre, 'triplet_valid': tripvalid, 'cpu_grid_used': cpu_grid, 'gpu_grid_used': gpu_grid, 'full_mask': transformed, 'raw_boxes': box}
