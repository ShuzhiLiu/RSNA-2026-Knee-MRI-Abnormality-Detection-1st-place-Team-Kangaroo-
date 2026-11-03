"""Canonical native source pixel-edge to image-canvas geometry."""
from __future__ import annotations
import numpy as np


def source_to_canvas_affine(stage_meta: dict, size: int) -> np.ndarray:
    """3x3 affine: native source pixel-edge (x=col, y=row) -> canonical canvas pixel-edge at `size`.

    Mirrors canonicalize_inplane_orientation (transpose, then horizontal flip, then vertical flip) and
    physical_center_crop_resize (center crop/pad of crop_shape, resize to size).
    """
    o = stage_meta["orientation"]
    # crop trace source_shape is the ORIENTED shape (the crop receives already-oriented images)
    _, H, W = stage_meta["crop"]["source_shape"]
    M = np.eye(3)
    if o.get("transpose"):
        M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]]) @ M
    if o.get("flip_horizontal"):
        M = np.array([[-1, 0, W], [0, 1, 0], [0, 0, 1.0]]) @ M
    if o.get("flip_vertical"):
        M = np.array([[1, 0, 0], [0, -1, H], [0, 0, 1.0]]) @ M
    ch, cw = stage_meta["crop"]["crop_shape_pixels"]
    top = (H - ch) // 2
    left = (W - cw) // 2
    M = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1.0]]) @ M
    M = np.array([[size / cw, 0, 0], [0, size / ch, 0], [0, 0, 1.0]]) @ M
    return M

