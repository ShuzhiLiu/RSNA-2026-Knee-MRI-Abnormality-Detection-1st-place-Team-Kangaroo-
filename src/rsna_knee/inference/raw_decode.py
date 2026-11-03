import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
"""Shared raw DICOM decoder and role selector used by the final native32 reader."""
import copy
import io
import os
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=True
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA = Path(os.environ.get('KNEE_DATA', '/kaggle/input/competitions/rsna-knee-abnormality-detection'))
CANVAS_MM = 140.0  # physical canvas of the training input (knee_input_pipeline_256_v5.json, MXS256V59 / MXV2 N)
IO_THREADS = int(os.environ.get('KNEE_IO_THREADS', 16))  # concurrent file reads per decode worker (0 = sequential)
NATIVE = os.environ.get('KNEE_NATIVE', '0') == '1'  # direct uncompressed decode (guarded); off: no measured gain on Kaggle
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ROLES = ('SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS', 'SAG_STRUCT', 'COR_STRUCT')
PLANES = {'SAG': 'Sagittal', 'COR': 'Coronal', 'AX': 'Axial'}
SIZE, DEPTH, WINDOWS = 256, 18, 16
LEFT, POSTERIOR, INFERIOR = np.eye(3)[0], np.eye(3)[1], -np.eye(3)[2]  # DICOM LPS


# ----------------------------------------------------------------------------------------------- selection
def describe(row):
    """Selector inputs from the middle file's header."""
    files = sorted((DATA / 'test_series' / row['StudyInstanceUID'] / row['SeriesInstanceUID']).glob('*.dcm'))
    h = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
    sp = [float(v) for v in h.PixelSpacing]
    text = str(h.get('SeriesDescription') or '').casefold()
    return dict(uid=row['SeriesInstanceUID'], files=files, n=len(files), plane=row['Anatomical_Plane'],
                fluid=row['Fluid_Sensitive'] == 1, fat=row['Fat_Suppression'] == 1,
                t1='stir' not in text and re.search(r'(^|[^a-z])t1([^a-z]|$)', text) is not None,
                spacing=sum(sp) / 2, fov=min(h.Rows * sp[0], h.Columns * sp[1]))


def role_score(s, role):
    """Protocol preference first, then resolution, coverage and depth."""
    if s['plane'] != PLANES[role.split('_')[0]]:
        return None
    if role.endswith('FLUID_FS'):
        if not s['fluid']:
            return None
        score = 120.0 if s['fat'] else 90.0
    else:
        if s['fluid'] and s['fat']:
            return None
        score = (75.0 if s['fluid'] else 115.0) + 8.0 * s['t1']
    return score + 4.0 / s['spacing'] + 6.0 * min(s['fov'] / 180.0, 1.0) + 2.0 * min(s['n'] / 64.0, 1.0)


def select(candidates):
    chosen = {}
    for role in ROLES:
        scored = [(score, c['n'], c['uid'], c) for c in candidates
                  if c['uid'] not in {x['uid'] for x in chosen.values()}
                  and (score := role_score(c, role)) is not None]
        if scored:
            chosen[role] = max(scored, key=lambda t: t[:3])[3]
    return chosen


# ------------------------------------------------------------------------------------------ exact fast percentile
_SMALL_INT = {np.dtype(np.uint8): (np.uint8, 0), np.dtype(np.uint16): (np.uint16, 0),
              np.dtype(np.int8): (np.uint8, 0x80), np.dtype(np.int16): (np.uint16, 0x8000)}
_HIST_OK = None


def _percentile_from_counts(counts, values, q=(0.5, 99.5)):
    """np.percentile(x, q) for float32 x given as a histogram over sorted distinct values, via numpy's own helpers."""
    from numpy.lib import _function_base_impl as fb
    n = int(counts.sum())
    if n == 0:
        raise IndexError('no nonzero pixels')  # np.percentile of an empty array raises as well
    quantiles = np.true_divide(list(q), 100)
    method = fb._QuantileMethods['linear']
    virtual = np.asanyarray(method['get_virtual_index'](n, quantiles))
    previous, following = fb._get_indexes(np.empty(0, np.float32), virtual, n)
    cumulative = np.cumsum(counts)
    value = lambda ranks: values[np.searchsorted(cumulative, np.where(ranks < 0, ranks + n, ranks), side='right')]
    gamma = fb._get_gamma(virtual, previous, method).reshape(virtual.shape)
    return fb._lerp(value(previous), value(following), gamma)


def _value_map(dtype, slope, intercept):
    """float32 value of every possible integer pixel by the reference decode expression, or None if unusable."""
    view_dtype, flip = _SMALL_INT[dtype]
    bins = 1 << (8 * np.dtype(view_dtype).itemsize)
    values = (np.arange(bins) - flip).astype(dtype).astype(np.float32) * slope + intercept
    if values.dtype != np.float32 or not np.isfinite(values).all() or (np.diff(values) < 0).any():
        return None  # order statistics of the values must be the (monotone) image of those of the pixels
    return values


def fast_lohi(stack, slope, intercept):
    """(lo, hi, values) with lo, hi == np.percentile(decoded[decoded != 0], [0.5, 99.5]).astype(float) of the
    reference float32 decode of an 8/16-bit integer stack, or None when this shortcut does not apply."""
    values = _value_map(stack.dtype, slope, intercept)
    if values is None:
        return None
    view_dtype, flip = _SMALL_INT[stack.dtype]
    counts = np.zeros(len(values), np.int64)
    for plane in stack:  # exact integer counts; torch.bincount is ~1.7x faster than np.bincount here
        u = torch.from_numpy(plane.view(view_dtype)).to(torch.int32).ravel()
        counts += torch.bincount(u ^ flip if flip else u, minlength=len(values)).numpy()
    counts[values == 0] = 0  # pixels != 0 is taken on the decoded values
    lo, hi = _percentile_from_counts(counts, values).astype(float)
    return lo, hi, values


def _hist_selftest():
    """Bit-compare fast_lohi with the reference np.percentile; any mismatch or numpy API change -> reference path."""
    try:
        rng = np.random.default_rng(0)
        for trial in range(48):
            dtype = (np.uint16, np.int16, np.uint8)[trial % 3]
            slope, intercept = ((1.0, 0.0), (float(rng.uniform(0.3, 11)), 0.0), (1.0, float(rng.integers(-50, 50))),
                                (float(rng.uniform(0.1, 5)), float(rng.uniform(-100, 100))))[trial // 3 % 4]
            info = np.iinfo(dtype)
            n = int(rng.integers(1, 150000))
            v = rng.integers(max(info.min, -3000), min(info.max, 4096) + 1, size=(1, n)).astype(dtype)
            v[rng.random((1, n)) < 0.3] = 0
            if trial % 7 == 0:
                v[:] = rng.integers(1, 3)
            decoded = np.asarray(v, np.float32) * slope + intercept
            if not (decoded != 0).any():
                continue
            ref = np.percentile(decoded[decoded != 0], [0.5, 99.5]).astype(float)
            got = fast_lohi(v, slope, intercept)
            if got is None or not np.array_equal(np.array(got[:2]), ref):
                return False
        return True
    except Exception:
        return False


def _lut(values, lo, hi):
    """Normalised float32 value for every possible integer pixel, by the reference expression."""
    return np.rint((np.clip(values, lo, hi) - lo) / (hi - lo) * 65535).astype(np.float32) / 65535


# ---------------------------------------------------------------------------------------------- loading
def decode(d):
    try:
        return np.asarray(d.pixel_array, np.float32) * float(d.get('RescaleSlope') or 1) + float(d.get('RescaleIntercept') or 0)
    except Exception:
        return None


_NATIVE = {'1.2.840.10008.1.2', '1.2.840.10008.1.2.1'}  # implicit / explicit VR little endian


def native_pixels(d):
    """d.pixel_array for plain single-frame little-endian monochrome 8/16-bit data without pydicom's decoder
    machinery (frombuffer + pydicom's unused-bits shift); None = use d.pixel_array. Checked per series."""
    try:
        bits, stored, signed = d.BitsAllocated, d.BitsStored, d.PixelRepresentation
        if (d.file_meta.TransferSyntaxUID not in _NATIVE or 'PixelData' not in d or d.get('SamplesPerPixel', 1) != 1
                or int(d.get('NumberOfFrames') or 1) != 1 or bits not in (8, 16) or signed not in (0, 1)
                or not 0 < stored <= bits or d.PhotometricInterpretation not in ('MONOCHROME1', 'MONOCHROME2')):
            return None
        rows, cols, data = d.Rows, d.Columns, d.PixelData
        if not rows or not cols or len(data) not in (rows * cols * bits // 8, rows * cols * bits // 8 + 1):
            return None
        arr = np.frombuffer(data, np.dtype(f"<{'i' if signed else 'u'}{bits // 8}"), rows * cols).reshape(rows, cols).copy()
        if stored < bits:
            np.left_shift(arr, bits - stored, out=arr)
            np.right_shift(arr, bits - stored, out=arr)
        return arr
    except Exception:
        return None


def decode_raw(d, native=True):
    """(pixels, slope, intercept) with the stored integer pixels for 8/16-bit data, else the reference float decode."""
    try:
        px = (native_pixels(d) if native else None)
        px = d.pixel_array if px is None else px
        slope, intercept = float(d.get('RescaleSlope') or 1), float(d.get('RescaleIntercept') or 0)
    except Exception:
        return None
    if np.dtype(px.dtype) in _SMALL_INT:
        return px, slope, intercept
    return (f, None, None) if (f := decode(d)) is not None else None


def _native_agrees(d):
    """Guard: the native decode of one slice equals pydicom's pixel_array bit for bit (or does not apply)."""
    px = native_pixels(d)
    if px is None:
        return True
    try:
        ref = d.pixel_array
    except Exception:
        return False
    return ref.dtype == px.dtype and ref.shape == px.shape and np.array_equal(ref, px)


_POOL = None


def _pool():
    """Per-process I/O thread pool (created after the DataLoader fork)."""
    global _POOL
    if _POOL is None and IO_THREADS > 0:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(IO_THREADS)
    return _POOL


def _read(path):
    try:
        return path.read_bytes()
    except Exception:
        return None  # dcmread(path) then raises exactly as in the reference


def prefetch(files):
    """File bytes read concurrently (cold network storage is latency bound); {} when disabled."""
    pool = _pool()
    return dict(zip(files, pool.map(_read, files))) if pool else {}


def _dcmread(f, blobs):
    blob = blobs.get(f)
    return pydicom.dcmread(io.BytesIO(blob) if blob is not None else f)


def load_series(files, blobs=None):
    """Decode in spatial order, dropping up to max(1, 10%) undecodable slices (more fails the series)."""
    blobs = blobs or {}
    normal = lambda d: np.cross(*np.asarray(d.ImageOrientationPatient, float).reshape(2, 3))
    position = lambda d: float(np.asarray(d.ImagePositionPatient, float) @ normal(d))
    datasets = [(_dcmread(f, blobs), f.name) for f in files]
    measured = all(hasattr(d, 'ImagePositionPatient') and hasattr(d, 'ImageOrientationPatient') for d,_ in datasets)
    ds = [d for d,_ in sorted(datasets, key=(lambda t:(position(t[0]),t[1])) if measured else (lambda t:(int(t[0].get('InstanceNumber',0)),t[1])))]
    native = NATIVE and _native_agrees(ds[len(ds) // 2])
    decoded = [(d, px) for d in ds if (px := decode_raw(d, native)) is not None]
    assert decoded and len(ds) - len(decoded) <= max(1, int(len(ds) * 0.10))
    ds, raws = [d for d, _ in decoded], [px for _, px in decoded]
    global _HIST_OK
    if _HIST_OK is None:
        _HIST_OK = _hist_selftest()
    fast = None
    if _HIST_OK and len({(r.dtype, slope, intercept) for r, slope, intercept in raws}) == 1 and raws[0][1] is not None:
        pixels = np.stack([r for r, _, _ in raws])
        fast = fast_lohi(pixels, raws[0][1], raws[0][2])
    if fast is not None:  # one integer dtype and one rescale for the whole series
        lo, hi, values = fast
        lut = _lut(values, lo, hi)
    else:  # mixed/float/non-monotone series: exactly the reference path
        pixels = np.stack([r if slope is None else np.asarray(r, np.float32) * slope + intercept for r, slope, intercept in raws])
        lo, hi = np.percentile(pixels[pixels != 0], [0.5, 99.5]).astype(float)
        lut = None
    first = ds[0]
    return dict(pixels=pixels, lut=lut, lo=lo, hi=hi,
                positions=np.array([position(d) for d in ds]) if measured else None,
                source_metadata=dict(image_positions_patient=[list(map(float,d.ImagePositionPatient)) for d in ds] if all(hasattr(d,'ImagePositionPatient') for d in ds) else None,
                    slice_positions_mm=[position(d) for d in ds] if measured else None,
                    spacing_between_slices=float(first.SpacingBetweenSlices) if hasattr(first,'SpacingBetweenSlices') else None,
                    slice_thickness=float(first.SliceThickness) if hasattr(first,'SliceThickness') else None),
                ipp=np.asarray([d.ImagePositionPatient for d in ds], float) if all(hasattr(d,'ImagePositionPatient') for d in ds) else None,
                iop=np.asarray(first.ImageOrientationPatient, float) if hasattr(first,'ImageOrientationPatient') else None,
                spacing=[float(v) for v in first.PixelSpacing], rows=first.Rows, cols=first.Columns,
                tag=(str(first.get('ImageLaterality', '')) or str(first.get('Laterality', ''))).upper())


def laterality(series):
    """Consistent L/R tag, else patient-coordinate centre beyond 30 mm; any conflict -> unknown."""
    tags = {s['tag'][0] for s in series if s['tag'] in ('L', 'LEFT', 'R', 'RIGHT')}
    x = np.median([np.median(s['ipp'], axis=0)[0] + s['iop'][0] * s['spacing'][1] * (s['cols'] - 1) / 2
                   + s['iop'][3] * s['spacing'][0] * (s['rows'] - 1) / 2 for s in series if s['ipp'] is not None and s['iop'] is not None] or [float('nan')])
    geometry = 'L' if x > 30 else 'R' if x < -30 else None
    if len(tags) > 1 or (tags and geometry and geometry not in tags):
        return None
    return next(iter(tags)) if tags else geometry








class Studies(Dataset):
    """Shared study file discovery; native32 subclass constructs model inputs."""

    def __init__(self, study_ids, series_rows):
        self.study_ids, self.series_rows = study_ids, series_rows

    def __len__(self):
        return len(self.study_ids)

    def _io(self, i):
        """File work of study i: headers of every series, then the bytes of the first-choice series."""
        rows, pool = self.series_rows[self.study_ids[i]], _pool()
        candidates = list(pool.map(describe, rows)) if pool else [describe(row) for row in rows]
        return candidates, prefetch([f for c in select(candidates).values() for f in c['files']])


