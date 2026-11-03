import ctypes
from pathlib import Path
import numpy as np
def _library_path():
 import hashlib, os, subprocess, tempfile
 source=Path(__file__).with_name('polygon_kernel.cpp')
 digest=hashlib.sha256(source.read_bytes()).hexdigest()
 cache=Path(os.environ.get('RSNA_KERNEL_CACHE', tempfile.gettempdir()))/'rsna_polygon'/digest
 cache.mkdir(parents=True,exist_ok=True)
 target=cache/'polygon_kernel.so'
 if not target.exists():
  temporary=cache/f'polygon_kernel.{os.getpid()}.so'
  subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(source),'-o',str(temporary)],check=True)
  os.replace(temporary,target)
 return target
_lib=ctypes.CDLL(str(_library_path()))
_ptr=np.ctypeslib.ndpointer
_lib.overlap.argtypes=[_ptr(np.float64,flags='C_CONTIGUOUS'),_ptr(np.int32,flags='C_CONTIGUOUS'),ctypes.c_int,_ptr(np.float64,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,ctypes.c_int,_ptr(np.float32,flags='C_CONTIGUOUS')]
_lib.overlap.restype=None
def overlap(polys,valid,G,size=256):
 out=np.zeros((G,G),np.float32)
 if not polys or len(valid)<3:return out
 points=np.ascontiguousarray(np.concatenate(polys),dtype=np.float64);lengths=np.asarray([len(p) for p in polys],np.int32);valid=np.ascontiguousarray(valid,dtype=np.float64)
 _lib.overlap(points,lengths,len(polys),valid,len(valid),G,size,out);return out

_compiled_overlap=overlap
fallback_count=0
def overlap(polys,valid,G,size=256):
 # Tiny/degenerate polygons suffer float64 cancellation under the NumPy dot
 # reduction order. Preserve the exact frozen Python oracle for these inputs.
 global fallback_count
 if any(len(p)<3 or np.any(np.ptp(p,axis=0)<.01) for p in polys):
  from .box_model import oracle_overlap
  fallback_count+=1
  return oracle_overlap(polys,valid,G,size)
 return _compiled_overlap(polys,valid,G,size)
