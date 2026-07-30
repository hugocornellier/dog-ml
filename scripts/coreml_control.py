"""Control test: can this ctypes harness drive the CoreML delegate on ANY model?

Both landmark graphs hang past 600s under CoreML here, but
flutter_litert's own doc/delegate_verification.md reports CoreML *completing* on the
same models (as a no-op, dev = 0.0) when driven through Dart. A harness that hangs
where the reference completes is the more likely culprit, so this checks whether the
harness can drive CoreML at all.

Uses whatever model is passed, reads its input shape from the graph and feeds zeros,
so it works on the simple static models (species classifier, ssdlite) as well as the
deconv-headed ones. If CoreML hangs even on a plain classifier with no
TRANSPOSE_CONV, the harness is at fault and the landmark hangs say nothing about the
library.

A plausible mechanism for a harness-side hang: CoreML model compilation on macOS may
require an active Cocoa run loop to pump, which a bare Python process does not have.
That would be unrelated to the model.
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_litert_macos import DYLIB, _bind  # noqa: E402

import os as _os
# Allow pointing at a patched build for before/after comparison.
COREML_DYLIB = Path(_os.environ.get(
    "COREML_DYLIB", str(DYLIB.parent / "libtensorflowlite_coreml-mac.dylib")))


def try_coreml(model_path: Path, max_partitions: int = 0,
               min_nodes: int = 2) -> str:
    lib = ctypes.CDLL(str(DYLIB))
    _bind(lib)
    lib.TfLiteInterpreterGetInputTensorCount.restype = ctypes.c_int32
    lib.TfLiteInterpreterGetInputTensorCount.argtypes = [ctypes.c_void_p]
    lib.TfLiteTensorNumDims.restype = ctypes.c_int32
    lib.TfLiteTensorNumDims.argtypes = [ctypes.c_void_p]
    lib.TfLiteTensorDim.restype = ctypes.c_int32
    lib.TfLiteTensorDim.argtypes = [ctypes.c_void_p, ctypes.c_int32]

    c = ctypes.CDLL(str(COREML_DYLIB))
    c.TfLiteCoreMlDelegateCreate.restype = ctypes.c_void_p
    c.TfLiteCoreMlDelegateCreate.argtypes = [ctypes.c_void_p]

    model = lib.TfLiteModelCreateFromFile(str(model_path).encode())
    if not model:
        return "model load failed"
    opts = lib.TfLiteInterpreterOptionsCreate()
    lib.TfLiteInterpreterOptionsSetNumThreads(opts, 4)

    o = (ctypes.c_ubyte * 256)()
    struct.pack_into("<i", o, 0, 1)              # enabled_devices = AllDevices
    struct.pack_into("<i", o, 8, max_partitions)  # max_delegated_partitions
    struct.pack_into("<i", o, 12, min_nodes)      # min_nodes_per_partition
    print("  creating CoreML delegate ...", flush=True)
    d = c.TfLiteCoreMlDelegateCreate(ctypes.byref(o))
    if not d:
        return "delegate creation returned null"
    print("  delegate created, creating interpreter ...", flush=True)
    lib.TfLiteInterpreterOptionsAddDelegate(opts, d)

    interp = lib.TfLiteInterpreterCreate(model, opts)
    if not interp:
        return "interpreter creation failed"
    print("  interpreter created, allocating ...", flush=True)
    if lib.TfLiteInterpreterAllocateTensors(interp) != 0:
        return "AllocateTensors failed"

    in_t = lib.TfLiteInterpreterGetInputTensor(interp, 0)
    dims = [lib.TfLiteTensorDim(in_t, i)
            for i in range(lib.TfLiteTensorNumDims(in_t))]
    print(f"  allocated, input dims {dims}, invoking ...", flush=True)
    buf = np.zeros(dims, dtype=np.float32)
    lib.TfLiteTensorCopyFromBuffer(in_t, buf.ctypes.data_as(ctypes.c_void_p),
                                   buf.nbytes)
    rc = lib.TfLiteInterpreterInvoke(interp)
    return f"invoke rc={rc} (0 == ok)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--max-partitions", type=int, default=0)
    ap.add_argument("--min-nodes", type=int, default=2)
    args = ap.parse_args()
    print(f"model: {args.model.name} max_partitions={args.max_partitions} "
          f"min_nodes={args.min_nodes}", flush=True)
    print("result:", try_coreml(args.model, args.max_partitions, args.min_nodes),
          flush=True)


if __name__ == "__main__":
    main()
