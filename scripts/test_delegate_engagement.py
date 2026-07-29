"""Does the Metal/CoreML delegate actually run this model, or silently no-op?

flutter_litert 3.7.0 ships doc/delegate_verification.md, which reports that on
`landmarks_v3l_384` and `cat_face_landmarks_full` the GPU and CoreML delegates
attach, delegate **zero ops**, and fall back to bare CPU without warning. Their
detection trick is the only reliable one: timing cannot tell "ran on GPU and was
slow" from "never ran on GPU", but output can. A delegate that did real work
produces small numerical deviation from the CPU reference; a delegate that
no-opped produces deviation of *exactly* 0.0.

The doc root-causes their `CompiledModel` corruption to a **dynamic
(runtime-shaped) model output tensor**, which is precisely what Keras'
`PACK`-derived `TRANSPOSE_CONV` output shapes create and precisely what the
static re-export removes. Their matrix was measured on the old dynamic-shape
files, so this script re-runs the same test on both the old and the new export to
see whether the fix also unblocks the delegates.

Interpretation:
  dev == 0.0        delegate no-opped, ran on CPU (the bug)
  dev small (1e-6)  delegate engaged, numerics agree
  dev large         delegate engaged but is computing something wrong

Usage:
  python scripts/test_delegate_engagement.py old.tflite new.tflite
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_litert_macos import (  # noqa: E402
    DYLIB, LITERT_VERSION, XNNPackOptions, NUM_THREADS_OFFSET, _bind,
)

RESOURCES = DYLIB.parent
GPU_DYLIB = RESOURCES / "libtensorflowlite_gpu-mac.dylib"
COREML_DYLIB = RESOURCES / "libtensorflowlite_coreml-mac.dylib"


def _infer(model_path: Path, x: np.ndarray, make_delegate=None):
    """Run one input through the model, optionally with a delegate attached."""
    lib = ctypes.CDLL(str(DYLIB))
    _bind(lib)

    model = lib.TfLiteModelCreateFromFile(str(model_path).encode())
    if not model:
        raise RuntimeError(f"could not load {model_path}")
    opts = lib.TfLiteInterpreterOptionsCreate()
    lib.TfLiteInterpreterOptionsSetNumThreads(opts, 4)

    delegate = None
    if make_delegate is not None:
        delegate = make_delegate()
        if not delegate:
            return None, "delegate creation failed"
        lib.TfLiteInterpreterOptionsAddDelegate(opts, delegate)

    interp = lib.TfLiteInterpreterCreate(model, opts)
    if not interp:
        return None, "interpreter creation failed"
    if lib.TfLiteInterpreterAllocateTensors(interp) != 0:
        return None, "AllocateTensors failed"

    in_t = lib.TfLiteInterpreterGetInputTensor(interp, 0)
    out_t = lib.TfLiteInterpreterGetOutputTensor(interp, 0)
    n_out = lib.TfLiteTensorByteSize(out_t) // 4
    out = np.zeros(n_out, dtype=np.float32)

    buf = np.ascontiguousarray(x, dtype=np.float32)
    lib.TfLiteTensorCopyFromBuffer(in_t, buf.ctypes.data_as(ctypes.c_void_p),
                                   buf.nbytes)
    rc = lib.TfLiteInterpreterInvoke(interp)
    if rc != 0:
        return None, f"Invoke failed rc={rc}"
    lib.TfLiteTensorCopyToBuffer(out_t, out.ctypes.data_as(ctypes.c_void_p),
                                 out.nbytes)
    return out.copy(), None


def _cpu_reference(model_path: Path, x: np.ndarray):
    """Plain CPU, no delegate at all."""
    return _infer(model_path, x, None)


def _xnnpack(model_path: Path, x: np.ndarray):
    def make():
        lib = ctypes.CDLL(str(DYLIB))
        _bind(lib)
        o = lib.TfLiteXNNPackDelegateOptionsDefault()
        struct.pack_into("<i", o.raw, NUM_THREADS_OFFSET, 4)
        return lib.TfLiteXNNPackDelegateCreate(ctypes.byref(o))
    return _infer(model_path, x, make)


def _metal(model_path: Path, x: np.ndarray):
    """Same call flutter_litert's GpuDelegate() makes on macOS."""
    def make():
        g = ctypes.CDLL(str(GPU_DYLIB))
        g.TFLGpuDelegateCreate.restype = ctypes.c_void_p
        g.TFLGpuDelegateCreate.argtypes = [ctypes.c_void_p]
        return g.TFLGpuDelegateCreate(None)
    return _infer(model_path, x, make)


def _coreml(model_path: Path, x: np.ndarray):
    """Reproduce InterpreterFactory._createCoreml exactly.

    It builds `CoreMlDelegateOptions(enabledDevices: 1)` and takes the Dart
    defaults for everything else. Those defaults are NOT all zero, and the one that
    matters is `minNodesPerPartition = 2`. Zeroing it lets CoreML create a
    partition per node; with 279 ops and both DEQUANTIZE and TRANSPOSE_CONV
    unsupported, that means a very large number of tiny partitions, each compiled
    separately on first use. An earlier version of this function left it at 0 and
    hung for over 15 minutes, which was this bug and not a library defect.

    TfLiteCoreMlDelegateOptions layout: four consecutive ints,
    enabled_devices / coreml_version / max_delegated_partitions /
    min_nodes_per_partition. Over-allocated for the same reason as the XNNPACK
    options struct.
    """
    def make():
        c = ctypes.CDLL(str(COREML_DYLIB))
        c.TfLiteCoreMlDelegateCreate.restype = ctypes.c_void_p
        c.TfLiteCoreMlDelegateCreate.argtypes = [ctypes.c_void_p]
        opts = (ctypes.c_ubyte * 256)()
        struct.pack_into("<i", opts, 0, 1)   # enabled_devices = AllDevices
        struct.pack_into("<i", opts, 4, 0)   # coreml_version = highest available
        struct.pack_into("<i", opts, 8, 0)   # max_delegated_partitions = all
        struct.pack_into("<i", opts, 12, 2)  # min_nodes_per_partition = Dart default
        return c.TfLiteCoreMlDelegateCreate(ctypes.byref(opts))
    return _infer(model_path, x, make)


BACKENDS = [("xnnpack", _xnnpack), ("metal_gpu", _metal), ("coreml", _coreml)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--input", choices=["real", "zeros"], default="real")
    args = ap.parse_args()

    from pareto_harness import load_val_cache
    crops, _ = load_val_cache()
    x = (np.asarray(crops[0:1]) if args.input == "real"
         else np.zeros((1, 384, 384, 3), np.float32))

    print(f"flutter_litert {LITERT_VERSION}, input regime: {args.input}")
    print("dev = max |backend - cpu|.  dev == 0.0 means the delegate did nothing.\n")
    print(f"{'model':38s} {'backend':10s} {'dev':>12s}  note")
    for m in args.models:
        ref, err = _cpu_reference(m, x)
        if err:
            print(f"{m.name:38s} {'cpu':10s} {'--':>12s}  {err}")
            continue
        for name, fn in BACKENDS:
            out, err = fn(m, x)
            if err:
                print(f"{m.name:38s} {name:10s} {'--':>12s}  {err}")
                continue
            dev = float(np.abs(out - ref).max())
            note = "NO-OP (ran on CPU)" if dev == 0.0 else "engaged"
            print(f"{m.name:38s} {name:10s} {dev:12.3e}  {note}")
        print()


if __name__ == "__main__":
    main()
