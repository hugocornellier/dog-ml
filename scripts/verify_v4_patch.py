"""Does the patched GPU dylib accept TRANSPOSE_CONV v4 on the UNMODIFIED shipped asset?

This is the end-to-end test of the flutter_litert patch. Until now the 5.11 ms GPU
figure required rewriting the model to move its ReLU out of the deconv, which drops
the opcode to version 3. If the patch works, the *unmodified* static export runs at
the same speed with no model surgery at all.

Three models, to separate the effects:

  static_fp16          v4 with a fused ReLU. Rejected by the stock dylib
                       ("Max version supported: 3"), so the deconv fell to CPU and it
                       measured 30.25 ms. With the patch it should be accepted and
                       land near the unfused figure.
  static_unfused       v3, already accepted by the stock dylib at 5.11 ms. Should be
                       unchanged, which is the control proving the patched dylib did
                       not break the path that already worked.
  shipped dynamic      still expected to fail interpreter creation, because its
                       dynamic-shaped tensors are a separate blocker that this patch
                       does not touch.

Accuracy is checked over the full 480-image val split, not just latency, because a
delegate that computes the wrong thing quickly is the failure mode this whole
investigation kept running into. A correct result means both the version gate and
MaybeFuseActivation are doing their jobs: if the gate were raised without applying
the activation, the ReLU would be silently dropped and NME would blow up.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from bench_litert_macos import DYLIB, _bind, NUM_THREADS_OFFSET  # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-hugocornellier-IdeaProjects-dog-detection"
             "/ee61cbfd-dd84-4c4a-82ea-cb1141e124d6/scratchpad/valcache")
NL, LEFT, RIGHT = 46, 18, 19


def run(model: Path, gpu_dylib: Path | None, n: int, runs: int):
    lib = ctypes.CDLL(str(DYLIB))
    _bind(lib)
    m = lib.TfLiteModelCreateFromFile(str(model).encode())
    if not m:
        return None, None, "model load failed"
    o = lib.TfLiteInterpreterOptionsCreate()
    lib.TfLiteInterpreterOptionsSetNumThreads(o, 4)
    if gpu_dylib is None:
        xo = lib.TfLiteXNNPackDelegateOptionsDefault()
        struct.pack_into("<i", xo.raw, NUM_THREADS_OFFSET, 4)
        lib.TfLiteInterpreterOptionsAddDelegate(
            o, lib.TfLiteXNNPackDelegateCreate(ctypes.byref(xo)))
    else:
        g = ctypes.CDLL(str(gpu_dylib))
        g.TFLGpuDelegateCreate.restype = ctypes.c_void_p
        g.TFLGpuDelegateCreate.argtypes = [ctypes.c_void_p]
        d = g.TFLGpuDelegateCreate(None)
        if not d:
            return None, None, "metal delegate null"
        lib.TfLiteInterpreterOptionsAddDelegate(o, d)
    it = lib.TfLiteInterpreterCreate(m, o)
    if not it:
        return None, None, "interpreter creation FAILED"
    if lib.TfLiteInterpreterAllocateTensors(it) != 0:
        return None, None, "AllocateTensors failed"

    crops = np.load(CACHE / "crops_384_0.05_0.1.npy", mmap_mode="r")
    ti = lib.TfLiteInterpreterGetInputTensor(it, 0)
    to = lib.TfLiteInterpreterGetOutputTensor(it, 0)
    nout = lib.TfLiteTensorByteSize(to) // 4
    out = np.zeros(nout, np.float32)
    preds = np.zeros((n, nout), np.float32)
    ts = []
    for i in range(n):
        b = np.ascontiguousarray(crops[i], np.float32)
        lib.TfLiteTensorCopyFromBuffer(ti, b.ctypes.data_as(ctypes.c_void_p), b.nbytes)
        t0 = time.perf_counter()
        rc = lib.TfLiteInterpreterInvoke(it)
        dt = (time.perf_counter() - t0) * 1000.0
        lib.TfLiteTensorCopyToBuffer(to, out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
        if rc != 0:
            return None, None, f"invoke rc={rc}"
        preds[i] = out
        if i >= 8:
            ts.append(dt)
    return statistics.median(ts), np.clip(preds, 0, 1), None


def nme(preds):
    gt = np.load(CACHE / "gt_384_0.05_0.1.npy")[:preds.shape[0]]
    g = gt.reshape(-1, NL, 2).astype(np.float64)
    q = preds.reshape(-1, NL, 2).astype(np.float64)
    iod = np.sqrt(np.sum((g[:, LEFT] - g[:, RIGHT]) ** 2, axis=-1) + 1e-8)
    d = np.sqrt(np.sum((q - g) ** 2, axis=-1) + 1e-8)
    return float((d / np.maximum(iod, 1e-8)[:, None] * 100).mean(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched-gpu", type=Path, required=True,
                    help="the dylib built by the patched build-metal-macos workflow")
    ap.add_argument("--n", type=int, default=480)
    args = ap.parse_args()

    stock = DYLIB.parent / "libtensorflowlite_gpu-mac.dylib"
    models = [
        ("static_fp16 (v4, fused)", REPO / "artifacts/pareto/static_fp16.tflite"),
        ("static_unfused (v3)", REPO / "artifacts/pareto/static_unfused.tflite"),
        ("shipped dynamic", REPO / "artifacts/small_v3large_384_long"
                                   "/dog_face_landmarks_384_float16.tflite"),
    ]
    print(f"{args.n} val images | XNNPACK reference, then stock vs patched Metal\n")
    print(f"{'model':26s} {'backend':16s} {'ms':>8s} {'NME_IOD':>9s}  note")
    for tag, p in models:
        if not p.exists():
            print(f"{tag:26s} missing: {p}")
            continue
        for blabel, dyl in (("xnnpack", None), ("metal stock", stock),
                            ("metal PATCHED", args.patched_gpu)):
            ms, preds, err = run(p, dyl, args.n, 30)
            if err:
                print(f"{tag:26s} {blabel:16s} {'--':>8s} {'--':>9s}  {err}")
            else:
                print(f"{tag:26s} {blabel:16s} {ms:8.2f} {nme(preds):9.4f}")
        print()


if __name__ == "__main__":
    main()
