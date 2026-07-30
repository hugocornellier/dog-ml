"""Solve the deconv-to-subpixel weight mapping numerically instead of deriving it.

Claim under test: a Conv2DTranspose with kernel 4, stride 2, padding 'same' is exactly
reproducible as a plain Conv2D producing 4*C_out channels followed by
tf.nn.depth_to_space(block=2).

Deriving the index mapping on paper is error-prone: Keras stores transpose-conv kernels
as (kh, kw, C_out, C_in) rather than Conv2D's (kh, kw, C_in, C_out), the gradient
definition flips the kernel, TF pads even-sized 'same' kernels asymmetrically, and
depth_to_space has its own channel order. Getting any one wrong yields a model that is
off by a sub-pixel shift and still looks plausible.

So this solves for the Conv2D kernel by least squares from random probes. Both
operators are linear in the input, so an exact equivalent, if one exists, is found and
the residual is ~0.

Two disciplines matter here, both learned the hard way earlier in this session:

  1. **Validate the solver before trusting a negative result.** `--self-test` generates
     the target *from* a known conv+depth_to_space and checks the solver recovers it. If
     that fails, any "no equivalence" verdict is a bug in this file, not a fact.
  2. **Sweep the configuration.** A single failing configuration does not refute the
     idea. Conv kernel size and padding both matter, because the two sub-positions of a
     stride-2 deconv may need different input offsets, which one 'SAME' conv cannot
     express even when a larger or differently-padded one can.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import tensorflow as tf

STRIDE = 2


def deconv_out(x, W, out_hw):
    return tf.nn.conv2d_transpose(
        x, W, output_shape=[x.shape[0], out_hw[0], out_hw[1], W.shape[2]],
        strides=[1, STRIDE, STRIDE, 1], padding="SAME",
    ).numpy()


def subpixel_out(x, K, pad):
    """Conv to 4*C_out channels then interleave. pad is 'SAME' or an explicit
    ((top,bottom),(left,right)) applied before a VALID conv."""
    if pad == "SAME":
        y = tf.nn.conv2d(x, K, strides=[1, 1, 1, 1], padding="SAME")
    else:
        (t, b), (l, r) = pad
        xp = tf.pad(x, [[0, 0], [t, b], [l, r], [0, 0]])
        y = tf.nn.conv2d(xp, K, strides=[1, 1, 1, 1], padding="VALID")
    return tf.nn.depth_to_space(y, STRIDE).numpy()


def _solve_for(X, Y, cin, cout, kk, pad):
    n_unk = kk * kk * cin * (STRIDE * STRIDE * cout)
    cols = []
    for u in range(n_unk):
        k = np.zeros(n_unk, np.float64)
        k[u] = 1.0
        Ku = k.reshape(kk, kk, cin, STRIDE * STRIDE * cout)
        out = subpixel_out(tf.constant(X, tf.float32), tf.constant(Ku, tf.float32), pad)
        if out.shape != Y.shape:
            return None, None, out.shape
        cols.append(out.reshape(-1))
    A = np.stack(cols, axis=1)
    b = Y.reshape(-1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = float(np.abs(A @ sol - b).max())
    return sol.reshape(kk, kk, cin, STRIDE * STRIDE * cout), \
        resid / max(float(np.abs(b).max()), 1e-12), Y.shape


def sweep(cin=2, cout=3, h=5, w=6, probes=120, seed=0, self_test=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((probes, h, w, cin)).astype(np.float64)

    if self_test:
        kk = 2
        Ktrue = rng.standard_normal(
            (kk, kk, cin, STRIDE * STRIDE * cout)).astype(np.float64)
        Y = subpixel_out(tf.constant(X, tf.float32),
                         tf.constant(Ktrue, tf.float32), "SAME").astype(np.float64)
        K, rel, _ = _solve_for(X, Y, cin, cout, kk, "SAME")
        print(f"SELF-TEST  relative residual {rel:.3e}  "
              f"max |K-Ktrue| {np.abs(K - Ktrue).max():.3e}")
        print("solver is trustworthy" if rel < 1e-6 else "SOLVER IS BROKEN")
        return

    W = rng.standard_normal((4, 4, cout, cin)).astype(np.float64)
    Y = deconv_out(tf.constant(X, tf.float32), tf.constant(W, tf.float32),
                   (h * STRIDE, w * STRIDE)).astype(np.float64)
    print(f"target: Conv2DTranspose k=4 s=2 SAME, out {Y.shape}\n")

    pads = [("SAME", "SAME")]
    for kk in (2, 3, 4):
        for t, b, l, r in itertools.product(range(2), repeat=4):
            pads.append((f"pad t{t} b{b} l{l} r{r}", ((t, b), (l, r))))

    print(f"{'conv k':>7s} {'padding':>18s} {'rel residual':>14s}")
    best = None
    for kk in (2, 3, 4):
        for label, pad in pads:
            try:
                K, rel, shape = _solve_for(X, Y, cin, cout, kk, pad)
            except Exception:
                continue
            if K is None:
                continue
            if rel < 1e-6:
                print(f"{kk:7d} {label:>18s} {rel:14.3e}   <-- EXACT")
                if best is None:
                    best = (kk, label, pad, K)
            elif rel < 0.5:
                print(f"{kk:7d} {label:>18s} {rel:14.3e}")
    if best is None:
        print("\nNo exact equivalence found in the swept space.")
    else:
        kk, label, pad, K = best
        print(f"\nEXACT with conv kernel {kk}, {label}")
        flat_w = W.reshape(-1)
        for p, q, sub, co in itertools.product(range(kk), range(kk),
                                               range(STRIDE * STRIDE), range(1)):
            val = K[p, q, 0, sub * cout + co]
            hits = np.argwhere(np.isclose(flat_w, val, atol=1e-9))
            if len(hits) == 1:
                kh, kw, wco, wci = np.unravel_index(hits[0][0], W.shape)
                da, db = divmod(sub, STRIDE)
                print(f"  conv[{p},{q},ci,sub({da},{db})co{co}] = W[{kh},{kw},{wco},{wci}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    sweep(self_test=a.self_test)
