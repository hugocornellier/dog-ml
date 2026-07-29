"""Sweep coordinate-extraction methods on the shipped model's raw heatmaps.

SoftArgmax2D costs 0.5 ms of the model's 57 ms, so anything that replaces it is
free on both size and latency. That makes extraction the cheapest possible place
to look for accuracy, which is why it gets tested even though there is a strong
prior against it: the model was trained with coordinate MSE *through* a global
beta=1 soft-argmax, so its heatmaps are shaped to make that particular estimator
correct, not to put their mode on the landmark.

Methods:
  soft_beta_B      global spatial softmax at temperature B, expectation (B=1 is
                   what the model ships with)
  argmax           hard argmax, no subpixel
  argmax_parabola  argmax + 1D parabolic fit per axis (the classic subpixel fix)
  local_R_B        argmax, then softmax-expectation restricted to a (2R+1)^2
                   window around the peak at temperature B

Reads heatmaps from a heatmap-output TFLite export so the numbers land on the
same converted graph the accuracy table uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from pareto_harness import load_val_cache, summarize, paired_delta  # noqa: E402

HM_TFLITE = REPO / "artifacts" / "pareto" / "profile" / "heatmap_conv.tflite"


def soft_argmax(hm: np.ndarray, beta: float) -> np.ndarray:
    """hm: (H, W, K) logits -> (K*2,) coords in [0,1], matching SoftArgmax2D."""
    h, w, k = hm.shape
    flat = hm.reshape(h * w, k).astype(np.float64) * beta
    flat -= flat.max(axis=0, keepdims=True)
    e = np.exp(flat)
    p = e / e.sum(axis=0, keepdims=True)
    p = p.reshape(h, w, k)
    xs = np.linspace(0.0, 1.0, w)
    ys = np.linspace(0.0, 1.0, h)
    x = (p.sum(axis=0) * xs[:, None]).sum(axis=0)
    y = (p.sum(axis=1) * ys[:, None]).sum(axis=0)
    return np.stack([x, y], axis=-1).reshape(-1)


def _peaks(hm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w, k = hm.shape
    idx = hm.reshape(h * w, k).argmax(axis=0)
    return idx // w, idx % w  # (row, col) per landmark


def argmax_coords(hm: np.ndarray) -> np.ndarray:
    h, w, _ = hm.shape
    py, px = _peaks(hm)
    return np.stack([px / (w - 1), py / (h - 1)], axis=-1).reshape(-1)


def argmax_parabola(hm: np.ndarray) -> np.ndarray:
    h, w, k = hm.shape
    py, px = _peaks(hm)
    ki = np.arange(k)
    dx = np.zeros(k)
    dy = np.zeros(k)

    ok = (px > 0) & (px < w - 1)
    if ok.any():
        c = hm[py[ok], px[ok], ki[ok]]
        l = hm[py[ok], px[ok] - 1, ki[ok]]
        r = hm[py[ok], px[ok] + 1, ki[ok]]
        denom = 2.0 * c - l - r
        d = np.where(np.abs(denom) > 1e-8, 0.5 * (r - l) / np.where(denom == 0, 1, denom), 0.0)
        dx[ok] = np.clip(d, -0.5, 0.5)

    ok = (py > 0) & (py < h - 1)
    if ok.any():
        c = hm[py[ok], px[ok], ki[ok]]
        t = hm[py[ok] - 1, px[ok], ki[ok]]
        b = hm[py[ok] + 1, px[ok], ki[ok]]
        denom = 2.0 * c - t - b
        d = np.where(np.abs(denom) > 1e-8, 0.5 * (b - t) / np.where(denom == 0, 1, denom), 0.0)
        dy[ok] = np.clip(d, -0.5, 0.5)

    return np.stack([(px + dx) / (w - 1), (py + dy) / (h - 1)], axis=-1).reshape(-1)


def local_soft_argmax(hm: np.ndarray, radius: int, beta: float) -> np.ndarray:
    """Softmax-expectation restricted to a window around the argmax peak."""
    h, w, k = hm.shape
    py, px = _peaks(hm)
    pad = radius
    padded = np.pad(hm, ((pad, pad), (pad, pad), (0, 0)), mode="constant",
                    constant_values=-1e9)
    size = 2 * radius + 1
    offs = np.arange(size)
    # Window per landmark: (K, size, size)
    rows = (py[:, None] + offs[None, :])            # already shifted by pad
    cols = (px[:, None] + offs[None, :])
    win = padded[rows[:, :, None], cols[:, None, :], np.arange(k)[:, None, None]]

    z = win.astype(np.float64) * beta
    z -= z.max(axis=(1, 2), keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=(1, 2), keepdims=True)

    rel = offs - radius
    dy = (p.sum(axis=2) * rel[None, :]).sum(axis=1)
    dx = (p.sum(axis=1) * rel[None, :]).sum(axis=1)
    return np.stack([(px + dx) / (w - 1), (py + dy) / (h - 1)], axis=-1).reshape(-1)


def main():
    import tensorflow as tf

    crops, gt = load_val_cache()
    interp = tf.lite.Interpreter(model_path=str(HM_TFLITE), num_threads=4)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print("heatmap output shape:", out_det["shape"])

    methods: dict[str, callable] = {}
    for b in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        methods[f"soft_beta_{b}"] = (lambda hm, b=b: soft_argmax(hm, b))
    methods["argmax"] = argmax_coords
    methods["argmax_parabola"] = argmax_parabola
    for r in (3, 5, 9, 15, 25):
        for b in (1.0, 2.0, 4.0):
            methods[f"local_r{r}_b{b}"] = (
                lambda hm, r=r, b=b: local_soft_argmax(hm, r, b)
            )

    n = crops.shape[0]
    preds = {name: np.zeros((n, 92), dtype=np.float32) for name in methods}
    for i in range(n):
        interp.set_tensor(in_det["index"], crops[i:i + 1].astype(in_det["dtype"]))
        interp.invoke()
        hm = interp.get_tensor(out_det["index"])[0]  # (H, W, K)
        for name, fn in methods.items():
            preds[name][i] = fn(hm)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}")

    base = preds["soft_beta_1.0"]
    rows = []
    for name in methods:
        p = np.clip(preds[name], 0.0, 1.0)
        s = summarize(gt, p)
        d = paired_delta(gt, base, p)
        rows.append((name, s["nme_iod"], d["mean_delta"], d["sem"], d["t"]))

    rows.sort(key=lambda r: r[1])
    print()
    print(f"{'method':22s} {'NME_IOD':>9s} {'delta':>9s} {'+-sem':>8s} {'t':>7s}")
    for name, nme, dm, ds, t in rows:
        print(f"{name:22s} {nme:9.4f} {dm:+9.4f} {ds:8.4f} {t:+7.2f}")

    np.save(REPO / "artifacts" / "pareto" / "extraction_best.npy",
            preds[rows[0][0]])


if __name__ == "__main__":
    main()
