"""Shared measurement harness for the ~11MB landmark model Pareto push.

Everything a candidate has to beat the baseline on lives here, so that accuracy
and latency are always produced by the same code path:

  * ``load_val_cache``   -- the 480 DogFLW test crops at 384px, preprocessed
                            exactly the way ``train_dog_face_landmarks.py``
                            does it (lm_margin=0.05, crop_margin=0.10), cached
                            to disk so every candidate sees identical input.
  * ``eval_tflite``      -- full-val NME_IOD (not a 16-image sample) plus the
                            per-region breakdown, on the converted TFLite file.
  * ``bench_tflite``     -- median latency, fixed thread count, same process.

The baseline's ``tflite_sanity`` in model_metadata.json is a 16-image crop-space
NME, which is not comparable to the 480-image NME_IOD the Keras model reports.
Judging candidates needs both computed the same way, which is what this does.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
)
CACHE_DIR = Path("/private/tmp/claude-501/-Users-hugocornellier-IdeaProjects-dog-detection"
                 "/ee61cbfd-dd84-4c4a-82ea-cb1141e124d6/scratchpad/valcache")

NUM_LANDMARKS = 46
LEFT_OUTER_EYE_IDX = 18
RIGHT_OUTER_EYE_IDX = 19

REGIONS = {
    "right_ear":     list(range(0, 9)),
    "left_ear":      list(range(9, 18)),
    "right_eye":     list(range(18, 24)),
    "left_eye":      list(range(24, 30)),
    "nose_bridge":   list(range(30, 34)),
    "nose_nostrils": list(range(34, 42)),
    "mouth":         list(range(42, 46)),
}

# Verified involution from LANDMARK_DETECTION_REPORT.md.
FLIP_INDEX = np.array([
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    17, 16, 19, 18, 21, 20, 23, 22, 24, 25, 27, 26, 29, 28,
    31, 30, 32, 34, 33, 35, 37, 36, 38, 40, 39, 41, 42, 44,
    43, 45,
], dtype=np.int32)


# ---------------------------------------------------------------------------
# Validation cache
# ---------------------------------------------------------------------------

def build_val_cache(img_size: int = 384, lm_margin: float = 0.05,
                    crop_margin: float = 0.10) -> tuple[Path, Path]:
    """Preprocess the DogFLW test split once and memmap it to disk.

    Reuses the training script's own ``load_split_records`` and
    ``crop_and_normalize`` so there is no chance of the eval crop drifting from
    the training crop.
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import tensorflow as tf
    import train_dog_face_landmarks as T

    tag = f"{img_size}_{lm_margin}_{crop_margin}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    crops_path = CACHE_DIR / f"crops_{tag}.npy"
    gt_path = CACHE_DIR / f"gt_{tag}.npy"
    if crops_path.exists() and gt_path.exists():
        return crops_path, gt_path

    records = T.load_split_records(DATA_ROOT, "test", lm_margin)
    n = len(records)
    crops = np.lib.format.open_memmap(
        crops_path, mode="w+", dtype=np.float32, shape=(n, img_size, img_size, 3)
    )
    gt = np.zeros((n, NUM_LANDMARKS * 2), dtype=np.float32)
    # Crop window (x1, y1, w, h) in original image pixels, so predictions can be
    # mapped back for the absolute-pixel-space metric the package publishes.
    boxes = np.zeros((n, 4), dtype=np.float32)
    gt_abs = np.zeros((n, NUM_LANDMARKS * 2), dtype=np.float32)

    for i, rec in enumerate(records):
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32
        )
        crop, lm_norm = T.crop_and_normalize(
            image, tf.constant(rec.bbox_xyxy_abs, tf.float32), lm_flat,
            img_size, crop_margin,
        )
        crops[i] = crop.numpy()
        gt[i] = lm_norm.numpy()

        # Reproduce crop_and_normalize's integer crop window exactly.
        iw, ih = rec.orig_size_wh
        x1, y1, x2, y2 = rec.bbox_xyxy_abs
        mx, my = (x2 - x1) * crop_margin, (y2 - y1) * crop_margin
        cx1i = int(np.floor(max(0.0, x1 - mx)))
        cy1i = int(np.floor(max(0.0, y1 - my)))
        cx2i = min(int(np.ceil(min(float(iw), x2 + mx))), iw)
        cy2i = min(int(np.ceil(min(float(ih), y2 + my))), ih)
        boxes[i] = (cx1i, cy1i, max(cx2i - cx1i, 1), max(cy2i - cy1i, 1))
        gt_abs[i] = np.asarray(rec.landmarks_abs, dtype=np.float32).reshape(-1)

        if (i + 1) % 100 == 0:
            print(f"  cached {i + 1}/{n}")

    crops.flush()
    np.save(gt_path, gt)
    np.save(CACHE_DIR / f"boxes_{tag}.npy", boxes)
    np.save(CACHE_DIR / f"gtabs_{tag}.npy", gt_abs)
    print(f"Cached {n} val crops -> {crops_path}")
    return crops_path, gt_path


def load_val_cache(img_size: int = 384, lm_margin: float = 0.05,
                   crop_margin: float = 0.10):
    crops_path, gt_path = build_val_cache(img_size, lm_margin, crop_margin)
    return np.load(crops_path, mmap_mode="r"), np.load(gt_path)


def load_abs_refs(img_size: int = 384, lm_margin: float = 0.05,
                  crop_margin: float = 0.10):
    """(boxes, gt_abs) for the absolute-image-pixel-space metric."""
    tag = f"{img_size}_{lm_margin}_{crop_margin}"
    return (np.load(CACHE_DIR / f"boxes_{tag}.npy"),
            np.load(CACHE_DIR / f"gtabs_{tag}.npy"))


def nme_iod_abs(pred_norm: np.ndarray, boxes: np.ndarray,
                gt_abs: np.ndarray) -> dict:
    """NME_IOD computed in original image pixels, not normalized crop units.

    The crop is generally not square, so normalizing x by crop width and y by
    crop height (what the training metric does) anisotropically rescales every
    distance. The Flutter package publishes the undistorted absolute-pixel
    figure, so shipping needs both numbers.
    """
    p = pred_norm.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    g = gt_abs.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    x0, y0, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    px = p[:, :, 0] * bw[:, None] + x0[:, None]
    py = p[:, :, 1] * bh[:, None] + y0[:, None]
    pa = np.stack([px, py], axis=-1)

    iod = np.sqrt(np.sum(
        (g[:, LEFT_OUTER_EYE_IDX] - g[:, RIGHT_OUTER_EYE_IDX]) ** 2, axis=-1) + 1e-8)
    dist = np.sqrt(np.sum((pa - g) ** 2, axis=-1) + 1e-8)
    per_lm = dist / np.maximum(iod, 1e-8)[:, None] * 100.0
    per_sample = per_lm.mean(axis=1)
    return {
        "nme_iod_abs": float(per_sample.mean()),
        "sem": float(per_sample.std(ddof=1) / np.sqrt(len(per_sample))),
        "regions": {r: float(per_lm[:, idx].mean()) for r, idx in REGIONS.items()},
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nme_iod_per_landmark(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """[N, 46] per-landmark NME_IOD in percent, IOD taken from ground truth."""
    g = gt.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    p = pred.reshape(-1, NUM_LANDMARKS, 2).astype(np.float64)
    iod = np.sqrt(
        np.sum((g[:, LEFT_OUTER_EYE_IDX] - g[:, RIGHT_OUTER_EYE_IDX]) ** 2, axis=-1)
        + 1e-8
    )
    dist = np.sqrt(np.sum((p - g) ** 2, axis=-1) + 1e-8)
    return dist / np.maximum(iod, 1e-8)[:, None] * 100.0


def summarize(gt: np.ndarray, pred: np.ndarray) -> dict:
    per_lm = nme_iod_per_landmark(gt, pred)
    per_sample = per_lm.mean(axis=1)
    out = {
        "nme_iod": float(per_sample.mean()),
        # Standard error over the 480 val images: the yardstick for whether a
        # delta is real. Paired comparisons should use paired_delta() instead.
        "sem": float(per_sample.std(ddof=1) / np.sqrt(len(per_sample))),
        "n": int(len(per_sample)),
        "regions": {
            r: float(per_lm[:, idx].mean()) for r, idx in REGIONS.items()
        },
    }
    return out


def paired_delta(gt: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Paired per-image delta (b - a). Far tighter than comparing two SEMs."""
    a = nme_iod_per_landmark(gt, pred_a).mean(axis=1)
    b = nme_iod_per_landmark(gt, pred_b).mean(axis=1)
    d = b - a
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "mean_delta": float(d.mean()),
        "sem": float(sem),
        "t": float(d.mean() / sem) if sem > 0 else 0.0,
        "n_better": int((d < 0).sum()),
        "n_worse": int((d > 0).sum()),
    }


# ---------------------------------------------------------------------------
# TFLite
# ---------------------------------------------------------------------------

def _interpreter(path: Path, threads: int = 4):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    interp.allocate_tensors()
    return interp


def predict_tflite(path: Path, crops: np.ndarray, threads: int = 4) -> np.ndarray:
    interp = _interpreter(path, threads)
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    n = crops.shape[0]
    preds = np.zeros((n, NUM_LANDMARKS * 2), dtype=np.float32)
    for i in range(n):
        interp.set_tensor(in_det["index"], crops[i:i + 1].astype(in_det["dtype"]))
        interp.invoke()
        preds[i] = interp.get_tensor(out_det["index"])[0].astype(np.float32)
    return np.clip(preds, 0.0, 1.0)


def eval_tflite(path: Path, crops: np.ndarray, gt: np.ndarray,
                threads: int = 4) -> tuple[dict, np.ndarray]:
    preds = predict_tflite(path, crops, threads)
    return summarize(gt, preds), preds


def bench_tflite(path: Path, crops: np.ndarray, threads: int = 4,
                 warmup: int = 10, runs: int = 60) -> dict:
    """Median single-image latency. Same process, same thread count, always."""
    interp = _interpreter(path, threads)
    in_det = interp.get_input_details()[0]
    sample = crops[0:1].astype(in_det["dtype"])
    for _ in range(warmup):
        interp.set_tensor(in_det["index"], sample)
        interp.invoke()
    times = []
    for i in range(runs):
        img = crops[i % crops.shape[0]:i % crops.shape[0] + 1].astype(in_det["dtype"])
        interp.set_tensor(in_det["index"], img)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "median_ms": float(np.median(times)),
        "p10_ms": float(np.percentile(times, 10)),
        "p90_ms": float(np.percentile(times, 90)),
        "mean_ms": float(times.mean()),
        "threads": threads,
        "runs": runs,
    }


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def report(name: str, path: Path, crops, gt, threads: int = 4):
    acc, preds = eval_tflite(path, crops, gt, threads)
    lat = bench_tflite(path, crops, threads)
    row = {
        "name": name,
        "path": str(path),
        "size_mb": size_mb(path),
        **acc,
        "latency": lat,
    }
    print(json.dumps(row, indent=2))
    return row, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--save-preds", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    crops, gt = load_val_cache(args.img_size)
    row, preds = report(args.name or args.tflite.stem, args.tflite, crops, gt,
                        args.threads)
    if args.save_preds:
        np.save(args.save_preds, preds)
    if args.out:
        args.out.write_text(json.dumps(row, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
