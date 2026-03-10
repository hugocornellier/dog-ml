#!/usr/bin/env python3
"""Generate GT and predicted landmark visualizations for test images.

Uses 3-model ensemble (256+320+384) with multi-scale+flip TTA (18 passes)
matching the best config: NME_IOD = 8.04.

For each image produces:
  {stem}_landmarks_true.png  — ground-truth landmarks + bbox from DogFLW labels
  {stem}_landmarks_pred.png  — predicted landmarks + bbox from the TFLite pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# Reuse helpers from the inference script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_dog_landmarks_tflite import (
    NUM_LANDMARKS,
    _LM_GROUPS,
    _LM_EDGES,
    _lm_colour,
    letterbox_image,
    run_bbox_tflite,
    deletterbox_bbox,
    crop_for_landmarks,
    denormalize_landmarks,
    load_gt,
    draw_bbox,
    draw_landmarks,
)

# Paths
DATA_ROOT = (
    Path.home() / ".cache" / "kagglehub" / "datasets"
    / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
)
BBOX_MODEL = Path("artifacts/dog_face_detector/dog_face_localizer_224_float16.tflite")
LM_MODELS = [
    {"path": Path("artifacts/tight_margin_256/dog_face_landmarks_256_float16.tflite"), "img_size": 256},
    {"path": Path("artifacts/tight_margin_320/dog_face_landmarks_320_float16.tflite"), "img_size": 320},
    {"path": Path("artifacts/tight_margin_384/dog_face_landmarks_384_float16.tflite"), "img_size": 384},
]
OUT_DIR = Path("artifacts/dog_face_landmarks/inference_examples")

BBOX_IMG_SIZE = 224
LM_MARGIN = 0.05
CROP_MARGIN = 0.10
SCALES = [0.9, 1.0, 1.1]

FLIP_INDEX = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    17, 16, 19, 18, 21, 20, 23, 22, 24, 25, 27, 26, 29, 28,
    31, 30, 32, 34, 33, 35, 37, 36, 38, 40, 39, 41, 42, 44,
    43, 45,
]

# Use all test images.
STEMS = sorted(
    p.stem for p in (DATA_ROOT / "test" / "images").glob("*.png")
    if (DATA_ROOT / "test" / "labels" / f"{p.stem}.json").exists()
)


# ---------------------------------------------------------------------------
# TFLite interpreter cache (avoid reloading per image)
# ---------------------------------------------------------------------------
_interp_cache: dict[str, tf.lite.Interpreter] = {}


def get_interpreter(model_path: Path) -> tf.lite.Interpreter:
    key = str(model_path)
    if key not in _interp_cache:
        interp = tf.lite.Interpreter(model_path=key)
        interp.allocate_tensors()
        _interp_cache[key] = interp
    return _interp_cache[key]


def run_landmark_tflite_cached(model_path: Path, arr: np.ndarray) -> np.ndarray:
    interp = get_interpreter(model_path)
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    inp = np.expand_dims(arr, 0).astype(in_d["dtype"])
    interp.set_tensor(in_d["index"], inp)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])[0].astype(np.float32)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Scale / flip helpers (numpy versions of the TF ops in eval scripts)
# ---------------------------------------------------------------------------

def scale_crop_np(arr: np.ndarray, scale: float) -> np.ndarray:
    """Apply scale transform to a [H,W,3] float32 crop array.

    scale < 1.0: zoom in (center-crop then resize back)
    scale > 1.0: zoom out (reflect-pad then resize back)
    """
    if abs(scale - 1.0) < 1e-6:
        return arr
    h, w = arr.shape[:2]
    if scale < 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        off_y = (h - new_h) // 2
        off_x = (w - new_w) // 2
        cropped = arr[off_y:off_y + new_h, off_x:off_x + new_w]
        img = Image.fromarray((cropped * 255).astype(np.uint8))
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(img).astype(np.float32) / 255.0
    else:
        pad_h = int(round(h * (scale - 1.0) / 2.0))
        pad_w = int(round(w * (scale - 1.0) / 2.0))
        padded = np.pad(arr, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="reflect")
        img = Image.fromarray((padded * 255).astype(np.uint8))
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(img).astype(np.float32) / 255.0


def unscale_coords(coords_2d: np.ndarray, scale: float) -> np.ndarray:
    """Map coords from scaled frame back to original: p_orig = (p_scaled - 0.5) * s + 0.5"""
    if abs(scale - 1.0) < 1e-6:
        return coords_2d
    return (coords_2d - 0.5) * scale + 0.5


def flip_coords(coords_2d: np.ndarray) -> np.ndarray:
    """Flip landmark coordinates: reorder with FLIP_INDEX and mirror x."""
    remapped = coords_2d[FLIP_INDEX]
    remapped[:, 0] = 1.0 - remapped[:, 0]
    return remapped


# ---------------------------------------------------------------------------
# Ensemble prediction
# ---------------------------------------------------------------------------

def predict_ensemble(image: Image.Image, pred_bbox: np.ndarray) -> np.ndarray:
    """Run 3-model ensemble with multi-scale+flip TTA (18 passes).

    Returns predicted landmarks in original image coordinates [46, 2].
    """
    all_preds = []

    for model_info in LM_MODELS:
        model_path = model_info["path"]
        img_size = model_info["img_size"]

        # Get crop for this model's resolution
        crop_arr, crop_meta = crop_for_landmarks(image, pred_bbox, CROP_MARGIN, img_size)

        for scale in SCALES:
            # Normal orientation
            scaled = scale_crop_np(crop_arr, scale)
            lm_norm = run_landmark_tflite_cached(model_path, scaled)
            coords = lm_norm.reshape(NUM_LANDMARKS, 2)
            coords = unscale_coords(coords, scale)
            all_preds.append(coords)

            # Flipped orientation
            flipped = np.ascontiguousarray(scaled[:, ::-1, :])
            lm_norm_f = run_landmark_tflite_cached(model_path, flipped)
            coords_f = lm_norm_f.reshape(NUM_LANDMARKS, 2)
            coords_f = flip_coords(coords_f)
            coords_f = unscale_coords(coords_f, scale)
            all_preds.append(coords_f)

    # Average all 18 predictions (normalized coords)
    avg_norm = np.mean(np.stack(all_preds, axis=0), axis=0)

    # Denormalize using the first model's crop meta (both share same crop_margin
    # so the crop region is identical — only resize target differs)
    _, ref_meta = crop_for_landmarks(image, pred_bbox, CROP_MARGIN, LM_MODELS[0]["img_size"])
    avg_flat = avg_norm.reshape(-1)
    pred_pts = denormalize_landmarks(avg_flat, **ref_meta)
    return pred_pts


def make_true_image(image: Image.Image, label_path: Path) -> Image.Image:
    """Draw GT bbox (green) and GT landmarks on a copy of the image."""
    gt_bbox, gt_pts = load_gt(label_path, image.size, LM_MARGIN)
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw_bbox(draw, gt_bbox, (64, 255, 64), "gt")
    draw_landmarks(draw, gt_pts, radius=3, draw_edges=True)
    return vis


def make_pred_image(image: Image.Image) -> Image.Image:
    """Run 3-model ensemble + ms+flip TTA and draw predicted bbox + landmarks."""
    arr_lb, lb_meta = letterbox_image(image, BBOX_IMG_SIZE)
    pred_bbox_norm = run_bbox_tflite(BBOX_MODEL, arr_lb)
    pred_bbox = deletterbox_bbox(pred_bbox_norm, **lb_meta)

    pred_pts = predict_ensemble(image, pred_bbox)

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw_bbox(draw, pred_bbox, (255, 64, 64), "pred")
    draw_landmarks(draw, pred_pts, radius=3, draw_edges=True)
    return vis


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(STEMS)
    print(f"Generating {total} image pairs using 3-model ensemble (256+320+384) + ms+flip TTA")
    print(f"  Models: {[m['path'].name for m in LM_MODELS]}")
    print(f"  Scales: {SCALES}, flip=True -> 18 passes per image")
    print(f"  Output: {OUT_DIR}\n")

    for i, stem in enumerate(STEMS):
        img_path = DATA_ROOT / "test" / "images" / f"{stem}.png"
        lbl_path = DATA_ROOT / "test" / "labels" / f"{stem}.json"
        if not img_path.exists() or not lbl_path.exists():
            print(f"SKIP {stem} (missing image or label)")
            continue

        image = Image.open(img_path).convert("RGB")

        true_img = make_true_image(image, lbl_path)
        true_img.save(OUT_DIR / f"{stem}_landmarks_true.png")

        pred_img = make_pred_image(image)
        pred_img.save(OUT_DIR / f"{stem}_landmarks_pred.png")

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] OK  {stem}")

    print(f"\nDone. {total} pairs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
