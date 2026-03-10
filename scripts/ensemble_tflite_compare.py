#!/usr/bin/env python3
"""Run 3-model TFLite ensemble with multi-scale + flip TTA on a single image.

Mirrors the Flutter EnsembleLandmarkModel pipeline exactly:
  - Stage 1: bbox via dog_face_localizer_224_float16.tflite
  - Stage 2: 3 models x 3 scales x 2 flips = 18 passes, averaged

Outputs JSON with bbox and all 46 landmark coordinates for comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

NUM_LANDMARKS = 46
SCALES = [0.9, 1.0, 1.1]
CROP_MARGIN = 0.20  # Match Flutter's default

FLIP_INDEX = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    17, 16, 19, 18, 21, 20, 23, 22, 24, 25, 27, 26, 29, 28,
    31, 30, 32, 34, 33, 35, 37, 36, 38, 40, 39, 41, 42, 44,
    43, 45,
]

BBOX_MODEL = Path("artifacts/dog_face_detector/dog_face_localizer_224_float16.tflite")
LM_MODELS = [
    ("256", Path("artifacts/tight_margin_256/dog_face_landmarks_256_float16.tflite"), 256),
    ("320", Path("artifacts/tight_margin_320/dog_face_landmarks_320_float16.tflite"), 320),
    ("384", Path("artifacts/tight_margin_384/dog_face_landmarks_384_float16.tflite"), 384),
]


def letterbox_image(image, img_size=224):
    image = image.convert("RGB")
    w, h = image.size
    scale = min(img_size / w, img_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (img_size - new_w) // 2, (img_size - new_h) // 2
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    return arr, {"orig_w": float(w), "orig_h": float(h), "scale": float(scale),
                 "pad_x": float(pad_x), "pad_y": float(pad_y), "img_size": float(img_size)}


def run_bbox(arr):
    interp = tf.lite.Interpreter(model_path=str(BBOX_MODEL))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    inp = np.expand_dims(arr, 0).astype(in_d["dtype"])
    interp.set_tensor(in_d["index"], inp)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])[0].astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def deletterbox_bbox(bbox_norm, orig_w, orig_h, scale, pad_x, pad_y, img_size):
    b = bbox_norm * img_size
    b[[0, 2]] -= pad_x
    b[[1, 3]] -= pad_y
    b /= scale
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    return np.array([
        np.clip(x1, 0, orig_w), np.clip(y1, 0, orig_h),
        np.clip(x2, 0, orig_w), np.clip(y2, 0, orig_h),
    ], dtype=np.float32)


def crop_for_landmarks(image, bbox_abs, lm_img_size):
    w, h = image.size
    x1, y1, x2, y2 = bbox_abs.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0.0, x1 - bw * CROP_MARGIN)
    cy1 = max(0.0, y1 - bh * CROP_MARGIN)
    cx2 = min(float(w), x2 + bw * CROP_MARGIN)
    cy2 = min(float(h), y2 + bh * CROP_MARGIN)
    cropped = image.crop((cx1, cy1, cx2, cy2))
    crop_w, crop_h = cropped.size
    resized = cropped.resize((lm_img_size, lm_img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    return arr, {"cx1": cx1, "cy1": cy1, "crop_w": float(crop_w), "crop_h": float(crop_h)}


def scale_crop(crop_arr, scale, img_size):
    if abs(scale - 1.0) < 1e-6:
        return crop_arr.copy()
    h, w = img_size, img_size
    if scale < 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        offset_y = (h - new_h) // 2
        offset_x = (w - new_w) // 2
        cropped = crop_arr[offset_y:offset_y+new_h, offset_x:offset_x+new_w, :]
        pil = Image.fromarray((cropped * 255).astype(np.uint8))
        resized = pil.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(resized).astype(np.float32) / 255.0
    else:
        pad_h = int(round(h * (scale - 1.0) / 2.0))
        pad_w = int(round(w * (scale - 1.0) / 2.0))
        padded = np.pad(crop_arr, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="reflect")
        pil = Image.fromarray((padded * 255).astype(np.uint8))
        resized = pil.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(resized).astype(np.float32) / 255.0


def unscale_coords(coords, scale):
    if abs(scale - 1.0) < 1e-6:
        return coords
    return (coords - 0.5) * scale + 0.5


def run_lm_model(interp, arr):
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    inp = np.expand_dims(arr, 0).astype(in_d["dtype"])
    interp.set_tensor(in_d["index"], inp)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])[0].astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not image_path:
        print("Usage: python ensemble_tflite_compare.py <image_path>")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")

    # Stage 1: bbox
    arr_lb, lb_meta = letterbox_image(image, 224)
    bbox_norm = run_bbox(arr_lb)
    bbox_abs = deletterbox_bbox(bbox_norm, **lb_meta)
    print(f"Bbox: {bbox_abs.tolist()}", file=sys.stderr)

    # Load all 3 landmark interpreters
    interpreters = {}
    for name, path, size in LM_MODELS:
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        interpreters[name] = (interp, size)

    # Crop at 384px (largest), then resize for smaller models
    crop384_arr, crop_meta = crop_for_landmarks(image, bbox_abs, 384)

    # Also crop at native sizes for 256 and 320
    crop256_arr, _ = crop_for_landmarks(image, bbox_abs, 256)
    crop320_arr, _ = crop_for_landmarks(image, bbox_abs, 320)

    crops = {"256": crop256_arr, "320": crop320_arr, "384": crop384_arr}

    # Run 18 passes: 3 models x 3 scales x 2 orientations
    all_preds = []
    for name, (interp, size) in interpreters.items():
        crop = crops[name]
        for scale in SCALES:
            # Normal
            scaled = scale_crop(crop, scale, size)
            raw = run_lm_model(interp, scaled)
            coords = raw.reshape(NUM_LANDMARKS, 2)
            coords = unscale_coords(coords, scale)
            all_preds.append(coords)

            # Flipped
            flipped = np.fliplr(scaled)
            raw_f = run_lm_model(interp, flipped)
            coords_f = raw_f.reshape(NUM_LANDMARKS, 2)
            remapped = coords_f[FLIP_INDEX]
            remapped = np.stack([1.0 - remapped[:, 0], remapped[:, 1]], axis=-1)
            remapped = unscale_coords(remapped, scale)
            all_preds.append(remapped)

    # Average
    avg = np.mean(np.stack(all_preds), axis=0)

    # Map back to original image coords
    cx1 = crop_meta["cx1"]
    cy1 = crop_meta["cy1"]
    crop_w = crop_meta["crop_w"]
    crop_h = crop_meta["crop_h"]

    pts = avg.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * crop_w + cx1
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * crop_h + cy1

    result = {
        "bbox": bbox_abs.tolist(),
        "crop_meta": crop_meta,
        "num_passes": len(all_preds),
        "landmarks": [
            {"idx": i, "x": round(float(pts[i, 0]), 2), "y": round(float(pts[i, 1]), 2)}
            for i in range(NUM_LANDMARKS)
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
