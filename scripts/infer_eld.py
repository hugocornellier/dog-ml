#!/usr/bin/env python3
"""End-to-end ELD inference: face detector -> center detector -> 5 specialists.

Usage:
  python scripts/infer_eld.py --image path/to/dog.jpg
  python scripts/infer_eld.py --image path/to/dog.jpg --output-image result.png
  python scripts/infer_eld.py --image path/to/dog.jpg --label-json path/to/label.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_eld import REGION_DEFS, REGION_ORDER, FACE_CROP_SIZE, FACE_CROP_MARGIN
from train_dog_face_landmarks import NUM_LANDMARKS, LEFT_OUTER_EYE_IDX, RIGHT_OUTER_EYE_IDX


# ---------------------------------------------------------------------------
# TFLite helpers
# ---------------------------------------------------------------------------

def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def run_tflite(interp, input_np):
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    interp.set_tensor(in_det["index"], input_np.astype(in_det["dtype"]))
    interp.invoke()
    return interp.get_tensor(out_det["index"])[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Face detection (letterbox approach from existing inference script)
# ---------------------------------------------------------------------------

def letterbox_image(image_np, target_size=224):
    """Pad image to square, resize to target_size. Returns (resized, scale, pad_x, pad_y)."""
    h, w = image_np.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = np.array(Image.fromarray(
        (image_np * 255).astype(np.uint8)
    ).resize((new_w, new_h), Image.BILINEAR)) / 255.0

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas = np.zeros((target_size, target_size, 3), dtype=np.float32)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def deletterbox_bbox(bbox_norm, orig_w, orig_h, target_size=224):
    """Convert normalized bbox from letterboxed space to original image pixel coords."""
    scale = target_size / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    x1 = (bbox_norm[0] * target_size - pad_x) / scale
    y1 = (bbox_norm[1] * target_size - pad_y) / scale
    x2 = (bbox_norm[2] * target_size - pad_x) / scale
    y2 = (bbox_norm[3] * target_size - pad_y) / scale

    return np.clip([x1, y1, x2, y2], [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h])


# ---------------------------------------------------------------------------
# Face cropping
# ---------------------------------------------------------------------------

def crop_face(image_np, bbox_abs, margin, target_size=224):
    """Crop face from image, resize to target_size. Returns (crop, meta)."""
    h, w = image_np.shape[:2]
    x1, y1, x2, y2 = bbox_abs
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin

    cx1 = max(0, int(np.floor(x1 - mx)))
    cy1 = max(0, int(np.floor(y1 - my)))
    cx2 = min(w, int(np.ceil(x2 + mx)))
    cy2 = min(h, int(np.ceil(y2 + my)))
    cw, ch = max(cx2 - cx1, 1), max(cy2 - cy1, 1)

    crop = image_np[cy1:cy1 + ch, cx1:cx1 + cw]
    resized = np.array(Image.fromarray(
        (crop * 255).astype(np.uint8)
    ).resize((target_size, target_size), Image.BILINEAR)) / 255.0

    meta = {"cx1": cx1, "cy1": cy1, "cw": cw, "ch": ch}
    return resized.astype(np.float32), meta


def denormalize_face_landmarks(lm_face_2d, face_meta):
    """Convert face-crop [0,1] landmarks to original image pixel coords."""
    lm_abs = lm_face_2d.copy()
    lm_abs[:, 0] = lm_abs[:, 0] * face_meta["cw"] + face_meta["cx1"]
    lm_abs[:, 1] = lm_abs[:, 1] * face_meta["ch"] + face_meta["cy1"]
    return lm_abs


# ---------------------------------------------------------------------------
# Region cropping
# ---------------------------------------------------------------------------

def compute_region_bbox(center_xy, crop_scale, margin):
    half = crop_scale / 2.0 * (1.0 + margin)
    cx, cy = center_xy
    return (max(0.0, cx - half), max(0.0, cy - half),
            min(1.0, cx + half), min(1.0, cy + half))


def crop_region(face_crop_np, center_xy, crop_scale, margin, target_size=224):
    """Crop region from face crop. Returns (region_crop, meta)."""
    h, w = face_crop_np.shape[:2]
    rx1, ry1, rx2, ry2 = compute_region_bbox(center_xy, crop_scale, margin)

    rx1_px = int(np.floor(rx1 * w))
    ry1_px = int(np.floor(ry1 * h))
    rx2_px = min(int(np.ceil(rx2 * w)), w)
    ry2_px = min(int(np.ceil(ry2 * h)), h)
    rw_px = max(rx2_px - rx1_px, 1)
    rh_px = max(ry2_px - ry1_px, 1)

    region = face_crop_np[ry1_px:ry1_px + rh_px, rx1_px:rx1_px + rw_px]
    resized = np.array(Image.fromarray(
        (region * 255).astype(np.uint8)
    ).resize((target_size, target_size), Image.BILINEAR)) / 255.0

    meta = {"rx1_f": rx1_px / w, "ry1_f": ry1_px / h,
            "rw_f": rw_px / w, "rh_f": rh_px / h}
    return resized.astype(np.float32), meta


def transform_region_to_face(lm_region_flat, meta):
    """Region-local [0,1] -> face-crop [0,1]."""
    lm = lm_region_flat.reshape(-1, 2)
    lm_face_x = lm[:, 0] * meta["rw_f"] + meta["rx1_f"]
    lm_face_y = lm[:, 1] * meta["rh_f"] + meta["ry1_f"]
    return np.stack([lm_face_x, lm_face_y], axis=-1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

REGION_COLORS = {
    "right_ear": (255, 100, 100),
    "left_ear": (100, 100, 255),
    "right_eye": (255, 200, 0),
    "left_eye": (0, 200, 255),
    "nose_mouth": (0, 255, 100),
}


def draw_landmarks(image_pil, landmarks_abs, bbox_abs=None, gt_landmarks=None):
    """Draw predicted landmarks (and optionally GT) on the image."""
    draw = ImageDraw.Draw(image_pil)

    if bbox_abs is not None:
        x1, y1, x2, y2 = bbox_abs
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    for rname in REGION_ORDER:
        rdef = REGION_DEFS[rname]
        color = REGION_COLORS[rname]
        for idx in rdef["indices"]:
            x, y = landmarks_abs[idx]
            r = 3
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    if gt_landmarks is not None:
        for i in range(NUM_LANDMARKS):
            x, y = gt_landmarks[i]
            r = 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill="white", outline="gray")

    return image_pil


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_eld_pipeline(
    image_np,
    bbox_interp,
    center_interp,
    specialist_interps,
    bbox_abs=None,
):
    """Run full ELD pipeline.

    Args:
        image_np: [H, W, 3] float32 in [0, 1]
        bbox_interp: face detector TFLite (or None if bbox_abs provided)
        center_interp: center detector TFLite
        specialist_interps: dict {region_name: TFLite interpreter}
        bbox_abs: optional pre-computed face bbox [x1, y1, x2, y2] in pixels

    Returns:
        landmarks_abs: [46, 2] in original image pixel coords
        face_bbox_abs: [4] face bbox in pixel coords
    """
    h, w = image_np.shape[:2]

    # Stage 1: Face detection
    if bbox_abs is None:
        lb_img, _, _, _ = letterbox_image(image_np, 224)
        bbox_norm = run_tflite(bbox_interp, np.expand_dims(lb_img, 0))
        bbox_abs = deletterbox_bbox(bbox_norm, w, h, 224)

    # Stage 2: Face crop + center detection
    face_crop, face_meta = crop_face(image_np, bbox_abs, FACE_CROP_MARGIN, FACE_CROP_SIZE)
    centers_flat = run_tflite(center_interp, np.expand_dims(face_crop, 0))
    centers_flat = np.clip(centers_flat, 0.0, 1.0)
    centers = centers_flat.reshape(5, 2)

    # Stage 3: Regional specialists
    pred_face_2d = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)

    for j, rname in enumerate(REGION_ORDER):
        rdef = REGION_DEFS[rname]
        region_crop, region_meta = crop_region(
            face_crop, centers[j], rdef["crop_scale"], rdef["margin"],
        )
        region_pred = run_tflite(specialist_interps[rname], np.expand_dims(region_crop, 0))
        region_pred = np.clip(region_pred, 0.0, 1.0)

        lm_face = transform_region_to_face(region_pred, region_meta)
        for k, global_idx in enumerate(rdef["indices"]):
            pred_face_2d[global_idx] = lm_face[k]

    # Denormalize to original image coords
    landmarks_abs = denormalize_face_landmarks(pred_face_2d, face_meta)
    return landmarks_abs, np.array(bbox_abs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_root = (
        Path.home() / ".cache" / "kagglehub" / "datasets"
        / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
    )
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True, help="Input image path")
    p.add_argument("--output-image", type=Path, default=None, help="Save visualization")
    p.add_argument("--label-json", type=Path, default=None,
                   help="GT label JSON (for NME computation)")
    p.add_argument("--eld-dir", type=Path, default=Path("artifacts/eld"))
    p.add_argument("--bbox-model", type=Path,
                   default=Path("artifacts/dog_face_detector/dog_face_localizer_224_float16.tflite"))
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="Manual face bbox (skip face detector)")
    args = p.parse_args()

    # Load image
    img_pil = Image.open(args.image).convert("RGB")
    image_np = np.array(img_pil, dtype=np.float32) / 255.0

    # Load models
    eld_dir = Path(args.eld_dir)
    bbox_interp = None
    bbox_abs = args.bbox

    if bbox_abs is None:
        if not args.bbox_model.exists():
            raise FileNotFoundError(f"Face detector not found: {args.bbox_model}")
        bbox_interp = load_tflite(args.bbox_model)

    center_path = eld_dir / "center_detector" / "center_detector_224_float16.tflite"
    if not center_path.exists():
        raise FileNotFoundError(f"Center detector not found: {center_path}")
    center_interp = load_tflite(center_path)

    specialist_interps = {}
    for rname in REGION_ORDER:
        sp_path = eld_dir / "specialists" / rname / f"specialist_{rname}_224_float16.tflite"
        if not sp_path.exists():
            raise FileNotFoundError(f"Specialist not found: {sp_path}")
        specialist_interps[rname] = load_tflite(sp_path)

    # Run pipeline
    landmarks_abs, face_bbox = run_eld_pipeline(
        image_np, bbox_interp, center_interp, specialist_interps,
        bbox_abs=np.array(bbox_abs) if bbox_abs else None,
    )

    # Output
    result = {
        "face_bbox": face_bbox.tolist(),
        "landmarks": landmarks_abs.tolist(),
    }

    # GT comparison
    if args.label_json and args.label_json.exists():
        ann = json.loads(args.label_json.read_text())
        gt_lm = np.array(ann["landmarks"], dtype=np.float32)
        left_eye = gt_lm[LEFT_OUTER_EYE_IDX]
        right_eye = gt_lm[RIGHT_OUTER_EYE_IDX]
        iod = np.linalg.norm(left_eye - right_eye)
        if iod > 1e-8:
            dists = np.linalg.norm(landmarks_abs - gt_lm, axis=-1)
            nme_iod = float(np.mean(dists) / iod * 100.0)
            result["nme_iod"] = nme_iod
            print(f"NME_IOD: {nme_iod:.2f}")

    print(json.dumps(result, indent=2))

    # Visualization
    if args.output_image:
        gt_lm = None
        if args.label_json and args.label_json.exists():
            ann = json.loads(args.label_json.read_text())
            gt_lm = np.array(ann["landmarks"], dtype=np.float32)

        vis = draw_landmarks(img_pil.copy(), landmarks_abs, face_bbox, gt_lm)
        vis.save(args.output_image)
        print(f"Saved visualization: {args.output_image}")


if __name__ == "__main__":
    main()
