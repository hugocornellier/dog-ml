#!/usr/bin/env python3
"""Two-stage dog face landmark inference using TFLite models.

Stage 1  (bbox model)    : letterbox image -> predict face bbox in original coords
Stage 2  (landmark model): crop to bbox + margin -> predict 46 (x,y) landmarks
                           -> map landmarks back to original image coords

Draws bbox (red) and all 46 landmarks (coloured dots) on the output image.
If --label-json is provided it also draws the GT bbox (green) and GT landmarks
(white dots) and prints per-landmark and overall NME.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

NUM_LANDMARKS = 46

# ---------------------------------------------------------------------------
# DogFLW 46-landmark face mesh
# Inferred from coordinate geometry: pts 0-13 are paired left/right eye
# contours (even=left, odd=right), pts 14-15 nose bridge, 16-31 outer nose
# ring, 32-45 mouth/chin contour.
# ---------------------------------------------------------------------------

# Per-group colour and point indices
_LM_GROUPS: list[tuple[str, tuple[int, int, int], list[int]]] = [
    ("left_ear",   (100, 200, 255), [8, 6, 4, 2, 0, 10, 12]),   # left ear contour
    ("right_ear",  (255, 180,  60), [9, 7, 5, 3, 1, 11, 13]),   # right ear contour
    ("nose_ridge", (180, 255, 100), [14, 15]),                    # nose bridge
    ("left_eye",   (255, 255,  80), [16, 18, 20, 22]),            # left eye
    ("right_eye",  (255, 255,  80), [17, 19, 21, 23]),            # right eye
    ("nose",       (255, 200,  60), [24, 28, 29]),                # nose
    ("nose_tip",   (255, 140, 220), [25, 26, 27, 30, 31, 32, 33, 34]),  # nose tip
    ("mouth",      (255,  80,  80), [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]), # mouth
]

# Edges to draw as lines (index pairs)
_LM_EDGES: list[tuple[tuple[int, int, int], list[tuple[int, int]]]] = [
    ((100, 200, 255), [(8,6),(6,4),(4,2),(2,0),(0,12),(12,10),(10,8)]),     # left ear
    ((255, 180,  60), [(9,7),(7,5),(5,3),(3,1),(1,13),(13,11),(11,9)]),   # right ear
    ((255, 255,  80), [(18,20),(20,16),(16,22),(22,18)]),                  # left eye
    ((255, 255,  80), [(19,21),(21,17),(17,23),(23,19)]),                  # right eye

]


def _lm_colour(idx: int) -> tuple[int, int, int]:
    for _, colour, indices in _LM_GROUPS:
        if idx in indices:
            return colour
    return (200, 200, 200)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bbox-model", type=Path,
        default=Path("artifacts/dog_face_detector/dog_face_localizer_224_float16.tflite"),
    )
    p.add_argument(
        "--landmark-model", type=Path,
        default=Path("artifacts/dog_face_landmarks/dog_face_landmarks_224_float16.tflite"),
    )
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output-image", type=Path, default=None)
    p.add_argument("--label-json", type=Path, default=None,
                   help="DogFLW label JSON for GT overlay and NME computation.")
    p.add_argument("--bbox-img-size", type=int, default=224,
                   help="Input size used by the bbox model.")
    p.add_argument("--lm-img-size", type=int, default=224,
                   help="Input size used by the landmark model.")
    p.add_argument("--lm-margin", type=float, default=0.12,
                   help="Margin used to derive GT bbox from landmarks (for GT overlay).")
    p.add_argument("--crop-margin", type=float, default=0.20,
                   help="Extra margin added around pred bbox before landmark crop.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage 1 helpers  (bbox)
# ---------------------------------------------------------------------------

def letterbox_image(
    image: Image.Image, img_size: int
) -> tuple[np.ndarray, dict[str, float]]:
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


def deletterbox_bbox(
    bbox_norm: np.ndarray, *, orig_w: float, orig_h: float,
    scale: float, pad_x: float, pad_y: float, img_size: float,
) -> np.ndarray:
    b = np.clip(bbox_norm, 0.0, 1.0) * img_size
    b[[0, 2]] -= pad_x
    b[[1, 3]] -= pad_y
    b /= scale
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    return np.array([
        np.clip(x1, 0, orig_w), np.clip(y1, 0, orig_h),
        np.clip(x2, 0, orig_w), np.clip(y2, 0, orig_h),
    ], dtype=np.float32)


def run_bbox_tflite(
    model_path: Path, arr: np.ndarray
) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    inp = np.expand_dims(arr, 0).astype(in_d["dtype"])
    interp.set_tensor(in_d["index"], inp)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])[0].astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    x1, y1, x2, y2 = out
    return np.array([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 2 helpers  (landmarks)
# ---------------------------------------------------------------------------

def crop_for_landmarks(
    image: Image.Image,
    bbox_abs: np.ndarray,
    crop_margin: float,
    lm_img_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Crop image to bbox + margin and resize to (lm_img_size, lm_img_size).

    Returns the float32 [H,W,3] array and crop metadata needed to map
    landmark predictions back to original image coordinates.
    """
    w, h = image.size
    x1, y1, x2, y2 = bbox_abs.tolist()
    bw, bh = x2 - x1, y2 - y1

    cx1 = max(0.0, x1 - bw * crop_margin)
    cy1 = max(0.0, y1 - bh * crop_margin)
    cx2 = min(float(w), x2 + bw * crop_margin)
    cy2 = min(float(h), y2 + bh * crop_margin)

    cropped = image.crop((cx1, cy1, cx2, cy2))
    crop_w, crop_h = cropped.size
    resized = cropped.resize((lm_img_size, lm_img_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0

    meta = {
        "cx1": cx1, "cy1": cy1,
        "crop_w": float(crop_w), "crop_h": float(crop_h),
        "lm_img_size": float(lm_img_size),
    }
    return arr, meta


def run_landmark_tflite(
    model_path: Path, arr: np.ndarray
) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    inp = np.expand_dims(arr, 0).astype(in_d["dtype"])
    interp.set_tensor(in_d["index"], inp)
    interp.invoke()
    out = interp.get_tensor(out_d["index"])[0].astype(np.float32)
    return np.clip(out, 0.0, 1.0)  # [92] normalized


def denormalize_landmarks(
    lm_norm: np.ndarray,
    cx1: float, cy1: float,
    crop_w: float, crop_h: float,
    **_: Any,
) -> np.ndarray:
    """Map normalized [0,1] landmark predictions back to original image coords."""
    pts = lm_norm.reshape(NUM_LANDMARKS, 2).copy()
    pts[:, 0] = pts[:, 0] * crop_w + cx1
    pts[:, 1] = pts[:, 1] * crop_h + cy1
    return pts  # [46, 2]


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------

def load_gt(
    label_json: Path,
    image_size: tuple[int, int],
    lm_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (gt_bbox_abs [4], gt_landmarks_abs [46,2])."""
    d = json.loads(label_json.read_text(encoding="utf-8"))
    lms = d.get("landmarks", [])
    if len(lms) != NUM_LANDMARKS:
        raise ValueError(f"Expected {NUM_LANDMARKS} landmarks, got {len(lms)}")
    w, h = image_size
    pts = np.array([[np.clip(p[0], 0, w), np.clip(p[1], 0, h)] for p in lms],
                   dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = x2 - x1, y2 - y1
    gt_bbox = np.array([
        np.clip(x1 - bw * lm_margin, 0, w),
        np.clip(y1 - bh * lm_margin, 0, h),
        np.clip(x2 + bw * lm_margin, 0, w),
        np.clip(y2 + bh * lm_margin, 0, h),
    ], dtype=np.float32)
    return gt_bbox, pts


def nme(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """NME normalised by the diagonal of the GT landmark bounding box."""
    x1, y1 = gt_pts[:, 0].min(), gt_pts[:, 1].min()
    x2, y2 = gt_pts[:, 0].max(), gt_pts[:, 1].max()
    diag = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) + 1e-8
    dist = np.sqrt(np.sum((pred_pts - gt_pts) ** 2, axis=-1))
    return float(np.mean(dist) / diag)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return float(inter / max(ua, 1e-8))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: np.ndarray,
    colour: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
    draw.text((x1 + 4, max(0.0, y1 - 14)), label, fill=colour)


def draw_landmarks(
    draw: ImageDraw.ImageDraw,
    pts: np.ndarray,
    radius: int = 4,
    draw_edges: bool = True,
    alpha_edges: float = 0.85,
) -> None:
    # Draw edges first (underneath dots)
    if draw_edges:
        for colour, edges in _LM_EDGES:
            for a, b in edges:
                if a < len(pts) and b < len(pts):
                    x1, y1 = float(pts[a][0]), float(pts[a][1])
                    x2, y2 = float(pts[b][0]), float(pts[b][1])
                    draw.line([x1, y1, x2, y2], fill=colour, width=2)
    # Draw dots on top
    for i, (x, y) in enumerate(pts):
        c = _lm_colour(i)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    for p, name in [(args.bbox_model, "bbox model"), (args.landmark_model, "landmark model"),
                    (args.image, "image")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    image = Image.open(args.image).convert("RGB")

    # Stage 1 — bbox
    arr_lb, lb_meta = letterbox_image(image, args.bbox_img_size)
    pred_bbox_norm = run_bbox_tflite(args.bbox_model, arr_lb)
    pred_bbox = deletterbox_bbox(pred_bbox_norm, **lb_meta)

    # Stage 2 — landmarks
    crop_arr, crop_meta = crop_for_landmarks(image, pred_bbox, args.crop_margin, args.lm_img_size)
    lm_norm = run_landmark_tflite(args.landmark_model, crop_arr)
    pred_pts = denormalize_landmarks(lm_norm, **crop_meta)

    result: dict[str, Any] = {
        "image": str(args.image),
        "image_size": {"width": image.width, "height": image.height},
        "prediction": {
            "bbox_xyxy_pixels": pred_bbox.tolist(),
            "landmarks_xy_pixels": pred_pts.tolist(),
        },
    }

    # GT comparison
    gt_bbox, gt_pts = None, None
    if args.label_json is not None:
        gt_bbox, gt_pts = load_gt(args.label_json, image.size, args.lm_margin)
        result["ground_truth"] = {
            "bbox_xyxy_pixels": gt_bbox.tolist(),
            "landmarks_xy_pixels": gt_pts.tolist(),
            "bbox_iou": bbox_iou(pred_bbox, gt_bbox),
            "landmark_nme": nme(pred_pts, gt_pts),
        }

    print(json.dumps(result, indent=2))

    if args.output_image is not None:
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        draw = ImageDraw.Draw(vis)

        # GT (if available)
        if gt_bbox is not None:
            draw_bbox(draw, gt_bbox, (64, 255, 64), "gt")
        if gt_pts is not None:
            draw_landmarks(draw, gt_pts, radius=2, draw_edges=False)

        # Predictions
        draw_bbox(draw, pred_bbox, (255, 64, 64), "pred")
        draw_landmarks(draw, pred_pts, radius=3, draw_edges=False)

        vis.save(args.output_image)
        print(f"Saved: {args.output_image}")


if __name__ == "__main__":
    main()
