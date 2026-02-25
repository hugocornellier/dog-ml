#!/usr/bin/env python3
"""Run dog-face TFLite inference on a single image and optionally draw the box.

This script uses the same letterbox geometry as `train_dog_face_detector.py`:
- input image -> letterbox to square (`img_size`)
- model outputs normalized xyxy in letterboxed coordinates
- output bbox is mapped back to original image pixel coordinates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/dog_face_detector/dog_face_localizer_224_float16.tflite"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, default=None)
    parser.add_argument(
        "--label-json",
        type=Path,
        default=None,
        help="Optional DogFLW label json for GT overlay/IoU.",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=0.12,
        help="Margin used when deriving GT bbox from landmarks (matches training default).",
    )
    parser.add_argument("--img-size", type=int, default=224)
    return parser.parse_args()


def letterbox_image(image: Image.Image, img_size: int) -> tuple[np.ndarray, dict[str, float]]:
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image size")

    scale = min(img_size / w, img_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas).astype(np.float32) / 255.0
    meta = {
        "orig_w": float(w),
        "orig_h": float(h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "img_size": float(img_size),
    }
    return arr, meta


def deletterbox_bbox(
    bbox_norm_xyxy_lb: np.ndarray,
    *,
    orig_w: float,
    orig_h: float,
    scale: float,
    pad_x: float,
    pad_y: float,
    img_size: float,
) -> np.ndarray:
    bbox = np.asarray(bbox_norm_xyxy_lb, dtype=np.float32).copy()
    bbox = np.clip(bbox, 0.0, 1.0)
    x1, y1, x2, y2 = bbox * img_size
    x1 -= pad_x
    x2 -= pad_x
    y1 -= pad_y
    y2 -= pad_y
    x1 /= scale
    x2 /= scale
    y1 /= scale
    y2 /= scale
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = float(np.clip(x1, 0.0, orig_w))
    x2 = float(np.clip(x2, 0.0, orig_w))
    y1 = float(np.clip(y1, 0.0, orig_h))
    y2 = float(np.clip(y2, 0.0, orig_h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(np.float32)
    bx1, by1, bx2, by2 = b.astype(np.float32)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-8)
    return float(inter / union)


def derive_gt_bbox_from_dogflw(label_json: Path, image_size: tuple[int, int], margin: float) -> np.ndarray:
    data = json.loads(label_json.read_text(encoding="utf-8"))
    landmarks = data.get("landmarks", [])
    if not landmarks:
        raise ValueError(f"No landmarks in {label_json}")
    w, h = image_size
    xs = [float(np.clip(pt[0], 0.0, w)) for pt in landmarks]
    ys = [float(np.clip(pt[1], 0.0, h)) for pt in landmarks]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw = x2 - x1
    bh = y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    x1 = float(np.clip(x1, 0.0, w))
    x2 = float(np.clip(x2, 0.0, w))
    y1 = float(np.clip(y1, 0.0, h))
    y2 = float(np.clip(y2, 0.0, h))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    tx = x1 + 4
    ty = max(0.0, y1 - 14)
    draw.text((tx, ty), label, fill=color, font=font)


def run_tflite(model_path: Path, image_arr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    inp = np.expand_dims(image_arr, axis=0)
    if in_det["dtype"] == np.uint8:
        scale, zero = in_det["quantization"]
        if scale == 0:
            raise RuntimeError("Invalid input quantization scale")
        inp = np.clip(np.round(inp / scale + zero), 0, 255).astype(np.uint8)
    else:
        inp = inp.astype(in_det["dtype"])

    interpreter.set_tensor(in_det["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det["index"])[0].astype(np.float32)
    out = np.clip(out, 0.0, 1.0)
    x1, y1, x2, y2 = out.tolist()
    out = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)

    meta = {
        "input_shape": in_det["shape"].tolist(),
        "input_dtype": str(np.dtype(in_det["dtype"])),
        "output_shape": out_det["shape"].tolist(),
        "output_dtype": str(np.dtype(out_det["dtype"])),
        "output_bbox_norm_xyxy_letterbox": out.tolist(),
    }
    return out, meta


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = Image.open(args.image).convert("RGB")
    img_arr_lb, lb_meta = letterbox_image(image, img_size=args.img_size)
    pred_lb_norm, tflite_meta = run_tflite(args.model, img_arr_lb)
    pred_orig = deletterbox_bbox(pred_lb_norm, **lb_meta)

    result: dict[str, Any] = {
        "image": str(args.image),
        "image_size": {"width": image.width, "height": image.height},
        "model": str(args.model),
        "letterbox": lb_meta,
        "prediction": {
            "bbox_xyxy_pixels": [float(x) for x in pred_orig.tolist()],
            "bbox_xyxy_norm_letterbox": [float(x) for x in pred_lb_norm.tolist()],
        },
        "tflite": tflite_meta,
    }

    gt_bbox = None
    if args.label_json is not None:
        gt_bbox = derive_gt_bbox_from_dogflw(args.label_json, image.size, margin=args.bbox_margin)
        result["ground_truth"] = {
            "label_json": str(args.label_json),
            "bbox_xyxy_pixels": [float(x) for x in gt_bbox.tolist()],
            "iou_pred_vs_gt": bbox_iou_xyxy(pred_orig, gt_bbox),
        }

    # Print machine-readable result for shell usage.
    print(json.dumps(result, indent=2))

    if args.output_image is not None:
        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        draw_bbox(draw, pred_orig, (255, 64, 64), "pred")
        if gt_bbox is not None:
            draw_bbox(draw, gt_bbox, (64, 255, 64), "gt")
        vis.save(args.output_image)
        print(f"Saved annotated image: {args.output_image}")


if __name__ == "__main__":
    main()
