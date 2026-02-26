#!/usr/bin/env python3
"""Generate separate GT and predicted landmark visualizations for test images.

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
    run_landmark_tflite,
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
LM_MODEL = Path("artifacts/dog_face_landmarks/dog_face_landmarks_128_float16.tflite")
OUT_DIR = Path("artifacts/dog_face_landmarks/inference_examples")

BBOX_IMG_SIZE = 224
LM_IMG_SIZE = 128
LM_MARGIN = 0.12
CROP_MARGIN = 0.20

# Same images used in the existing examples.
STEMS = [
    "n02085620_1073", "n02086240_3921", "n02086646_45",
    "n02088364_4281", "n02089867_2382", "n02092339_2752",
    "n02097474_7545", "n02098286_4250", "n02099267_3900",
    "n02099849_4287", "n02100583_3713", "n02102177_3917",
    "n02102480_4743", "n02105505_2570", "n02110185_712",
    "n02111500_7983", "n02112706_1995", "n02113712_459",
]


def make_true_image(image: Image.Image, label_path: Path) -> Image.Image:
    """Draw GT bbox (green) and GT landmarks on a copy of the image."""
    gt_bbox, gt_pts = load_gt(label_path, image.size, LM_MARGIN)
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw_bbox(draw, gt_bbox, (64, 255, 64), "gt")
    draw_landmarks(draw, gt_pts, radius=3, draw_edges=False)
    return vis


def make_pred_image(image: Image.Image) -> Image.Image:
    """Run the two-stage TFLite pipeline and draw predicted bbox + landmarks."""
    arr_lb, lb_meta = letterbox_image(image, BBOX_IMG_SIZE)
    pred_bbox_norm = run_bbox_tflite(BBOX_MODEL, arr_lb)
    pred_bbox = deletterbox_bbox(pred_bbox_norm, **lb_meta)

    crop_arr, crop_meta = crop_for_landmarks(image, pred_bbox, CROP_MARGIN, LM_IMG_SIZE)
    lm_norm = run_landmark_tflite(LM_MODEL, crop_arr)
    pred_pts = denormalize_landmarks(lm_norm, **crop_meta)

    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw_bbox(draw, pred_bbox, (255, 64, 64), "pred")
    draw_landmarks(draw, pred_pts, radius=3, draw_edges=False)
    return vis


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stem in STEMS:
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

        print(f"OK  {stem}")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
