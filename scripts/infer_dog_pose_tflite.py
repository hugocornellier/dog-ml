#!/usr/bin/env python3
"""SuperAnimal dog pose inference using TFLite models.

Runs inference with either the HRNet-w32 or RTMPose-S TFLite model, decodes
39-keypoint predictions, and saves a visualization on the original image with
skeleton connections and keypoint labels.

Without --detector: runs pose model on the full image (resized to 256x256).
With --detector:    runs SSDLite TFLite first to find dogs, then runs pose
                    on each detected bounding box crop (letterboxed to 256x256).
                    Also runs species classifier on each crop (optional).
With --face-ensemble (requires --detector):
                    Additionally runs the 3-model face landmark ensemble
                    (46 landmarks) and removes face keypoints 0-14 from pose.

Usage:
    python scripts/infer_dog_pose_tflite.py --model hrnet --image path/to/dog.jpg
    python scripts/infer_dog_pose_tflite.py --model rtmpose --image path/to/dog.jpg --output out.png
    python scripts/infer_dog_pose_tflite.py --model hrnet --image path/to/dog.jpg --detector
    python scripts/infer_dog_pose_tflite.py --model hrnet --image path/to/dog.jpg --detector --no-classify
    python scripts/infer_dog_pose_tflite.py --model hrnet --image path/to/dog.jpg --detector --face-ensemble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle", "tail_base", "tail_end",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai", "back_right_thai",
    "back_left_knee", "back_right_knee", "back_right_paw",
    "belly_bottom", "body_middle_right", "body_middle_left",
]

NUM_KEYPOINTS = 39
INPUT_SIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SIMCC_SPLIT_RATIO = 2.0

# SSD detector constants
DETECTOR_INPUT_SIZE = 320
SSD_WEIGHTS = (10.0, 10.0, 5.0, 5.0)   # wx, wy, ww, wh
SSD_BBOX_CLIP = 4.135166556742356
SSD_LEVEL_SIZES = [(20, 20), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
SSD_ANCHORS_PER_LOC = 6

# Body-only skeleton (keypoints 0-14 removed — those are face/head)
BODY_SKELETON = [
    # Neck/Throat
    (17, 18), # throat_base -> throat_end
    # Spine
    (19, 21), # back_base -> back_middle
    (21, 20), # back_middle -> back_end
    (20, 22), # back_end -> tail_base
    (22, 23), # tail_base -> tail_end
    # Front legs
    (24, 25), (25, 26), (27, 28), (28, 29),
    # Back legs
    (31, 33), (33, 30), (32, 34), (34, 35),
]

# Full skeleton (used when face-ensemble is NOT active)
SKELETON = [
    # Head
    (0, 1),   # nose -> upper_jaw
    (0, 2),   # nose -> lower_jaw
    (0, 5),   # nose -> right_eye
    (0, 10),  # nose -> left_eye
    (5, 6),   # right_eye -> right_earbase
    (6, 7),   # right_earbase -> right_earend
    (10, 11), # left_eye -> left_earbase
    (11, 12), # left_earbase -> left_earend
    # Mouth
    (3, 1),   # mouth_end_right -> upper_jaw
    (4, 1),   # mouth_end_left -> upper_jaw
    (3, 2),   # mouth_end_right -> lower_jaw
    (4, 2),   # mouth_end_left -> lower_jaw
    # Neck/Throat
    (17, 18), # throat_base -> throat_end
    # Spine
    (19, 21), # back_base -> back_middle
    (21, 20), # back_middle -> back_end
    (20, 22), # back_end -> tail_base
    (22, 23), # tail_base -> tail_end
    # Front legs
    (24, 25), # front_left_thai -> front_left_knee
    (25, 26), # front_left_knee -> front_left_paw
    (27, 28), # front_right_thai -> front_right_knee
    (28, 29), # front_right_knee -> front_right_paw
    # Back legs
    (31, 33), # back_left_thai -> back_left_knee
    (33, 30), # back_left_knee -> back_left_paw
    (32, 34), # back_right_thai -> back_right_knee
    (34, 35), # back_right_knee -> back_right_paw
]

# Face landmark skeleton connections
FACE_SKELETON = [
    # Left ear contour (closed loop)
    (8, 6), (6, 4), (4, 2), (2, 0), (0, 12), (12, 10), (10, 8),
    # Right ear contour (closed loop)
    (9, 7), (7, 5), (5, 3), (3, 1), (1, 13), (13, 11), (11, 9),
    # Left eye (quad)
    (18, 20), (20, 16), (16, 22), (22, 18),
    # Right eye (quad)
    (19, 21), (21, 17), (17, 23), (23, 19),
]

# Face landmark model paths
FACE_LOCALIZER_PATH = Path("artifacts/dog_face_detector/dog_face_localizer_224_float16.tflite")
FACE_MODEL_CONFIGS = [
    (Path("artifacts/tight_margin_256/dog_face_landmarks_256_float16.tflite"), 256),
    (Path("artifacts/tight_margin_320/dog_face_landmarks_320_float16.tflite"), 320),
    (Path("artifacts/tight_margin_384/dog_face_landmarks_384_float16.tflite"), 384),
]

# TTA configuration
FACE_SCALES = [0.9, 1.0, 1.1]
FACE_FLIP_INDEX = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    17, 16, 19, 18, 21, 20, 23, 22, 24, 25, 27, 26, 29, 28,
    31, 30, 32, 34, 33, 35, 37, 36, 38, 40, 39, 41, 42, 44,
    43, 45,
]

# Color per keypoint index: (R, G, B)
def _kp_color(idx: int) -> tuple[int, int, int]:
    if idx <= 14:
        return (220, 50, 50)
    if idx <= 18:
        return (255, 140, 0)
    if idx <= 21:
        return (50, 200, 50)
    if idx <= 23:
        return (0, 200, 200)
    if idx <= 29:
        return (50, 100, 220)
    if idx <= 35:
        return (150, 50, 200)
    return (220, 200, 0)


def _skeleton_color(a: int, b: int) -> tuple[int, int, int]:
    return _kp_color(a)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["hrnet", "rtmpose"], required=True,
                   help="Which TFLite model to use.")
    p.add_argument("--image", type=Path, required=True,
                   help="Path to input dog image.")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to save visualization. Defaults to "
                        "artifacts/superanimal_pose/viz_{model}_{imagename}.png")
    p.add_argument("--conf-threshold", type=float, default=None,
                   help="Minimum confidence to draw a keypoint. Defaults to 0.05 for hrnet "
                        "and 1e-5 for rtmpose (RTMPose SimCC softmax confidences are much "
                        "smaller due to 512-bin distribution).")
    p.add_argument("--detector", action="store_true",
                   help="Use SSDLite TFLite detector to find dogs first, then run pose "
                        "on each detected bounding box crop.")
    p.add_argument("--det-threshold", type=float, default=0.5,
                   help="Minimum detection confidence to use a bounding box. Default: 0.5")
    p.add_argument("--no-classify", action="store_true",
                   help="Skip species classification even if the TFLite model exists.")
    p.add_argument("--face-ensemble", action="store_true",
                   help="Run 3-model face landmark ensemble (18 passes) and replace pose "
                        "face keypoints 0-14 with 46 detailed face landmarks. Requires "
                        "--detector. Adds ~165MB model loading + 18 inference passes.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image) -> np.ndarray:
    """Resize to INPUT_SIZE x INPUT_SIZE, ImageNet-normalize, return [1,H,W,3] float32."""
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, 0).astype(np.float32)


def letterbox_resize(
    image: Image.Image, target_size: int = INPUT_SIZE
) -> tuple[Image.Image, float, int, int]:
    """Resize preserving aspect ratio, pad shorter side with black.

    Returns:
        canvas: target_size x target_size PIL image
        scale:  scale factor applied to the original image
        pad_x:  horizontal padding added (pixels)
        pad_y:  vertical padding added (pixels)
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def preprocess_letterbox(image: Image.Image) -> tuple[np.ndarray, float, int, int]:
    """Letterbox-resize to INPUT_SIZE, ImageNet-normalize, return [1,H,W,3] and transform params."""
    canvas, scale, pad_x, pad_y = letterbox_resize(image, INPUT_SIZE)
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, 0).astype(np.float32), scale, pad_x, pad_y


def unletterbox_keypoints(
    keypoints: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    crop_x: int,
    crop_y: int,
) -> np.ndarray:
    """Map keypoints from INPUT_SIZE letterboxed space back to original image.

    Args:
        keypoints: [N, 2] array in letterboxed-crop space (pixels in 0..INPUT_SIZE)
        scale:     letterbox scale factor (crop_size → INPUT_SIZE)
        pad_x:     horizontal padding in letterboxed space
        pad_y:     vertical padding in letterboxed space
        crop_x:    x-offset of the crop in original image
        crop_y:    y-offset of the crop in original image

    Returns:
        keypoints in original image pixel coordinates
    """
    pts = keypoints.copy()
    pts[:, 0] = (pts[:, 0] - pad_x) / scale + crop_x
    pts[:, 1] = (pts[:, 1] - pad_y) / scale + crop_y
    return pts


# ---------------------------------------------------------------------------
# Pose inference
# ---------------------------------------------------------------------------

def run_hrnet(model_path: Path, inp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns keypoints [39, 2] in [0, INPUT_SIZE] space, confidences [39]."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    interp.set_tensor(in_d[0]['index'], inp)
    interp.invoke()
    heatmaps = interp.get_tensor(out_d[0]['index'])  # [1, 64, 64, 39]

    heatmap_size = heatmaps.shape[1]  # 64
    scale = float(INPUT_SIZE) / float(heatmap_size)

    keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    confidences = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    for kp in range(NUM_KEYPOINTS):
        hm = heatmaps[0, :, :, kp]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        keypoints[kp, 1] = idx[0] * scale  # y
        keypoints[kp, 0] = idx[1] * scale  # x
        confidences[kp] = float(hm[idx])
    return keypoints, confidences


def run_rtmpose(model_path: Path, inp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns keypoints [39, 2] in [0, INPUT_SIZE] space, confidences [39]."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    interp.set_tensor(in_d[0]['index'], inp)
    interp.invoke()
    simcc_x = interp.get_tensor(out_d[0]['index'])  # [1, 39, 512]
    simcc_y = interp.get_tensor(out_d[1]['index'])  # [1, 39, 512]

    keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    confidences = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    for kp in range(NUM_KEYPOINTS):
        x_idx = int(np.argmax(simcc_x[0, kp]))
        y_idx = int(np.argmax(simcc_y[0, kp]))
        keypoints[kp, 0] = x_idx / SIMCC_SPLIT_RATIO
        keypoints[kp, 1] = y_idx / SIMCC_SPLIT_RATIO

        x_probs = np.exp(simcc_x[0, kp].astype(np.float64))
        x_probs /= x_probs.sum()
        y_probs = np.exp(simcc_y[0, kp].astype(np.float64))
        y_probs /= y_probs.sum()
        confidences[kp] = float(x_probs[x_idx]) * float(y_probs[y_idx])
    return keypoints, confidences


# ---------------------------------------------------------------------------
# Scale keypoints to original image (full-image mode)
# ---------------------------------------------------------------------------

def scale_keypoints(
    keypoints: np.ndarray, orig_w: int, orig_h: int
) -> np.ndarray:
    """Scale from INPUT_SIZE space to original image dimensions."""
    pts = keypoints.copy()
    pts[:, 0] = pts[:, 0] * orig_w / INPUT_SIZE
    pts[:, 1] = pts[:, 1] * orig_h / INPUT_SIZE
    return pts


# ---------------------------------------------------------------------------
# SSD detector
# ---------------------------------------------------------------------------

def _load_ssd_anchors(anchors_path: Path) -> np.ndarray:
    """Load precomputed SSD anchor boxes [3234, 4] in x1y1x2y2 format (320px space)."""
    return np.load(str(anchors_path))


def _decode_ssd_boxes(
    reg_flat: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """Decode SSD regression deltas to absolute boxes (320px space).

    Args:
        reg_flat: [3234, 4] regression outputs
        anchors:  [3234, 4] prior boxes x1y1x2y2

    Returns:
        [3234, 4] decoded boxes x1y1x2y2
    """
    wx, wy, ww, wh = SSD_WEIGHTS
    ax1, ay1, ax2, ay2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    aw = ax2 - ax1
    ah = ay2 - ay1
    acx = ax1 + 0.5 * aw
    acy = ay1 + 0.5 * ah

    dx = reg_flat[:, 0] / wx
    dy = reg_flat[:, 1] / wy
    dw = np.clip(reg_flat[:, 2] / ww, None, SSD_BBOX_CLIP)
    dh = np.clip(reg_flat[:, 3] / wh, None, SSD_BBOX_CLIP)

    pred_cx = dx * aw + acx
    pred_cy = dy * ah + acy
    pred_w = np.exp(dw) * aw
    pred_h = np.exp(dh) * ah

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h

    return np.stack([x1, y1, x2, y2], axis=1)


def run_detector(
    detector_path: Path,
    anchors_path: Path,
    image: Image.Image,
    det_threshold: float = 0.5,
) -> list[tuple[int, int, int, int, float]]:
    """Run SSDLite TFLite detector on image.

    Returns:
        List of (x1, y1, x2, y2, score) tuples in original image pixel space,
        sorted by score descending.
    """
    from torchvision.ops import nms as torch_nms
    import torch

    orig_w, orig_h = image.size
    anchors = _load_ssd_anchors(anchors_path)

    # Preprocess: resize to 320x320, ImageNet normalize
    img_320 = image.convert("RGB").resize(
        (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE), Image.Resampling.BILINEAR
    )
    arr = np.array(img_320, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    inp = np.expand_dims(arr, 0).astype(np.float32)  # [1, 320, 320, 3]

    # Run TFLite
    interp = tf.lite.Interpreter(model_path=str(detector_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()

    interp.set_tensor(in_d[0]['index'], inp)
    interp.invoke()

    # Collect per-level outputs, group by (H, W)
    by_hw: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for d in out_d:
        t = interp.get_tensor(d['index'])[0]  # [H, W, C]
        hw = (t.shape[0], t.shape[1])
        if hw not in by_hw:
            by_hw[hw] = {}
        if t.shape[2] == 24:
            by_hw[hw]['reg'] = t
        else:
            by_hw[hw]['cls'] = t

    # Flatten to [3234, 4] and [3234, 2]
    all_reg, all_cls = [], []
    for hw in SSD_LEVEL_SIZES:
        level = by_hw.get(hw)
        if level is None:
            raise RuntimeError(f"Missing detector output for level {hw}")
        H, W = hw
        reg = level['reg'].reshape(H, W, SSD_ANCHORS_PER_LOC, 4).reshape(-1, 4)
        cls = level['cls'].reshape(H, W, SSD_ANCHORS_PER_LOC, 2).reshape(-1, 2)
        all_reg.append(reg)
        all_cls.append(cls)

    reg_all = np.concatenate(all_reg, axis=0)  # [3234, 4]
    cls_all = np.concatenate(all_cls, axis=0)  # [3234, 2]

    # Decode boxes (in 320px space)
    boxes_320 = _decode_ssd_boxes(reg_all, anchors)

    # Compute foreground softmax scores
    cls_f = cls_all.astype(np.float64)
    exp_cls = np.exp(cls_f - cls_f.max(axis=1, keepdims=True))
    probs = exp_cls / exp_cls.sum(axis=1, keepdims=True)
    scores = probs[:, 1].astype(np.float32)

    # Scale boxes to original image
    scale_x = orig_w / DETECTOR_INPUT_SIZE
    scale_y = orig_h / DETECTOR_INPUT_SIZE
    boxes_orig = boxes_320.copy()
    boxes_orig[:, [0, 2]] *= scale_x
    boxes_orig[:, [1, 3]] *= scale_y
    # Clamp to image bounds
    boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clip(0, orig_w)
    boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clip(0, orig_h)

    # Filter by threshold
    keep_mask = scores > det_threshold
    if not keep_mask.any():
        return []

    kept_boxes = boxes_orig[keep_mask]
    kept_scores = scores[keep_mask]

    # NMS
    nms_indices = torch_nms(
        torch.from_numpy(kept_boxes).float(),
        torch.from_numpy(kept_scores).float(),
        iou_threshold=0.55,
    ).numpy()

    results = []
    for i in nms_indices:
        box = kept_boxes[i]
        score = float(kept_scores[i])
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        results.append((x1, y1, x2, y2, score))

    results.sort(key=lambda x: -x[4])
    return results


def expand_box(
    x1: int, y1: int, x2: int, y2: int,
    margin: float, img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Expand a bounding box by `margin` fraction on each side, clamped to image bounds."""
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * margin)
    dy = int(bh * margin)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w, x2 + dx),
        min(img_h, y2 + dy),
    )


# ---------------------------------------------------------------------------
# Species classification
# ---------------------------------------------------------------------------

CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_PATH = Path("artifacts/superanimal_pose/species_classifier_float16.tflite")
SPECIES_MAPPING_PATH = Path("artifacts/superanimal_pose/species_mapping.json")


def load_species_mapping(mapping_path: Path) -> dict[int, str]:
    """Load species_mapping.json and return {imagenet_idx: species_name}."""
    with open(mapping_path) as f:
        data = json.load(f)
    lookup: dict[int, str] = {}
    for species, info in data["species"].items():
        for idx in info.get("imagenet_ids", []):
            lookup[idx] = species
    return lookup


def classify_species(
    classifier_path: Path,
    species_lookup: dict[int, str],
    imagenet_categories: list[str],
    crop_image: Image.Image,
) -> tuple[str, str, float]:
    """Run species classifier on a cropped image.

    Args:
        classifier_path:     Path to species_classifier_float16.tflite
        species_lookup:      {imagenet_idx: species_name} from species_mapping.json
        imagenet_categories: List of 1000 ImageNet class name strings
        crop_image:          PIL Image of the cropped animal region

    Returns:
        (species_category, specific_class_name, confidence)
        e.g. ("dog", "golden retriever", 0.87)
    """
    img = crop_image.convert("RGB").resize(
        (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), Image.Resampling.BILINEAR
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    inp = np.expand_dims(arr, 0).astype(np.float32)

    interp = tf.lite.Interpreter(model_path=str(classifier_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    interp.set_tensor(in_d[0]["index"], inp)
    interp.invoke()
    logits = interp.get_tensor(out_d[0]["index"])[0]  # [1000]

    logits_f64 = logits.astype(np.float64)
    exp_l = np.exp(logits_f64 - logits_f64.max())
    probs = exp_l / exp_l.sum()

    top1_idx = int(np.argmax(probs))
    top1_conf = float(probs[top1_idx])
    specific_class = imagenet_categories[top1_idx] if imagenet_categories else str(top1_idx)
    species = species_lookup.get(top1_idx, "unknown_animal")
    return species, specific_class, top1_conf


# ---------------------------------------------------------------------------
# Face landmark pipeline
# ---------------------------------------------------------------------------

def run_face_localizer(localizer_path: Path, image: Image.Image) -> np.ndarray | None:
    """Run face localizer on an image, return bbox [x1, y1, x2, y2] in image pixel coords.

    Args:
        localizer_path: Path to dog_face_localizer_224_float16.tflite
        image:          PIL RGB image (the dog crop from SSDLite)

    Returns:
        [4] float32 array [x1, y1, x2, y2] in original image pixels, or None if invalid.
    """
    w, h = image.size
    scale = 224 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (224, 224), (0, 0, 0))
    pad_x = (224 - new_w) // 2
    pad_y = (224 - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.array(canvas, dtype=np.float32) / 255.0  # [0, 1], NOT ImageNet
    inp = np.expand_dims(arr, 0).astype(np.float32)

    interp = tf.lite.Interpreter(model_path=str(localizer_path))
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    interp.set_tensor(in_d[0]['index'], inp)
    interp.invoke()
    bbox_norm = interp.get_tensor(out_d[0]['index'])[0]  # [4] normalized [0, 1]
    bbox_norm = np.clip(bbox_norm, 0.0, 1.0)

    # De-letterbox: from normalized 224-space back to original image coords
    x1_px = (bbox_norm[0] * 224 - pad_x) / scale
    y1_px = (bbox_norm[1] * 224 - pad_y) / scale
    x2_px = (bbox_norm[2] * 224 - pad_x) / scale
    y2_px = (bbox_norm[3] * 224 - pad_y) / scale

    x1 = float(np.clip(min(x1_px, x2_px), 0, w))
    y1 = float(np.clip(min(y1_px, y2_px), 0, h))
    x2 = float(np.clip(max(x1_px, x2_px), 0, w))
    y2 = float(np.clip(max(y1_px, y2_px), 0, h))

    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return None

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _scale_crop_arr(crop_arr: np.ndarray, scale: float) -> np.ndarray:
    """Scale a [H, W, 3] float32 [0,1] crop array for TTA.

    scale < 1: center-crop (zoom in). scale > 1: pad (zoom out).
    """
    if abs(scale - 1.0) < 1e-6:
        return crop_arr
    h, w = crop_arr.shape[:2]
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        cropped = crop_arr[y0:y0 + new_h, x0:x0 + new_w]
        return np.array(
            Image.fromarray((cropped * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        ) / 255.0
    else:
        pad_h = int(round(h * (scale - 1.0) / 2.0))
        pad_w = int(round(w * (scale - 1.0) / 2.0))
        padded = np.pad(crop_arr, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="reflect")
        return np.array(
            Image.fromarray((padded * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        ) / 255.0


def run_face_ensemble(
    model_configs: list[tuple[Path, int]],
    face_crop: Image.Image,
) -> np.ndarray:
    """Run 3-model face landmark ensemble with multi-scale + flip TTA.

    Args:
        model_configs: List of (model_path, input_size) for each landmark model.
        face_crop:     PIL RGB image of the face region (with margin).

    Returns:
        [46, 2] float32 landmarks normalized [0, 1] relative to face_crop.
    """
    crop_arr = np.array(face_crop.convert("RGB"), dtype=np.float32) / 255.0  # [0, 1]
    all_preds: list[np.ndarray] = []

    for model_path, model_size in model_configs:
        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        in_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]
        in_idx = in_d['index']
        out_idx = out_d['index']

        for scale in FACE_SCALES:
            scaled = _scale_crop_arr(crop_arr, scale)
            # Resize to model input size
            scaled_resized = np.array(
                Image.fromarray((scaled * 255).astype(np.uint8)).resize(
                    (model_size, model_size), Image.Resampling.BILINEAR
                )
            ) / 255.0

            # Normal pass
            inp = np.expand_dims(scaled_resized, 0).astype(np.float32)
            interp.set_tensor(in_idx, inp)
            interp.invoke()
            raw = interp.get_tensor(out_idx)[0]  # [92]
            coords = raw.reshape(46, 2).copy()
            coords = (coords - 0.5) * scale + 0.5  # unscale
            all_preds.append(coords)

            # Flipped pass
            flipped = np.fliplr(scaled_resized).copy()
            inp_f = np.expand_dims(flipped, 0).astype(np.float32)
            interp.set_tensor(in_idx, inp_f)
            interp.invoke()
            raw_f = interp.get_tensor(out_idx)[0]
            coords_f = raw_f.reshape(46, 2).copy()
            coords_f = coords_f[FACE_FLIP_INDEX]  # Swap L/R landmarks
            coords_f[:, 0] = 1.0 - coords_f[:, 0]  # Mirror x
            coords_f = (coords_f - 0.5) * scale + 0.5  # unscale
            all_preds.append(coords_f)

    # Average all 18 predictions (3 models × 3 scales × 2 flips)
    return np.mean(all_preds, axis=0)  # [46, 2] normalized [0, 1]


def get_face_landmarks_in_image(
    localizer_path: Path,
    model_configs: list[tuple[Path, int]],
    dog_crop: Image.Image,
    dog_crop_offset: tuple[int, int],
    crop_margin: float = 0.20,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run the full face landmark pipeline on a dog crop.

    Args:
        localizer_path:   Path to face localizer model.
        model_configs:    List of (path, size) for ensemble landmark models.
        dog_crop:         PIL image of the dog (expanded crop from SSDLite).
        dog_crop_offset:  (crop_x, crop_y) offset of dog_crop in original image.
        crop_margin:      Margin to add around face bbox for landmark crop.

    Returns:
        (face_bbox_in_orig, landmarks_in_orig) where face_bbox is [4] and
        landmarks is [46, 2] in original image pixel coords.
        Returns (None, None) if face is not detected.
    """
    # Stage 1: localize face within dog crop
    face_bbox_in_crop = run_face_localizer(localizer_path, dog_crop)
    if face_bbox_in_crop is None:
        return None, None

    fx1, fy1, fx2, fy2 = face_bbox_in_crop
    bw = fx2 - fx1
    bh = fy2 - fy1

    # Crop face region with margin
    crop_w, crop_h = dog_crop.size
    cx1 = max(0.0, fx1 - bw * crop_margin)
    cy1 = max(0.0, fy1 - bh * crop_margin)
    cx2 = min(float(crop_w), fx2 + bw * crop_margin)
    cy2 = min(float(crop_h), fy2 + bh * crop_margin)

    face_region = dog_crop.crop((cx1, cy1, cx2, cy2))
    actual_crop_w = cx2 - cx1
    actual_crop_h = cy2 - cy1

    # Stage 2: run ensemble
    landmarks_norm = run_face_ensemble(model_configs, face_region)  # [46, 2] in [0, 1]

    # Denormalize to face-crop coords, then to dog-crop coords, then to original image
    dog_crop_x, dog_crop_y = dog_crop_offset
    landmarks_orig = landmarks_norm.copy()
    landmarks_orig[:, 0] = landmarks_norm[:, 0] * actual_crop_w + cx1 + dog_crop_x
    landmarks_orig[:, 1] = landmarks_norm[:, 1] * actual_crop_h + cy1 + dog_crop_y

    # Face bbox in original image coords
    face_bbox_orig = np.array([
        fx1 + dog_crop_x, fy1 + dog_crop_y,
        fx2 + dog_crop_x, fy2 + dog_crop_y,
    ], dtype=np.float32)

    return face_bbox_orig, landmarks_orig


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_pose(
    image: Image.Image,
    keypoints: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = 0.05,
    boxes: list[tuple[int, int, int, int, float]] | None = None,
    species_labels: list[tuple[str, str, float]] | None = None,
    face_landmarks: np.ndarray | None = None,
    face_bbox: np.ndarray | None = None,
    use_body_only_skeleton: bool = False,
) -> Image.Image:
    """Draw pose, boxes, optional species labels, and optional face landmarks.

    Args:
        face_landmarks:         [46, 2] face landmarks in image coords, or None.
        face_bbox:              [4] face bbox in image coords, or None.
        use_body_only_skeleton: If True, draw BODY_SKELETON and skip kps 0-14.
        species_labels: list of (species_category, specific_class, confidence),
                        one per box in ``boxes``. If None or shorter than boxes,
                        falls back to the generic "dog <score>" label.
    """
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=9)
    except Exception:
        font = ImageFont.load_default()

    # Draw bounding boxes if provided
    if boxes:
        for box_idx, (x1, y1, x2, y2, score) in enumerate(boxes):
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            if species_labels and box_idx < len(species_labels):
                species, specific, conf = species_labels[box_idx]
                label = f"{species} ({specific}, {conf:.2f})"
            else:
                label = f"dog {score:.2f}"
            draw.text((x1 + 3, y1 + 3), label, fill=(0, 255, 0), font=font)

    # Draw face bbox if provided
    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox]
        draw.rectangle([fx1, fy1, fx2, fy2], outline=(220, 50, 50), width=2)

    # Select skeleton
    active_skeleton = BODY_SKELETON if use_body_only_skeleton else SKELETON

    # Draw skeleton connections first (underneath dots)
    for a, b in active_skeleton:
        if confidences[a] < conf_threshold or confidences[b] < conf_threshold:
            continue
        x1, y1 = float(keypoints[a, 0]), float(keypoints[a, 1])
        x2, y2 = float(keypoints[b, 0]), float(keypoints[b, 1])
        color = _skeleton_color(a, b)
        draw.line([x1, y1, x2, y2], fill=color, width=2)

    # Draw keypoint dots and labels on top
    radius = max(4, min(image.width, image.height) // 100)
    kp_start = 15 if use_body_only_skeleton else 0
    for kp in range(kp_start, NUM_KEYPOINTS):
        if confidences[kp] < conf_threshold:
            continue
        x, y = float(keypoints[kp, 0]), float(keypoints[kp, 1])
        color = _kp_color(kp)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 255),
        )
        draw.text((x + radius + 1, y - 5), KEYPOINT_NAMES[kp], fill=color, font=font)

    # Draw face landmarks if provided
    if face_landmarks is not None:
        face_color = (220, 50, 50)
        face_radius = max(2, radius - 2)

        # Draw face skeleton edges
        for a, b in FACE_SKELETON:
            if a < len(face_landmarks) and b < len(face_landmarks):
                x1, y1 = float(face_landmarks[a, 0]), float(face_landmarks[a, 1])
                x2, y2 = float(face_landmarks[b, 0]), float(face_landmarks[b, 1])
                draw.line([x1, y1, x2, y2], fill=face_color, width=1)

        # Draw face landmark dots
        for i in range(len(face_landmarks)):
            x, y = float(face_landmarks[i, 0]), float(face_landmarks[i, 1])
            draw.ellipse(
                [x - face_radius, y - face_radius, x + face_radius, y + face_radius],
                fill=face_color,
                outline=(255, 200, 200),
            )

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.face_ensemble and not args.detector:
        raise ValueError("--face-ensemble requires --detector")

    model_paths = {
        "hrnet": Path("artifacts/superanimal_pose/superanimal_hrnet_w32_float16.tflite"),
        "rtmpose": Path("artifacts/superanimal_pose/superanimal_rtmpose_s_float16.tflite"),
    }
    model_path = model_paths[args.model]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    detector_path = Path("artifacts/superanimal_pose/superanimal_ssdlite_float16.tflite")
    anchors_path = Path("checkpoints/ssdlite_anchors.npy")

    # Validate face ensemble model paths if requested
    if args.face_ensemble:
        if not FACE_LOCALIZER_PATH.exists():
            raise FileNotFoundError(f"Face localizer not found: {FACE_LOCALIZER_PATH}")
        for model_p, _ in FACE_MODEL_CONFIGS:
            if not model_p.exists():
                raise FileNotFoundError(f"Face landmark model not found: {model_p}")
        print(f"Face localizer: {FACE_LOCALIZER_PATH}")
        print(f"Face landmark models: {[str(p) for p, _ in FACE_MODEL_CONFIGS]}")

    # Load species classifier if available and not suppressed
    use_classifier = (
        args.detector
        and not args.no_classify
        and CLASSIFIER_PATH.exists()
        and SPECIES_MAPPING_PATH.exists()
    )
    species_lookup: dict[int, str] = {}
    imagenet_categories: list[str] = []
    if use_classifier:
        species_lookup = load_species_mapping(SPECIES_MAPPING_PATH)
        # Load category names from torchvision if available, else empty list
        try:
            from torchvision.models import MobileNet_V3_Small_Weights
            imagenet_categories = list(
                MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]
            )
        except Exception:
            pass
        print(f"Classifier: {CLASSIFIER_PATH}")
    elif args.detector and not args.no_classify and not CLASSIFIER_PATH.exists():
        print(f"Classifier not found at {CLASSIFIER_PATH} — skipping species classification.")

    if args.conf_threshold is None:
        args.conf_threshold = 1e-5 if args.model == "rtmpose" else 0.05

    if args.output is None:
        stem = args.image.stem
        suffix = "_pipeline" if args.detector else ""
        args.output = Path(f"artifacts/superanimal_pose/viz{suffix}_{args.model}_{stem}.png")

    image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Image: {args.image}  ({orig_w}x{orig_h})")
    print(f"Model: {model_path}")

    # -----------------------------------------------------------------------
    # Pipeline mode: detector → crop → pose (→ species classify) (→ face ensemble)
    # -----------------------------------------------------------------------
    if args.detector:
        if not detector_path.exists():
            raise FileNotFoundError(f"Detector not found: {detector_path}")
        if not anchors_path.exists():
            raise FileNotFoundError(f"Anchors not found: {anchors_path}")

        print(f"Detector: {detector_path}  (threshold={args.det_threshold})")
        detections = run_detector(
            detector_path, anchors_path, image, det_threshold=args.det_threshold
        )
        print(f"Detected {len(detections)} dog(s)")

        if not detections:
            print("No dogs detected — falling back to full-image pose.")
            detections = None

        if detections:
            # Combine all per-dog keypoints into single arrays for visualization
            all_keypoints = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
            all_confidences = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            all_species_labels: list[tuple[str, str, float]] = []
            best_face_landmarks: np.ndarray | None = None
            best_face_bbox: np.ndarray | None = None

            for det_idx, (x1, y1, x2, y2, det_score) in enumerate(detections):
                print(f"\n  Dog {det_idx}: box=({x1},{y1},{x2},{y2})  score={det_score:.3f}")

                # Expand box with 10% margin
                cx1, cy1, cx2, cy2 = expand_box(x1, y1, x2, y2, 0.10, orig_w, orig_h)
                crop = image.crop((cx1, cy1, cx2, cy2))
                print(f"  Crop: ({cx1},{cy1},{cx2},{cy2})  size={crop.size}")

                # Species classification on the (unexpanded) crop
                if use_classifier:
                    raw_crop = image.crop((x1, y1, x2, y2))
                    species, specific, cls_conf = classify_species(
                        CLASSIFIER_PATH, species_lookup, imagenet_categories, raw_crop
                    )
                    all_species_labels.append((species, specific, cls_conf))
                    print(f"  Species: {species} ({specific}, {cls_conf:.4f})")

                # Letterbox-resize crop to 256x256
                inp, scale, pad_x, pad_y = preprocess_letterbox(crop)

                # Run pose model
                if args.model == "hrnet":
                    kps_256, confs = run_hrnet(model_path, inp)
                else:
                    kps_256, confs = run_rtmpose(model_path, inp)

                # Un-letterbox: map from 256x256 back to original image
                kps_orig = unletterbox_keypoints(kps_256, scale, pad_x, pad_y, cx1, cy1)

                # Face landmark ensemble (runs on dog crop, not full image)
                if args.face_ensemble and (det_idx == 0 or confs.mean() > all_confidences.mean()):
                    print(f"  Running face landmark ensemble on dog crop...")
                    face_bbox_orig, face_lm_orig = get_face_landmarks_in_image(
                        FACE_LOCALIZER_PATH,
                        FACE_MODEL_CONFIGS,
                        crop,
                        dog_crop_offset=(cx1, cy1),
                        crop_margin=0.20,
                    )
                    if face_lm_orig is not None:
                        best_face_landmarks = face_lm_orig
                        best_face_bbox = face_bbox_orig
                        print(f"  Face detected: bbox={face_bbox_orig.tolist()}")
                        print(f"  Face landmarks: {len(face_lm_orig)} points")
                    else:
                        print(f"  Face NOT detected in dog crop.")

                # Accumulate (use highest-confidence dog if multiple detected)
                if det_idx == 0 or confs.mean() > all_confidences.mean():
                    all_keypoints = kps_orig
                    all_confidences = confs

                high_conf = (confs >= args.conf_threshold).sum()
                print(f"  Keypoints with conf >= {args.conf_threshold}: {high_conf}/{NUM_KEYPOINTS}")
                for kp in range(NUM_KEYPOINTS):
                    if confs[kp] >= args.conf_threshold:
                        x, y = kps_orig[kp]
                        print(f"    {KEYPOINT_NAMES[kp]:25s}: ({x:6.1f}, {y:6.1f})  conf={confs[kp]:.4f}")

            use_body_only = args.face_ensemble and best_face_landmarks is not None
            if args.face_ensemble:
                body_kps = (all_confidences[15:] >= args.conf_threshold).sum()
                face_kps = len(best_face_landmarks) if best_face_landmarks is not None else 0
                print(f"\n  Combined: {face_kps} face landmarks + {body_kps} body keypoints (15-38)")

            vis = draw_pose(
                image, all_keypoints, all_confidences, args.conf_threshold,
                boxes=detections,
                species_labels=all_species_labels if use_classifier else None,
                face_landmarks=best_face_landmarks,
                face_bbox=best_face_bbox,
                use_body_only_skeleton=use_body_only,
            )
        else:
            # Fallback
            inp = preprocess(image)
            if args.model == "hrnet":
                kps_256, confidences = run_hrnet(model_path, inp)
            else:
                kps_256, confidences = run_rtmpose(model_path, inp)
            kps_orig = scale_keypoints(kps_256, orig_w, orig_h)
            vis = draw_pose(image, kps_orig, confidences, args.conf_threshold)

    # -----------------------------------------------------------------------
    # Full-image mode (no detector)
    # -----------------------------------------------------------------------
    else:
        inp = preprocess(image)

        if args.model == "hrnet":
            keypoints_256, confidences = run_hrnet(model_path, inp)
        else:
            keypoints_256, confidences = run_rtmpose(model_path, inp)

        keypoints_orig = scale_keypoints(keypoints_256, orig_w, orig_h)

        print(f"\nKeypoint predictions (conf >= {args.conf_threshold}):")
        for kp in range(NUM_KEYPOINTS):
            if confidences[kp] >= args.conf_threshold:
                x, y = keypoints_orig[kp]
                print(f"  {KEYPOINT_NAMES[kp]:25s}: ({x:6.1f}, {y:6.1f})  conf={confidences[kp]:.4f}")

        vis = draw_pose(image, keypoints_orig, confidences, args.conf_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vis.save(args.output)
    print(f"\nSaved visualization: {args.output}")


if __name__ == "__main__":
    main()
