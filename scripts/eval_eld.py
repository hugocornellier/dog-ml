#!/usr/bin/env python3
"""Evaluate the full ELD pipeline on DogFLW test set.

Runs the 3-stage cascade (face bbox -> region centers -> 5 specialists)
and computes NME_IOD with per-region breakdown.

Usage:
  python scripts/eval_eld.py
  python scripts/eval_eld.py --use-gt-bbox      # skip face detector, use GT bbox
  python scripts/eval_eld.py --use-gt-centers    # also skip center detector
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dog_face_landmarks import (
    NUM_LANDMARKS, LEFT_OUTER_EYE_IDX, RIGHT_OUTER_EYE_IDX,
    load_split_records, crop_and_normalize,
    set_seed, configure_ca_bundle,
)
from train_eld import (
    REGION_DEFS, REGION_ORDER,
    FACE_CROP_SIZE, FACE_CROP_MARGIN,
    compute_region_centers,
)


# ---------------------------------------------------------------------------
# TFLite helpers
# ---------------------------------------------------------------------------

def load_tflite(path):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def run_tflite(interp, input_np):
    """Run TFLite model. input_np: [1, H, W, 3] float32."""
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    interp.set_tensor(in_det["index"], input_np.astype(in_det["dtype"]))
    interp.invoke()
    return interp.get_tensor(out_det["index"])[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Region cropping (numpy, matching train_eld.py logic)
# ---------------------------------------------------------------------------

def compute_region_bbox(center_xy, crop_scale, margin):
    """Compute region bbox in face-crop [0,1] space."""
    half = crop_scale / 2.0 * (1.0 + margin)
    cx, cy = center_xy
    rx1 = max(0.0, cx - half)
    ry1 = max(0.0, cy - half)
    rx2 = min(1.0, cx + half)
    ry2 = min(1.0, cy + half)
    return rx1, ry1, rx2, ry2


def crop_region_from_face(face_crop_np, center_xy, crop_scale, margin, target_size=224):
    """Crop a region from the face crop image (numpy).

    Returns (region_crop, region_meta) where region_meta has the transform info.
    """
    h, w = face_crop_np.shape[:2]
    rx1, ry1, rx2, ry2 = compute_region_bbox(center_xy, crop_scale, margin)

    rx1_px = int(np.floor(rx1 * w))
    ry1_px = int(np.floor(ry1 * h))
    rx2_px = min(int(np.ceil(rx2 * w)), w)
    ry2_px = min(int(np.ceil(ry2 * h)), h)
    rw_px = max(rx2_px - rx1_px, 1)
    rh_px = max(ry2_px - ry1_px, 1)

    region = face_crop_np[ry1_px:ry1_px + rh_px, rx1_px:rx1_px + rw_px]
    region_resized = tf.image.resize(region, [target_size, target_size], antialias=True).numpy()
    region_resized = np.clip(region_resized, 0.0, 1.0)

    meta = {
        "rx1_f": rx1_px / w,
        "ry1_f": ry1_px / h,
        "rw_f": rw_px / w,
        "rh_f": rh_px / h,
    }
    return region_resized, meta


def transform_region_to_face(lm_region, meta):
    """Transform region-local [0,1] landmarks back to face-crop [0,1] space."""
    lm = lm_region.reshape(-1, 2)
    lm_face_x = lm[:, 0] * meta["rw_f"] + meta["rx1_f"]
    lm_face_y = lm[:, 1] * meta["rh_f"] + meta["ry1_f"]
    return np.stack([lm_face_x, lm_face_y], axis=-1)


# ---------------------------------------------------------------------------
# NME computation
# ---------------------------------------------------------------------------

REGIONS = {
    "right_ear": list(range(0, 9)),
    "left_ear": list(range(9, 18)),
    "right_eye": list(range(18, 24)),
    "left_eye": list(range(24, 30)),
    "nose_bridge": list(range(30, 34)),
    "nose_nostrils": list(range(34, 42)),
    "mouth": list(range(42, 46)),
}


def compute_nme_iod(pred_2d, gt_2d):
    """Compute NME_IOD for a single sample. pred_2d, gt_2d: [46, 2]."""
    left_eye = gt_2d[LEFT_OUTER_EYE_IDX]
    right_eye = gt_2d[RIGHT_OUTER_EYE_IDX]
    iod = np.linalg.norm(left_eye - right_eye)
    if iod < 1e-8:
        return float("nan")
    dists = np.linalg.norm(pred_2d - gt_2d, axis=-1)
    return float(np.mean(dists) / iod * 100.0)


def compute_per_region_nme(pred_2d, gt_2d):
    """Per-region NME_IOD for a single sample."""
    left_eye = gt_2d[LEFT_OUTER_EYE_IDX]
    right_eye = gt_2d[RIGHT_OUTER_EYE_IDX]
    iod = np.linalg.norm(left_eye - right_eye)
    if iod < 1e-8:
        return {}
    result = {}
    for rname, indices in REGIONS.items():
        dists = np.linalg.norm(pred_2d[indices] - gt_2d[indices], axis=-1)
        result[rname] = float(np.mean(dists) / iod * 100.0)
    return result


# ---------------------------------------------------------------------------
# Full ELD evaluation
# ---------------------------------------------------------------------------

def evaluate_eld(
    val_records,
    center_interp=None,
    specialist_interps=None,
    use_gt_bbox=True,
    use_gt_centers=False,
    bbox_interp=None,
):
    """Run full ELD on validation set and compute NME_IOD.

    Args:
        val_records: list of Record
        center_interp: TFLite interpreter for center detector (or None if use_gt_centers)
        specialist_interps: dict {region_name: TFLite interpreter}
        use_gt_bbox: if True, use GT bbox instead of face detector
        use_gt_centers: if True, use GT centers instead of center detector
        bbox_interp: TFLite interpreter for face detector (only used if not use_gt_bbox)
    """
    all_nmes = []
    all_per_region = {rname: [] for rname in REGIONS}

    for i, rec in enumerate(val_records):
        # Load image
        image = tf.image.convert_image_dtype(
            tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3), tf.float32
        )
        lm_flat_gt = tf.constant(
            [c for pt in rec.landmarks_abs for c in pt], dtype=tf.float32
        )

        # Step 1: Face crop
        if use_gt_bbox:
            bbox = tf.constant(rec.bbox_xyxy_abs, dtype=tf.float32)
        else:
            raise NotImplementedError("Auto bbox not implemented yet. Use --use-gt-bbox.")

        face_crop, lm_face_gt = crop_and_normalize(
            image, bbox, lm_flat_gt, FACE_CROP_SIZE, FACE_CROP_MARGIN,
        )
        face_crop_np = face_crop.numpy()
        lm_face_gt_2d = lm_face_gt.numpy().reshape(NUM_LANDMARKS, 2)

        # Step 2: Region centers
        if use_gt_centers:
            lm_gt_tf = tf.reshape(tf.constant(lm_face_gt_2d, tf.float32), [NUM_LANDMARKS, 2])
            centers_flat = compute_region_centers(lm_gt_tf).numpy()
        else:
            face_input = np.expand_dims(face_crop_np, 0)
            centers_flat = run_tflite(center_interp, face_input)
            centers_flat = np.clip(centers_flat, 0.0, 1.0)

        centers = centers_flat.reshape(5, 2)

        # Step 3: Run specialists
        pred_face_2d = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)

        for j, rname in enumerate(REGION_ORDER):
            rdef = REGION_DEFS[rname]
            center_xy = centers[j]

            region_crop, meta = crop_region_from_face(
                face_crop_np, center_xy, rdef["crop_scale"], rdef["margin"],
                target_size=FACE_CROP_SIZE,
            )
            region_input = np.expand_dims(region_crop, 0)
            region_pred = run_tflite(specialist_interps[rname], region_input)
            region_pred = np.clip(region_pred, 0.0, 1.0)

            lm_face = transform_region_to_face(region_pred, meta)
            for k, global_idx in enumerate(rdef["indices"]):
                pred_face_2d[global_idx] = lm_face[k]

        # Compute NME
        nme = compute_nme_iod(pred_face_2d, lm_face_gt_2d)
        if not np.isnan(nme):
            all_nmes.append(nme)
            per_region = compute_per_region_nme(pred_face_2d, lm_face_gt_2d)
            for rn, val in per_region.items():
                all_per_region[rn].append(val)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(val_records)}] running NME_IOD = {np.mean(all_nmes):.2f}")

    return all_nmes, all_per_region


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
    p.add_argument("--data-root", type=Path, default=default_root)
    p.add_argument("--eld-dir", type=Path, default=Path("artifacts/eld"),
                   help="Base ELD artifact dir.")
    p.add_argument("--use-gt-bbox", action="store_true", default=True,
                   help="Use GT face bbox (default, fair comparison to single-model results).")
    p.add_argument("--use-gt-centers", action="store_true", default=False,
                   help="Use GT region centers (measures specialist quality in isolation).")
    args = p.parse_args()

    configure_ca_bundle()
    set_seed(42)

    if not args.data_root.exists():
        raise FileNotFoundError(f"DogFLW not found at {args.data_root}")

    val_records = load_split_records(args.data_root, "test", lm_margin=0.05)
    print(f"Loaded {len(val_records)} test records.")

    eld_dir = Path(args.eld_dir)

    # Load center detector
    center_interp = None
    if not args.use_gt_centers:
        center_path = eld_dir / "center_detector" / f"center_detector_{FACE_CROP_SIZE}_float16.tflite"
        if not center_path.exists():
            raise FileNotFoundError(f"Center detector not found: {center_path}")
        center_interp = load_tflite(center_path)
        print(f"Loaded center detector: {center_path}")

    # Load specialists
    specialist_interps = {}
    for rname in REGION_ORDER:
        sp_path = eld_dir / "specialists" / rname / f"specialist_{rname}_{FACE_CROP_SIZE}_float16.tflite"
        if not sp_path.exists():
            raise FileNotFoundError(f"Specialist not found: {sp_path}")
        specialist_interps[rname] = load_tflite(sp_path)
        print(f"Loaded specialist [{rname}]: {sp_path}")

    # Run evaluation
    mode_str = "GT bbox"
    if args.use_gt_centers:
        mode_str += " + GT centers"
    else:
        mode_str += " + predicted centers"
    print(f"\nEvaluating ELD ({mode_str})...")

    t0 = time.time()
    all_nmes, all_per_region = evaluate_eld(
        val_records,
        center_interp=center_interp,
        specialist_interps=specialist_interps,
        use_gt_bbox=args.use_gt_bbox,
        use_gt_centers=args.use_gt_centers,
    )
    elapsed = time.time() - t0

    # Print results
    mean_nme = float(np.mean(all_nmes))
    print(f"\n{'=' * 60}")
    print(f"ELD Results ({mode_str})")
    print(f"{'=' * 60}")
    print(f"Overall NME_IOD:  {mean_nme:.2f}")
    print(f"Median NME_IOD:   {float(np.median(all_nmes)):.2f}")
    print(f"Std NME_IOD:      {float(np.std(all_nmes)):.2f}")
    print(f"Samples:          {len(all_nmes)}")
    print(f"Time:             {elapsed:.1f}s ({elapsed/len(all_nmes)*1000:.0f}ms/sample)")

    print(f"\nPer-region NME_IOD:")
    print(f"  {'Region':<16} {'NME_IOD':>8}")
    print(f"  {'-'*16} {'-'*8}")
    for rname in ["right_ear", "left_ear", "right_eye", "left_eye",
                   "nose_bridge", "nose_nostrils", "mouth"]:
        vals = all_per_region[rname]
        if vals:
            print(f"  {rname:<16} {np.mean(vals):>8.2f}")

    # Save results
    results = {
        "mode": mode_str,
        "overall_nme_iod": mean_nme,
        "median_nme_iod": float(np.median(all_nmes)),
        "std_nme_iod": float(np.std(all_nmes)),
        "num_samples": len(all_nmes),
        "per_region": {rn: float(np.mean(v)) for rn, v in all_per_region.items() if v},
        "time_seconds": elapsed,
    }
    out_path = eld_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
