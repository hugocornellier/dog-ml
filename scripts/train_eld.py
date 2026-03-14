#!/usr/bin/env python3
"""Train the ELD (Ensemble Landmark Detector) for dog facial landmark detection.

The ELD is a 3-stage cascade with 7 EfficientNetV2S models:
  1. Face detector (already trained) -> face bounding box
  2. Region center detector -> 5 region centers in face-crop space
  3. 5 regional specialists -> per-region landmarks at 224x224 zoom

Usage:
  python scripts/train_eld.py --mode center
  python scripts/train_eld.py --mode specialist --region right_ear
  python scripts/train_eld.py --mode specialist --region all
  python scripts/train_eld.py --mode all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Import shared utilities from main landmark training script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dog_face_landmarks import (
    ExperimentConfig, Record, NUM_LANDMARKS, FLIP_INDEX,
    load_split_records, crop_and_normalize,
    set_seed, configure_ca_bundle,
    photometric_augment,
    SoftArgmax2D, _build_deconv_head, WarmupSchedule,
    SWACallback, get_backbone, export_tflite,
    jitter_crop_box, scale_crop_box,
    rotate_augment, flip_augment, _rotation_matrix,
)

# ---------------------------------------------------------------------------
# Region definitions
# ---------------------------------------------------------------------------

REGION_DEFS = {
    "right_ear":  {"indices": list(range(0, 9)),   "crop_scale": 0.65, "margin": 0.15},
    "left_ear":   {"indices": list(range(9, 18)),  "crop_scale": 0.65, "margin": 0.15},
    "right_eye":  {"indices": list(range(18, 24)), "crop_scale": 0.25, "margin": 0.20},
    "left_eye":   {"indices": list(range(24, 30)), "crop_scale": 0.25, "margin": 0.20},
    "nose_mouth": {"indices": list(range(30, 46)), "crop_scale": 0.50, "margin": 0.15},
}

REGION_ORDER = ["right_ear", "left_ear", "right_eye", "left_eye", "nose_mouth"]

FACE_CROP_SIZE = 320
FACE_CROP_MARGIN = 0.10


def _compute_region_flip_map(region_indices):
    """Compute local flip map for a region. Returns None if cross-region."""
    local_map = []
    for global_idx in region_indices:
        flipped_global = FLIP_INDEX[global_idx]
        if flipped_global not in region_indices:
            return None
        local_map.append(region_indices.index(flipped_global))
    return local_map


REGION_FLIP_MAPS = {}
for _rname, _rdef in REGION_DEFS.items():
    _fm = _compute_region_flip_map(_rdef["indices"])
    if _fm is not None:
        REGION_FLIP_MAPS[_rname] = _fm
# Result: right_eye, left_eye, nose_mouth have clean flip maps.
# right_ear, left_ear have None (landmark 8<->9 crosses regions).


# ---------------------------------------------------------------------------
# Regional augmentation helpers
# ---------------------------------------------------------------------------

def rotate_augment_regional(crop, lm_2d, num_lm, img_size, max_deg):
    """Rotate region crop and landmarks (variable num_lm)."""
    angle_deg = tf.random.uniform((), -max_deg, max_deg)
    angle_rad = angle_deg * (math.pi / 180.0)

    crop = tf.expand_dims(crop, 0)
    crop = tf.raw_ops.ImageProjectiveTransformV3(
        images=crop,
        transforms=_rotation_matrix(angle_rad, img_size),
        output_shape=tf.constant([img_size, img_size], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    crop = tf.squeeze(crop, 0)
    crop = tf.clip_by_value(crop, 0.0, 1.0)

    cos_a = tf.cos(-angle_rad)
    sin_a = tf.sin(-angle_rad)
    cx = lm_2d[:, 0] - 0.5
    cy = lm_2d[:, 1] - 0.5
    rx = cx * cos_a - cy * sin_a + 0.5
    ry = cx * sin_a + cy * cos_a + 0.5
    lm_rot = tf.clip_by_value(tf.stack([rx, ry], axis=-1), 0.0, 1.0)
    return crop, tf.reshape(lm_rot, [num_lm * 2])


def flip_augment_regional(crop, lm_flat, flip_map_tf, num_lm):
    """Horizontal flip with region-local landmark index swapping."""
    do_flip = tf.random.uniform(()) < 0.5

    flipped_img = tf.image.flip_left_right(crop)
    lm = tf.reshape(lm_flat, [num_lm, 2])
    lm_flipped = tf.gather(lm, flip_map_tf, axis=0)
    lm_flipped = tf.stack([1.0 - lm_flipped[:, 0], lm_flipped[:, 1]], axis=-1)
    lm_flipped_flat = tf.reshape(lm_flipped, [num_lm * 2])

    crop_out = tf.where(do_flip, flipped_img, crop)
    lm_out = tf.where(do_flip, lm_flipped_flat, lm_flat)
    return crop_out, lm_out


# ---------------------------------------------------------------------------
# Region center computation
# ---------------------------------------------------------------------------

def compute_region_centers(lm_face_2d):
    """Compute 5 region centers from face-crop-normalized landmarks [46,2].

    Returns [10] tensor: [cx0,cy0, cx1,cy1, ...] for REGION_ORDER.
    """
    centers = []
    for rname in REGION_ORDER:
        idx = REGION_DEFS[rname]["indices"]
        region_lm = tf.gather(lm_face_2d, idx)
        centers.append(tf.reduce_mean(region_lm, axis=0))
    return tf.reshape(tf.stack(centers), [10])


# ---------------------------------------------------------------------------
# Dataset: center detector
# ---------------------------------------------------------------------------

def build_center_dataset(records, cfg, training):
    """Face crop (224x224) -> 5 region centers (10 values)."""
    paths = np.array([r.image_path for r in records], dtype=object)
    boxes = np.array([r.bbox_xyxy_abs for r in records], dtype=np.float32)
    lmarks = np.array(
        [[c for pt in r.landmarks_abs for c in pt] for r in records],
        dtype=np.float32,
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, boxes, lmarks))
    if training:
        ds = ds.shuffle(len(records), seed=cfg.seed, reshuffle_each_iteration=True)

    def _process(path, box_abs, lm_flat):
        image = tf.image.convert_image_dtype(
            tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32
        )
        box_aug = box_abs
        if training and cfg.aug_scale:
            box_aug = scale_crop_box(box_aug, image, cfg.aug_scale_range)
        if training and cfg.aug_crop_jitter:
            box_aug = jitter_crop_box(box_aug, image, cfg.aug_crop_jitter_frac)

        face_crop, lm_face = crop_and_normalize(
            image, box_aug, lm_flat, FACE_CROP_SIZE, FACE_CROP_MARGIN,
        )

        if training:
            if cfg.aug_rotation:
                face_crop, lm_face = rotate_augment(
                    face_crop, lm_face, FACE_CROP_SIZE, cfg.aug_rotation_deg,
                )
            if cfg.aug_flip:
                face_crop, lm_face = flip_augment(face_crop, lm_face)
            face_crop = photometric_augment(face_crop, cfg)

        lm_2d = tf.reshape(lm_face, [NUM_LANDMARKS, 2])
        centers = compute_region_centers(lm_2d)
        return face_crop, centers

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Dataset: specialist
# ---------------------------------------------------------------------------

def build_specialist_dataset(records, region_name, cfg, training):
    """Region crop (224x224) -> region landmarks in region-crop [0,1] space."""
    rdef = REGION_DEFS[region_name]
    region_indices = rdef["indices"]
    num_lm = len(region_indices)
    crop_scale = rdef["crop_scale"]
    region_margin = rdef["margin"]
    flip_map = REGION_FLIP_MAPS.get(region_name)

    region_idx_tf = tf.constant(region_indices, dtype=tf.int32)
    flip_map_tf = tf.constant(flip_map, dtype=tf.int32) if flip_map else None

    paths = np.array([r.image_path for r in records], dtype=object)
    boxes = np.array([r.bbox_xyxy_abs for r in records], dtype=np.float32)
    lmarks = np.array(
        [[c for pt in r.landmarks_abs for c in pt] for r in records],
        dtype=np.float32,
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, boxes, lmarks))
    if training:
        ds = ds.shuffle(len(records), seed=cfg.seed, reshuffle_each_iteration=True)

    def _process(path, box_abs, lm_flat):
        image = tf.image.convert_image_dtype(
            tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32
        )
        box_aug = box_abs
        if training and cfg.aug_scale:
            box_aug = scale_crop_box(box_aug, image, cfg.aug_scale_range)
        if training and cfg.aug_crop_jitter:
            box_aug = jitter_crop_box(box_aug, image, cfg.aug_crop_jitter_frac)

        # Level 1: face crop
        face_crop, lm_face = crop_and_normalize(
            image, box_aug, lm_flat, FACE_CROP_SIZE, FACE_CROP_MARGIN,
        )

        # Region center in face-crop [0,1] space
        lm_2d = tf.reshape(lm_face, [NUM_LANDMARKS, 2])
        region_lm = tf.gather(lm_2d, region_idx_tf)
        center = tf.reduce_mean(region_lm, axis=0)

        # Region bbox in face-crop [0,1] space
        half = crop_scale / 2.0 * (1.0 + region_margin)
        rx1 = tf.maximum(0.0, center[0] - half)
        ry1 = tf.maximum(0.0, center[1] - half)
        rx2 = tf.minimum(1.0, center[0] + half)
        ry2 = tf.minimum(1.0, center[1] + half)

        # Level 2: region crop from face crop
        fh = tf.cast(tf.shape(face_crop)[0], tf.float32)
        fw = tf.cast(tf.shape(face_crop)[1], tf.float32)
        rx1_px = tf.cast(tf.math.floor(rx1 * fw), tf.int32)
        ry1_px = tf.cast(tf.math.floor(ry1 * fh), tf.int32)
        rx2_px = tf.minimum(tf.cast(tf.math.ceil(rx2 * fw), tf.int32), tf.shape(face_crop)[1])
        ry2_px = tf.minimum(tf.cast(tf.math.ceil(ry2 * fh), tf.int32), tf.shape(face_crop)[0])
        rw_px = tf.maximum(rx2_px - rx1_px, 1)
        rh_px = tf.maximum(ry2_px - ry1_px, 1)

        region_crop = tf.image.crop_to_bounding_box(face_crop, ry1_px, rx1_px, rh_px, rw_px)
        region_crop = tf.image.resize(region_crop, [FACE_CROP_SIZE, FACE_CROP_SIZE], antialias=True)
        region_crop = tf.clip_by_value(region_crop, 0.0, 1.0)

        # Normalize region landmarks to region-crop [0,1]
        rx1_f = tf.cast(rx1_px, tf.float32) / fw
        ry1_f = tf.cast(ry1_px, tf.float32) / fh
        rw_f = tf.cast(rw_px, tf.float32) / fw
        rh_f = tf.cast(rh_px, tf.float32) / fh
        lm_rx = (region_lm[:, 0] - rx1_f) / tf.maximum(rw_f, 1e-8)
        lm_ry = (region_lm[:, 1] - ry1_f) / tf.maximum(rh_f, 1e-8)
        lm_region = tf.clip_by_value(tf.stack([lm_rx, lm_ry], axis=-1), 0.0, 1.0)

        # Augmentation on region crop
        if training:
            if cfg.aug_rotation:
                region_crop, lm_flat_out = rotate_augment_regional(
                    region_crop, lm_region, num_lm, FACE_CROP_SIZE, cfg.aug_rotation_deg,
                )
                lm_region = tf.reshape(lm_flat_out, [num_lm, 2])
            if flip_map_tf is not None and cfg.aug_flip:
                lm_flat_r = tf.reshape(lm_region, [num_lm * 2])
                region_crop, lm_flat_r = flip_augment_regional(
                    region_crop, lm_flat_r, flip_map_tf, num_lm,
                )
                lm_region = tf.reshape(lm_flat_r, [num_lm, 2])
            region_crop = photometric_augment(region_crop, cfg)

        lm_out = tf.reshape(lm_region, [num_lm * 2])
        return region_crop, lm_out

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_center_detector(face_crop_size=320, num_deconv=3, pretrained=True):
    """EfficientNetV2S + heatmap head -> 5 center coordinates (10 values).

    Uses the same deconv + SoftArgmax2D approach as specialists for precise
    spatial localization, instead of GAP+Dense which destroys spatial info.
    """
    inputs = tf.keras.Input(shape=(face_crop_size, face_crop_size, 3), name="face_crop")
    x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)
    backbone = tf.keras.applications.EfficientNetV2S(
        input_shape=(face_crop_size, face_crop_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    _heatmaps, coords = _build_deconv_head(
        x, 5, 256,  # 5 region centers
        dropout=0.1, num_deconv=num_deconv,
        softargmax_beta=1.0,
    )
    coords = tf.keras.layers.Identity(name="centers")(coords)
    return tf.keras.Model(inputs=inputs, outputs=coords, name="region_center_detector")


def build_specialist(num_landmarks, channels=256, dropout=0.1,
                     num_deconv=3, beta=1.0, pretrained=True,
                     face_crop_size=320):
    """EfficientNetV2S + heatmap head -> num_landmarks*2 coords."""
    inputs = tf.keras.Input(shape=(face_crop_size, face_crop_size, 3), name="region_crop")
    x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)
    backbone = tf.keras.applications.EfficientNetV2S(
        input_shape=(face_crop_size, face_crop_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    _heatmaps, coords = _build_deconv_head(
        x, num_landmarks, channels,
        dropout=dropout, num_deconv=num_deconv,
        softargmax_beta=beta,
    )
    coords = tf.keras.layers.Identity(name="landmarks_xy")(coords)
    return tf.keras.Model(inputs=inputs, outputs=coords,
                          name=f"specialist_{num_landmarks}lm")


# ---------------------------------------------------------------------------
# Metrics (parameterized by num_landmarks)
# ---------------------------------------------------------------------------

def make_nme_metric(num_lm):
    """Create a crop-normalized NME metric for variable num_landmarks."""
    def landmark_nme(y_true, y_pred):
        diff = tf.reshape(y_true - y_pred, [-1, num_lm, 2])
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
        return tf.reduce_mean(dist)
    landmark_nme.__name__ = "landmark_nme"
    return landmark_nme


# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------

def _make_optimizer(lr, cfg):
    if cfg.optimizer == "adamw":
        try:
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=cfg.weight_decay)
        except AttributeError:
            return tf.keras.optimizers.Adam(learning_rate=lr)
    try:
        return tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    except AttributeError:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def compile_specialist_fn(num_lm, cfg):
    """Return a compile function for a specialist with num_lm landmarks."""
    def _compile(model, lr):
        model.compile(
            optimizer=_make_optimizer(lr, cfg),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[make_nme_metric(num_lm)],
        )
    return _compile


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_eld_model(
    model, train_ds, val_ds, out_dir, cfg, num_train,
    compile_fn,
):
    """Train an ELD component (center detector or specialist).

    compile_fn(model, lr) is called to compile the model with the given lr.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = "val_landmark_nme"
    score_key = "landmark_nme"

    if cfg.finetune_epochs == 0:
        # --- single phase ---
        compile_fn(model, cfg.learning_rate)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, mode="min",
                patience=cfg.patience, restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, mode="min",
                factor=0.5, patience=5, min_lr=1e-6, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
        ]
        print(f"\n=== Single-phase training: {cfg.epochs} epochs ===")
        model.fit(train_ds, validation_data=val_ds,
                  epochs=cfg.epochs, callbacks=callbacks, verbose=2)
        best_state = [np.array(w, copy=True) for w in model.get_weights()]
    else:
        # --- two phase ---
        # Phase 1: frozen backbone
        compile_fn(model, cfg.learning_rate)
        p1_cbs = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, mode="min",
                patience=min(cfg.patience, 6), restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, mode="min",
                factor=0.5, patience=3, min_lr=1e-6, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
        ]
        print(f"\n=== Phase 1: frozen backbone, {cfg.epochs} epochs ===")
        model.fit(train_ds, validation_data=val_ds,
                  epochs=cfg.epochs, callbacks=p1_cbs, verbose=2)

        vals = model.evaluate(val_ds, verbose=0)
        if isinstance(vals, (int, float)):
            vals = [vals]
        p1_score = float(dict(zip(model.metrics_names, vals)).get(score_key, float("inf")))
        best_score = p1_score
        best_state = [np.array(w, copy=True) for w in model.get_weights()]
        print(f"Phase 1 best {score_key}={best_score:.6f}")

        # Phase 2: fine-tune backbone tail
        print(f"\n=== Phase 2: fine-tune backbone, {cfg.finetune_epochs} epochs ===")
        backbone = get_backbone(model)
        backbone.trainable = True
        if cfg.finetune_last_layers > 0:
            for layer in backbone.layers[:-cfg.finetune_last_layers]:
                layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        steps_per_epoch = math.ceil(num_train / cfg.batch_size)
        ft_lr = WarmupSchedule(cfg.finetune_learning_rate, steps_per_epoch * 5)
        compile_fn(model, ft_lr)

        p2_cbs = [
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor, mode="min",
                patience=cfg.patience, restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=True),
        ]
        model.fit(train_ds, validation_data=val_ds,
                  epochs=cfg.finetune_epochs, callbacks=p2_cbs, verbose=2)

        vals = model.evaluate(val_ds, verbose=0)
        if isinstance(vals, (int, float)):
            vals = [vals]
        p2_score = float(dict(zip(model.metrics_names, vals)).get(score_key, float("inf")))

        if p2_score <= best_score:
            best_state = [np.array(w, copy=True) for w in model.get_weights()]
            print(f"Phase 2 improved: {p2_score:.6f} < {best_score:.6f}")
        else:
            print(f"Keeping Phase 1: {best_score:.6f} < {p2_score:.6f}")

    model.set_weights(best_state)
    model.save(out_dir / "best.keras")
    model.save_weights(out_dir / "best.weights.h5")
    print(f"Saved best model to {out_dir}")
    return model


# ---------------------------------------------------------------------------
# Default training config for ELD components
# ---------------------------------------------------------------------------

ELD_BASE_CFG = ExperimentConfig(
    name="eld",
    backbone="efficientnetv2s",
    head_type="heatmap",
    heatmap_dropout=0.1,
    num_deconv_layers=3,
    epochs=100,
    finetune_epochs=200,
    finetune_learning_rate=1e-5,
    finetune_last_layers=50,
    batch_size=8,
    learning_rate=1e-4,
    lr_schedule="constant",
    loss="mse",
    optimizer="adamw",
    weight_decay=1e-4,
    use_swa=False,
    img_size=320,
    lm_margin=0.05,
    crop_margin=0.10,
    aug_rotation=True,
    aug_rotation_deg=15.0,
    aug_flip=True,
    aug_crop_jitter=True,
    aug_crop_jitter_frac=0.08,
    aug_scale=True,
    aug_brightness=True,
    aug_contrast=True,
    aug_saturation=True,
    aug_color_balance=True,
    aug_sharpness=True,
    aug_blur=True,
    aug_noise=True,
    nme_mode="iod",
    patience=50,
)

# Center detector uses a simpler config (fewer epochs, less augmentation).
ELD_CENTER_CFG = ExperimentConfig(
    name="eld_center",
    backbone="efficientnetv2s",
    head_type="heatmap",
    heatmap_dropout=0.1,
    num_deconv_layers=3,
    softargmax_beta=1.0,
    epochs=100,
    finetune_epochs=200,
    finetune_learning_rate=1e-5,
    finetune_last_layers=50,
    batch_size=8,
    learning_rate=1e-4,
    lr_schedule="constant",
    loss="mse",
    optimizer="adamw",
    weight_decay=1e-4,
    use_swa=False,
    img_size=320,
    lm_margin=0.05,
    crop_margin=0.10,
    aug_rotation=True,
    aug_rotation_deg=15.0,
    aug_flip=True,
    aug_crop_jitter=True,
    aug_crop_jitter_frac=0.08,
    aug_scale=True,
    aug_brightness=True,
    aug_contrast=True,
    aug_saturation=True,
    aug_color_balance=True,
    aug_sharpness=True,
    aug_blur=True,
    aug_noise=True,
    nme_mode="iod",
    patience=50,
)


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def train_center_detector(data_root, out_dir, cfg=None):
    """Train the region center detector."""
    cfg = cfg or ELD_CENTER_CFG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recs = load_split_records(data_root, "train", cfg.lm_margin)
    val_recs = load_split_records(data_root, "test", cfg.lm_margin)

    train_ds = build_center_dataset(train_recs, cfg, training=True)
    val_ds = build_center_dataset(val_recs, cfg, training=False)

    model = build_center_detector(
        face_crop_size=FACE_CROP_SIZE,
        num_deconv=cfg.num_deconv_layers,
    )
    model.summary()

    compile_fn = compile_specialist_fn(5, cfg)  # 5 centers treated as 5 landmarks

    model = train_eld_model(
        model, train_ds, val_ds, out_dir, cfg,
        num_train=len(train_recs), compile_fn=compile_fn,
    )

    # Export TFLite
    tflite_path = out_dir / f"center_detector_{FACE_CROP_SIZE}_float16.tflite"
    export_tflite(model, tflite_path)

    # Metadata
    vals = model.evaluate(val_ds, verbose=0)
    if isinstance(vals, (int, float)):
        vals = [vals]
    val_metrics = dict(zip(model.metrics_names, [float(v) for v in vals]))
    meta = {
        "model": "region_center_detector",
        "num_centers": 5,
        "region_order": REGION_ORDER,
        "input_shape": [1, FACE_CROP_SIZE, FACE_CROP_SIZE, 3],
        "output_shape": [1, 10],
        "val_metrics": val_metrics,
        "config": asdict(cfg),
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nCenter detector done. TFLite: {tflite_path}")
    return model


def train_specialist(data_root, region_name, out_dir, cfg=None):
    """Train one regional specialist."""
    cfg = cfg or ELD_BASE_CFG
    rdef = REGION_DEFS[region_name]
    num_lm = len(rdef["indices"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recs = load_split_records(data_root, "train", cfg.lm_margin)
    val_recs = load_split_records(data_root, "test", cfg.lm_margin)

    train_ds = build_specialist_dataset(train_recs, region_name, cfg, training=True)
    val_ds = build_specialist_dataset(val_recs, region_name, cfg, training=False)

    model = build_specialist(
        num_lm,
        channels=cfg.heatmap_channels,
        dropout=cfg.heatmap_dropout,
        num_deconv=cfg.num_deconv_layers,
        beta=cfg.softargmax_beta,
        face_crop_size=FACE_CROP_SIZE,
    )
    model.summary()

    compile_fn = compile_specialist_fn(num_lm, cfg)

    model = train_eld_model(
        model, train_ds, val_ds, out_dir, cfg,
        num_train=len(train_recs), compile_fn=compile_fn,
    )

    # Export TFLite
    tflite_path = out_dir / f"specialist_{region_name}_{FACE_CROP_SIZE}_float16.tflite"
    export_tflite(model, tflite_path)

    # Metadata
    vals = model.evaluate(val_ds, verbose=0)
    if isinstance(vals, (int, float)):
        vals = [vals]
    val_metrics = dict(zip(model.metrics_names, [float(v) for v in vals]))
    meta = {
        "model": f"specialist_{region_name}",
        "region": region_name,
        "num_landmarks": num_lm,
        "landmark_indices": rdef["indices"],
        "crop_scale": rdef["crop_scale"],
        "region_margin": rdef["margin"],
        "input_shape": [1, FACE_CROP_SIZE, FACE_CROP_SIZE, 3],
        "output_shape": [1, num_lm * 2],
        "val_metrics": val_metrics,
        "config": asdict(cfg),
    }
    (out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSpecialist [{region_name}] done. TFLite: {tflite_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    default_root = (
        Path.home() / ".cache" / "kagglehub" / "datasets"
        / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
    )
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=default_root)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/eld"),
                   help="Base output dir. Subdirs created per component.")
    p.add_argument("--mode", choices=["center", "specialist", "all"], required=True)
    p.add_argument("--region", type=str, default=None,
                   help="Region name (or 'all') for specialist mode.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--finetune-learning-rate", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_overrides(cfg, args):
    """Apply CLI overrides to config."""
    overrides = {
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "finetune_learning_rate": args.finetune_learning_rate,
        "patience": args.patience,
        "seed": args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def main():
    args = parse_args()
    configure_ca_bundle()
    set_seed(args.seed)
    tf.config.optimizer.set_jit(False)

    if not args.data_root.exists():
        raise FileNotFoundError(f"DogFLW not found at {args.data_root}")

    print(f"TensorFlow: {tf.__version__}")
    print(f"ELD training mode: {args.mode}")

    base_out = Path(args.out_dir)

    if args.mode in ("center", "all"):
        import copy
        cfg = copy.deepcopy(ELD_CENTER_CFG)
        cfg = apply_overrides(cfg, args)
        print("\n" + "=" * 60)
        print("Training REGION CENTER DETECTOR")
        print("=" * 60)
        train_center_detector(
            args.data_root,
            out_dir=base_out / "center_detector",
            cfg=cfg,
        )

    if args.mode == "specialist":
        if args.region is None:
            raise ValueError("--region required for specialist mode")
        regions = REGION_ORDER if args.region == "all" else [args.region]
        for rname in regions:
            if rname not in REGION_DEFS:
                raise ValueError(f"Unknown region: {rname}. Choose from {list(REGION_DEFS)}")

    if args.mode in ("specialist", "all"):
        import copy
        regions = REGION_ORDER if (args.mode == "all" or args.region == "all") else [args.region]
        for rname in regions:
            cfg = copy.deepcopy(ELD_BASE_CFG)
            cfg = apply_overrides(cfg, args)
            print("\n" + "=" * 60)
            print(f"Training SPECIALIST: {rname} ({len(REGION_DEFS[rname]['indices'])} landmarks)")
            print("=" * 60)
            train_specialist(
                args.data_root,
                region_name=rname,
                out_dir=base_out / "specialists" / rname,
                cfg=cfg,
            )

    print("\n" + "=" * 60)
    print("ELD training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
