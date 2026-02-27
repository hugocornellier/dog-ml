#!/usr/bin/env python3
"""Train a dog facial landmark regressor on DogFLW and export TFLite.

Two-stage inference pipeline:
  1) Run the face bbox model  (train_dog_face_detector.py)  -> face bbox in original coords
  2) Crop + resize to landmark model input size (--img-size, default 128)
  3) Run this landmark model  -> 46 (x, y) pairs normalized in [0, 1] relative to crop
  4) Map back to original image coordinates

DogFLW provides 46 facial landmarks per image (eyes, nose, mouth contour).
Ground-truth boxes used for cropping at train time are derived from landmarks
with a margin (--crop-margin).  At inference time the predicted bbox from
stage-1 is used with the same margin.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

NUM_LANDMARKS = 46


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Record:
    image_path: str
    bbox_xyxy_abs: tuple[float, float, float, float]   # landmark-derived GT box
    landmarks_abs: tuple                                # ((x0,y0), ..., (x45,y45))
    orig_size_wh: tuple[int, int]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_root = (
        Path.home() / ".cache" / "kagglehub" / "datasets"
        / "georgemartvel" / "dogflw" / "versions" / "1" / "DogFLW"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=default_root)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/dog_face_landmarks"))
    p.add_argument("--img-size", type=int, default=224,
                   help="Square crop size fed to landmark model")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15,
                   help="Frozen-backbone epochs")
    p.add_argument("--finetune-epochs", type=int, default=60)
    p.add_argument("--finetune-last-layers", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    p.add_argument("--lm-margin", type=float, default=0.12,
                   help="Margin applied to landmark span to derive GT bbox")
    p.add_argument("--crop-margin", type=float, default=0.20,
                   help="Extra margin added around GT bbox before cropping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--tflite-only", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_ca_bundle() -> None:
    try:
        import certifi
    except Exception:
        return
    ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _landmark_bbox(
    landmarks: list[list[float]],
    img_w: int,
    img_h: int,
    margin: float,
) -> tuple[float, float, float, float] | None:
    xs, ys = [], []
    for pt in landmarks:
        if len(pt) != 2:
            continue
        xs.append(float(np.clip(pt[0], 0.0, img_w)))
        ys.append(float(np.clip(pt[1], 0.0, img_h)))
    if not xs:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None
    x1 = max(0.0, x1 - bw * margin)
    y1 = max(0.0, y1 - bh * margin)
    x2 = min(float(img_w), x2 + bw * margin)
    y2 = min(float(img_h), y2 + bh * margin)
    if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
        return None
    return (x1, y1, x2, y2)


def load_split_records(data_root: Path, split: str, lm_margin: float) -> list[Record]:
    image_dir = data_root / split / "images"
    label_dir = data_root / split / "labels"
    records: list[Record] = []
    skipped = 0
    for img_path in sorted(image_dir.glob("*.png")):
        lbl_path = label_dir / f"{img_path.stem}.json"
        if not lbl_path.exists():
            skipped += 1
            continue
        ann = json.loads(lbl_path.read_text(encoding="utf-8"))
        landmarks = ann.get("landmarks", [])
        if len(landmarks) != NUM_LANDMARKS:
            skipped += 1
            continue
        # Reject records where any landmark coordinate is NaN (present in 2 DogFLW samples).
        if any(
            not (isinstance(pt[0], (int, float)) and isinstance(pt[1], (int, float))
                 and pt[0] == pt[0] and pt[1] == pt[1])
            for pt in landmarks
        ):
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            skipped += 1
            continue
        bbox = _landmark_bbox(landmarks, img_w, img_h, lm_margin)
        if bbox is None:
            skipped += 1
            continue
        lm_clean = tuple(
            (float(np.clip(pt[0], 0.0, img_w)), float(np.clip(pt[1], 0.0, img_h)))
            for pt in landmarks
        )
        records.append(Record(
            image_path=str(img_path),
            bbox_xyxy_abs=bbox,
            landmarks_abs=lm_clean,
            orig_size_wh=(img_w, img_h),
        ))
    print(f"[{split}] usable={len(records)} skipped={skipped}")
    return records


# ---------------------------------------------------------------------------
# TF dataset pipeline
# ---------------------------------------------------------------------------

def build_tf_dataset(
    records: list[Record],
    img_size: int,
    crop_margin: float,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = np.array([r.image_path for r in records], dtype=object)
    boxes = np.array([r.bbox_xyxy_abs for r in records], dtype=np.float32)
    lmarks = np.array(
        [[coord for pt in r.landmarks_abs for coord in pt] for r in records],
        dtype=np.float32,
    )  # shape [N, 92]

    ds = tf.data.Dataset.from_tensor_slices((paths, boxes, lmarks))
    if training:
        ds = ds.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)

    def _load_and_crop(
        path: tf.Tensor,
        box_abs: tf.Tensor,
        lm_flat: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(path)
        image = tf.io.decode_png(img_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        box_aug = box_abs
        if training:
            box_aug = jitter_crop_box(box_abs, image)
        crop, lm_norm = crop_and_normalize(image, box_aug, lm_flat, img_size, crop_margin)
        if training:
            crop, lm_norm = rotate_augment(crop, lm_norm, img_size)
            crop = photometric_augment(crop)
        return crop, lm_norm

    ds = ds.map(_load_and_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def crop_and_normalize(
    image: tf.Tensor,
    bbox_abs: tf.Tensor,
    lm_flat: tf.Tensor,
    img_size: int,
    crop_margin: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Crop image to bbox + margin, resize to (img_size, img_size).

    Normalizes landmark coordinates relative to the crop window so that
    each value is in [0, 1].  At inference time, call with the predicted bbox.
    """
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)

    x1, y1, x2, y2 = bbox_abs[0], bbox_abs[1], bbox_abs[2], bbox_abs[3]
    bw = x2 - x1
    bh = y2 - y1

    mx = bw * crop_margin
    my = bh * crop_margin
    cx1 = tf.maximum(0.0, x1 - mx)
    cy1 = tf.maximum(0.0, y1 - my)
    cx2 = tf.minimum(img_w, x2 + mx)
    cy2 = tf.minimum(img_h, y2 + my)

    cx1i = tf.cast(tf.math.floor(cx1), tf.int32)
    cy1i = tf.cast(tf.math.floor(cy1), tf.int32)
    cx2i = tf.minimum(tf.cast(tf.math.ceil(cx2), tf.int32), tf.shape(image)[1])
    cy2i = tf.minimum(tf.cast(tf.math.ceil(cy2), tf.int32), tf.shape(image)[0])
    crop_w = tf.maximum(cx2i - cx1i, 1)
    crop_h = tf.maximum(cy2i - cy1i, 1)

    cropped = tf.image.crop_to_bounding_box(image, cy1i, cx1i, crop_h, crop_w)
    resized = tf.image.resize(cropped, [img_size, img_size], antialias=True)
    resized = tf.cast(tf.clip_by_value(resized, 0.0, 1.0), tf.float32)

    # Normalize landmarks relative to crop
    cx1f = tf.cast(cx1i, tf.float32)
    cy1f = tf.cast(cy1i, tf.float32)
    crop_wf = tf.cast(crop_w, tf.float32)
    crop_hf = tf.cast(crop_h, tf.float32)

    lm = tf.reshape(lm_flat, [NUM_LANDMARKS, 2])  # [46, 2]
    lm_x = (lm[:, 0] - cx1f) / crop_wf
    lm_y = (lm[:, 1] - cy1f) / crop_hf
    lm_norm = tf.clip_by_value(tf.stack([lm_x, lm_y], axis=-1), 0.0, 1.0)
    lm_norm_flat = tf.reshape(lm_norm, [NUM_LANDMARKS * 2])

    return resized, lm_norm_flat


def jitter_crop_box(
    bbox_abs: tf.Tensor, image: tf.Tensor, max_frac: float = 0.05
) -> tf.Tensor:
    """Randomly shift the crop box by up to ±max_frac of its width/height.

    Simulates the imperfect bboxes the landmark model will receive from the
    detector at inference time.
    """
    img_h = tf.cast(tf.shape(image)[0], tf.float32)
    img_w = tf.cast(tf.shape(image)[1], tf.float32)
    x1, y1, x2, y2 = bbox_abs[0], bbox_abs[1], bbox_abs[2], bbox_abs[3]
    bw, bh = x2 - x1, y2 - y1
    dx = tf.random.uniform((), -max_frac, max_frac) * bw
    dy = tf.random.uniform((), -max_frac, max_frac) * bh
    x1 = tf.clip_by_value(x1 + dx, 0.0, img_w)
    y1 = tf.clip_by_value(y1 + dy, 0.0, img_h)
    x2 = tf.clip_by_value(x2 + dx, 0.0, img_w)
    y2 = tf.clip_by_value(y2 + dy, 0.0, img_h)
    return tf.stack([x1, y1, x2, y2])


def rotate_augment(
    crop: tf.Tensor, lm_norm_flat: tf.Tensor, img_size: int,
    max_deg: float = 15.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Random rotation of the crop with corresponding landmark transform."""
    angle_deg = tf.random.uniform((), -max_deg, max_deg)
    angle_rad = angle_deg * (math.pi / 180.0)

    # Rotate image about its centre.
    crop = tf.expand_dims(crop, 0)  # [1, H, W, 3]
    # angles is counter-clockwise in radians
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

    # Rotate landmarks (pivot = centre of unit square).
    lm = tf.reshape(lm_norm_flat, [NUM_LANDMARKS, 2])
    cos_a = tf.cos(-angle_rad)
    sin_a = tf.sin(-angle_rad)
    cx = lm[:, 0] - 0.5
    cy = lm[:, 1] - 0.5
    rx = cx * cos_a - cy * sin_a + 0.5
    ry = cx * sin_a + cy * cos_a + 0.5
    lm_rot = tf.clip_by_value(tf.stack([rx, ry], axis=-1), 0.0, 1.0)
    return crop, tf.reshape(lm_rot, [NUM_LANDMARKS * 2])


def _rotation_matrix(angle_rad: tf.Tensor, img_size: int) -> tf.Tensor:
    """Build a [1, 8] projective transform matrix for tf.raw_ops.ImageProjectiveTransformV3."""
    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)
    half = tf.cast(img_size, tf.float32) / 2.0
    # Translate to origin, rotate, translate back.
    tx = half - half * cos_a + half * sin_a
    ty = half - half * sin_a - half * cos_a
    return tf.expand_dims(
        tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0]), 0
    )


def photometric_augment(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    image = tf.image.random_saturation(image, lower=0.80, upper=1.20)
    return tf.clip_by_value(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def wing_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    omega: float = 0.1,
    epsilon: float = 0.02,
) -> tf.Tensor:
    """Wing loss (Feng et al. 2018) adapted for normalized [0,1] coords.

    omega and epsilon are in the same units as the predictions (normalized).
    """
    delta = tf.abs(y_true - y_pred)
    C = omega - omega * tf.math.log(1.0 + omega / epsilon)
    loss = tf.where(
        delta < omega,
        omega * tf.math.log(1.0 + delta / epsilon),
        delta - C,
    )
    return tf.reduce_mean(loss, axis=-1)


def landmark_nme(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean per-landmark Euclidean distance in normalized crop coordinates.

    Equivalent to NME normalized by crop size.  Lower is better.
    Typical good values are < 0.05 (i.e. < 5% of the crop dimension).
    """
    diff = tf.reshape(y_true - y_pred, [-1, NUM_LANDMARKS, 2])
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    return tf.reduce_mean(dist)


# ---------------------------------------------------------------------------
# SWA (Stochastic Weight Averaging)
# ---------------------------------------------------------------------------

class SWACallback(tf.keras.callbacks.Callback):
    """Collect weights from `start_epoch` onward and average them."""

    def __init__(self, start_epoch: int = 0):
        super().__init__()
        self.start_epoch = start_epoch
        self._weight_sum: list[np.ndarray] | None = None
        self._count = 0

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch >= self.start_epoch:
            weights = self.model.get_weights()
            if self._weight_sum is None:
                self._weight_sum = [np.zeros_like(w) for w in weights]
            for s, w in zip(self._weight_sum, weights):
                s += w
            self._count += 1

    def get_averaged_weights(self) -> list[np.ndarray] | None:
        if self._weight_sum is None or self._count == 0:
            return None
        return [s / self._count for s in self._weight_sum]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(img_size: int, pretrained: bool) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="crop")
    # EfficientNetB0 has its own preprocessing (expects [0, 255]).
    x = tf.keras.layers.Rescaling(scale=255.0, offset=0.0, name="to_0_255")(inputs)

    backbone = tf.keras.applications.EfficientNetB2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None if not pretrained else "imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(
        NUM_LANDMARKS * 2, activation="sigmoid", name="landmarks_xy"
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dog_face_landmark_regressor")


def compile_model(model: tf.keras.Model, lr) -> None:
    try:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=wing_loss,
        metrics=[landmark_nme],
        run_eagerly=True,
    )


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
            return layer
    raise RuntimeError("EfficientNet backbone not found")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    out_dir: Path,
    args: argparse.Namespace,
    num_train: int,
) -> tf.keras.Model:
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_landmark_nme",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_landmark_nme",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
    ]

    print("\n=== Phase 1: frozen backbone ===")
    compile_model(model, lr=args.learning_rate)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(args.epochs, 0),
        callbacks=phase1_callbacks,
        verbose=2,
    )
    p1_metrics = evaluate_model(model, val_ds)
    p1_score = float(p1_metrics.get("landmark_nme", float("inf")))
    p1_state = [np.array(w, copy=True) for w in model.get_weights()]

    best_score = p1_score
    best_source = "phase1"
    best_state = p1_state

    if not args.skip_finetune and args.finetune_epochs > 0:
        print("\n=== Phase 2: fine-tune backbone tail ===")
        backbone = get_backbone(model)
        backbone.trainable = True
        if args.finetune_last_layers > 0:
            for layer in backbone.layers[: -args.finetune_last_layers]:
                layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # Cosine decay LR schedule
        steps_per_epoch = math.ceil(num_train / args.batch_size)
        total_steps = steps_per_epoch * args.finetune_epochs
        cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.finetune_learning_rate,
            decay_steps=total_steps,
            alpha=1e-6,
        )
        compile_model(model, lr=cosine_lr)

        swa_cb = SWACallback(start_epoch=args.finetune_epochs // 2)
        phase2_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_landmark_nme",
                mode="min",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                str(out_dir / "train_log.csv"), append=True
            ),
            swa_cb,
        ]
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(args.finetune_epochs, 0),
            callbacks=phase2_callbacks,
            verbose=2,
        )
        p2_metrics = evaluate_model(model, val_ds)
        p2_score = float(p2_metrics.get("landmark_nme", float("inf")))
        p2_weights = [np.array(w, copy=True) for w in model.get_weights()]
        if p2_score <= best_score:
            best_score = p2_score
            best_source = "phase2"
            best_state = p2_weights

        # Try SWA averaged weights
        swa_weights = swa_cb.get_averaged_weights()
        if swa_weights is not None:
            model.set_weights(swa_weights)
            swa_metrics = evaluate_model(model, val_ds)
            swa_score = float(swa_metrics.get("landmark_nme", float("inf")))
            print(f"SWA val_landmark_nme={swa_score:.6f}")
            if swa_score <= best_score:
                best_score = swa_score
                best_source = "swa"
                best_state = [np.array(w, copy=True) for w in swa_weights]

    print(f"Selecting {best_source} model (val_landmark_nme={best_score:.6f})")
    model.set_weights(best_state)
    model.save(out_dir / "best.keras")
    model.save_weights(out_dir / "best.weights.h5")
    return model


def evaluate_model(model: tf.keras.Model, val_ds: tf.data.Dataset) -> dict[str, float]:
    values = model.evaluate(val_ds, verbose=0)
    metrics = {n: float(v) for n, v in zip(model.metrics_names, values)}
    print("Validation metrics:", metrics)
    return metrics


# ---------------------------------------------------------------------------
# TFLite export + sanity check
# ---------------------------------------------------------------------------

def export_tflite(model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    print(f"Saved TFLite: {out_path} ({len(tflite_bytes)/1024/1024:.2f} MB)")


def tflite_sanity_check(
    tflite_path: Path,
    val_records: list[Record],
    img_size: int,
    crop_margin: float,
    num_samples: int = 16,
) -> dict[str, float]:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print("TFLite input:", in_det["shape"], in_det["dtype"],
          "output:", out_det["shape"], out_det["dtype"])

    nmes: list[float] = []
    for rec in val_records[:num_samples]:
        image = tf.io.decode_png(tf.io.read_file(rec.image_path), channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        lm_flat = tf.constant(
            [coord for pt in rec.landmarks_abs for coord in pt], dtype=tf.float32
        )
        crop, lm_norm = crop_and_normalize(
            image,
            tf.constant(rec.bbox_xyxy_abs, tf.float32),
            lm_flat,
            img_size,
            crop_margin,
        )
        inp = tf.expand_dims(crop, 0).numpy().astype(in_det["dtype"])
        interp.set_tensor(in_det["index"], inp)
        interp.invoke()
        pred_flat = interp.get_tensor(out_det["index"])[0].astype(np.float32)
        pred_flat = np.clip(pred_flat, 0.0, 1.0)

        diff = lm_norm.numpy() - pred_flat  # [92]
        diff_2d = diff.reshape(NUM_LANDMARKS, 2)
        nme = float(np.mean(np.sqrt(np.sum(diff_2d**2, axis=-1))))
        nmes.append(nme)

    result = {"tflite_mean_nme_sample": float(np.mean(nmes)) if nmes else math.nan}
    print("TFLite sanity:", result)
    return result


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_metadata(
    out_dir: Path,
    img_size: int,
    crop_margin: float,
    train_records: list[Record],
    val_records: list[Record],
    val_metrics: dict[str, float],
    tflite_path: Path,
    tflite_sanity: dict[str, float],
    args: argparse.Namespace,
) -> None:
    meta = {
        "model_name": "dog_face_landmark_regressor",
        "num_landmarks": NUM_LANDMARKS,
        "data_root": str(args.data_root),
        "dataset": "DogFLW",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "input": {
            "shape": [1, img_size, img_size, 3],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "preprocessing": "crop to GT/pred bbox + margin, resize to square, rescale to [0,255] for EfficientNetB0 in-model",
        },
        "output": {
            "name": "landmarks_xy",
            "format": "flattened [x0,y0, x1,y1, ..., x45,y45] normalized in [0,1] relative to crop",
            "shape": [1, NUM_LANDMARKS * 2],
        },
        "training": {
            "lm_margin": args.lm_margin,
            "crop_margin": crop_margin,
            "img_size": img_size,
            "batch_size": args.batch_size,
            "epochs_frozen": args.epochs,
            "epochs_finetune": 0 if args.skip_finetune else args.finetune_epochs,
            "pretrained_backbone": not args.no_pretrained,
            "seed": args.seed,
        },
        "validation_metrics": val_metrics,
        "tflite_sanity": tflite_sanity,
        "artifacts": {
            "tflite": str(tflite_path),
            "keras_best": str(out_dir / "best.keras"),
            "train_log_csv": str(out_dir / "train_log.csv"),
        },
    }
    (out_dir / "model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    configure_ca_bundle()
    set_seed(args.seed)
    tf.config.optimizer.set_jit(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"DogFLW not found at {args.data_root}")

    train_records = load_split_records(args.data_root, "train", args.lm_margin)
    val_records = load_split_records(args.data_root, "test", args.lm_margin)
    if not train_records or not val_records:
        raise RuntimeError("Empty train or val records.")

    train_ds = build_tf_dataset(
        train_records, img_size=args.img_size, crop_margin=args.crop_margin,
        batch_size=args.batch_size, training=True, seed=args.seed,
    )
    val_ds = build_tf_dataset(
        val_records, img_size=args.img_size, crop_margin=args.crop_margin,
        batch_size=args.batch_size, training=False, seed=args.seed,
    )

    model = build_model(img_size=args.img_size, pretrained=not args.no_pretrained)
    model.summary()

    if not args.tflite_only:
        model = train_model(model, train_ds, val_ds, out_dir=args.out_dir, args=args,
                            num_train=len(train_records))
    else:
        best_w = args.out_dir / "best.weights.h5"
        if not best_w.exists():
            raise FileNotFoundError(f"--tflite-only: {best_w} not found")
        model.load_weights(str(best_w))

    compile_model(model, lr=args.finetune_learning_rate)
    val_metrics = evaluate_model(model, val_ds)

    tflite_path = args.out_dir / f"dog_face_landmarks_{args.img_size}_float16.tflite"
    export_tflite(model, tflite_path)

    tflite_sanity = tflite_sanity_check(
        tflite_path, val_records,
        img_size=args.img_size, crop_margin=args.crop_margin,
    )
    save_metadata(
        out_dir=args.out_dir,
        img_size=args.img_size,
        crop_margin=args.crop_margin,
        train_records=train_records,
        val_records=val_records,
        val_metrics=val_metrics,
        tflite_path=tflite_path,
        tflite_sanity=tflite_sanity,
        args=args,
    )

    print("\nDone.")
    print(f"TFLite: {tflite_path}")
    print(f"Metadata: {args.out_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
