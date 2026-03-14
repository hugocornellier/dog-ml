#!/usr/bin/env python3
"""Convert MobileNetV3-Small (ImageNet pretrained) to TFLite species classifier.

Uses torchvision's MobileNet_V3_Small_Weights.IMAGENET1K_V1 weights.
Exports via ONNX → onnx2tf → TF SavedModel → float16 TFLite.

Falls back to pure Keras path if onnx2tf conversion fails.

Output:
    artifacts/superanimal_pose/species_classifier_float16.tflite
    artifacts/superanimal_pose/species_mapping.json

Usage:
    cd /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml
    source .venv/bin/activate
    python scripts/convert_species_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path("/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml")
ONNX_PATH = ROOT / "checkpoints" / "mobilenetv3_small.onnx"
TF_DIR = ROOT / "checkpoints" / "mobilenetv3_small_tf"
TFLITE_OUT = ROOT / "artifacts" / "superanimal_pose" / "species_classifier_float16.tflite"
MAPPING_OUT = ROOT / "artifacts" / "superanimal_pose" / "species_mapping.json"

TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)

# ImageNet class indices verified from torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1
# categories list (1000 classes, 0-indexed).
#
# Dogs: 151-267 plus a few outliers (275=African hunting dog). The main
# contiguous block is 151-267 (117 classes), all are dog breeds except a
# small number of gaps where non-dog classes appear (159=Rhodesian ridgeback
# area, etc). We verified every index above — all dog breeds fall in 151-267
# plus 275.
#
# Cats (domestic): 281-285 (tabby, tiger cat, Persian, Siamese, Egyptian)
# Foxes: 277-280
# Bears: 294-297
# Rabbits/hares: 330-331, 332=Angora (rabbit)
# Horses: 339=sorrel, 340=zebra (we include sorrel; zebra is separate)
# Cattle/ungulates: 345-349 (ox, water buffalo, bison, ram, bighorn)
# Deer-like: 351-353 (hartebeest, impala, gazelle)

SPECIES_MAPPING = {
    "species": {
        "dog": {
            "imagenet_ids": list(range(151, 268)) + [275],
            "description": "All domestic dog breeds plus African hunting dog"
        },
        "cat": {
            "imagenet_ids": list(range(281, 286)),
            "description": "Domestic cat breeds (tabby, tiger cat, Persian, Siamese, Egyptian)"
        },
        "fox": {
            "imagenet_ids": [277, 278, 279, 280],
            "description": "Wild foxes (red, kit, Arctic, grey)"
        },
        "bear": {
            "imagenet_ids": [294, 295, 296, 297],
            "description": "Bears (brown, American black, polar/ice, sloth)"
        },
        "rabbit": {
            "imagenet_ids": [330, 331, 332],
            "description": "Rabbits and hares (wood rabbit, hare, Angora)"
        },
        "horse": {
            "imagenet_ids": [339],
            "description": "Horse (sorrel)"
        },
        "zebra": {
            "imagenet_ids": [340],
            "description": "Zebra"
        },
        "cow": {
            "imagenet_ids": [345, 346, 347],
            "description": "Cattle (ox, water buffalo, bison)"
        },
        "sheep": {
            "imagenet_ids": [348, 349, 350],
            "description": "Sheep and wild sheep (ram, bighorn, ibex)"
        },
        "deer": {
            "imagenet_ids": [351, 352, 353],
            "description": "Deer-like animals (hartebeest, impala, gazelle)"
        },
        "unknown_animal": {
            "imagenet_ids": [],
            "description": "Detected animal but species not in known set"
        }
    },
    "notes": (
        "ImageNet class indices follow torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1 "
        "ordering (0-999). Verified from weights.meta['categories']."
    )
}

# Build lookup: imagenet_idx -> species name
_SPECIES_LOOKUP: dict[int, str] = {}
for _species, _info in SPECIES_MAPPING["species"].items():
    for _idx in _info.get("imagenet_ids", []):
        _SPECIES_LOOKUP[_idx] = _species


# ---------------------------------------------------------------------------
# Step 1: Load model
# ---------------------------------------------------------------------------

def load_model() -> tuple[torch.nn.Module, MobileNet_V3_Small_Weights]:
    print("Loading MobileNetV3-Small (ImageNet pretrained)...")
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    model.eval()
    print(f"  Categories: {len(weights.meta['categories'])}")
    return model, weights


# ---------------------------------------------------------------------------
# Step 2: Export to ONNX
# ---------------------------------------------------------------------------

def export_onnx(model: torch.nn.Module) -> bool:
    print("\n--- ONNX Export ---")
    dummy = torch.randn(1, 3, 224, 224)
    try:
        import torch.onnx.utils as onnx_utils
        onnx_utils.export(
            model,
            (dummy,),
            str(ONNX_PATH),
            opset_version=11,
            input_names=["input"],
            output_names=["logits"],
            do_constant_folding=True,
        )
        size_mb = ONNX_PATH.stat().st_size / 1e6
        print(f"  Saved ONNX: {ONNX_PATH}  ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 3: ONNX -> TF SavedModel via onnx2tf
# ---------------------------------------------------------------------------

def convert_onnx_to_tf() -> bool:
    print("\n--- ONNX -> TF SavedModel ---")
    if not ONNX_PATH.exists():
        print(f"  ONNX not found: {ONNX_PATH}")
        return False
    import shutil
    if TF_DIR.exists():
        shutil.rmtree(TF_DIR)
    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=str(ONNX_PATH),
            output_folder_path=str(TF_DIR),
            non_verbose=True,
        )
        print(f"  SavedModel written to: {TF_DIR}")
        return True
    except Exception as e:
        print(f"  onnx2tf conversion failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 4a: TF SavedModel -> float16 TFLite
# ---------------------------------------------------------------------------

def convert_tf_to_tflite_from_savedmodel() -> bool:
    print("\n--- TF SavedModel -> float16 TFLite ---")
    if not TF_DIR.exists():
        print(f"  SavedModel not found: {TF_DIR}")
        return False
    try:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_DIR))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        TFLITE_OUT.write_bytes(tflite_model)
        size_mb = len(tflite_model) / 1e6
        print(f"  TFLite written: {TFLITE_OUT}  ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  TFLite from SavedModel failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 4b: Fallback — pure Keras -> TFLite (no ONNX)
# ---------------------------------------------------------------------------

def convert_keras_to_tflite() -> bool:
    """Fallback: build MobileNetV3Small via Keras and quantize to float16."""
    print("\n--- Keras -> float16 TFLite (fallback) ---")
    try:
        import tensorflow as tf

        # Build model from local weights if available, else download
        keras_model = tf.keras.applications.MobileNetV3Small(
            weights="imagenet", include_top=True, input_shape=(224, 224, 3)
        )
        print("  Keras model loaded.")

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        TFLITE_OUT.write_bytes(tflite_model)
        size_mb = len(tflite_model) / 1e6
        print(f"  TFLite written: {TFLITE_OUT}  ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  Keras TFLite conversion failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 5: Write species_mapping.json
# ---------------------------------------------------------------------------

def write_mapping() -> None:
    MAPPING_OUT.write_text(json.dumps(SPECIES_MAPPING, indent=2))
    print(f"\nWrote species mapping: {MAPPING_OUT}")


# ---------------------------------------------------------------------------
# Step 6: Verify TFLite
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DACHSHUND_PATH = Path(
    "/Users/hugocornellier/Downloads/"
    "cream-long-haired-dachshund-outside_Valeria-Head_Shutterstock.jpg"
)


def verify_tflite(test_image_path: Path | None = None) -> None:
    print("\n--- Verify TFLite ---")
    if not TFLITE_OUT.exists():
        print(f"  TFLite not found: {TFLITE_OUT}")
        return

    import tensorflow as tf
    from PIL import Image

    interp = tf.lite.Interpreter(model_path=str(TFLITE_OUT))
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()

    print("  Input:", in_d[0]["shape"], in_d[0]["dtype"].__name__)
    print("  Output:", out_d[0]["shape"], out_d[0]["dtype"].__name__)
    size_mb = TFLITE_OUT.stat().st_size / 1e6
    print(f"  File size: {size_mb:.2f} MB")

    img_path = test_image_path or DACHSHUND_PATH
    if not img_path.exists():
        print(f"  Test image not found: {img_path} — skipping inference check.")
        return

    # Preprocess
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    inp = np.expand_dims(arr, 0).astype(np.float32)

    interp.set_tensor(in_d[0]["index"], inp)
    interp.invoke()
    logits = interp.get_tensor(out_d[0]["index"])[0]  # [1000]

    # Softmax
    logits_f64 = logits.astype(np.float64)
    exp_l = np.exp(logits_f64 - logits_f64.max())
    probs = exp_l / exp_l.sum()

    # Top-5
    top5 = np.argsort(probs)[::-1][:5]
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]

    print(f"\n  Top-5 predictions on: {img_path.name}")
    for rank, idx in enumerate(top5):
        species = _SPECIES_LOOKUP.get(idx, "unknown_animal")
        print(f"    {rank+1}. [{idx:4d}] {categories[idx]:35s}  {probs[idx]:.4f}  -> species: {species}")

    top1_idx = int(top5[0])
    top1_species = _SPECIES_LOOKUP.get(top1_idx, "unknown_animal")
    top1_conf = float(probs[top1_idx])
    print(f"\n  Classification: {top1_species} ({categories[top1_idx]}, {top1_conf:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model, weights = load_model()

    onnx_ok = export_onnx(model)

    tflite_ok = False
    if onnx_ok:
        tf_ok = convert_onnx_to_tf()
        if tf_ok:
            tflite_ok = convert_tf_to_tflite_from_savedmodel()

    if not tflite_ok:
        print("\nONNX path failed or incomplete — trying Keras fallback...")
        tflite_ok = convert_keras_to_tflite()

    if not tflite_ok:
        print("\nAll conversion paths failed.")
        sys.exit(1)

    write_mapping()
    verify_tflite()
    print("\nDone.")


if __name__ == "__main__":
    main()
