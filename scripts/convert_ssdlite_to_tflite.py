#!/usr/bin/env python3
"""Convert SuperAnimal SSDLite detector to TFLite (float16).

The SSDLite model is a torchvision ssdlite320_mobilenet_v3_large with 2 classes
(background + animal). It takes a 320x320 RGB image (ImageNet normalized) and
outputs bounding box regressions and class logits for 6 feature pyramid levels.

Conversion approach:
  - Export the backbone + per-level convolution heads as 4D NCHW tensors (no
    in-model reshape/permute that confuses onnx2tf)
  - Convert ONNX -> TF SavedModel via onnx2tf (outputs become NHWC automatically)
  - Quantize to float16 TFLite

Output TFLite tensors (12 total):
  6 regression tensors:  reg_i  NHWC [1, H_i, W_i, 24]  (6 anchors × 4 coords)
  6 classification tensors: cls_i NHWC [1, H_i, W_i, 12]  (6 anchors × 2 classes)

Feature level sizes (H, W): 20x20, 10x10, 5x5, 3x3, 2x2, 1x1

Post-processing (done in Python / app):
  For each level i, for each spatial position (y, x), for each anchor a:
    reg = reg_tensor[0, y, x, a*4:(a+1)*4]   # [dx, dy, dw, dh]
    cls = cls_tensor[0, y, x, a*2:(a+1)*2]   # [bg_logit, fg_logit]
    score = softmax(cls)[1]                   # foreground probability
    box = decode_ssd_box(reg, prior_box[level,y,x,a])

Usage:
    cd /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml
    source .venv/bin/activate
    python scripts/convert_ssdlite_to_tflite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path("/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml")
CHECKPOINT = ROOT / "checkpoints" / "superanimal_quadruped_ssdlite.pt"
ONNX_PATH = ROOT / "checkpoints" / "ssdlite_4d.onnx"
TF_DIR = ROOT / "checkpoints" / "ssdlite_tf_4d"
TFLITE_OUT = ROOT / "artifacts" / "superanimal_pose" / "superanimal_ssdlite_float16.tflite"

TFLITE_OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Download checkpoint if missing
# ---------------------------------------------------------------------------

def download_checkpoint():
    if CHECKPOINT.exists():
        print(f"Checkpoint already exists: {CHECKPOINT}")
        return
    print("Downloading SSDLite checkpoint from HuggingFace...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
        filename="superanimal_quadruped_ssdlite.pt",
        local_dir=str(ROOT / "checkpoints"),
    )
    print(f"Downloaded: {CHECKPOINT}")


# ---------------------------------------------------------------------------
# Step 2: Build and load the model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """Build SSDLite via DLC and load SuperAnimal weights."""
    print("Building SSDLite model...")
    from deeplabcut.pose_estimation_pytorch.models.detectors import SSDLite

    detector = SSDLite(box_score_thresh=0.01)
    model = detector.model  # raw torchvision ssdlite320_mobilenet_v3_large

    print(f"Loading weights from: {CHECKPOINT}")
    snapshot = torch.load(str(CHECKPOINT), map_location="cpu")
    state_dict = snapshot["model"]

    # Checkpoint keys are prefixed "model." — strip it.
    new_state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(new_state, strict=True)
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}")
    print("  Weights loaded successfully.")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Step 3: Export wrapper
#
# torchvision's SSD head applies:
#   conv -> reshape([B, anchors_per_loc, out_ch, H, W])
#        -> permute([0,3,4,1,2])
#        -> reshape([B, H*W*anchors_per_loc, out_ch])
#
# The intermediate 3D tensors confuse onnx2tf (it transposes NCW→NWC).
#
# Fix: bypass the head entirely.  Output the per-level conv results as plain
# 4D NCHW tensors [1, C*anchors, H, W].  onnx2tf transposes these to NHWC
# [1, H, W, C*anchors], which is exactly what we want for TFLite.
#
# Output order: reg_0, cls_0, reg_1, cls_1, ..., reg_5, cls_5
# Level sizes: 20x20, 10x10, 5x5, 3x3, 2x2, 1x1
# reg channels: 24 (6 anchors × 4 coords)
# cls channels: 12 (6 anchors × 2 classes)
# ---------------------------------------------------------------------------

class SSDLite4DWrapper(nn.Module):
    """Output per-level 4D NCHW tensors for TFLite-friendly onnx2tf conversion."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.backbone = base_model.backbone
        self.reg_heads = base_model.head.regression_head.module_list
        self.cls_heads = base_model.head.classification_head.module_list

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.backbone(x)
        feat_list = list(features.values())
        outputs = []
        for feat, reg_h, cls_h in zip(feat_list, self.reg_heads, self.cls_heads):
            outputs.append(reg_h(feat))   # [1, 24, H, W]
            outputs.append(cls_h(feat))   # [1, 12, H, W]
        return tuple(outputs)


def export_onnx(model: nn.Module) -> bool:
    """Export to ONNX. Returns True on success."""
    print("\n--- ONNX Export ---")
    wrapper = SSDLite4DWrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 3, 320, 320)

    with torch.no_grad():
        outs = wrapper(dummy)
    for i, o in enumerate(outs):
        level = i // 2
        kind = "reg" if i % 2 == 0 else "cls"
        print(f"  {kind}_{level}: {tuple(o.shape)}")

    output_names = []
    for i in range(6):
        output_names.append(f"reg_{i}")
        output_names.append(f"cls_{i}")

    try:
        # Use torch.onnx.utils.export (legacy path) to avoid onnxscript/ml_dtypes
        # incompatibility with torch 2.10.0 + ml_dtypes 0.2.0.
        import torch.onnx.utils as onnx_utils
        onnx_utils.export(
            wrapper,
            (dummy,),
            str(ONNX_PATH),
            opset_version=12,
            input_names=["images"],
            output_names=output_names,
            do_constant_folding=True,
        )
        size_mb = ONNX_PATH.stat().st_size / 1e6
        print(f"\n  Saved ONNX: {ONNX_PATH}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 4: ONNX → TF SavedModel via onnx2tf
# ---------------------------------------------------------------------------

def convert_onnx_to_tf() -> bool:
    """Convert ONNX to TF SavedModel. Returns True on success."""
    print("\n--- ONNX → TF SavedModel ---")
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
            disable_strict_mode=True,
        )
        print(f"  SavedModel written to: {TF_DIR}")
        return True
    except Exception as e:
        print(f"  onnx2tf conversion failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 5: TF SavedModel → float16 TFLite
# ---------------------------------------------------------------------------

def convert_tf_to_tflite() -> bool:
    """Quantize SavedModel to float16 TFLite. Returns True on success."""
    print("\n--- TF SavedModel → float16 TFLite ---")
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
        print(f"  TFLite conversion failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 6: Verify TFLite
# ---------------------------------------------------------------------------

def verify_tflite():
    """Load TFLite, print tensor shapes, run a forward pass."""
    print("\n--- Verify TFLite ---")
    if not TFLITE_OUT.exists():
        print(f"  TFLite not found: {TFLITE_OUT}")
        return

    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(TFLITE_OUT))
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()

    print("  Input:")
    for d in in_details:
        print(f"    [{d['index']}] {d['name']:40s} {list(d['shape'])}  dtype={d['dtype'].__name__}")

    dummy = np.zeros(in_details[0]["shape"], dtype=np.float32)
    interp.set_tensor(in_details[0]["index"], dummy)
    interp.invoke()

    print("  Outputs (sorted by shape for readability):")
    output_info = []
    for d in out_details:
        t = interp.get_tensor(d["index"])
        output_info.append((d["name"], list(d["shape"]), d["dtype"].__name__))
    for name, shape, dtype in sorted(output_info, key=lambda x: (-x[1][1], -x[1][3])):
        print(f"    {name:50s} {shape}  dtype={dtype}")

    size_mb = TFLITE_OUT.stat().st_size / 1e6
    print(f"\n  File: {TFLITE_OUT}  ({size_mb:.2f} MB)")
    print("\n  NOTE: TFLite outputs are NHWC [1, H, W, C] per feature level.")
    print("  Post-processing: flatten to [H*W*6, 4] (reg) and [H*W*6, 2] (cls), decode anchors, apply NMS.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    download_checkpoint()
    model = build_model()

    onnx_ok = export_onnx(model)
    if not onnx_ok:
        print("\nONNX export failed.")
        sys.exit(1)

    tf_ok = convert_onnx_to_tf()
    if not tf_ok:
        print("\nONNX→TF conversion failed.")
        sys.exit(1)

    tflite_ok = convert_tf_to_tflite()
    if not tflite_ok:
        print("\nTFLite conversion failed.")
        sys.exit(1)

    verify_tflite()
    print("\nDone.")


if __name__ == "__main__":
    main()
