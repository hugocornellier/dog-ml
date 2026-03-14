"""
Convert SuperAnimal RTMPose-S to ONNX and TFLite.

RTMPose uses SimCC (Simulated Coordinate Classification).
Output: bodypart.x [1, 39, 512], bodypart.y [1, 39, 512]
These are raw logits over coordinate bins. To decode:
  softmax(x) -> argmax -> divide by simcc_split_ratio (2.0) -> x pixel coordinate
  softmax(y) -> argmax -> divide by simcc_split_ratio (2.0) -> y pixel coordinate

Post-processing in Dart:
  x_coord = argmax(softmax(simcc_x[kp])) / 2.0
  y_coord = argmax(softmax(simcc_y[kp])) / 2.0
  confidence = max(softmax(simcc_x[kp])) * max(softmax(simcc_y[kp]))
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn

from deeplabcut.pose_estimation_pytorch.modelzoo.utils import load_super_animal_config
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel

CHECKPOINT = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/checkpoints/superanimal_quadruped_rtmpose_s.pt"
ONNX_PATH = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/checkpoints/rtmpose_s.onnx"
TF_SAVED_MODEL_DIR = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/checkpoints/rtmpose_s_tf"
TFLITE_DIR = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/artifacts/superanimal_pose"
TFLITE_PATH = os.path.join(TFLITE_DIR, "superanimal_rtmpose_s_float16.tflite")


# ---------------------------------------------------------------------------
# Wrapper: returns (simcc_x, simcc_y) as a 2-tuple for ONNX export
# ---------------------------------------------------------------------------
class RTMPoseExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        simcc_x = out["bodypart"]["x"]  # [B, 39, 512]
        simcc_y = out["bodypart"]["y"]  # [B, 39, 512]
        return simcc_x, simcc_y


# ---------------------------------------------------------------------------
# Step 1: Load model
# ---------------------------------------------------------------------------
def load_model() -> nn.Module:
    print("Loading RTMPose-S config...")
    config = load_super_animal_config(
        super_animal="superanimal_quadruped",
        model_name="rtmpose_s",
        detector_name="fasterrcnn_resnet50_fpn_v2",
    )

    print("Building model...")
    model = PoseModel.build(config["model"])
    model.eval()

    print(f"Loading weights from {CHECKPOINT}...")
    snapshot = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(snapshot["model"])
    print("Model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Step 2: Inspect output structure
# ---------------------------------------------------------------------------
def inspect_output(model: nn.Module):
    print("\n--- Output structure ---")
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))

    def _print(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                _print(v, prefix + "  ")
            elif isinstance(v, torch.Tensor):
                print(f"{prefix}{k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{prefix}{k}: {type(v)}")

    _print(out)
    print("------------------------\n")
    return out


# ---------------------------------------------------------------------------
# Step 3: Export to ONNX
# ---------------------------------------------------------------------------
def export_onnx(model: nn.Module) -> None:
    wrapper = RTMPoseExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 256, 256)

    # Verify wrapper forward pass
    with torch.no_grad():
        sx, sy = wrapper(dummy)
    print(f"Wrapper output: simcc_x={sx.shape}, simcc_y={sy.shape}")

    print(f"Exporting to ONNX: {ONNX_PATH}")
    for opset in (17, 13, 11):
        try:
            torch.onnx.export(
                wrapper,
                dummy,
                ONNX_PATH,
                opset_version=opset,
                input_names=["input"],
                output_names=["simcc_x", "simcc_y"],
                dynamic_axes={
                    "input": {0: "batch"},
                    "simcc_x": {0: "batch"},
                    "simcc_y": {0: "batch"},
                },
            )
            print(f"ONNX export succeeded with opset {opset}.")
            break
        except Exception as e:
            print(f"opset {opset} failed: {e}")
    else:
        raise RuntimeError("ONNX export failed for all opsets.")


# ---------------------------------------------------------------------------
# Step 4: Validate ONNX with onnxruntime
# ---------------------------------------------------------------------------
def validate_onnx(model: nn.Module) -> None:
    import onnx
    import onnxruntime as ort

    print("Validating ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    dummy = torch.randn(1, 3, 256, 256)

    # PyTorch reference
    wrapper = RTMPoseExportWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        pt_sx, pt_sy = wrapper(dummy)

    # ONNX Runtime
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy.numpy()}
    ort_sx, ort_sy = sess.run(None, ort_inputs)

    diff_x = np.abs(pt_sx.numpy() - ort_sx).max()
    diff_y = np.abs(pt_sy.numpy() - ort_sy).max()
    print(f"Max |PT - ONNX| diff: simcc_x={diff_x:.6f}, simcc_y={diff_y:.6f}")

    assert diff_x < 1e-3, f"simcc_x mismatch too large: {diff_x}"
    assert diff_y < 1e-3, f"simcc_y mismatch too large: {diff_y}"
    print("ONNX validation passed.")
    return float(max(diff_x, diff_y))


# ---------------------------------------------------------------------------
# Step 5: Convert ONNX -> TFLite (via onnx2tf)
# ---------------------------------------------------------------------------
def convert_tflite() -> None:
    # onnx_graphsurgeon 0.5.8 references onnx.helper.float32_to_bfloat16 which
    # was removed in onnx 1.17+. Patch it back before importing onnx2tf.
    import struct
    import onnx.helper as _onnx_helper
    if not hasattr(_onnx_helper, "float32_to_bfloat16"):
        def _float32_to_bfloat16(val: float) -> int:
            bits = struct.pack("f", val)
            return struct.unpack("H", bits[2:])[0]
        _onnx_helper.float32_to_bfloat16 = _float32_to_bfloat16

    import onnx2tf

    print(f"Converting ONNX -> TF SavedModel: {TF_SAVED_MODEL_DIR}")
    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=TF_SAVED_MODEL_DIR,
        non_verbose=True,
    )

    # onnx2tf may already emit a .tflite; check
    tflite_candidates = [
        f for f in os.listdir(TF_SAVED_MODEL_DIR) if f.endswith(".tflite")
    ]
    if tflite_candidates:
        print(f"onnx2tf already produced TFLite files: {tflite_candidates}")

    print("Converting SavedModel -> float16 TFLite...")
    import tensorflow as tf

    os.makedirs(TFLITE_DIR, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {TFLITE_PATH}")


# ---------------------------------------------------------------------------
# Step 6: Verify TFLite
# ---------------------------------------------------------------------------
def verify_tflite() -> None:
    import tensorflow as tf

    print("Verifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input tensors:")
    for d in input_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

    print("Output tensors:")
    for i, d in enumerate(output_details):
        print(f"  Output {i} ({d['name']}): shape={d['shape']}, dtype={d['dtype']}")

    size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
    print(f"File size: {size_mb:.1f} MB")

    # Quick inference sanity check
    input_data = np.random.randn(*input_details[0]["shape"]).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    for i, d in enumerate(output_details):
        out = interpreter.get_tensor(d["index"])
        print(f"  Output {i} value range: [{out.min():.3f}, {out.max():.3f}]")

    print("TFLite verification passed.")
    return size_mb, input_details, output_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RTMPose-S -> ONNX -> TFLite pipeline")
    print("=" * 60)

    model = load_model()
    inspect_output(model)

    # Task #4: ONNX export
    export_onnx(model)
    max_diff = validate_onnx(model)
    print(f"\nTask #4 COMPLETE. Max numerical diff PyTorch vs ONNX: {max_diff:.6f}")

    # Task #6: TFLite conversion
    convert_tflite()
    result = verify_tflite()
    size_mb = result[0] if isinstance(result, tuple) else result
    print(f"\nTask #6 COMPLETE. TFLite file: {TFLITE_PATH} ({size_mb:.1f} MB)")

    print("\nSimCC decoding notes:")
    print("  Input size: 256x256, simcc_split_ratio=2.0")
    print("  Bins: 512 per axis (256 * 2.0)")
    print("  x_pixel = argmax(softmax(simcc_x[kp])) / 2.0")
    print("  y_pixel = argmax(softmax(simcc_y[kp])) / 2.0")
    print("  confidence = max(softmax(simcc_x[kp])) * max(softmax(simcc_y[kp]))")


if __name__ == "__main__":
    main()
