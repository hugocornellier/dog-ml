"""
Convert SuperAnimal quadruped HRNet-w32 to TFLite (float16).

Pipeline:
  1. Load HRNet-w32 from .pt snapshot
  2. Export to ONNX (opset 11)
  3. Validate ONNX with onnxruntime
  4. Convert ONNX -> TFLite (float16) via onnx2tf
  5. Verify TFLite model

Outputs:
  checkpoints/hrnet_w32.onnx
  checkpoints/hrnet_w32_tf/   (SavedModel + onnx2tf TFLite files)
  artifacts/superanimal_pose/superanimal_hrnet_w32_float16.tflite
"""

import os
import glob
import shutil

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Step 1: Load model
# ---------------------------------------------------------------------------
print("=== Step 1: Loading HRNet-w32 model ===")

from deeplabcut.pose_estimation_pytorch.modelzoo.utils import load_super_animal_config
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel

CHECKPOINT = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/checkpoints/superanimal_quadruped_hrnet_w32.pt"
ONNX_PATH = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/checkpoints/hrnet_w32.onnx"
TF_FOLDER = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/checkpoints/hrnet_w32_tf"
ARTIFACT_DIR = "/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/artifacts/superanimal_pose"
TFLITE_OUT = os.path.join(ARTIFACT_DIR, "superanimal_hrnet_w32_float16.tflite")

config = load_super_animal_config(
    super_animal="superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
)
model = PoseModel.build(config["model"])
model.eval()

snapshot = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(snapshot["model"])
print("Weights loaded OK")

# ---------------------------------------------------------------------------
# Step 2: Wrapper + verify output structure
# ---------------------------------------------------------------------------
print("\n=== Step 2: Verify output structure ===")

dummy = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    raw_out = model(dummy)

assert isinstance(raw_out, dict) and "bodypart" in raw_out, f"Unexpected output: {raw_out}"
assert "heatmap" in raw_out["bodypart"], f"Missing heatmap key: {raw_out['bodypart'].keys()}"
heatmap = raw_out["bodypart"]["heatmap"]
print(f"Output shape: {heatmap.shape}  (expected [1, 39, 64, 64])")


class HRNetExportWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        out = self.model(x)
        return out["bodypart"]["heatmap"]


wrapper = HRNetExportWrapper(model)
wrapper.eval()

with torch.no_grad():
    torch_out = wrapper(dummy)
print(f"Wrapper output shape: {torch_out.shape}")

# ---------------------------------------------------------------------------
# Step 3: Export to ONNX
# ---------------------------------------------------------------------------
print("\n=== Step 3: Exporting to ONNX ===")

torch.onnx.export(
    wrapper,
    dummy,
    ONNX_PATH,
    opset_version=11,
    input_names=["input"],
    output_names=["heatmaps"],
    dynamo=False,
)
print(f"ONNX saved to: {ONNX_PATH}")
print(f"ONNX file size: {os.path.getsize(ONNX_PATH) / 1024 / 1024:.1f} MB")

# ---------------------------------------------------------------------------
# Step 4: Validate ONNX with onnxruntime
# ---------------------------------------------------------------------------
print("\n=== Step 4: Validating ONNX ===")

import onnx
import onnxruntime as ort

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model check passed")

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
dummy_np = dummy.numpy()
ort_out = sess.run(None, {"input": dummy_np})[0]
torch_np = torch_out.detach().numpy()
max_diff = np.max(np.abs(ort_out - torch_np))
print(f"Max diff PyTorch vs ONNX: {max_diff:.2e}")
assert max_diff < 1e-3, f"Max diff too large: {max_diff}"
print("ONNX validation passed")

# ---------------------------------------------------------------------------
# Step 5: Convert ONNX -> TFLite via onnx2tf
# ---------------------------------------------------------------------------
print("\n=== Step 5: Converting ONNX -> TFLite via onnx2tf ===")

import onnx2tf

onnx2tf.convert(
    input_onnx_file_path=ONNX_PATH,
    output_folder_path=TF_FOLDER,
    batch_size=1,
    non_verbose=True,
)
print(f"onnx2tf conversion complete. Output folder: {TF_FOLDER}")
print("Contents:", os.listdir(TF_FOLDER))

# ---------------------------------------------------------------------------
# Step 6: Check if onnx2tf produced a float16 TFLite already
# ---------------------------------------------------------------------------
print("\n=== Step 6: Producing float16 TFLite ===")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# onnx2tf may have already generated float16 tflite files
float16_candidates = glob.glob(os.path.join(TF_FOLDER, "*float16*.tflite"))
print(f"onnx2tf float16 candidates: {float16_candidates}")

if float16_candidates:
    shutil.copy(float16_candidates[0], TFLITE_OUT)
    print(f"Copied existing float16 TFLite to: {TFLITE_OUT}")
else:
    # Build float16 TFLite from the SavedModel
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(TF_FOLDER)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(TFLITE_OUT, "wb") as f:
        f.write(tflite_model)
    print(f"float16 TFLite saved to: {TFLITE_OUT}")

# ---------------------------------------------------------------------------
# Step 7: Verify TFLite
# ---------------------------------------------------------------------------
print("\n=== Step 7: Verifying TFLite model ===")

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path=TFLITE_OUT)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input:  shape={input_details[0]['shape']}  dtype={input_details[0]['dtype']}")
print(f"Output: shape={output_details[0]['shape']}  dtype={output_details[0]['dtype']}")
file_mb = os.path.getsize(TFLITE_OUT) / 1024 / 1024
print(f"File size: {file_mb:.1f} MB")

print("\n=== All steps complete ===")
print(f"ONNX:   {ONNX_PATH}")
print(f"TFLite: {TFLITE_OUT}")
