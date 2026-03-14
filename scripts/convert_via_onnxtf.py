"""
Convert RTMPose static ONNX to TFLite via onnx-tf backend.
"""
import os
import sys

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
LOG_FILE = os.path.join(BASE, 'checkpoints/onnxtf_rtmpose_log.txt')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            try:
                f.write(s)
                f.flush()
            except Exception:
                pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except Exception: pass

log = open(LOG_FILE, 'w')
sys.stdout = Tee(sys.__stdout__, log)
sys.stderr = Tee(sys.__stderr__, log)

import traceback
import numpy as np

try:
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    import onnxruntime as ort

    print('Loading static ONNX...')
    onnx_model = onnx.load(os.path.join(BASE, 'checkpoints/rtmpose_s_static.onnx'))
    print(f'opset: {onnx_model.opset_import[0].version}')

    print('Converting to TF...')
    tf_rep = prepare(onnx_model)

    saved_model_dir = os.path.join(BASE, 'checkpoints/rtmpose_s_tf_onnxtf')
    print(f'Exporting SavedModel to {saved_model_dir}...')
    tf_rep.export_graph(saved_model_dir)
    print('SavedModel exported OK')

    # Convert to float16 TFLite
    tflite_dir = os.path.join(BASE, 'artifacts/superanimal_pose')
    os.makedirs(tflite_dir, exist_ok=True)
    tflite_path = os.path.join(tflite_dir, 'superanimal_rtmpose_s_float16.tflite')

    print('Converting SavedModel -> float16 TFLite...')
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f'TFLite saved: {tflite_path} ({size_mb:.1f} MB)')

    # Verify
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('Input tensors:')
    for d in input_details:
        print(f"  {d['name']}: shape={d['shape']}, dtype={d['dtype']}")

    print('Output tensors:')
    for i, d in enumerate(output_details):
        print(f"  Output {i} ({d['name']}): shape={d['shape']}, dtype={d['dtype']}")

    print('CONVERSION_SUCCESS')

except Exception as e:
    print(f'CONVERSION_FAILED: {e}')
    traceback.print_exc()

log.close()
