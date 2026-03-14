#!/bin/zsh
source /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/.venv/bin/activate
cd /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml

python > /tmp/rtmpose_onnx2tf_replacement.log 2>&1 << 'PYEOF'
import struct, onnx.helper as h
if not hasattr(h, 'float32_to_bfloat16'):
    def _f(v):
        b = struct.pack('f', v)
        return struct.unpack('H', b[2:])[0]
    h.float32_to_bfloat16 = _f

import tensorflow as tf
print(f'TF: {tf.__version__}', flush=True)
import onnx2tf
print(f'onnx2tf: {onnx2tf.__version__}', flush=True)

import os, shutil
out_dir = 'checkpoints/rtmpose_s_tf'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

print('Starting conversion with param replacement...', flush=True)
onnx2tf.convert(
    input_onnx_file_path='checkpoints/rtmpose_s_opset11_sym.onnx',
    output_folder_path=out_dir,
    batch_size=1,
    param_replacement_file='checkpoints/rtmpose_s_param_replacement.json',
    non_verbose=True,
)
print('Conversion complete!', flush=True)
print('Output:', os.listdir(out_dir), flush=True)
PYEOF

echo "exit code: $?" >> /tmp/rtmpose_onnx2tf_replacement.log
