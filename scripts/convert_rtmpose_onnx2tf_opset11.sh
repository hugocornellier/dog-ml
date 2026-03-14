#!/bin/zsh
# Run onnx2tf on RTMPose opset11 symbolic ONNX
# Output goes to /tmp/rtmpose_onnx2tf_opset11.log

source /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/.venv/bin/activate
cd /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml

python > /tmp/rtmpose_onnx2tf_opset11.log 2>&1 << 'PYEOF'
import struct, onnx.helper as h
if not hasattr(h, 'float32_to_bfloat16'):
    def _f(v):
        b = struct.pack('f', v)
        return struct.unpack('H', b[2:])[0]
    h.float32_to_bfloat16 = _f

print('Importing onnx2tf...', flush=True)
import onnx2tf
print('onnx2tf imported OK', flush=True)

import os, shutil
out_dir = 'checkpoints/rtmpose_s_tf'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

print('Starting conversion...', flush=True)
onnx2tf.convert(
    input_onnx_file_path='checkpoints/rtmpose_s_opset11_sym.onnx',
    output_folder_path=out_dir,
    batch_size=1,
    non_verbose=True,
)
print('Conversion complete!', flush=True)
print('Output:', os.listdir(out_dir), flush=True)
PYEOF

echo "exit code: $?" >> /tmp/rtmpose_onnx2tf_opset11.log
