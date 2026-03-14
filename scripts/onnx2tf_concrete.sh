#!/bin/zsh
source /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/.venv/bin/activate
cd /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml

python -c "
import struct, onnx.helper as h
if not hasattr(h, 'float32_to_bfloat16'):
    def _f(v):
        b = struct.pack('f', v)
        return struct.unpack('H', b[2:])[0]
    h.float32_to_bfloat16 = _f
import onnx2tf
import os, shutil

out_dir = 'checkpoints/rtmpose_s_tf'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

print('Starting onnx2tf conversion...', flush=True)
onnx2tf.convert(
    input_onnx_file_path='checkpoints/rtmpose_s_concrete.onnx',
    output_folder_path=out_dir,
    non_verbose=True,
)
print('SUCCESS', flush=True)
" > /tmp/onnx2tf_rtmpose_run.log 2>&1
echo "exit: $?" >> /tmp/onnx2tf_rtmpose_run.log
