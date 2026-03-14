#!/bin/zsh
source /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml/.venv/bin/activate
cd /Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml

python > /tmp/rtmpose_onnx2tf_verbose.log 2>&1 << 'PYEOF'
import struct, onnx.helper as h
if not hasattr(h, 'float32_to_bfloat16'):
    def _f(v):
        b = struct.pack('f', v)
        return struct.unpack('H', b[2:])[0]
    h.float32_to_bfloat16 = _f

import tensorflow as tf
import onnx2tf, os, shutil

out_dir = 'checkpoints/rtmpose_s_tf'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

try:
    onnx2tf.convert(
        input_onnx_file_path='checkpoints/rtmpose_s_gau_fixed_v2.onnx',
        output_folder_path=out_dir,
        batch_size=1,
        non_verbose=False,  # VERBOSE
    )
    print('SUCCESS', flush=True)
except Exception as e:
    print(f'FAILED: {e}', flush=True)
PYEOF
echo "exit: $?" >> /tmp/rtmpose_onnx2tf_verbose.log
