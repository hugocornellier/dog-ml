"""Diagnostic script to run onnx2tf and capture error."""
import struct
import onnx.helper as h
if not hasattr(h, 'float32_to_bfloat16'):
    def _f(v):
        b = struct.pack('f', v)
        return struct.unpack('H', b[2:])[0]
    h.float32_to_bfloat16 = _f

import os
import shutil
import traceback

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
LOG_FILE = os.path.join(BASE, 'checkpoints/onnx2tf_rtmpose_log.txt')

import sys
# Tee stdout/stderr to file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            f.write(s)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log = open(LOG_FILE, 'w')
sys.stdout = Tee(sys.__stdout__, log)
sys.stderr = Tee(sys.__stderr__, log)

try:
    import onnx2tf

    out_dir = os.path.join(BASE, 'checkpoints/rtmpose_s_tf')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    onnx2tf.convert(
        input_onnx_file_path=os.path.join(BASE, 'checkpoints/rtmpose_s_static.onnx'),
        output_folder_path=out_dir,
        non_verbose=False,
    )
    print('ONNX2TF_SUCCESS')
except Exception as e:
    print(f'ONNX2TF_FAILED: {e}')
    traceback.print_exc()

log.close()
