"""Run onnx2tf on the concrete ONNX model."""
import os, sys, shutil

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
LOG = os.path.join(BASE, 'checkpoints/onnx2tf_concrete_log.txt')

log = open(LOG, 'w', buffering=1)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files:
            try: f.write(s); f.flush()
            except: pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except: pass

sys.stdout = Tee(sys.__stdout__, log)
sys.stderr = Tee(sys.__stderr__, log)

import struct
import onnx.helper as h
if not hasattr(h, 'float32_to_bfloat16'):
    def _f(v):
        b = struct.pack('f', v)
        return struct.unpack('H', b[2:])[0]
    h.float32_to_bfloat16 = _f

import traceback
import onnx2tf

out_dir = os.path.join(BASE, 'checkpoints/rtmpose_s_tf')
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

try:
    onnx2tf.convert(
        input_onnx_file_path=os.path.join(BASE, 'checkpoints/rtmpose_s_concrete.onnx'),
        output_folder_path=out_dir,
        non_verbose=True,
    )
    print('ONNX2TF_SUCCESS')
except Exception as e:
    print(f'ONNX2TF_FAILED: {type(e).__name__}: {e}')
    traceback.print_exc()

log.close()
