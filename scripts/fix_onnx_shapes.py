"""
Fix symbolic dimensions in RTMPose ONNX by running a concrete inference
and updating value_info shapes, then re-export.
"""
import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import TensorProto, helper

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
ONNX_IN = os.path.join(BASE, 'checkpoints/rtmpose_s_static.onnx')
ONNX_OUT = os.path.join(BASE, 'checkpoints/rtmpose_s_concrete.onnx')

print(f'Loading {ONNX_IN}')
m = onnx.load(ONNX_IN)

# Run ORT to get all intermediate shapes
import onnxruntime as ort

# Enable getting intermediate outputs
sess_options = ort.SessionOptions()
sess_options.enable_profiling = False

# Get all intermediate output names
intermediate_names = [vi.name for vi in m.graph.value_info]

# Add all intermediates to outputs temporarily
m_extended = onnx.ModelProto()
m_extended.CopyFrom(m)

# Build a model that exposes all intermediates as outputs
for vi in m.graph.value_info:
    output = onnx.helper.make_tensor_value_info(
        vi.name, vi.type.tensor_type.elem_type,
        [dim.dim_value if dim.HasField('dim_value') else None
         for dim in vi.type.tensor_type.shape.dim]
    )
    m_extended.graph.output.append(output)

print('Running ORT with dummy input to get concrete shapes...')
sess = ort.InferenceSession(m_extended.SerializeToString(), providers=['CPUExecutionProvider'])
dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
outputs = sess.run(None, {'input': dummy_input})

# Build name -> concrete shape mapping
all_output_names = [o.name for o in m_extended.graph.output]
concrete_shapes = {}
for name, arr in zip(all_output_names, outputs):
    concrete_shapes[name] = arr.shape

print(f'Got {len(concrete_shapes)} concrete shapes')

# Now update the model's value_info with concrete shapes
new_model = onnx.ModelProto()
new_model.CopyFrom(m)
new_model.graph.ClearField('value_info')

for vi in m.graph.value_info:
    if vi.name in concrete_shapes:
        shape = concrete_shapes[vi.name]
        elem_type = vi.type.tensor_type.elem_type
        new_vi = helper.make_tensor_value_info(vi.name, elem_type, list(shape))
        new_model.graph.value_info.append(new_vi)
    else:
        new_model.graph.value_info.append(vi)

# Save
onnx.save(new_model, ONNX_OUT)
print(f'Saved concrete ONNX to {ONNX_OUT}')

# Verify the problematic nodes
shape_dict = {}
for vi in list(new_model.graph.value_info) + list(new_model.graph.input):
    try:
        dims = [d.dim_value if d.HasField('dim_value') else d.dim_param
                for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    except:
        pass

# Check GAU tensors
for node in new_model.graph.node:
    for inp in node.input:
        if 'beta' in inp:
            print(f'GAU Add node: {node.name}')
            for i in node.input:
                print(f'  input {i}: {shape_dict.get(i, "?")}')
            for o in node.output:
                print(f'  output {o}: {shape_dict.get(o, "?")}')

print('Done!')
