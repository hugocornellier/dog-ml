"""
Make all intermediate ONNX shapes concrete for batch=1, 256x256 input.
Uses ONNX Runtime with extended outputs to get concrete shapes.
Saves a new ONNX with all value_info shapes filled in concretely.
"""
import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import TensorProto, helper, shape_inference
import onnxruntime as ort
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
ONNX_IN = os.path.join(BASE, 'checkpoints/rtmpose_s_static.onnx')
ONNX_OUT = os.path.join(BASE, 'checkpoints/rtmpose_s_concrete.onnx')

print(f'Loading {ONNX_IN}')
m = onnx.load(ONNX_IN)
print(f'opset: {m.opset_import[0].version}')
print(f'# nodes: {len(m.graph.node)}')

# Step 1: Run symbolic shape inference (onnxruntime)
print('Running symbolic shape inference...')
try:
    m_sym = SymbolicShapeInference.infer_shapes(
        m, auto_merge=True, int_max=2**31 - 1, guess_output_rank=False, verbose=0
    )
    print('Symbolic shape inference succeeded')
except Exception as e:
    print(f'Symbolic shape inference failed: {e}, falling back to onnx shape_inference')
    m_sym = shape_inference.infer_shapes(m)

# Collect all intermediate shapes after symbolic inference
shape_dict = {}
for vi in list(m_sym.graph.value_info) + list(m_sym.graph.input):
    try:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField('dim_value') and d.dim_value > 0:
                dims.append(d.dim_value)
            elif d.HasField('dim_param'):
                dims.append(d.dim_param)
            else:
                dims.append(None)
        shape_dict[vi.name] = dims
    except Exception:
        pass
for ini in m_sym.graph.initializer:
    shape_dict[ini.name] = list(ini.dims)

# Check problematic node
for node in m_sym.graph.node:
    for inp in node.input:
        if 'beta' in inp:
            print(f'GAU Add node after sym inference: {node.name}')
            for i in node.input:
                print(f'  input {i}: {shape_dict.get(i, "?")}')
            for o in node.output:
                print(f'  output {o}: {shape_dict.get(o, "?")}')

# Step 2: Run ORT to get concrete shapes for any remaining symbolic/unknown dims
print('\nCreating extended model for intermediate shape collection...')

# Build a model that exposes ALL intermediate values as outputs
m_ext = onnx.ModelProto()
m_ext.CopyFrom(m)
all_vi_names = {vi.name for vi in m_ext.graph.value_info}

for node in m_ext.graph.node:
    for out in node.output:
        if out and out not in all_vi_names:
            # Add as output if not already there
            pass

# Add all value_info as graph outputs
existing_out_names = {o.name for o in m_ext.graph.output}
for vi in m.graph.value_info:
    if vi.name not in existing_out_names:
        m_ext.graph.output.append(vi)

print(f'Extended model has {len(m_ext.graph.output)} outputs')
print('Running ORT on extended model...')

try:
    sess = ort.InferenceSession(
        m_ext.SerializeToString(),
        providers=['CPUExecutionProvider'],
        sess_options=ort.SessionOptions()
    )
    dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
    out_names = [o.name for o in m_ext.graph.output]
    results = sess.run(None, {'input': dummy})

    concrete_shapes = {}
    for name, arr in zip(out_names, results):
        concrete_shapes[name] = arr.shape

    print(f'Got {len(concrete_shapes)} concrete shapes from ORT')

    # Check the GAU tensors
    gau_tensors = [n for n in concrete_shapes if 'gau' in n.lower() or 'Mul_output_0' in n]
    for n in gau_tensors[:5]:
        print(f'  {n}: {concrete_shapes[n]}')

except Exception as e:
    print(f'ORT extended run failed: {e}')
    concrete_shapes = {}

# Step 3: Build updated model with concrete shapes
print('\nBuilding concrete ONNX model...')
new_m = onnx.ModelProto()
new_m.CopyFrom(m)
new_m.graph.ClearField('value_info')

for vi in m.graph.value_info:
    elem_type = vi.type.tensor_type.elem_type

    if vi.name in concrete_shapes:
        shape = concrete_shapes[vi.name]
        new_vi = helper.make_tensor_value_info(vi.name, elem_type, list(shape))
    elif vi.name in shape_dict:
        new_vi = helper.make_tensor_value_info(vi.name, elem_type, shape_dict[vi.name])
    else:
        new_vi = vi

    new_m.graph.value_info.append(new_vi)

# Final check
final_shape_dict = {}
for vi in new_m.graph.value_info:
    try:
        dims = [d.dim_value if d.HasField('dim_value') else d.dim_param
                for d in vi.type.tensor_type.shape.dim]
        final_shape_dict[vi.name] = dims
    except:
        pass

for node in new_m.graph.node:
    for inp in node.input:
        if 'beta' in inp:
            print(f'Final GAU Add node: {node.name}')
            for i in node.input:
                print(f'  input {i}: {final_shape_dict.get(i, "?")}')

onnx.save(new_m, ONNX_OUT)
print(f'\nSaved concrete ONNX to {ONNX_OUT}')
print('Done!')
