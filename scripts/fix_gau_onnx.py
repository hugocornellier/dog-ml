"""
Fix the RTMPose GAU head in ONNX by avoiding 4D non-image tensors that break onnx2tf.

The GAU Mul op takes [1,39,1,128] x [1,1,2,128] -> [1,39,2,128].
onnx2tf incorrectly NCHW-transposes these 4D tensors.

Fix: Remove the Unsqueeze before the Mul, modify the weight to [1,39,2,128],
and add a Reshape after the Mul to produce [1,39,2,128] from 3D tensors.

Actually the cleaner fix: add Flatten/Reshape before the GAU operations
to remove the problematic 4D tensors.
"""

import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import helper, TensorProto, numpy_helper

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
ONNX_IN = os.path.join(BASE, 'checkpoints/rtmpose_s_opset11_sym.onnx')
ONNX_OUT = os.path.join(BASE, 'checkpoints/rtmpose_s_gau_fixed.onnx')

print(f'Loading {ONNX_IN}')
m = onnx.load(ONNX_IN)
print(f'opset: {m.opset_import[0].version}, nodes: {len(m.graph.node)}')

# First run symbolic shape inference
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
m = SymbolicShapeInference.infer_shapes(m, auto_merge=True, int_max=2**31-1, guess_output_rank=False, verbose=0)

shape_dict = {}
for vi in list(m.graph.value_info) + list(m.graph.input):
    try:
        dims = [(d.dim_value if d.HasField('dim_value') else d.dim_param) for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    except: pass
for ini in m.graph.initializer:
    arr = nph.to_array(ini)
    shape_dict[ini.name] = list(arr.shape)

# Find the problematic Unsqueeze node
unsqueeze_node = None
mul_node = None
for node in m.graph.node:
    if node.name == '/model/bodypart/gau/Unsqueeze':
        unsqueeze_node = node
    if node.name == '/model/bodypart/gau/Mul':
        mul_node = node

print(f'Unsqueeze node: {unsqueeze_node.name}, inputs={list(unsqueeze_node.input)}, outputs={list(unsqueeze_node.output)}')
print(f'  Unsqueeze input shape: {shape_dict.get(unsqueeze_node.input[0], "?")}')
print(f'  Unsqueeze output shape: {shape_dict.get(unsqueeze_node.output[0], "?")}')
print(f'Mul node: {mul_node.name}, inputs={list(mul_node.input)}, outputs={list(mul_node.output)}')
print(f'  Mul weight shape: {shape_dict.get(mul_node.input[1], "?")}')
print(f'  Mul output shape: {shape_dict.get(mul_node.output[0], "?")}')

# Get the Mul weight initializer
mul_weight_name = mul_node.input[1]
mul_weight_init = None
for ini in m.graph.initializer:
    if ini.name == mul_weight_name:
        mul_weight_init = ini
        break
assert mul_weight_init is not None, f'Could not find initializer {mul_weight_name}'
weight = nph.to_array(mul_weight_init)
print(f'Weight array shape: {weight.shape}')  # should be [1, 1, 2, 128]

# Strategy:
# The Unsqueeze [1,39,128] → [1,39,1,128] just adds a dim
# The weight is [1,1,2,128]
# The output is [1,39,2,128]
#
# We can replace this with: Mul([1,39,128], reshape([2,128])) → [1,39,2,128]...
# but that's harder without Expand.
#
# Better: just replace the 4D Mul with a Reshape→Mul→Reshape pattern:
# 1. Reshape [1,39,1,128] -> [1,39,128] (remove the unsqueeze dim)
# 2. Mul with reshaped weight [128] ... but we need [2,128] broadcast
#
# Actually the operation is: z[1,39,1,128] * w[1,1,2,128] = out[1,39,2,128]
# This is: expand z along dim 2 by repeating 2 times
# Which = stack([z*w[0], z*w[1]], dim=2)
#
# Simplest equivalent using 3D ops:
# z_3d = z[1,39,128] (=Unsqueeze input)
# w0 = w[1,1,0,:] = [128] scalar vector
# w1 = w[1,1,1,:] = [128] scalar vector
# out0 = z_3d * w0  → [1,39,128]
# out1 = z_3d * w1  → [1,39,128]
# out = Stack([out0, out1], axis=2) → [1,39,2,128]
# But Stack is not standard ONNX...
#
# Alternative with Concat:
# out0 = Unsqueeze(z_3d*w0, axis=2) → [1,39,1,128]  -- same 4D problem
#
# Best approach: Add a Reshape+Transpose node to turn [1,39,1,128]*[1,1,2,128]
# into 3D ops.
#
# Actually: just add a Reshape to make the activations [39, 128] (remove batch=1
# since we have static batch), do the mul as [39,128]*[2,128] -> still 2D broadcast issue
#
# Real solution: use onnx_graphsurgeon to reshape to avoid onnx2tf's NCHW transposing.
#
# SIMPLEST FIX: pad both inputs to 5D tensors [1,1,39,1,128] and [1,1,1,2,128]
# onnx2tf only transposes 4D tensors (NCHW->NHWC), NOT 5D tensors (NCDHW->NDHWC)
# So making them 5D will prevent onnx2tf from applying the wrong transpose.

print('\nApplying 5D fix: adding Unsqueeze/Squeeze around the problematic Mul and Add nodes...')

new_nodes = []
new_initializers = []

# Map of (old node name -> new replacement node)
# We'll process nodes in order and replace the ones that need fixing

# Find which nodes need fixing: Mul and Add in GAU that have 4D tensors
problem_nodes = {'/model/bodypart/gau/Mul', '/model/bodypart/gau/Add'}

# Also find the preceding Unsqueeze (which is fine, keep it)
# and the following Split (keep as-is)

# NEW APPROACH: Insert Reshape 4D->2D before the Mul, and Reshape 2D->4D after
# The Unsqueeze output is [1,39,1,128] - we can reshape to [39,128] (drop batch=1, drop 1)
# The weight needs to be reshaped too: [1,1,2,128] -> needs careful handling

# Let me try a different simplest approach:
# Replace Unsqueeze+Mul+Add with:
# 1. Keep Unsqueeze as-is [1,39,128] -> [1,39,1,128]
# 2. Add a Reshape: [1,39,1,128] -> [39,1,128]  (merge batch dim)
# 3. Reshape weight: [1,1,2,128] -> [1,2,128]
# 4. Mul: [39,1,128] * [1,2,128] = [39,2,128]  (3D broadcast, onnx2tf safe)
# 5. Reshape: [39,2,128] -> [1,39,2,128]
# Similarly for Add:
# 6. Reshape: [1,39,2,128] -> [39,2,128]
# 7. Add: [39,2,128] + [2,128] = [39,2,128]  (2D broadcast, onnx2tf safe)
# 8. Reshape: [39,2,128] -> [1,39,2,128]

# Add the reshape weight initializer
mul_weight = nph.to_array(mul_weight_init)  # shape [1,1,2,128]
mul_weight_3d = mul_weight.reshape(1, 2, 128)  # [1,2,128]

weight_3d_name = 'rtmpose_fix_mul_weight_3d'
weight_3d_init = numpy_helper.from_array(mul_weight_3d, name=weight_3d_name)
new_initializers.append(weight_3d_init)

# Add beta initializer (for Add)
beta_name = 'model.heads.bodypart.gau.beta'
beta_init = None
for ini in m.graph.initializer:
    if ini.name == beta_name:
        beta_init = ini
        break
# beta shape is [2, 128] - this is 2D, onnx2tf handles 2D fine, no fix needed for Add

# Shape constants
unsq_input_name = unsqueeze_node.input[0]  # [1,39,128]

# Reshape [1,39,1,128] -> [39,1,128]
reshape_4d_to_3d_name = 'rtmpose_fix_unsq_out_3d'
reshape_4d_to_3d_shape_name = 'rtmpose_fix_shape_39_1_128'
shape_39_1_128 = np.array([39, 1, 128], dtype=np.int64)
shape_init = numpy_helper.from_array(shape_39_1_128, name=reshape_4d_to_3d_shape_name)
new_initializers.append(shape_init)

reshape_4d_to_3d_node = helper.make_node(
    'Reshape',
    inputs=[unsqueeze_node.output[0], reshape_4d_to_3d_shape_name],
    outputs=[reshape_4d_to_3d_name],
    name='rtmpose_fix_reshape_4d_to_3d'
)

# Mul [39,1,128] * [1,2,128] = [39,2,128]
mul_3d_out_name = 'rtmpose_fix_mul_3d_out'
mul_3d_node = helper.make_node(
    'Mul',
    inputs=[reshape_4d_to_3d_name, weight_3d_name],
    outputs=[mul_3d_out_name],
    name='rtmpose_fix_mul_3d'
)

# Reshape [39,2,128] -> [1,39,2,128]
reshape_3d_to_4d_name = mul_node.output[0]  # use the original Mul output name
reshape_3d_to_4d_shape_name = 'rtmpose_fix_shape_1_39_2_128'
shape_1_39_2_128 = np.array([1, 39, 2, 128], dtype=np.int64)
shape_init2 = numpy_helper.from_array(shape_1_39_2_128, name=reshape_3d_to_4d_shape_name)
new_initializers.append(shape_init2)

reshape_3d_to_4d_node = helper.make_node(
    'Reshape',
    inputs=[mul_3d_out_name, reshape_3d_to_4d_shape_name],
    outputs=[reshape_3d_to_4d_name],
    name='rtmpose_fix_reshape_3d_to_4d'
)

# Build the new node list
skipped_nodes = {'/model/bodypart/gau/Mul'}
extra_nodes = [reshape_4d_to_3d_node, mul_3d_node, reshape_3d_to_4d_node]
extra_nodes_added = False

new_graph_nodes = []
for node in m.graph.node:
    if node.name == '/model/bodypart/gau/Unsqueeze':
        new_graph_nodes.append(node)  # keep as-is
        # Add our fix nodes right after
        new_graph_nodes.extend(extra_nodes)
        extra_nodes_added = True
    elif node.name in skipped_nodes:
        # Skip the original Mul
        pass
    else:
        new_graph_nodes.append(node)

assert extra_nodes_added, 'Could not find Unsqueeze node to insert after'
print(f'Original nodes: {len(m.graph.node)}, New nodes: {len(new_graph_nodes)}')

# Build new model
new_m = onnx.ModelProto()
new_m.CopyFrom(m)
new_m.graph.ClearField('node')
for node in new_graph_nodes:
    new_m.graph.node.append(node)

# Add new initializers
for init in new_initializers:
    new_m.graph.initializer.append(init)

# Clear and re-add value_info (it may be stale)
new_m.graph.ClearField('value_info')

# Run shape inference on new model
print('Running ONNX checker...')
try:
    onnx.checker.check_model(new_m)
    print('ONNX check passed')
except Exception as e:
    print(f'ONNX check warning: {e}')

onnx.save(new_m, ONNX_OUT)
print(f'Saved to {ONNX_OUT}')

# Quick ORT validation
import onnxruntime as ort
import numpy as np

print('Running ORT validation...')
sess = ort.InferenceSession(ONNX_OUT, providers=['CPUExecutionProvider'])
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
result = sess.run(None, {'input': dummy})
print(f'ORT output shapes: {[r.shape for r in result]}')
print('ORT validation passed!')
