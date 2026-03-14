"""
Fix the RTMPose GAU head in ONNX v2.

Replace the Unsqueeze + 4D Mul + Add with equivalent 3D operations:

Original:
  z[1,39,128] -> Unsqueeze -> [1,39,1,128]
  [1,39,1,128] * weight[1,1,2,128] -> [1,39,2,128]

Replacement:
  z[1,39,128] -> MatMul with weight_T[128, 2*128] -> [1,39,2*128]
  [1,39,2*128] -> Reshape -> [1,39,2,128]

The weight transformation:
  weight [1,1,2,128] -> squeeze -> [2,128] -> view -> diagonal matrices

Actually simpler: the broadcast multiply is equivalent to:
  for each of the 2 rows w[i] of shape [128]:
    z_i = z * w[i]   -> [1,39,128]
  stack([z_0, z_1], axis=2) -> [1,39,2,128]

This can be done as:
  z_expanded = Tile(z, [1, 1, 2]) -> [1,39,256] ... but that multiplies ALL features

Actually: the exact operation is element-wise:
  out[b,s,i,d] = z[b,s,d] * w[0,0,i,d]  for i in [0,1], d in [0..127]

This is equivalent to:
  w2d = w.reshape(2, 128)  -> [2, 128]
  w_T = w2d.T              -> [128, 2]
  # Then: einsum('bsd,dc->bscd', z, w_T)... which is not a simple matmul

Alternative: diag weights
  z_diag = Einsum... too complex for ONNX opset 11

Better: use Expand + Mul with 3D
  z3d = Unsqueeze(z, axis=1)  -> [1,1,39,128] ... still 4D

Actually, the cleanest approach for opset 11:
  - Create a weight matrix W_flat = diag(w[0]) concat diag(w[1]) ... but that's huge

SIMPLEST CORRECT FIX for onnx2tf:
  Keep the existing Unsqueeze and Mul nodes, but add Squeeze nodes before them
  to make onnx2tf think these are 3D tensors, not 4D.

  OR: manually create a new Mul node that takes 3D inputs:
  z[1,39,128] * reshaped_weight[2,128] -> doesn't broadcast to [1,39,2,128]

Let me think about this differently using Reshape + Gemm:
  z_flat = Reshape(z, [39, 128])  -> [39, 128]
  w_T = w.reshape(2, 128).T -> [128, 2]  (transposed weight)
  dot = MatMul(z_flat, w_T) -> [39, 2]  ... but this LOSES the head_dim info

No, the multiply preserves all 128 dims. Each element is multiplied by the same scalar per dim per head. This is a channel-wise scaling.

CORRECT UNDERSTANDING:
  For head i (i=0,1), feature d (d=0..127):
    out[b,s,i,d] = z[b,s,d] * w[i,d]

  This is: broadcast multiply z[B,S,D] with w[H,D] along a new axis.

For onnx2tf to handle this correctly:
  1. We need to avoid 4D representations
  2. We can reshape z to [B*S, D] and use a matrix where each row is repeated

The cleanest solution:
  Use concat of two separate Mul operations, all in 3D:

  z[1,39,128] * w0[128] -> [1,39,128]  (head 0)
  z[1,39,128] * w1[128] -> [1,39,128]  (head 1)
  Concat([head0, head1], axis=2)  ... axis=2 would concat along feature dim

  But we want output [1,39,2,128], not [1,39,256].
  We need Unsqueeze then Concat:
  head0_4d = Unsqueeze(head0, axis=2)  -> [1,39,1,128]  # 4D again!

OK, the real solution is to let onnx2tf see these as NHWC-compatible 4D tensors.

Actually: use the `keep_ncw_or_nchw_or_ncdhw_input_names` option in onnx2tf to
tell it that the `input` tensor (spatial image) is in NCHW, but create a
SEPARATE input for the GAU part that bypasses the transposing... Not possible.

FINAL APPROACH: Simply avoid Unsqueeze+4D entirely by computing the output differently.

The operation z[1,39,128] * w[1,1,2,128] = out[1,39,2,128] means:
  out[b,s,k,d] = z[b,s,d] * w[0,0,k,d]

This is: for each head k, multiply z by w[k].

We can compute this as:
  1. Compute head0 = z * w0   [1,39,128] (w0 = w[0,0,0,:])
  2. Compute head1 = z * w1   [1,39,128] (w1 = w[0,0,1,:])
  3. Stack along a new axis = Concat after Unsqueeze

But Unsqueeze creates 4D again...

ALTERNATIVE: compute as [1,39,256] and then reshape to [1,39,2,128]:
  1. Tile z from [1,39,128] to [1,39,256] -> Tile([1,1,2])
  2. Create weight_tiled [256] = concat(w0, w1)
  3. out_flat = Tile(z) * weight_tiled -> [1,39,256]
  4. out = Reshape(out_flat, [1,39,2,128])

  But Tile then multiply gives z*w0 in [0:128] and z*w1 in [128:256], which when
  reshaped gives [1,39,2,128] where [b,s,0,:]=z*w0 and [b,s,1,:]=z*w1. CORRECT!

This is FULLY 3D/2D, no 4D intermediate, onnx2tf handles this fine!
"""

import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
ONNX_IN = os.path.join(BASE, 'checkpoints/rtmpose_s_opset11_sym.onnx')
ONNX_OUT = os.path.join(BASE, 'checkpoints/rtmpose_s_gau_fixed_v2.onnx')

print(f'Loading {ONNX_IN}')
m = onnx.load(ONNX_IN)
m = SymbolicShapeInference.infer_shapes(m, auto_merge=True, int_max=2**31-1, guess_output_rank=False, verbose=0)

# Get all initializers as numpy arrays
init_dict = {ini.name: nph.to_array(ini) for ini in m.graph.initializer}

# Find the Unsqueeze and Mul nodes
unsqueeze_node = None
mul_node = None
add_node = None
for node in m.graph.node:
    if node.name == '/model/bodypart/gau/Unsqueeze':
        unsqueeze_node = node
    if node.name == '/model/bodypart/gau/Mul':
        mul_node = node
    if node.name == '/model/bodypart/gau/Add':
        add_node = node

print(f'Unsqueeze: input={list(unsqueeze_node.input)}, output={list(unsqueeze_node.output)}')
print(f'Mul: input={list(mul_node.input)}, output={list(mul_node.output)}')
print(f'Add: input={list(add_node.input)}, output={list(add_node.output)}')

# z is the Unsqueeze input [1,39,128]
z_name = unsqueeze_node.input[0]  # [1,39,128]

# Mul weight [1,1,2,128] -> needs to become [256] (concat of w0 and w1)
mul_weight = init_dict[mul_node.input[1]]  # [1,1,2,128]
w0 = mul_weight[0, 0, 0, :]  # [128]
w1 = mul_weight[0, 0, 1, :]  # [128]
weight_tiled_values = np.concatenate([w0, w1], axis=0)  # [256]
print(f'w0 shape: {w0.shape}, w1 shape: {w1.shape}, weight_tiled shape: {weight_tiled_values.shape}')

# Add bias [2, 128] -> needs to become [256] (concat of b0 and b1)
add_weight = init_dict[add_node.input[1]]  # [2,128]
b0 = add_weight[0, :]  # [128]
b1 = add_weight[1, :]  # [128]
bias_tiled_values = np.concatenate([b0, b1], axis=0)  # [256]
print(f'add_weight shape: {add_weight.shape}, bias_tiled shape: {bias_tiled_values.shape}')

# Create new initializer names
weight_tiled_name = 'rtmpose_fix_v2_mul_weight_256'
bias_tiled_name = 'rtmpose_fix_v2_add_bias_256'
shape_1_39_256_name = 'rtmpose_fix_v2_shape_1_39_256'
shape_1_39_2_128_name = 'rtmpose_fix_v2_shape_1_39_2_128'

# Values
weight_tiled_init = numpy_helper.from_array(weight_tiled_values, name=weight_tiled_name)
bias_tiled_init = numpy_helper.from_array(bias_tiled_values, name=bias_tiled_name)
shape_1_39_256_init = numpy_helper.from_array(np.array([1, 39, 256], dtype=np.int64), name=shape_1_39_256_name)
shape_1_39_2_128_init = numpy_helper.from_array(np.array([1, 39, 2, 128], dtype=np.int64), name=shape_1_39_2_128_name)

# Build replacement nodes:
# 1. Tile z[1,39,128] -> [1,39,256]:  Tile(z, [1,1,2])
tile_repeats_name = 'rtmpose_fix_v2_tile_repeats'
tile_repeats_init = numpy_helper.from_array(np.array([1, 1, 2], dtype=np.int64), name=tile_repeats_name)
tile_out_name = 'rtmpose_fix_v2_tile_out'
tile_node = helper.make_node(
    'Tile',
    inputs=[z_name, tile_repeats_name],
    outputs=[tile_out_name],
    name='rtmpose_fix_v2_tile'
)

# 2. Mul: [1,39,256] * [256] -> [1,39,256]
mul_flat_out_name = 'rtmpose_fix_v2_mul_flat_out'
mul_flat_node = helper.make_node(
    'Mul',
    inputs=[tile_out_name, weight_tiled_name],
    outputs=[mul_flat_out_name],
    name='rtmpose_fix_v2_mul_flat'
)

# 3. Add: [1,39,256] + [256] -> [1,39,256]
add_flat_out_name = add_node.output[0]  # Use the original Add output name so downstream works
add_flat_node = helper.make_node(
    'Add',
    inputs=[mul_flat_out_name, bias_tiled_name],
    outputs=['rtmpose_fix_v2_add_flat_out_tmp'],
    name='rtmpose_fix_v2_add_flat'
)

# 4. Reshape: [1,39,256] -> [1,39,2,128]
reshape_final_node = helper.make_node(
    'Reshape',
    inputs=['rtmpose_fix_v2_add_flat_out_tmp', shape_1_39_2_128_name],
    outputs=[add_flat_out_name],
    name='rtmpose_fix_v2_reshape_final'
)

# Nodes to skip
skipped_nodes = {
    '/model/bodypart/gau/Unsqueeze',
    '/model/bodypart/gau/Mul',
    '/model/bodypart/gau/Add'
}

new_graph_nodes = []
insertion_done = False
for node in m.graph.node:
    if node.name == '/model/bodypart/gau/Split' and not insertion_done:
        # Insert BEFORE the Split_1 which comes after the Add
        pass
    if node.name in skipped_nodes:
        if not insertion_done:
            # Insert replacement nodes here
            new_graph_nodes.extend([tile_node, mul_flat_node, add_flat_node, reshape_final_node])
            insertion_done = True
        # Skip the original node
        continue
    new_graph_nodes.append(node)

print(f'Original: {len(m.graph.node)}, New: {len(new_graph_nodes)}')

# Build new model
new_m = onnx.ModelProto()
new_m.CopyFrom(m)
new_m.graph.ClearField('node')
for node in new_graph_nodes:
    new_m.graph.node.append(node)

# Add new initializers
for init in [weight_tiled_init, bias_tiled_init, tile_repeats_init, shape_1_39_256_init, shape_1_39_2_128_init]:
    new_m.graph.initializer.append(init)

# Remove old initializers that are no longer used
old_init_names_to_remove = {mul_node.input[1], add_node.input[1]}
new_initializers = [ini for ini in new_m.graph.initializer if ini.name not in old_init_names_to_remove]
new_m.graph.ClearField('initializer')
for ini in new_initializers:
    new_m.graph.initializer.append(ini)

# Clear value_info (stale)
new_m.graph.ClearField('value_info')

# Validate
print('Running ONNX checker...')
try:
    onnx.checker.check_model(new_m)
    print('ONNX check passed')
except Exception as e:
    print(f'ONNX check warning: {e}')

onnx.save(new_m, ONNX_OUT)
print(f'Saved to {ONNX_OUT}')

# ORT validation
import onnxruntime as ort
print('Running ORT validation...')
sess = ort.InferenceSession(ONNX_OUT, providers=['CPUExecutionProvider'])
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
result = sess.run(None, {'input': dummy})
print(f'ORT output shapes: {[r.shape for r in result]}')

# Compare with original
sess_orig = ort.InferenceSession(ONNX_IN, providers=['CPUExecutionProvider'])
result_orig = sess_orig.run(None, {'input': dummy})
for i, (r, r_orig) in enumerate(zip(result, result_orig)):
    diff = np.abs(r - r_orig).max()
    print(f'Output {i}: max diff = {diff:.6f}')

print('Done!')
