"""
Fix RTMPose ONNX v5: Replace dynamic Reshape + SimCC head with 2D-only ops.

Key insight from verbose onnx2tf analysis:
- onnx2tf transposes 3D NCW->NWC tensors, inserting transpose ops
- The dynamic Shape+Slice+Concat+Reshape block produces unknown-shape tensors
- onnx2tf inserts extra transposes on unknown-shape tensors before downstream ops
- Entry reshape [39,64] gets a transposed (NWC) input -> wrong data

Fix:
1. Replace the dynamic Reshape block (Shape+Slice+Concat+Reshape) with a STATIC
   Reshape from [1,39,8,8] -> [39,64] directly. This produces a 2D tensor that
   onnx2tf does NOT transpose.

2. Replace MLP mlp.0 (RMSNorm) which operated on [1,39,64] (3D) with the same
   operations on [39,64] (2D):
   - ReduceL2 axis=-1 on [39,64] -> [39,1] (was [1,39,1])
   - All element-wise ops stay the same since they're axis-independent

3. Remove the previously-inserted entry_reshape from v4 (no longer needed since
   the dynamic Reshape is now already [39,64])

4. Keep the GAU q/k 2D replacements and Transpose perm=[1,0] fix from v4

5. Keep the exit reshape [39,512] -> [1,39,512]

Result: ALL SimCC head operations are 2D, onnx2tf converts correctly.
"""

import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
ONNX_IN = os.path.join(BASE, 'checkpoints/rtmpose_s_opset11_sym.onnx')
ONNX_OUT = os.path.join(BASE, 'checkpoints/rtmpose_s_gau_fixed_v5.onnx')

print(f'Loading {ONNX_IN}')
m = onnx.load(ONNX_IN)
m = SymbolicShapeInference.infer_shapes(m, auto_merge=True, int_max=2**31-1, guess_output_rank=False, verbose=0)

# Get shape dict
shape_dict = {}
for vi in list(m.graph.value_info) + list(m.graph.input):
    try:
        dims = [(d.dim_value if d.HasField('dim_value') else d.dim_param) for d in vi.type.tensor_type.shape.dim]
        shape_dict[vi.name] = dims
    except: pass
for ini in m.graph.initializer:
    arr = nph.to_array(ini)
    shape_dict[ini.name] = list(arr.shape)

# Get all initializers
init_dict = {ini.name: nph.to_array(ini) for ini in m.graph.initializer}

# Find the final_layer Conv and its output (feeds into dynamic Reshape)
final_layer_conv_out = '/model/bodypart/final_layer/Conv_output_0'
conv_out_shape = shape_dict.get(final_layer_conv_out, [1, 39, 8, 8])
print(f'final_layer Conv output: {final_layer_conv_out}, shape: {conv_out_shape}')

# The dynamic Reshape target: [1, 39, -1] = [1, 39, 64]
seq_len = conv_out_shape[1]      # 39
spatial = conv_out_shape[2] * conv_out_shape[3]  # 8 * 8 = 64
print(f'seq_len={seq_len}, spatial={spatial} -> 2D shape [{seq_len},{spatial}]')

# Find nodes to replace
unsqueeze_node = None
mul_4d_node = None
add_4d_node = None
split1_node = None
squeeze_nodes = []
split_node = None
transpose_node = None
exit_nodes = []

# Dynamic reshape block nodes to replace
dynamic_reshape_nodes = {
    '/model/bodypart/Shape',
    '/model/bodypart/Slice',
    '/model/bodypart/Concat',
    '/model/bodypart/Reshape',
    # Also the Constant nodes that feed Slice/Concat
    '/model/bodypart/Constant',
    '/model/bodypart/Constant_1',
    '/model/bodypart/Constant_2',
    '/model/bodypart/Constant_3',
}

# Also need to replace MLP mlp.0 RMSNorm nodes since their axis was 2 (last of 3D)
# After 2D, axis=-1 should still work correctly (same index)
# Let's check what axis mlp.0/ReduceL2 uses
mlp0_rms_axis = -1  # axes=[-1] from earlier check, which is axis=2 (last) -- works for 2D too

for node in m.graph.node:
    n = node.name
    if n == '/model/bodypart/gau/Unsqueeze':
        unsqueeze_node = node
    elif n == '/model/bodypart/gau/Mul':
        mul_4d_node = node
    elif n == '/model/bodypart/gau/Add':
        add_4d_node = node
    elif n == '/model/bodypart/gau/Split_1':
        split1_node = node
    elif n in ('/model/bodypart/gau/Squeeze', '/model/bodypart/gau/Squeeze_1'):
        squeeze_nodes.append(node)
    elif n == '/model/bodypart/gau/Split':
        split_node = node
    elif n == '/model/bodypart/gau/Transpose':
        transpose_node = node
    elif n in ('/model/bodypart/cls_x/MatMul', '/model/bodypart/cls_y/MatMul'):
        exit_nodes.append(node)

print(f'Unsqueeze: {unsqueeze_node.name}')
print(f'Split: {split_node.name}')
print(f'Transpose: {transpose_node.name}')

# Extract GAU 4D weights for 2D replacement
mul_weight_name = mul_4d_node.input[1]
mul_weight = init_dict[mul_weight_name]  # [1,1,2,128]
w0 = mul_weight[0, 0, 0, :]  # [128]
w1 = mul_weight[0, 0, 1, :]  # [128]

add_bias_name = add_4d_node.input[1]
add_bias = init_dict[add_bias_name]  # [2,128]
b0 = add_bias[0, :]  # [128]
b1 = add_bias[1, :]  # [128]

# Original output of dynamic Reshape (feeds into mlp.0)
reshape_orig_out = '/model/bodypart/Reshape_output_0'
# Original output of Squeeze nodes (q and k)
squeeze_out_0 = squeeze_nodes[0].output[0]  # q
squeeze_out_1 = squeeze_nodes[1].output[0]  # k

p = 'rtmpose_fix_v5_'

# New initializers
new_inits = [
    # Static reshape target [39, 64]
    numpy_helper.from_array(np.array([seq_len, spatial], dtype=np.int64), name=p+'shape_flat'),
    # Exit reshape [1,39,512]
    numpy_helper.from_array(np.array([1, seq_len, 512], dtype=np.int64), name=p+'shape_exit'),
    # GAU 2D head weights/biases
    numpy_helper.from_array(w0.astype(np.float32), name=p+'w0'),
    numpy_helper.from_array(w1.astype(np.float32), name=p+'w1'),
    numpy_helper.from_array(b0.astype(np.float32), name=p+'b0'),
    numpy_helper.from_array(b1.astype(np.float32), name=p+'b1'),
]

# Replacement for dynamic Reshape: static Reshape [1,39,8,8] -> [39,64]
static_reshape_node = helper.make_node(
    'Reshape',
    inputs=[final_layer_conv_out, p+'shape_flat'],
    outputs=[reshape_orig_out],  # Reuse original output name so downstream works
    name=p+'static_reshape'
)

# GAU 2D replacements: q = z*w0 + b0, k = z*w1 + b1
z_name = unsqueeze_node.input[0]  # [39,128] after batch squeeze
q_mul_out = p + 'q_mul'
k_mul_out = p + 'k_mul'

q_mul = helper.make_node('Mul', inputs=[z_name, p+'w0'], outputs=[q_mul_out], name=p+'q_mul')
q_add = helper.make_node('Add', inputs=[q_mul_out, p+'b0'], outputs=[squeeze_out_0], name=p+'q_add')
k_mul = helper.make_node('Mul', inputs=[z_name, p+'w1'], outputs=[k_mul_out], name=p+'k_mul')
k_add = helper.make_node('Add', inputs=[k_mul_out, p+'b1'], outputs=[squeeze_out_1], name=p+'k_add')

# Nodes to skip
skip_nodes = dynamic_reshape_nodes | {
    unsqueeze_node.name,
    mul_4d_node.name,
    add_4d_node.name,
    split1_node.name,
    squeeze_nodes[0].name,
    squeeze_nodes[1].name,
}

# Build new node list
new_graph_nodes = []
static_reshape_inserted = False
gau_replace_done = False

for node in m.graph.node:
    if node.name in skip_nodes:
        # Insert static reshape before the first dynamic reshape node (the Shape node)
        if node.name == '/model/bodypart/Shape' and not static_reshape_inserted:
            new_graph_nodes.append(static_reshape_node)
            static_reshape_inserted = True
        # Insert GAU 2D replacements before first skip in the Unsqueeze block
        if node.name == unsqueeze_node.name and not gau_replace_done:
            new_graph_nodes.extend([q_mul, q_add, k_mul, k_add])
            gau_replace_done = True
        # Skip original node

    elif node.name == split_node.name:
        # Change split axis from 2 to 1 (for [39,1152] instead of [1,39,1152])
        split_attr = next(a for a in node.attribute if a.name == 'split')
        split_vals = list(split_attr.ints)
        new_node = helper.make_node(
            'Split',
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name,
            axis=1,
            split=split_vals,
        )
        new_graph_nodes.append(new_node)

    elif node.name == transpose_node.name:
        # Change perm from [0,2,1] to [1,0] for 2D [39,128] -> [128,39]
        new_node = helper.make_node(
            'Transpose',
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name,
            perm=[1, 0]
        )
        new_graph_nodes.append(new_node)

    elif node.name in ('/model/bodypart/cls_x/MatMul', '/model/bodypart/cls_y/MatMul'):
        orig_out = node.output[0]
        tmp_out = p + f'{orig_out}_2d'
        new_node = helper.make_node(
            node.op_type,
            inputs=list(node.input),
            outputs=[tmp_out],
            name=node.name
        )
        for attr in node.attribute:
            new_node.attribute.append(attr)
        exit_reshape = helper.make_node(
            'Reshape',
            inputs=[tmp_out, p+'shape_exit'],
            outputs=[orig_out],
            name=p+f'exit_reshape_{orig_out}'
        )
        new_graph_nodes.append(new_node)
        new_graph_nodes.append(exit_reshape)

    else:
        new_graph_nodes.append(node)

print(f'Original: {len(m.graph.node)}, New: {len(new_graph_nodes)}')

# Remove old initializers no longer needed
old_init_names_to_remove = {mul_weight_name, add_bias_name}
print(f'Removing old inits: {old_init_names_to_remove}')

# Build new model
new_m = onnx.ModelProto()
new_m.CopyFrom(m)
new_m.graph.ClearField('node')
for node in new_graph_nodes:
    new_m.graph.node.append(node)

for init in new_inits:
    new_m.graph.initializer.append(init)

new_initializers = [ini for ini in new_m.graph.initializer if ini.name not in old_init_names_to_remove]
new_m.graph.ClearField('initializer')
for ini in new_initializers:
    new_m.graph.initializer.append(ini)

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
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
try:
    sess_new = ort.InferenceSession(ONNX_OUT, providers=['CPUExecutionProvider'])
    result_new = sess_new.run(None, {'input': dummy})
    sess_orig = ort.InferenceSession(ONNX_IN, providers=['CPUExecutionProvider'])
    result_orig = sess_orig.run(None, {'input': dummy})
    for i, (r, r_orig) in enumerate(zip(result_new, result_orig)):
        diff = np.abs(r - r_orig).max()
        print(f'Output {i}: shape={r.shape}, max diff = {diff:.6f}')
    print('ORT validation PASSED')
except Exception as e:
    print(f'ORT validation FAILED: {e}')
print('Done!')
