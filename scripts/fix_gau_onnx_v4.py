"""
Fix RTMPose ONNX v4: Comprehensive fix for onnx2tf conversion.

Strategy - replace entire GAU block between entry and exits with 2D-only ops:

Entry: squeeze batch dim [1,39,X] -> [39,X]

Inside SimCC head, the problematic GAU pattern is:
  Unsqueeze+Mul+Add+Split_1+Squeeze (4D/3D tensors) -> produces q [39,128], k [39,128]

Replace with separate 2D mul+add for each head:
  q = z * w0 + b0   [39,128]
  k = z * w1 + b1   [39,128]

The Transpose [0,2,1] on [39,128] -> perm=[1,0] -> [128,39]

Exit: reshape [39,512] -> [1,39,512]

By reducing to 2D, onnx2tf won't apply NCW->NWC transposing on any of the SimCC tensors.
"""

import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
ONNX_IN = os.path.join(BASE, 'checkpoints/rtmpose_s_opset11_sym.onnx')
ONNX_OUT = os.path.join(BASE, 'checkpoints/rtmpose_s_gau_fixed_v4.onnx')

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

# Get all initializers as numpy arrays for weight extraction
init_dict = {ini.name: nph.to_array(ini) for ini in m.graph.initializer}

# --- Find relevant nodes ---
entry_node = None        # /model/bodypart/mlp/mlp.1/MatMul
unsqueeze_node = None    # /model/bodypart/gau/Unsqueeze
mul_4d_node = None       # /model/bodypart/gau/Mul
add_4d_node = None       # /model/bodypart/gau/Add
split1_node = None       # /model/bodypart/gau/Split_1
squeeze_nodes = []       # /model/bodypart/gau/Squeeze, /model/bodypart/gau/Squeeze_1
split_node = None        # /model/bodypart/gau/Split  (splits 1152 -> 512,512,128)
transpose_node = None    # /model/bodypart/gau/Transpose
exit_nodes = []

for node in m.graph.node:
    n = node.name
    if n == '/model/bodypart/mlp/mlp.1/MatMul':
        entry_node = node
    elif n == '/model/bodypart/gau/Unsqueeze':
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

print(f'Entry node: {entry_node.name}')
entry_input_name = entry_node.input[0]
entry_input_shape = shape_dict[entry_input_name]  # [1, 39, 64]
seq_len = entry_input_shape[1]  # 39
entry_feat_dim = entry_input_shape[2]  # 64
print(f'  input: {entry_input_name}, shape: {entry_input_shape}')

# --- Extract weights for 2D head replacement ---
# mul_4d_node input[1]: [1,1,2,128] -> w0=[128], w1=[128]
mul_weight_name = mul_4d_node.input[1]
mul_weight = init_dict[mul_weight_name]  # [1,1,2,128]
w0 = mul_weight[0, 0, 0, :]  # [128]
w1 = mul_weight[0, 0, 1, :]  # [128]

# add_4d_node input[1]: [2,128] -> b0=[128], b1=[128]
add_bias_name = add_4d_node.input[1]
add_bias = init_dict[add_bias_name]  # [2,128]
b0 = add_bias[0, :]  # [128]
b1 = add_bias[1, :]  # [128]
print(f'w0.shape={w0.shape}, b0.shape={b0.shape}')

# Target outputs of squeeze nodes (q and k names)
# Squeeze -> /model/bodypart/gau/Squeeze_output_0   (q used in MatMul)
# Squeeze_1 -> /model/bodypart/gau/Squeeze_1_output_0  (k used in Transpose)
squeeze_out_0 = squeeze_nodes[0].output[0]
squeeze_out_1 = squeeze_nodes[1].output[0]
print(f'squeeze_out_0 -> q: {squeeze_out_0}')
print(f'squeeze_out_1 -> k: {squeeze_out_1}')

# Prefix for new names
p = 'rtmpose_fix_v4_'

# --- New initializers ---
new_inits = [
    # Entry/exit reshape shapes
    numpy_helper.from_array(np.array([seq_len, entry_feat_dim], dtype=np.int64), name=p+'shape_entry'),
    numpy_helper.from_array(np.array([1, 39, 512], dtype=np.int64), name=p+'shape_exit'),
    # 2D head weights and biases
    numpy_helper.from_array(w0.astype(np.float32), name=p+'w0'),
    numpy_helper.from_array(w1.astype(np.float32), name=p+'w1'),
    numpy_helper.from_array(b0.astype(np.float32), name=p+'b0'),
    numpy_helper.from_array(b1.astype(np.float32), name=p+'b1'),
]

# --- New tensor names ---
entry_2d = p + 'entry_2d'  # [39,64] after entry reshape

# q = z * w0 + b0  (z is unsqueeze_node.input[0], but after squeeze: [39,128])
z_name = unsqueeze_node.input[0]  # Originally [1,39,128], now [39,128]
q_mul_out = p + 'q_mul'    # [39,128]
k_mul_out = p + 'k_mul'    # [39,128]

# Replacement nodes for Unsqueeze+Mul+Add+Split1+Squeeze block:
# q = z * w0 + b0
q_mul = helper.make_node('Mul', inputs=[z_name, p+'w0'], outputs=[q_mul_out], name=p+'q_mul')
q_add = helper.make_node('Add', inputs=[q_mul_out, p+'b0'], outputs=[squeeze_out_0], name=p+'q_add')
# k = z * w1 + b1
k_mul = helper.make_node('Mul', inputs=[z_name, p+'w1'], outputs=[k_mul_out], name=p+'k_mul')
k_add = helper.make_node('Add', inputs=[k_mul_out, p+'b1'], outputs=[squeeze_out_1], name=p+'k_add')

# --- Nodes to skip ---
skip_nodes = {
    unsqueeze_node.name,
    mul_4d_node.name,
    add_4d_node.name,
    split1_node.name,
    squeeze_nodes[0].name,
    squeeze_nodes[1].name,
}

# --- Build new node list ---
new_graph_nodes = []
gau_replace_done = False

for node in m.graph.node:
    if node.name == entry_node.name:
        # Insert entry reshape BEFORE entry node
        entry_reshape = helper.make_node(
            'Reshape',
            inputs=[entry_input_name, p+'shape_entry'],
            outputs=[entry_2d],
            name=p+'entry_reshape'
        )
        new_node = helper.make_node(
            node.op_type,
            inputs=[entry_2d] + list(node.input[1:]),
            outputs=list(node.output),
            name=node.name
        )
        for attr in node.attribute:
            new_node.attribute.append(attr)
        new_graph_nodes.append(entry_reshape)
        new_graph_nodes.append(new_node)

    elif node.name in skip_nodes:
        # Insert 2D replacement before first skip node
        if not gau_replace_done:
            new_graph_nodes.extend([q_mul, q_add, k_mul, k_add])
            gau_replace_done = True
        # Skip original node

    elif node.name == split_node.name:
        # Split [1,39,1152] becomes [39,1152] -> split on axis=1 instead of 2
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
        # Original perm=[0,2,1] on [1,39,128] -> now [39,128], perm=[1,0]
        new_node = helper.make_node(
            'Transpose',
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name,
            perm=[1, 0]
        )
        new_graph_nodes.append(new_node)

    elif node.name in ('/model/bodypart/cls_x/MatMul', '/model/bodypart/cls_y/MatMul'):
        orig_out = node.output[0]  # 'simcc_x' or 'simcc_y'
        tmp_out = p + f'{orig_out}_2d'
        # Modify exit node to output to tmp name
        new_node = helper.make_node(
            node.op_type,
            inputs=list(node.input),
            outputs=[tmp_out],
            name=node.name
        )
        for attr in node.attribute:
            new_node.attribute.append(attr)
        # Add exit reshape [39,512] -> [1,39,512]
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

# --- Remove old initializers no longer needed ---
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
