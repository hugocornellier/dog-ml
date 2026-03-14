"""
Fix RTMPose ONNX v3: Replace the entire SimCC MLP+GAU+Linear section
with equivalent ops that don't use 3D tensors (which onnx2tf mis-handles).

The SimCC head takes backbone output and produces [1,39,512] x and y logits.
Before the GAU section, insert Reshape [1,39,D] -> [39,D] (merge batch).
After the GAU section, insert Reshape [39,D] -> [1,39,D] (restore batch).

Since batch=1 is static, this is safe.

The operations:
1. mlp/MatMul: [1,39,256] x [256,256] -> [1,39,256]
   Replace: [39,256] x [256,256] -> [39,256] (2D)
2. GAU (multiple 3D ops)
   Handle with 2D replacements
3. cls_x/MatMul: [1,39,256] x [256,512] -> [1,39,512]
   Replace: [39,256] x [256,512] -> [39,512] (2D)
4. cls_y/MatMul same

This approach: find the entry point to the SimCC head, insert Reshape,
then find the exit MatMuls and insert Reshape back.
"""

import os
import numpy as np
import onnx
import onnx.numpy_helper as nph
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

BASE = '/Users/hugocornellier/PycharmProjects/dogs-in-the-wild-ml'
ONNX_IN = os.path.join(BASE, 'checkpoints/rtmpose_s_opset11_sym.onnx')
ONNX_OUT = os.path.join(BASE, 'checkpoints/rtmpose_s_gau_fixed_v3.onnx')

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

# Find the SimCC head entry: mlp/mlp.1/MatMul
# This takes [1, 39, 256] from the backbone output (after GAU input reshape)
entry_node = None
for node in m.graph.node:
    if node.name == '/model/bodypart/mlp/mlp.1/MatMul':
        entry_node = node
        break
print(f'Entry node: {entry_node.name}, inputs={list(entry_node.input)}, output={list(entry_node.output)}')
print(f'  Input shape: {shape_dict.get(entry_node.input[0], "?")}')

# Find the exit nodes: cls_x/MatMul and cls_y/MatMul
exit_nodes = []
for node in m.graph.node:
    if node.name in ('/model/bodypart/cls_x/MatMul', '/model/bodypart/cls_y/MatMul'):
        exit_nodes.append(node)
        print(f'Exit node: {node.name}, output: {list(node.output)}, output_shape: {shape_dict.get(node.output[0], "?")}')

# Strategy: we want the entire SimCC head to work with 2D tensors [39, D]
# 1. Before mlp/mlp.1/MatMul: insert Reshape [1,39,256] -> [39,256]
# 2. After each cls_*/MatMul: insert Reshape [39,512] -> [1,39,512]
#    (the output tensor names are the original outputs: simcc_x, simcc_y)

# BUT: onnx2tf's problem is that 3D tensors are treated as NCW (channel-first).
# By making them 2D [39, D], onnx2tf treats them as [B, D] (2D matrix), no transposing.

# Entry: reshape [1,39,D] -> [39,D]
entry_input_name = entry_node.input[0]  # e.g., some 3D tensor
entry_input_shape = shape_dict.get(entry_input_name, [1, 39, 256])
seq_len = entry_input_shape[1]
entry_feat_dim = entry_input_shape[2]
print(f'Entry input: {entry_input_name}: {entry_input_shape}, seq_len={seq_len}, feat_dim={entry_feat_dim}')

# New names
reshape_entry_out = 'rtmpose_fix_v3_entry_2d'
reshape_39_D_shape_name = 'rtmpose_fix_v3_shape_seq_D'
reshape_1_39_512_shape_name = 'rtmpose_fix_v3_shape_1_39_512'

new_inits = [
    numpy_helper.from_array(np.array([seq_len, entry_feat_dim], dtype=np.int64), name=reshape_39_D_shape_name),
    numpy_helper.from_array(np.array([1, 39, 512], dtype=np.int64), name=reshape_1_39_512_shape_name),
]

# Entry reshape node
entry_reshape_node = helper.make_node(
    'Reshape',
    inputs=[entry_input_name, reshape_39_D_shape_name],
    outputs=[reshape_entry_out],
    name='rtmpose_fix_v3_entry_reshape'
)

# For exit nodes, collect original output names and create exit reshapes
exit_reshape_nodes = {}
for exit_node in exit_nodes:
    orig_out = exit_node.output[0]  # 'simcc_x' or 'simcc_y'
    tmp_out = f'rtmpose_fix_v3_{orig_out}_2d'  # [39, 512] 2D temp
    # Modify exit node to output to tmp name
    exit_reshape_nodes[exit_node.name] = (orig_out, tmp_out)

# Build new node list
new_graph_nodes = []
for node in m.graph.node:
    if node.name == entry_node.name:
        # Replace entry_node.input[0] with reshape_entry_out
        new_node = helper.make_node(
            node.op_type,
            inputs=[reshape_entry_out] + list(node.input[1:]),
            outputs=list(node.output),
            name=node.name
        )
        # Copy attributes
        for attr in node.attribute:
            new_node.attribute.append(attr)
        # Insert entry reshape BEFORE entry node
        new_graph_nodes.append(entry_reshape_node)
        new_graph_nodes.append(new_node)
    elif node.name in exit_reshape_nodes:
        orig_out, tmp_out = exit_reshape_nodes[node.name]
        # Modify exit node to output to tmp_out
        new_node = helper.make_node(
            node.op_type,
            inputs=list(node.input),
            outputs=[tmp_out],
            name=node.name
        )
        for attr in node.attribute:
            new_node.attribute.append(attr)
        new_graph_nodes.append(new_node)
        # Add reshape [39,512] -> [1,39,512]
        exit_reshape = helper.make_node(
            'Reshape',
            inputs=[tmp_out, reshape_1_39_512_shape_name],
            outputs=[orig_out],
            name=f'rtmpose_fix_v3_exit_reshape_{orig_out}'
        )
        new_graph_nodes.append(exit_reshape)
    else:
        new_graph_nodes.append(node)

print(f'Original: {len(m.graph.node)}, New: {len(new_graph_nodes)}')

# Build new model
new_m = onnx.ModelProto()
new_m.CopyFrom(m)
new_m.graph.ClearField('node')
for node in new_graph_nodes:
    new_m.graph.node.append(node)
for init in new_inits:
    new_m.graph.initializer.append(init)
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
sess_new = ort.InferenceSession(ONNX_OUT, providers=['CPUExecutionProvider'])
result_new = sess_new.run(None, {'input': dummy})
sess_orig = ort.InferenceSession(ONNX_IN, providers=['CPUExecutionProvider'])
result_orig = sess_orig.run(None, {'input': dummy})
for i, (r, r_orig) in enumerate(zip(result_new, result_orig)):
    diff = np.abs(r - r_orig).max()
    print(f'Output {i}: shape={r.shape}, max diff = {diff:.6f}')
print('Done!')
