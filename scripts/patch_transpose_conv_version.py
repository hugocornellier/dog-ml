"""Test whether TRANSPOSE_CONV's op version is the only thing blocking GPU delegation.

The Metal delegate refuses the deconv head with:

    TRANSPOSE_CONV: Max version supported: 3. Requested version 4.

The models declare version 4 and, crucially, they *use* the v4 feature: the converter
folded `deconv -> BatchNorm -> ReLU` into the op, giving 4 inputs (with a folded-BN
bias) and `fusedActivationFunction = 1` (RELU). So "the dylib is just stale" is not
obviously right, and relabelling the version alone would silently drop the ReLU.

This separates the two possibilities by producing a patched model with the activation
*unfused* in the only way a pure flatbuffer edit can: set
`fusedActivationFunction = NONE` and downgrade the opcode to version 3.

The patched model is numerically WRONG on purpose (the ReLUs are gone, not moved). It
is a probe, not a candidate. What it answers:

  * If Metal now accepts the TRANSPOSE_CONV ops, the version gate is the only
    delegation blocker, and the real fix is to emit deconv + a separate RELU op so the
    delegate can take both. That is worth pursuing.
  * If Metal still refuses them, the delegate lacks transpose-conv support for reasons
    beyond the fused activation, and unfusing buys nothing.

Either way the output deviation should be large, which doubles as a check that the
patch actually took effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema

TRANSPOSE_CONV = 67
ACTIVATION_NONE = 0


def patch(src: Path, dst: Path, target_version: int = 3,
          clear_activation: bool = True) -> None:
    buf = bytearray(src.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))

    tc_indices = set()
    for i, oc in enumerate(model.operatorCodes):
        code = oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode
        if code == TRANSPOSE_CONV:
            print(f"opcode[{i}] TRANSPOSE_CONV version {oc.version} -> {target_version}")
            oc.version = target_version
            tc_indices.add(i)

    if not tc_indices:
        raise SystemExit("no TRANSPOSE_CONV opcode in this model")

    touched = 0
    for sub in model.subgraphs:
        for op in sub.operators:
            if op.opcodeIndex in tc_indices and clear_activation:
                o = op.builtinOptions
                if o is not None and o.fusedActivationFunction != ACTIVATION_NONE:
                    print(f"  op fusedActivationFunction "
                          f"{o.fusedActivationFunction} -> {ACTIVATION_NONE}")
                    o.fusedActivationFunction = ACTIVATION_NONE
                    touched += 1
    print(f"cleared activation on {touched} ops")

    b = flatbuffers.Builder(0)
    b.Finish(model.Pack(b), file_identifier=b"TFL3")
    dst.write_bytes(bytes(b.Output()))
    print(f"wrote {dst} ({dst.stat().st_size/1024/1024:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--version", type=int, default=3)
    ap.add_argument("--keep-activation", action="store_true",
                    help="downgrade the version but leave the fused ReLU in place, "
                         "which is invalid but isolates the version check itself")
    args = ap.parse_args()
    patch(args.src, args.dst, args.version, clear_activation=not args.keep_activation)


if __name__ == "__main__":
    main()
