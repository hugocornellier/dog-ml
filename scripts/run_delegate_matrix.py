"""Run the delegate-engagement matrix with each cell in its own process.

test_delegate_engagement.py runs every (model, backend) pair in one process, which
is fine until a backend hangs. The Metal delegate on the dynamic-shape landmark
graph fails interpreter creation and then blocks forever in `mutex.cc RAW: Lock
blocking`, taking the whole run with it.

So each cell is a fresh subprocess with a hard timeout, and the outcome is one of
engaged / no-op / create-failed / hang. A hang is itself a result worth recording.

Caveat this driver cannot remove: if a GPU-using training job is active, the Metal
and CoreML rows are confounded. Check for that before trusting them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CELL = r"""
import sys, json, numpy as np
sys.path.insert(0, {scripts!r})
from pareto_harness import load_val_cache
import test_delegate_engagement as T

crops, _ = load_val_cache()
x = np.asarray(crops[0:1])
model = __import__("pathlib").Path(sys.argv[1])
backend = sys.argv[2]

ref, err = T._cpu_reference(model, x)
if err:
    print(json.dumps({{"outcome": "cpu-reference-failed", "detail": err}})); raise SystemExit
if backend == "cpu":
    print(json.dumps({{"outcome": "ok", "dev": 0.0}})); raise SystemExit

fn = dict(T.BACKENDS)[backend]
out, err = fn(model, x)
if err:
    print(json.dumps({{"outcome": "create-failed", "detail": err}})); raise SystemExit
dev = float(np.abs(out - ref).max())
print(json.dumps({{"outcome": "no-op" if dev == 0.0 else "engaged", "dev": dev}}))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--backends", default="xnnpack,metal_gpu,coreml")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    src = CELL.format(scripts=str(REPO / "scripts"))
    cell_path = Path("/tmp/_delegate_cell.py")
    cell_path.write_text(src)

    print(f"{'model':40s} {'backend':10s} {'outcome':16s} {'dev':>12s}")
    for m in args.models:
        for backend in args.backends.split(","):
            try:
                p = subprocess.run(
                    [sys.executable, str(cell_path), str(m), backend],
                    capture_output=True, text=True, timeout=args.timeout,
                    # Inherit the real environment. A stripped env (PATH and
                    # TF_CPP_MIN_LOG_LEVEL only) removes HOME and TMPDIR, and the
                    # CoreML delegate needs a writable temp location to compile
                    # into; without one it blocks for 10+ minutes instead of
                    # failing fast. That produced a spurious "CoreML hangs"
                    # result until it was tracked back to this line.
                    env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
                )
                line = [l for l in p.stdout.splitlines() if l.startswith("{")]
                if not line:
                    res = {"outcome": "no-output", "detail": p.stderr[-200:]}
                else:
                    res = json.loads(line[-1])
            except subprocess.TimeoutExpired:
                res = {"outcome": "HANG", "detail": f">{args.timeout}s"}
            dev = res.get("dev")
            devs = f"{dev:12.3e}" if isinstance(dev, float) else f"{'--':>12s}"
            print(f"{m.name:40s} {backend:10s} {res['outcome']:16s} {devs}"
                  + (f"  {res.get('detail','')}" if "detail" in res else ""))


if __name__ == "__main__":
    main()
