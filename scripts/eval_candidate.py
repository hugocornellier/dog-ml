"""Final three-axis verdict for a candidate against the shipped baseline.

Produces the whole Pareto table in one go so the decision is not assembled by
hand from several scripts:

  size          from the converted .tflite on disk
  accuracy      full 480-image val split, on the CONVERTED model, in both the
                crop-space metric the training script reports and the
                absolute-image-pixel metric dog_detection publishes
  latency       median invoke() through flutter_litert 3.6.0's own dylib, the
                runtime the package actually ships, same 4 threads as baseline

The accuracy call is decided by a *paired* per-image comparison, not by whether
two means look different. On 480 images the standard error of the mean is ~0.18
NME, so unpaired eyeballing cannot resolve the ~0.1 differences that matter here;
the paired test on the same images is roughly an order of magnitude tighter.

Usage:
  python scripts/eval_candidate.py --tflite artifacts/pareto/taper_full/*.tflite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from pareto_harness import (  # noqa: E402
    load_val_cache, load_abs_refs, summarize, nme_iod_abs, paired_delta,
)
from bench_litert_macos import run as litert_run  # noqa: E402

BASELINE_TFLITE = (REPO / "artifacts" / "small_v3large_384_long"
                   / "dog_face_landmarks_384_float16.tflite")
STATIC_TFLITE = REPO / "artifacts" / "pareto" / "static_fp16.tflite"

# The bar. Baseline crop-space NME_IOD on the converted model, all 480 images.
BASELINE_NME = 8.5664


def evaluate(path: Path, crops, gt, boxes, gtabs, runs: int):
    lat, preds = litert_run(path, crops, threads=4, warmup=10, runs=runs,
                            collect=True)
    preds = np.clip(preds, 0.0, 1.0)
    acc = summarize(gt, preds)
    return {
        "name": path.name,
        "size_mb": path.stat().st_size / 1024 / 1024,
        "nme_crop": acc["nme_iod"],
        "nme_abs": nme_iod_abs(preds, boxes, gtabs),
        "regions": acc["regions"],
        "median_ms": lat["median_ms"],
        "p10_ms": lat["p10_ms"],
        "p90_ms": lat["p90_ms"],
    }, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", type=Path, required=True,
                    help="candidate .tflite to judge")
    ap.add_argument("--label", default=None)
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()

    crops_mm, gt = load_val_cache()
    crops = np.asarray(crops_mm)
    boxes, gtabs = load_abs_refs()

    rows = []
    preds = {}
    targets = [("baseline (shipped, dynamic)", BASELINE_TFLITE),
               ("static re-export", STATIC_TFLITE),
               (args.label or args.tflite.stem, args.tflite)]
    for label, p in targets:
        if not p.exists():
            print(f"  skip {label}: {p} missing")
            continue
        r, pr = evaluate(p, crops, gt, boxes, gtabs, args.runs)
        r["label"] = label
        rows.append(r)
        preds[label] = pr

    print()
    print(f"{'candidate':30s} {'size MB':>8s} {'NME crop':>9s} {'NME abs':>8s} "
          f"{'median ms':>10s}")
    for r in rows:
        print(f"{r['label']:30s} {r['size_mb']:8.3f} {r['nme_crop']:9.4f} "
              f"{r['nme_abs']:8.4f} {r['median_ms']:10.2f}")

    base_label = targets[0][0]
    cand_label = targets[-1][0]
    if base_label in preds and cand_label in preds and cand_label != base_label:
        d = paired_delta(gt, preds[base_label], preds[cand_label])
        print()
        print("Paired per-image NME delta (candidate - baseline):")
        print(f"  mean {d['mean_delta']:+.4f}  +- {d['sem']:.4f} (sem)  "
              f"t = {d['t']:+.2f}")
        print(f"  better on {d['n_better']}/480, worse on {d['n_worse']}/480")

        cand = next(r for r in rows if r["label"] == cand_label)
        base = next(r for r in rows if r["label"] == base_label)
        # |t| < 2 means the 480-image split cannot tell these apart.
        within_noise = abs(d["t"]) < 2.0
        acc_ok = d["mean_delta"] <= 0 or within_noise
        size_ok = cand["size_mb"] <= base["size_mb"] + 0.05
        speed_ok = cand["median_ms"] <= base["median_ms"]

        print()
        print(f"  accuracy: {'OK' if acc_ok else 'REGRESSION'}"
              f"{' (within noise)' if within_noise and d['mean_delta'] > 0 else ''}")
        print(f"  size:     {'OK' if size_ok else 'REGRESSION'}")
        print(f"  latency:  {'OK' if speed_ok else 'REGRESSION'}")
        print()
        print("  VERDICT:", "PARETO WIN, ship" if (acc_ok and size_ok and speed_ok)
              else "TRADE, do not ship")

        print()
        print("Per-region NME_IOD (crop-space):")
        print(f"  {'region':16s} {'baseline':>9s} {'candidate':>10s} {'delta':>8s}")
        for reg in base["regions"]:
            b, c = base["regions"][reg], cand["regions"][reg]
            print(f"  {reg:16s} {b:9.2f} {c:10.2f} {c - b:+8.2f}")


if __name__ == "__main__":
    main()
