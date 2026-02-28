#!/usr/bin/env python3
"""Run ablation experiments for dog facial landmark detection.

Sequentially launches training runs with different configs,
collects results, and produces a summary table.

Usage:
    python scripts/run_experiments.py                    # run all experiments
    python scripts/run_experiments.py --only A B F       # run selected experiments
    python scripts/run_experiments.py --list              # show available experiments
    python scripts/run_experiments.py --max-epochs 50    # cap epochs for quick testing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Experiment definitions: name -> CLI flags for train_dog_face_landmarks.py.
# Order matters — experiments are run in definition order.
EXPERIMENTS: dict[str, dict[str, str]] = {
    "A_paper_baseline": {
        "--experiment": "paper_baseline",
    },
    "B_wing_loss": {
        "--experiment": "paper_baseline",
        "--loss": "wing",
    },
    "C_cosine_lr": {
        "--experiment": "paper_baseline",
        "--lr-schedule": "cosine",
    },
    "D_swa": {
        "--experiment": "paper_baseline",
        "--use-swa": "",
    },
    "E_crop_jitter": {
        "--experiment": "paper_baseline",
        "--aug-crop-jitter": "",
    },
    "F_wing_cosine_swa": {
        "--experiment": "paper_baseline",
        "--loss": "wing",
        "--lr-schedule": "cosine",
        "--use-swa": "",
    },
    "G_all_improvements": {
        "--experiment": "paper_baseline",
        "--loss": "wing",
        "--lr-schedule": "cosine",
        "--use-swa": "",
        "--aug-crop-jitter": "",
    },
    "H_paper_v2s": {
        "--experiment": "paper_baseline_v2s",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List experiments and exit")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only these experiments (by ID prefix, e.g. A B F)")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Cap total epochs for quick testing")
    p.add_argument("--out-base", type=Path, default=Path("artifacts/experiments"),
                   help="Base directory for experiment outputs")
    return p.parse_args()


def list_experiments() -> None:
    print(f"\n{'ID':<25} {'Description'}")
    print("-" * 60)
    descs = {
        "A_paper_baseline":     "Paper recipe with EfficientNetB2",
        "B_wing_loss":          "Paper baseline + Wing loss",
        "C_cosine_lr":          "Paper baseline + cosine LR decay",
        "D_swa":                "Paper baseline + SWA",
        "E_crop_jitter":        "Paper baseline + crop jitter",
        "F_wing_cosine_swa":    "Wing + cosine + SWA combined",
        "G_all_improvements":   "All improvements combined",
        "H_paper_v2s":          "Paper baseline with EfficientNetV2S",
    }
    for name in EXPERIMENTS:
        print(f"  {name:<23} {descs.get(name, '')}")
    print()


def run_experiment(
    name: str,
    extra_args: dict[str, str],
    out_dir: Path,
    max_epochs: int | None,
) -> dict:
    cmd = [
        sys.executable, "scripts/train_dog_face_landmarks.py",
        "--out-dir", str(out_dir),
    ]
    for k, v in extra_args.items():
        cmd.append(k)
        if v:
            cmd.append(v)

    if max_epochs is not None:
        cmd.extend(["--epochs", str(max_epochs)])

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    start = datetime.now()
    result = subprocess.run(cmd)
    elapsed = (datetime.now() - start).total_seconds()

    # Collect results from metadata.
    meta_path = out_dir / "model_metadata.json"
    metrics = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        metrics = meta.get("validation_metrics", {})

    return {
        "name": name,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()

    if args.list:
        list_experiments()
        return

    # Filter experiments if --only is provided.
    to_run = list(EXPERIMENTS.keys())
    if args.only:
        prefixes = [p.upper() for p in args.only]
        to_run = [n for n in to_run if any(n.upper().startswith(p) for p in prefixes)]
        if not to_run:
            print(f"No experiments matched --only {args.only}")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = args.out_base / timestamp

    results = []
    for name in to_run:
        out_dir = base_out / name
        r = run_experiment(name, EXPERIMENTS[name], out_dir, args.max_epochs)
        results.append(r)

        nme_iod = r["metrics"].get("landmark_nme_iod", "N/A")
        nme_crop = r["metrics"].get("landmark_nme", "N/A")
        status = "OK" if r["returncode"] == 0 else "FAIL"
        if isinstance(nme_iod, float):
            nme_iod = f"{nme_iod:.4f}"
        if isinstance(nme_crop, float):
            nme_crop = f"{nme_crop:.6f}"
        print(f"\n>> {name}: {status} | NME_IOD={nme_iod} | NME_crop={nme_crop} "
              f"| time={r['elapsed_seconds'] / 60:.1f}m")

    # Save summary.
    summary_path = base_out / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))

    # Print table.
    print(f"\n{'=' * 80}")
    print(f"{'Experiment':<28} {'NME_IOD':>10} {'NME_crop':>10} {'Status':>8} {'Time':>10}")
    print(f"{'-' * 80}")
    for r in results:
        nme_iod = r["metrics"].get("landmark_nme_iod", "N/A")
        nme_crop = r["metrics"].get("landmark_nme", "N/A")
        status = "OK" if r["returncode"] == 0 else "FAIL"
        time_str = f"{r['elapsed_seconds'] / 60:.1f}m"
        if isinstance(nme_iod, float):
            nme_iod = f"{nme_iod:.4f}"
        if isinstance(nme_crop, float):
            nme_crop = f"{nme_crop:.6f}"
        print(f"  {r['name']:<26} {nme_iod:>10} {nme_crop:>10} {status:>8} {time_str:>10}")
    print(f"{'=' * 80}")
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
