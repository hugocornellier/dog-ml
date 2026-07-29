"""Compare a running candidate's phase-2 curve against the baseline's, epoch for epoch.

The question during a long fine-tune is not "what is the number now" but "is the
gap to the baseline closing, holding, or widening". Reading that off two logs by
hand invites interpolation mistakes, so this lines the two curves up on matched
phase-2 epochs and prints the trend.

Usage:
  python scripts/track_gap.py artifacts/pareto/taper_full/train_log.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE_LOG = REPO / "artifacts" / "small_v3large_384_long" / "train_log.csv"


def phases(path: Path) -> tuple[list[float], list[float]]:
    """Split a train_log.csv into phase-1 and phase-2 val NME_IOD series.

    The logger drops the `lr` column on some rows, so the column index for
    val_landmark_nme_iod depends on the row width. Phase 2 is detected by the
    epoch counter restarting.
    """
    rows = [r for r in csv.reader(open(path)) if r and r[0] != "epoch"]
    p1: list[float] = []
    p2: list[float] = []
    cur, last = p1, -1
    for r in rows:
        ep = int(r[0])
        if ep < last:
            cur = p2
        last = ep
        cur.append(float(r[5]) if len(r) == 7 else float(r[6]))
    return p1, p2


def main():
    cand_log = Path(sys.argv[1])
    b1, b2 = phases(BASELINE_LOG)
    c1, c2 = phases(cand_log)

    print(f"phase 1: baseline best {min(b1):.4f} over {len(b1)} ep | "
          f"candidate best {min(c1):.4f} over {len(c1)} ep "
          f"({min(c1) - min(b1):+.4f})")

    if not c2:
        print("phase 2 has not started yet")
        return

    print(f"\nphase 2, matched epochs (candidate - baseline):")
    print(f"  {'epoch':>6s} {'baseline':>9s} {'candidate':>10s} {'gap':>8s}")
    marks = [e for e in (0, 15, 30, 50, 75, 100, 150, 200, 250, 300, 350, 399,
                         450, 500, 550, 599) if e < len(c2)]
    if len(c2) - 1 not in marks:
        marks.append(len(c2) - 1)
    for e in marks:
        if e < len(b2):
            print(f"  {e:6d} {b2[e]:9.4f} {c2[e]:10.4f} {c2[e] - b2[e]:+8.4f}")
        else:
            print(f"  {e:6d} {'--':>9s} {c2[e]:10.4f} "
                  f"{'(past baseline schedule)':>8s}")

    bb, cb = min(b2), min(c2)
    print(f"\nbest so far: baseline {bb:.4f} (final) | candidate {cb:.4f} "
          f"({cb - bb:+.4f})")
    print(f"candidate must reach <= 8.5664 on the converted TFLite to ship.")


if __name__ == "__main__":
    main()
