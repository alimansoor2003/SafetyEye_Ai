"""Measure the monitored classes across confidence thresholds.

config.yaml sets a single global conf_threshold (spec §1: 0.75). Validation defaults to conf=0.001,
so the headline precision/recall from training is NOT what the system will do at runtime. This
sweeps the real operating points so the threshold is chosen from data.

    python scripts/threshold_sweep.py --data datasets/construction-site-safety/data.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MONITORED = ["Person", "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"]
VIOLATIONS = ["NO-Hardhat", "NO-Safety Vest"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default=str(REPO / "models" / "ppe_yolov8.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", nargs="+", type=float,
                        default=[0.25, 0.40, 0.50, 0.60, 0.75])
    args = parser.parse_args()

    from ultralytics import YOLO

    rows: dict[float, dict[str, tuple[float, float]]] = {}
    for conf in args.conf:
        metrics = YOLO(args.weights).val(
            data=args.data, device=args.device, conf=conf, verbose=False, plots=False
        )
        names = metrics.names
        per_class = {}
        for i, class_index in enumerate(metrics.ap_class_index):
            per_class[names[class_index]] = (float(metrics.box.p[i]), float(metrics.box.r[i]))
        rows[conf] = per_class

    print("\n\nPRECISION / RECALL BY CONFIDENCE THRESHOLD\n")
    header = f"{'class':<17}" + "".join(f"{c:>16.2f}" for c in args.conf)
    print(header)
    print(f"{'':<17}" + "".join(f"{'P / R':>16}" for _ in args.conf))
    print("-" * len(header))
    for name in MONITORED:
        cells = ""
        for conf in args.conf:
            p, r = rows[conf].get(name, (float("nan"), float("nan")))
            cells += f"{p:>7.2f} /{r:>6.2f}  "
        print(f"{name:<17}{cells}")

    print("\nMISSED VIOLATIONS (1 - recall) — the false-negative rate that matters for HSE\n")
    for name in VIOLATIONS:
        cells = ""
        for conf in args.conf:
            _, r = rows[conf].get(name, (float("nan"), float("nan")))
            cells += f"{(1 - r) * 100:>13.0f}%  "
        print(f"{name:<17}{cells}")
    print(f"\n{'threshold:':<17}" + "".join(f"{c:>15.2f} " for c in args.conf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
