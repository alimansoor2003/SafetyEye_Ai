"""Train YOLOv8 PPE weights and report the per-class precision the spec asks for.

    python scripts/train.py --data datasets/construction-site-safety/data.yaml

Writes the best checkpoint to models/ppe_yolov8.pt. Spec risk #2 warns that NO-Safety Vest is
typically weaker than NO-Hardhat, so this prints per-class precision at the end — read it before
trusting the single global 0.75 threshold in config.yaml.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MONITORED = {"NO-Hardhat", "NO-Safety Vest", "Hardhat", "Safety Vest", "Person"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to the dataset's data.yaml")
    parser.add_argument("--base", default="yolov8s.pt", help="starting checkpoint")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="lower to 8 if VRAM is tight")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=str(REPO / "models" / "ppe_yolov8.pt"))
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable — training on CPU would take days. Aborting.", file=sys.stderr)
        return 1
    print(f"training on {torch.cuda.get_device_name(0)}")

    data = Path(args.data).resolve()
    if not data.exists():
        print(f"no data.yaml at {data} — run scripts/fetch_dataset.py first", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(REPO / "runs"),
        name="ppe",
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\nbest checkpoint {best}\ncopied to      {out}")

    _report_per_class(YOLO(str(out)).val(data=str(data), device=args.device))
    return 0


def _report_per_class(metrics) -> None:
    print("\nper-class validation (spec risk #2 — check before trusting one global threshold)")
    print(f"{'class':<18}{'precision':>11}{'recall':>9}{'mAP50':>9}")
    names = metrics.names
    for i, class_index in enumerate(metrics.ap_class_index):
        name = names[class_index]
        marker = " *" if name in MONITORED else ""
        p, r, ap50 = metrics.box.p[i], metrics.box.r[i], metrics.box.ap50[i]
        print(f"{name:<18}{p:>11.3f}{r:>9.3f}{ap50:>9.3f}{marker}")
    print("\n* = classes this system actually acts on")


if __name__ == "__main__":
    raise SystemExit(main())
