"""Download the Roboflow 'Construction Site Safety' dataset in YOLOv8 format.

Needs a free Roboflow account. Put your key in .env as ROBOFLOW_API_KEY — it is a training-time
credential only and is never read by the runtime pipeline.

    python scripts/fetch_dataset.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WORKSPACE = "roboflow-universe-projects"
PROJECT = "construction-site-safety"
DEFAULT_VERSION = 27


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--dest", default=str(REPO / "datasets" / "construction-site-safety"))
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        print(
            "ROBOFLOW_API_KEY is not set.\n"
            "  1. Create a free account at https://app.roboflow.com\n"
            "  2. Copy your key from Settings -> API Keys\n"
            "  3. Add ROBOFLOW_API_KEY=... to .env\n",
            file=sys.stderr,
        )
        return 1

    try:
        from roboflow import Roboflow
    except ModuleNotFoundError:
        print("pip install roboflow", file=sys.stderr)
        return 1

    project = Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(args.version).download("yolov8", location=args.dest)

    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"\ndataset at {dataset.location}")
    print(f"data.yaml  {data_yaml}")
    print(f"\nnext: python scripts/train.py --data \"{data_yaml}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
