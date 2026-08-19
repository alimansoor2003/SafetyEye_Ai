from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DetectionConfig

log = logging.getLogger(__name__)

PERSON = "Person"
HARDHAT = "Hardhat"
NO_HARDHAT = "NO-Hardhat"
SAFETY_VEST = "Safety Vest"
NO_SAFETY_VEST = "NO-Safety Vest"

CANONICAL = {
    "person": PERSON,
    "hardhat": HARDHAT,
    "helmet": HARDHAT,
    "no hardhat": NO_HARDHAT,
    "nohardhat": NO_HARDHAT,
    "no helmet": NO_HARDHAT,
    "safety vest": SAFETY_VEST,
    "vest": SAFETY_VEST,
    "no safety vest": NO_SAFETY_VEST,
    "nosafety vest": NO_SAFETY_VEST,
    "no vest": NO_SAFETY_VEST,
}

# Deliberately ignored. The spec's §1 class list assumed the 10-class Roboflow set; dataset v30
# ships 25, adding vehicle and equipment subtypes. None of them are actionable for PPE compliance.
IGNORED = {
    "mask", "no mask", "nomask", "machinery", "vehicle", "safety cone", "cone",
    "excavator", "gloves", "ladder", "suv", "bus", "dump truck", "fire hydrant",
    "mini van", "sedan", "semi", "trailer", "truck and trailer", "truck", "van",
    "wheel loader",
}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    label: str
    conf: float
    track_id: int | None = None

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


def normalize_label(raw: str) -> str | None:
    key = raw.strip().lower().replace("_", " ").replace("-", " ")
    key = " ".join(key.split())
    if key in IGNORED:
        return None
    return CANONICAL.get(key)


def assert_cuda(device: str) -> None:
    """Print the resolved GPU/CUDA build and hard-fail rather than silently using CPU."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is not installed. Install it from the CUDA wheel index BEFORE requirements.txt "
            "(cu121 for RTX 20/30/40, cu128 for RTX 50/Blackwell) — see README.md. Note that torch "
            "publishes no wheels for Python 3.14; use Python 3.12."
        ) from exc

    available = torch.cuda.is_available()
    log.info("torch %s | compiled CUDA %s | cuda.is_available()=%s",
             torch.__version__, torch.version.cuda, available)

    if device.startswith("cpu"):
        log.warning("device is %r — running on CPU, throughput will not meet the 15fps target", device)
        return

    if not available:
        raise RuntimeError(
            f"config requests device={device!r} but torch.cuda.is_available() is False. "
            f"This torch build reports CUDA {torch.version.cuda!r}. Reinstall from the CUDA wheel "
            f"index matching your GPU (cu121 for RTX 20/30/40, cu128 for RTX 50/Blackwell). "
            f"Refusing to fall back to CPU."
        )

    index = int(device.split(":")[1]) if ":" in device else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise RuntimeError(f"device={device!r} requested but only {count} CUDA device(s) present")

    log.info("CUDA device %d: %s", index, torch.cuda.get_device_name(index))


class Detector:
    """YOLOv8 + ByteTrack. Tracks every class; only Person IDs are load-bearing downstream."""

    def __init__(self, cfg: DetectionConfig):
        self.cfg = cfg
        assert_cuda(cfg.device)

        weights = Path(cfg.weights)
        if not weights.is_absolute():
            weights = Path(__file__).resolve().parent.parent / weights
        if not weights.exists():
            raise FileNotFoundError(
                f"weights not found at {weights}. Download the Roboflow Construction Site Safety "
                f"YOLOv8 PPE weights and place them there."
            )

        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.model.to(cfg.device)
        self.names: dict[int, str] = self.model.names
        self._warn_unmapped()

    def _warn_unmapped(self) -> None:
        unmapped = [n for n in self.names.values() if normalize_label(n) is None
                    and n.strip().lower().replace("_", " ").replace("-", " ") not in IGNORED]
        if unmapped:
            log.warning("weights expose classes with no canonical mapping (ignored): %s", unmapped)
        mapped = sorted({normalize_label(n) for n in self.names.values()} - {None})
        log.info("active classes: %s", mapped)

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.cfg.tracker,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(classes)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, cls, tid in zip(xyxy, confs, classes, ids):
            label = normalize_label(self.names.get(int(cls), ""))
            if label is None:
                continue
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    label=label,
                    conf=float(conf),
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return detections
