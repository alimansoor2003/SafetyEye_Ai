from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent


class CameraConfig(BaseModel):
    camera_id: str
    zone_id: str
    zone_label_en: str
    zone_label_ar: str
    source: Union[int, str]
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 30


class DetectionConfig(BaseModel):
    weights: str
    device: str = "cuda:0"
    imgsz: int = 640
    conf_threshold: float = 0.75
    iou_threshold: float = 0.45
    tracker: str = "bytetrack.yaml"
    monitored_violations: list[str]


class ViolationConfig(BaseModel):
    persist_seconds: float = 2.0
    clear_seconds: float = 1.5
    track_lost_seconds: float = 1.0
    cooldown_seconds: float = 120.0
    max_incidents_per_minute: int = 10


class StreamConfig(BaseModel):
    width: int = 1280
    height: int = 720
    fps: int = 15
    jpeg_quality: int = 70


class AgentConfig(BaseModel):
    # Amendment A5: Gemini primary. A6: Anthropic as the cross-provider budget fallback.
    model: str = "gemini-3.7-flash"
    fallback_model: str = "gemini-3.6-flash"
    claude_model: str = "claude-sonnet-5"
    use_fallback: bool = False
    max_tokens: int = 1500
    worker_count: int = 2
    max_retries: int = 4
    backoff_base_seconds: float = 2.0


class HardwareConfig(BaseModel):
    mqtt_enabled: bool = True
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    serial_fallback: bool = True
    serial_port: str = "COM3"
    serial_baud: int = 115200


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    history_replay: int = 25


class StorageConfig(BaseModel):
    db_path: str = "data/edgesentinel.db"
    evidence_dir: str = "evidence"
    retention_days: int = 30
    blur_faces: bool = False
    persist_detections: bool = True


class LocaleConfig(BaseModel):
    display_timezone: str = "Asia/Riyadh"


class Config(BaseModel):
    cameras: list[CameraConfig] = Field(min_length=1)
    detection: DetectionConfig
    violation: ViolationConfig
    stream: StreamConfig
    agent: AgentConfig
    hardware: HardwareConfig
    server: ServerConfig = ServerConfig()
    storage: StorageConfig
    locale: LocaleConfig

    def camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        known = ", ".join(c.camera_id for c in self.cameras)
        raise KeyError(f"camera_id {camera_id!r} not in config.yaml (have: {known})")


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh))
