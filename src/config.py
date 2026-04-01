"""Configuration loading and validation for FaceTrack PTZ."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml


DEFAULTS = {
    "scan.row_bucket": 80,
    "scan.max_zoom_stages": 5,
    "recognition.recognition_accept_threshold": 0.6,
    "recognition.min_similarity_gain": 0.02,
    "scan.ptz_retry": 3,
    "detection.detect_interval_ms": 500,
    "scan.ptz_stable_wait_ms": 1500,
    "recognition.top_k": 5,
    "reconnect.interval_ms": 5000,
    "logging.level": "INFO",
    "overlay.port": 8080,
    "overlay.host": "0.0.0.0",
    "device.box_percent": 0.5,
    "device.channel": 2,
}


class Config:
    """Application configuration with nested access."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        self._apply_defaults()
        self._validate()

    def _apply_defaults(self) -> None:
        for dotted_key, default_value in DEFAULTS.items():
            keys = dotted_key.split(".")
            d = self._data
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            if keys[-1] not in d:
                d[keys[-1]] = default_value

    def _validate(self) -> None:
        required_sections = ["video", "device", "triton", "vector_db"]
        for section in required_sections:
            if section not in self._data:
                raise ValueError(f"Missing required config section: {section}")

        v = self._data["video"]
        if not v.get("panoramic_url"):
            raise ValueError("video.panoramic_url is required")

        d = self._data["device"]
        for field in ("address", "username", "password"):
            if not d.get(field):
                raise ValueError(f"device.{field} is required")

        t = self._data["triton"]
        if not t.get("url"):
            raise ValueError("triton.url is required")

        if self._data["scan"]["max_zoom_stages"] < 1:
            raise ValueError("scan.max_zoom_stages must be >= 1")
        if self._data["detection"]["detect_interval_ms"] < 100:
            raise ValueError("detection.detect_interval_ms must be >= 100")

    def get(self, dotted_key: str, default: Any = None) -> Any:
        keys = dotted_key.split(".")
        d = self._data
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    @property
    def raw(self) -> Dict[str, Any]:
        return self._data


def load_config(path: str) -> Config:
    """Load YAML config file and return a Config object."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return Config(data)
