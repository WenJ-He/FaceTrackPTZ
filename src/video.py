"""Video stream reader with automatic reconnection."""

from __future__ import annotations

import time
import threading
from typing import Optional

import cv2
import numpy as np

from .config import Config
from . import logger as log


class VideoReader:
    """Reads frames from an RTSP video stream with auto-reconnect."""

    def __init__(self, url: str, reconnect_interval_ms: int = 5000) -> None:
        self._url = url
        self._reconnect_interval = reconnect_interval_ms / 1000.0
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def open(self) -> bool:
        if self._cap is not None:
            self._cap.release()
        log.log("Opening video stream", result=self._url)
        self._cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if self._cap.isOpened():
            self._connected = True
            log.log("Video stream opened", result="OK")
            return True
        self._connected = False
        log.log("Failed to open video stream", result="FAIL")
        return False

    def open_with_retry(self, max_retries: int = 0) -> bool:
        retries = 0
        while True:
            if self.open():
                return True
            retries += 1
            if max_retries > 0 and retries >= max_retries:
                return False
            log.log(f"Reconnecting in {self._reconnect_interval}s", result="RETRY")
            time.sleep(self._reconnect_interval)

    def read(self) -> Optional[np.ndarray]:
        """Read a single frame. Returns None on failure (triggers reconnect)."""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._connected = False
                return None
            return frame

    def read_with_reconnect(self) -> np.ndarray:
        """Read a frame, auto-reconnecting if needed. Blocks until success."""
        while True:
            frame = self.read()
            if frame is not None:
                return frame
            log.log("Lost video stream, reconnecting", result="RECONNECT")
            time.sleep(self._reconnect_interval)
            self.open()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Skip buffered frames and return the most recent one."""
        latest = None
        for _ in range(5):
            frame = self.read()
            if frame is not None:
                latest = frame
        return latest
