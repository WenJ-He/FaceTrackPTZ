"""PTZ controller: coordinate calculation + Hikvision device control.

Coordinate logic ported from /mnt/e/hwj/hkctl/src/camera.cpp.
Supports two modes (configurable):
  - isapi: HTTP ISAPI calls to Hikvision device (default, self-contained)
  - subprocess: calls the compiled ctrl binary from hkctl project
"""

from __future__ import annotations

import os
import subprocess
import time
from typing import Optional, Tuple

import requests
from requests.auth import HTTPDigestAuth

from .config import Config
from .models import BBox
from . import logger as log


class PTZController:
    """Controls Hikvision PTZ camera for face zoom-in."""

    def __init__(self, config: Config) -> None:
        self._device_addr = config.get("device.address", "192.168.0.248")
        self._port = config.get("device.port", 8000)
        self._username = config.get("device.username", "admin")
        self._password = config.get("device.password", "")
        self._channel = config.get("device.channel", 2)
        self._pano_cols = config.get("device.panoramic_resolution.cols", 3632)
        self._pano_rows = config.get("device.panoramic_resolution.rows", 1632)
        self._box_percent = config.get("device.box_percent", 0.5)
        self._stable_wait = config.get("scan.ptz_stable_wait_ms", 1500) / 1000.0
        self._max_retries = config.get("scan.ptz_retry", 3)
        self._mode = config.get("ptz.mode", "isapi")
        self._ctrl_binary = config.get("ptz.ctrl_binary", "")
        self._ctrl_lib_path = config.get("ptz.ctrl_lib_path", "")
        self._process: Optional[subprocess.Popen] = None

    def connect(self) -> bool:
        """Initialize PTZ connection based on mode."""
        if self._mode == "subprocess":
            return self._connect_subprocess()
        return self._connect_isapi()

    def _connect_isapi(self) -> bool:
        try:
            url = f"http://{self._device_addr}/ISAPI/System/deviceInfo"
            resp = requests.get(
                url,
                auth=HTTPDigestAuth(self._username, self._password),
                timeout=5,
            )
            if resp.status_code in (200, 401):
                log.log("PTZ ISAPI connection established", result="OK")
                return True
            log.log(f"PTZ ISAPI unexpected status: {resp.status_code}", result="WARN")
            return True  # proceed anyway
        except Exception as e:
            log.log(f"PTZ ISAPI connection failed: {e}", result="ERROR")
            return False

    def _connect_subprocess(self) -> bool:
        if not self._ctrl_binary or not os.path.isfile(self._ctrl_binary):
            log.log(f"ctrl binary not found: {self._ctrl_binary}", result="ERROR")
            return False
        try:
            env = os.environ.copy()
            if self._ctrl_lib_path:
                env["LD_LIBRARY_PATH"] = self._ctrl_lib_path + ":" + env.get("LD_LIBRARY_PATH", "")
            self._process = subprocess.Popen(
                [self._ctrl_binary],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=os.path.dirname(self._ctrl_binary),
            )
            log.log("PTZ subprocess started", result="OK")
            return True
        except Exception as e:
            log.log(f"PTZ subprocess failed: {e}", result="ERROR")
            return False

    def calculate_coordinates(
        self,
        bbox: BBox,
        stage_num: int = 1,
    ) -> Tuple[int, int, int, int]:
        """Convert panoramic bbox to PTZ relative coordinates (0-255).

        Ported from camera.cpp. For multi-stage zoom, we progressively
        reduce the view box to zoom in tighter.
        """
        cx = bbox.cx
        cy = bbox.cy
        box_w = bbox.width
        box_h = bbox.height

        # For later stages, increase box_percent to zoom in tighter
        effective_percent = min(self._box_percent * (1.0 + 0.3 * stage_num), 1.0)
        view_w = box_w / effective_percent
        view_h = box_h / effective_percent

        view_x1 = cx - view_w / 2.0
        view_y1 = cy - view_h / 2.0
        view_x2 = cx + view_w / 2.0
        view_y2 = cy + view_h / 2.0

        # Boundary clamp to panoramic image bounds
        view_x1 = max(0, view_x1)
        view_y1 = max(0, view_y1)
        view_x2 = min(self._pano_cols, view_x2)
        view_y2 = min(self._pano_rows, view_y2)

        # Map to 0-255 relative coordinates
        rect_x1 = int(view_x1 * 255.0 / self._pano_cols)
        rect_y1 = int(view_y1 * 255.0 / self._pano_rows)
        rect_x2 = int(view_x2 * 255.0 / self._pano_cols)
        rect_y2 = int(view_y2 * 255.0 / self._pano_rows)

        # Clamp to 0-255
        rect_x1 = max(0, min(255, rect_x1))
        rect_y1 = max(0, min(255, rect_y1))
        rect_x2 = max(0, min(255, rect_x2))
        rect_y2 = max(0, min(255, rect_y2))

        return rect_x1, rect_y1, rect_x2, rect_y2

    def move_to_target(self, target_id: int, rect: Tuple[int, int, int, int]) -> bool:
        """Send PTZ move command with retry."""
        for attempt in range(1, self._max_retries + 1):
            success = self._send_ptz_command(target_id, rect)
            if success:
                log.log(
                    f"PTZ moved to target {target_id}",
                    target_id=target_id,
                    result="OK",
                    extra_data={"rect": list(rect), "attempt": attempt},
                )
                return True
            log.log(
                f"PTZ attempt {attempt}/{self._max_retries} failed",
                target_id=target_id,
                result="RETRY",
            )
        log.log(
            f"PTZ failed after {self._max_retries} retries, skipping target",
            target_id=target_id,
            result="SKIP",
        )
        return False

    def _send_ptz_command(
        self, target_id: int, rect: Tuple[int, int, int, int]
    ) -> bool:
        if self._mode == "subprocess":
            return self._send_subprocess(target_id, rect)
        return self._send_isapi(rect)

    def _send_isapi(self, rect: Tuple[int, int, int, int]) -> bool:
        """Send PTZ SelZoomIn via ISAPI."""
        x1, y1, x2, y2 = rect
        channel = self._channel
        url = f"http://{self._device_addr}/ISAPI/PTZCtrl/channels/{channel}/position"

        # Hikvision ISAPI SelZoomIn XML body
        xml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<PTZPosition>
    <positionX>{x1}</positionX>
    <positionY>{y1}</positionY>
    <width>{x2 - x1}</width>
    <height>{y2 - y1}</height>
</PTZPosition>"""

        try:
            resp = requests.put(
                url,
                data=xml_body,
                headers={"Content-Type": "application/xml"},
                auth=HTTPDigestAuth(self._username, self._password),
                timeout=5,
            )
            return resp.status_code == 200
        except Exception as e:
            log.log(f"ISAPI PTZ request failed: {e}", result="ERROR")
            return False

    def _send_subprocess(
        self, target_id: int, rect: Tuple[int, int, int, int]
    ) -> bool:
        """Send PTZ command via ctrl subprocess."""
        if self._process is None or self._process.poll() is not None:
            return False
        x1, y1, x2, y2 = rect
        try:
            line = f"{target_id} {x1} {y1} {x2} {y2}\n"
            self._process.stdin.write(line.encode())
            self._process.stdin.flush()
            return True
        except Exception as e:
            log.log(f"Subprocess PTZ write failed: {e}", result="ERROR")
            return False

    def wait_stable(self) -> None:
        """Wait for PTZ movement to stabilize."""
        time.sleep(self._stable_wait)

    def disconnect(self) -> None:
        if self._process is not None:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
