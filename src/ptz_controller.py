"""PTZ controller: bridge Python detections to C++ hkctl PTZ control.

Modes:
  - hkctl: pipes pixel coordinates to C++ camera binary (recommended)
  - isapi: HTTP ISAPI calls to Hikvision device (fallback)
  - subprocess: pipes 0-255 coordinates to C++ ctrl binary (legacy)

hkctl mode is the primary mode. It starts the C++ camera program as a
subprocess and sends pixel-coordinate bboxes via stdin. The camera program
handles pixel -> 0-255 conversion internally and pipes to ctrl which calls
the Hikvision SDK (NET_DVR_PTZSelZoomIn_EX).

Input format to camera binary: "id x1 y1 x2 y2\n" (pixel coordinates)
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
        self._mode = config.get("ptz.mode", "hkctl")
        self._camera_binary = config.get("ptz.camera_binary", "")
        # C++ camera.cpp hardcoded resolution (what it expects for stdin input)
        self._cam_cols = config.get("ptz.camera_resolution.cols", 3632)
        self._cam_rows = config.get("ptz.camera_resolution.rows", 1632)
        self._ctrl_binary = config.get("ptz.ctrl_binary", "")
        self._ctrl_lib_path = config.get("ptz.ctrl_lib_path", "")
        self._process: Optional[subprocess.Popen] = None

    def connect(self) -> bool:
        """Initialize PTZ connection based on mode."""
        if self._mode == "hkctl":
            return self._connect_hkctl()
        if self._mode == "subprocess":
            return self._connect_subprocess()
        return self._connect_isapi()

    def _connect_hkctl(self) -> bool:
        """Start the C++ camera binary as a subprocess.

        The camera program reads 'id x1 y1 x2 y2' (pixel coords) from stdin,
        converts to 0-255 internally, and pipes to ctrl which calls the SDK.
        """
        camera_bin = self._camera_binary
        if not camera_bin or not os.path.isfile(camera_bin):
            log.log(f"camera binary not found: {camera_bin}", result="ERROR")
            return False
        try:
            env = os.environ.copy()
            if self._ctrl_lib_path:
                env["LD_LIBRARY_PATH"] = (
                    self._ctrl_lib_path + ":" + env.get("LD_LIBRARY_PATH", "")
                )
            cwd = os.path.dirname(camera_bin) or "."
            self._process = subprocess.Popen(
                [camera_bin],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
            log.log(f"PTZ hkctl camera started: {camera_bin}", result="OK")
            return True
        except Exception as e:
            log.log(f"PTZ hkctl failed: {e}", result="ERROR")
            return False

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
            log.log(f"PTZ ISAPI status: {resp.status_code}", result="WARN")
            return True
        except Exception as e:
            log.log(f"PTZ ISAPI failed: {e}", result="ERROR")
            return False

    def _connect_subprocess(self) -> bool:
        if not self._ctrl_binary or not os.path.isfile(self._ctrl_binary):
            log.log(f"ctrl binary not found: {self._ctrl_binary}", result="ERROR")
            return False
        try:
            env = os.environ.copy()
            if self._ctrl_lib_path:
                env["LD_LIBRARY_PATH"] = (
                    self._ctrl_lib_path + ":" + env.get("LD_LIBRARY_PATH", "")
                )
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

        Used for isapi/subprocess modes. For hkctl mode, the camera binary
        handles this conversion internally from pixel coordinates.
        """
        cx = bbox.cx
        cy = bbox.cy
        box_w = bbox.width
        box_h = bbox.height

        effective_percent = min(self._box_percent * (1.0 + 0.3 * stage_num), 1.0)
        view_w = box_w / effective_percent
        view_h = box_h / effective_percent

        view_x1 = cx - view_w / 2.0
        view_y1 = cy - view_h / 2.0
        view_x2 = cx + view_w / 2.0
        view_y2 = cy + view_h / 2.0

        view_x1 = max(0, view_x1)
        view_y1 = max(0, view_y1)
        view_x2 = min(self._pano_cols, view_x2)
        view_y2 = min(self._pano_rows, view_y2)

        rect_x1 = int(view_x1 * 255.0 / self._pano_cols)
        rect_y1 = int(view_y1 * 255.0 / self._pano_rows)
        rect_x2 = int(view_x2 * 255.0 / self._pano_cols)
        rect_y2 = int(view_y2 * 255.0 / self._pano_rows)

        rect_x1 = max(0, min(255, rect_x1))
        rect_y1 = max(0, min(255, rect_y1))
        rect_x2 = max(0, min(255, rect_x2))
        rect_y2 = max(0, min(255, rect_y2))

        return rect_x1, rect_y1, rect_x2, rect_y2

    def move_to_target(self, target_id: int, bbox: BBox,
                       frame_w: int = 0, frame_h: int = 0) -> bool:
        """Send PTZ move command with retry.

        For hkctl mode: sends pixel coordinates to camera binary.
        For other modes: calculates 0-255 coordinates and sends via ISAPI/subprocess.

        frame_w/frame_h: actual frame dimensions. If they differ from the
        panoramic_resolution in config, bbox is scaled to match C++ expectations.
        """
        for attempt in range(1, self._max_retries + 1):
            if self._mode == "hkctl":
                success = self._send_hkctl(target_id, bbox, frame_w, frame_h)
            else:
                rect = self.calculate_coordinates(bbox)
                if self._mode == "subprocess":
                    success = self._send_subprocess(target_id, rect)
                else:
                    success = self._send_isapi(rect)

            if success:
                log.log(
                    f"PTZ moved to target {target_id}",
                    target_id=target_id,
                    result="OK",
                    extra_data={"bbox": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                                "attempt": attempt},
                )
                return True
            log.log(
                f"PTZ attempt {attempt}/{self._max_retries} failed",
                target_id=target_id,
                result="RETRY",
            )
        log.log(
            f"PTZ failed after {self._max_retries} retries",
            target_id=target_id,
            result="SKIP",
        )
        return False

    def _send_hkctl(self, target_id: int, bbox: BBox, frame_w: int = 0, frame_h: int = 0) -> bool:
        """Send pixel bbox to C++ camera binary via stdin.

        Format: 'id x1 y1 x2 y2\n'
        camera.cpp expects coordinates in its hardcoded resolution
        (cam_cols x cam_rows, typically 3632x1632). If frame_w/frame_h
        differ from cam resolution, scale bbox accordingly.
        """
        if self._process is None or self._process.poll() is not None:
            log.log("camera process not running", result="ERROR")
            return False

        # Scale bbox to C++ expected resolution if actual frame size differs
        if frame_w > 0 and frame_h > 0 and (frame_w != self._cam_cols or frame_h != self._cam_rows):
            sx = self._cam_cols / frame_w
            sy = self._cam_rows / frame_h
            x1 = int(bbox.x1 * sx)
            y1 = int(bbox.y1 * sy)
            x2 = int(bbox.x2 * sx)
            y2 = int(bbox.y2 * sy)
            log.log(
                f"Scaled bbox {frame_w}x{frame_h} -> {self._cam_cols}x{self._cam_rows}",
                result="SCALE",
                extra_data={"original": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                            "scaled": [x1, y1, x2, y2]},
            )
        else:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        line = f"{target_id} {x1} {y1} {x2} {y2}\n"
        log.log(
            f"PTZ hkctl: bbox -> camera stdin",
            target_id=target_id,
            result="SEND",
            extra_data={"pixel_bbox": [x1, y1, x2, y2]},
        )
        try:
            self._process.stdin.write(line.encode())
            self._process.stdin.flush()
            return True
        except Exception as e:
            log.log(f"PTZ hkctl write failed: {e}", result="ERROR")
            return False

    def _send_isapi(self, rect: Tuple[int, int, int, int]) -> bool:
        """Send PTZ SelZoomIn via ISAPI."""
        x1, y1, x2, y2 = rect
        channel = self._channel
        url = f"http://{self._device_addr}/ISAPI/PTZCtrl/channels/{channel}/position"

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
            log.log(f"ISAPI PTZ failed: {e}", result="ERROR")
            return False

    def _send_subprocess(
        self, target_id: int, rect: Tuple[int, int, int, int]
    ) -> bool:
        """Send 0-255 coordinates to ctrl subprocess."""
        if self._process is None or self._process.poll() is not None:
            return False
        x1, y1, x2, y2 = rect
        try:
            line = f"{target_id} {x1} {y1} {x2} {y2}\n"
            self._process.stdin.write(line.encode())
            self._process.stdin.flush()
            return True
        except Exception as e:
            log.log(f"Subprocess PTZ failed: {e}", result="ERROR")
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
