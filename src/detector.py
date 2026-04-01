"""Face detection client using Triton Inference Server."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import tritonclient.grpc as grpc_client
    from tritonclient.grpc import InferInput, InferRequestedOutput
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from .config import Config
from .models import BBox, Detection
from . import logger as log


class Detector:
    """Face detection via Triton gRPC."""

    def __init__(self, config: Config) -> None:
        self._url = config.get("triton.url")
        self._model_name = config.get("triton.detection_model", "face_detection")
        input_size = config.get("triton.detection_input_size", [640, 640])
        self._input_w = int(input_size[0])
        self._input_h = int(input_size[1])
        self._score_threshold = config.get("detection.score_threshold", 0.5)
        self._min_face_w = config.get("detection.min_face_width", 30)
        self._min_face_h = config.get("detection.min_face_height", 30)
        self._client: Optional[object] = None
        self._ready = False

    def health_check(self) -> bool:
        """Check detection model availability and warm up."""
        if not HAS_TRITON:
            log.log("Triton client not installed, detection unavailable", result="SKIP")
            return False
        try:
            self._client = grpc_client.InferenceServerClient(url=self._url)
            ready = self._client.is_model_ready(self._model_name)
            if not ready:
                log.log(f"Detection model '{self._model_name}' not ready", result="NOT_READY")
                return False
            self._warmup()
            self._ready = True
            log.log("Detection model health check passed", result="OK")
            return True
        except Exception as e:
            log.log(f"Detection health check failed: {e}", result="ERROR")
            return False

    def _warmup(self) -> None:
        """Send a dummy inference to warm up the model."""
        dummy = np.zeros((1, 3, self._input_h, self._input_w), dtype=np.float32)
        inputs = [InferInput("input", dummy.shape, "FP32")]
        inputs[0].set_data_from_numpy(dummy)
        try:
            self._client.infer(self._model_name, inputs)
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run face detection on a frame. Returns empty list on failure."""
        if not self._ready or self._client is None:
            return []

        h, w = frame.shape[:2]
        input_h, input_w = self._input_h, self._input_w

        # Preprocess: resize + normalize to [0,1]
        resized = cv2.resize(frame, (input_w, input_h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # add batch dim

        inputs = [InferInput("input", blob.shape, "FP32")]
        inputs[0].set_data_from_numpy(blob)

        outputs = [InferRequestedOutput("output")]

        try:
            t0 = time.time()
            response = self._client.infer(self._model_name, inputs, outputs=outputs)
            infer_ms = (time.time() - t0) * 1000
            raw = response.as_numpy("output")
        except Exception as e:
            log.log(f"Detection inference failed: {e}", result="ERROR")
            return []

        if raw is None:
            return []

        detections = self._parse_output(raw, w, h, input_w, input_h)
        log.log(
            f"Detected {len(detections)} faces ({infer_ms:.0f}ms)",
            result="OK",
            extra_data={"face_count": len(detections), "infer_ms": round(infer_ms, 1)},
        )
        return detections

    def _parse_output(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        input_w: int,
        input_h: int,
    ) -> List[Detection]:
        """Parse detection output into Detection objects.

        Expects output shape: [batch, num_detections, 5+]
        where each row is [x1, y1, x2, y2, score, ...]
        Handles both [1, N, 5] and [N, 5] shapes.
        """
        if raw.ndim == 3:
            raw = raw[0]  # remove batch dim

        detections: List[Detection] = []
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        for row in raw:
            if len(row) < 5:
                continue
            score = float(row[4])
            if score < self._score_threshold:
                continue

            x1 = int(max(0, row[0]) * scale_x)
            y1 = int(max(0, row[1]) * scale_y)
            x2 = int(min(orig_w, row[2]) * scale_x)
            y2 = int(min(orig_h, row[3]) * scale_y)

            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            if bbox.width < self._min_face_w or bbox.height < self._min_face_h:
                continue
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                continue

            det = Detection(bbox=bbox, score=score)
            # Parse landmarks if available (5 points = 10 values)
            if len(row) >= 15:
                landmarks = []
                for i in range(5):
                    lx = float(row[5 + i * 2]) * scale_x
                    ly = float(row[5 + i * 2 + 1]) * scale_y
                    landmarks.append((lx, ly))
                det.landmark = landmarks

            detections.append(det)

        return detections

    @property
    def ready(self) -> bool:
        return self._ready
