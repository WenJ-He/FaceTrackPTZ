"""Face recognition client using Triton Inference Server."""

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
from .models import BBox, RecognitionResult
from .vector_db import VectorDB
from . import logger as log


class Recognizer:
    """Face recognition via Triton gRPC + vector DB search."""

    def __init__(self, config: Config, vector_db: VectorDB) -> None:
        self._url = config.get("triton.url")
        self._model_name = config.get("triton.recognition_model", "face_recognition")
        self._input_size = tuple(config.get("triton.recognition_input_size", [112, 112]))
        self._top_k = config.get("recognition.top_k", 5)
        self._client: Optional[object] = None
        self._vector_db = vector_db
        self._ready = False

    def health_check(self) -> bool:
        """Check recognition model availability (non-blocking)."""
        if not HAS_TRITON:
            log.log("Triton client not installed, recognition unavailable", result="SKIP")
            return False
        try:
            self._client = grpc_client.InferenceServerClient(url=self._url)
            ready = self._client.is_model_ready(self._model_name)
            if not ready:
                log.log(f"Recognition model '{self._model_name}' not ready", result="NOT_READY")
                return False
            self._ready = True
            log.log("Recognition model health check passed", result="OK")
            return True
        except Exception as e:
            log.log(f"Recognition health check failed: {e}", result="WARN")
            return False

    def extract_vector(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """Extract face embedding from a cropped face region."""
        if not self._ready or self._client is None:
            return None

        h, w = frame.shape[:2]
        x1 = max(0, bbox.x1)
        y1 = max(0, bbox.y1)
        x2 = min(w, bbox.x2)
        y2 = min(h, bbox.y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            log.log("Face crop too small for recognition", result="SKIP")
            return None

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        resized = cv2.resize(face_crop, (self._input_size[0], self._input_size[1]))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)

        inputs = [InferInput("input.1", blob.shape, "FP32")]
        inputs[0].set_data_from_numpy(blob)

        outputs = [InferRequestedOutput("1333")]

        try:
            response = self._client.infer(self._model_name, inputs, outputs=outputs)
            vector = response.as_numpy("1333")
            if vector is not None and vector.ndim >= 2:
                return vector[0].flatten()
            return None
        except Exception as e:
            log.log(f"Recognition inference failed: {e}", result="ERROR")
            return None

    def recognize(self, frame: np.ndarray, bbox: BBox) -> Optional[RecognitionResult]:
        """Extract embedding and search vector DB for matches."""
        vector = self.extract_vector(frame, bbox)
        if vector is None:
            return None

        results = self._vector_db.search(vector, top_k=self._top_k)
        if not results:
            log.log("Vector DB returned no results", result="EMPTY")
            return None

        top1_name, top1_sim = results[0]
        return RecognitionResult(
            identity=top1_name,
            similarity=float(top1_sim),
            top_k=[(name, float(sim)) for name, sim in results],
            vector=vector.tolist(),
        )

    def recognize_stage0(self, frame: np.ndarray, bbox: BBox) -> Optional[RecognitionResult]:
        """Perform Stage-0 recognition from panoramic image."""
        result = self.recognize(frame, bbox)
        if result is not None:
            result.fixed_identity_sim = result.similarity
        return result

    @property
    def ready(self) -> bool:
        return self._ready
