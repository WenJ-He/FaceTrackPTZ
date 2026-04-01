"""Data models for FaceTrack PTZ."""

from __future__ import annotations

import dataclasses
import time
from typing import List, Optional, Tuple


@dataclasses.dataclass
class BBox:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclasses.dataclass
class Detection:
    """Single face detection result."""
    bbox: BBox
    score: float
    landmark: Optional[List[Tuple[float, float]]] = None


@dataclasses.dataclass
class RecognitionResult:
    """Recognition result for a single stage."""
    identity: str
    similarity: float
    top_k: List[Tuple[str, float]]
    vector: Optional[List[float]] = None
    fixed_identity_sim: float = 0.0

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "similarity": self.similarity,
            "top_k": [(name, round(s, 4)) for name, s in self.top_k],
            "fixed_identity_sim": round(self.fixed_identity_sim, 4),
        }


@dataclasses.dataclass
class StageRecord:
    """Record for one recognition stage of a target."""
    stage_num: int
    timestamp: float
    bbox: BBox
    recognition: RecognitionResult
    similarity_delta_from_s0: float = 0.0
    similarity_delta_from_prev: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "recognition": self.recognition.to_dict(),
            "delta_from_s0": round(self.similarity_delta_from_s0, 4),
            "delta_from_prev": round(self.similarity_delta_from_prev, 4),
        }


@dataclasses.dataclass
class TargetRecord:
    """Full record for one scanned target."""
    target_id: int
    sort_index: int
    start_time: float
    end_time: Optional[float] = None
    stages: List[StageRecord] = dataclasses.field(default_factory=list)
    fixed_identity: Optional[str] = None
    best_similarity: float = 0.0
    best_stage: int = 0
    best_identity: Optional[str] = None
    final_identity: Optional[str] = None
    final_stage: int = 0
    final_similarity: float = 0.0
    ground_truth: Optional[str] = None

    @property
    def is_correct_top1(self) -> Optional[bool]:
        if self.ground_truth is None:
            return None
        return self.final_identity == self.ground_truth

    @property
    def is_correct_topk(self) -> Optional[bool]:
        if self.ground_truth is None:
            return None
        if not self.stages:
            return None
        last = self.stages[-1]
        return self.ground_truth in [name for name, _ in last.recognition.top_k]

    @property
    def total_gain(self) -> float:
        if len(self.stages) < 2:
            return 0.0
        s0_sim = self.stages[0].recognition.similarity
        final_sim = self.stages[-1].recognition.similarity
        return final_sim - s0_sim

    def add_stage(self, record: StageRecord) -> None:
        if record.stage_num == 0:
            self.fixed_identity = record.recognition.identity
            record.recognition.fixed_identity_sim = record.recognition.similarity
        else:
            s0_sim = self.stages[0].recognition.similarity
            record.similarity_delta_from_s0 = record.recognition.similarity - s0_sim
            if self.stages:
                record.similarity_delta_from_prev = (
                    record.recognition.similarity - self.stages[-1].recognition.similarity
                )
            fixed = self.fixed_identity
            for name, sim in record.recognition.top_k:
                if name == fixed:
                    record.recognition.fixed_identity_sim = sim
                    break

        self.stages.append(record)
        if record.recognition.similarity > self.best_similarity:
            self.best_similarity = record.recognition.similarity
            self.best_stage = record.stage_num
            self.best_identity = record.recognition.identity

    def finalize(self) -> None:
        self.end_time = time.time()
        if self.stages:
            last = self.stages[-1]
            self.final_identity = last.recognition.identity
            self.final_stage = last.stage_num
            self.final_similarity = last.recognition.similarity

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "sort_index": self.sort_index,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)) if self.end_time else None,
            "fixed_identity": self.fixed_identity,
            "stages": [s.to_dict() for s in self.stages],
            "best_similarity": round(self.best_similarity, 4),
            "best_stage": self.best_stage,
            "best_identity": self.best_identity,
            "final_identity": self.final_identity,
            "final_stage": self.final_stage,
            "final_similarity": round(self.final_similarity, 4),
            "gain": round(self.total_gain, 4),
            "ground_truth": self.ground_truth,
            "is_correct_top1": self.is_correct_top1,
            "is_correct_topk": self.is_correct_topk,
        }


@dataclasses.dataclass
class ScanCursor:
    """Tracks the current position in the left-to-right, top-to-bottom scan."""
    row_bucket: int = -1
    x_center: float = -1.0

    def is_before(self, face_row_bucket: int, face_x_center: float) -> bool:
        if face_row_bucket > self.row_bucket:
            return True
        if face_row_bucket == self.row_bucket and face_x_center > self.x_center:
            return True
        return False

    def update(self, row_bucket: int, x_center: float) -> None:
        self.row_bucket = row_bucket
        self.x_center = x_center

    def reset(self) -> None:
        self.row_bucket = -1
        self.x_center = -1.0
