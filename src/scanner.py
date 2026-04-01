"""Scan cursor and face sorting for left-to-right, top-to-bottom traversal.

Per 5.2:
- Sort by face center, row bucket first (top-down), then x (left-right)
- Scan cursor tracks position, never goes backward
- When no more targets after cursor, reset for next round
"""

from __future__ import annotations

from typing import List, Optional

from .config import Config
from .models import BBox, Detection, ScanCursor
from . import logger as log


class Scanner:
    """Manages face sorting and scan-cursor-based target selection."""

    def __init__(self, config: Config) -> None:
        self._row_bucket = config.get("scan.row_bucket", 80)
        self._cursor = ScanCursor()
        self._target_counter = 0

    @property
    def cursor(self) -> ScanCursor:
        return self._cursor

    def sort_faces(self, detections: List[Detection]) -> List[Detection]:
        """Sort detections: top-to-bottom (by row bucket), left-to-right within rows."""
        bucket = self._row_bucket
        sorted_dets = sorted(
            detections,
            key=lambda d: (int(d.bbox.cy // bucket), d.bbox.cx),
        )
        log.log(
            f"Sorted {len(sorted_dets)} faces",
            result="OK",
            extra_data={
                "order": [f"({d.bbox.cx:.0f},{d.bbox.cy:.0f})" for d in sorted_dets],
            },
        )
        return sorted_dets

    def select_next(
        self, sorted_faces: List[Detection]
    ) -> Optional[tuple[int, Detection]]:
        """Select the next face after the current scan cursor.

        Returns (target_id, detection) or None if no candidates remain.
        """
        bucket = self._row_bucket

        for det in sorted_faces:
            face_row_bucket = int(det.bbox.cy // bucket)
            face_x = det.bbox.cx
            if self._cursor.is_before(face_row_bucket, face_x):
                self._target_counter += 1
                tid = self._target_counter
                self._cursor.update(face_row_bucket, face_x)
                log.log(
                    f"Selected target {tid} at ({face_x:.0f},{det.bbox.cy:.0f})",
                    target_id=tid,
                    result="OK",
                    extra_data={
                        "cursor_row": face_row_bucket,
                        "cursor_x": round(face_x, 1),
                    },
                )
                return (tid, det)

        # No more targets after cursor
        return None

    def advance_cursor(self, detection: Detection) -> None:
        """Update cursor after processing a target."""
        bucket = self._row_bucket
        self._cursor.update(int(detection.bbox.cy // bucket), detection.bbox.cx)
        log.log(
            f"Cursor advanced to ({self._cursor.x_center:.0f}, row_bucket={self._cursor.row_bucket})",
            result="OK",
        )

    def reset_round(self) -> None:
        """Reset cursor to beginning for a new scan round."""
        self._cursor.reset()
        log.log("Scan cursor reset for new round", result="OK")

    @property
    def cursor_position(self) -> str:
        return f"(row_bucket={self._cursor.row_bucket}, x={self._cursor.x_center:.0f})"
