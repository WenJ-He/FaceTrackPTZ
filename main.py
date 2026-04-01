"""Minimal main.py — verify scanner sorting and scan-cursor logic only.

Uses mock detections, no video/Triton/PTZ/recognition.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import BBox, Detection
from src.scanner import Scanner


# ── Mock data ──────────────────────────────────────────────────────
MOCK_FACES = [
    # (x1, y1, x2, y2, score)  — deliberately out of scan order
    (800,  400, 880, 500, 0.92),   # mid-right, row 1
    (100,  50,  170, 130, 0.95),   # top-left,   row 0
    (500,  300, 580, 390, 0.88),   # mid-center, row 1
    (300,  160, 370, 240, 0.85),   # left-center,row 0
    (1200, 100, 1280, 190, 0.80),  # top-right,  row 0
]


def build_detections() -> list[Detection]:
    return [
        Detection(
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            score=score,
        )
        for x1, y1, x2, y2, score in MOCK_FACES
    ]


def print_det(idx: int, det: Detection) -> None:
    b = det.bbox
    print(
        f"  [{idx}] ({b.x1:4d},{b.y1:4d})-({b.x2:4d},{b.y2:4d})  "
        f"cx={b.cx:6.0f} cy={b.cy:5.0f}  score={det.score:.2f}"
    )


# ── Main ───────────────────────────────────────────────────────────
def main() -> None:
    # Minimal config needed by Scanner (only scan.row_bucket matters)
    from src.config import Config
    config = Config({"scan": {"row_bucket": 80}, "video": {"panoramic_url": "mock"}, "device": {"address": "mock", "username": "mock", "password": "mock"}, "triton": {"url": "mock"}, "vector_db": {"path": ":memory:"}})

    scanner = Scanner(config)

    detections = build_detections()

    print("=== Raw detections (input order) ===")
    for i, d in enumerate(detections):
        print_det(i, d)

    # ── Sort ───────────────────────────────────────────────────────
    sorted_faces = scanner.sort_faces(detections)

    print("\n=== Sorted (top→bottom, left→right, row_bucket=80) ===")
    for i, d in enumerate(sorted_faces):
        b = d.bbox
        row_bkt = int(b.cy // 80)
        print(f"  [{i}] cx={b.cx:6.0f} cy={b.cy:5.0f}  row_bucket={row_bkt}")

    # ── Simulate scan-cursor progression ───────────────────────────
    print("\n=== Scan-cursor simulation ===")
    round_num = 1

    while True:
        print(f"\n--- Round {round_num} ---")
        print(f"  cursor: {scanner.cursor_position}")

        target = scanner.select_next(sorted_faces)

        if target is None:
            print("  No more targets → reset cursor, start new round")
            scanner.reset_round()
            # Try one more round to prove reset works
            if round_num >= 2:
                print("\n  (second round re-selects same targets, stop here)")
                target = scanner.select_next(sorted_faces)
                if target:
                    tid, det = target
                    print(f"  -> Target #{tid} selected: cx={det.bbox.cx:.0f} cy={det.bbox.cy:.0f}")
                break
            round_num += 1
            continue

        tid, det = target
        b = det.bbox
        print(f"  -> Target #{tid}  cx={b.cx:.0f} cy={b.cy:.0f}")
        print(f"    cursor now: {scanner.cursor_position}")


def _label(det: Detection) -> str:
    """Human-readable label based on cx for easy tracking."""
    b = det.bbox
    if b.cx < 200:
        return "A"
    if b.cx < 400:
        return "B"
    if b.cx < 600:
        return "C"
    if b.cx < 1000:
        return "D"
    return "E"


def test_dynamic_input() -> None:
    """Verify cursor behavior when detections change between rounds.

    Round 1: faces = [A, B, C]  →  scan A then B
    Round 1 (re-detect): faces = [B, C, D]  →  cursor after B, should skip A (gone),
             skip B (already past), pick C then D
    """
    print("\n" + "=" * 60)
    print("TEST: Dynamic input change — cursor must not go backward")
    print("=" * 60)

    from src.config import Config
    config = Config({
        "scan": {"row_bucket": 80},
        "video": {"panoramic_url": "mock"},
        "device": {"address": "mock", "username": "mock", "password": "mock"},
        "triton": {"url": "mock"},
        "vector_db": {"path": ":memory:"},
    })
    scanner = Scanner(config)

    # ── Round 1: faces [A, B, C] ──────────────────────────────────
    faces_r1 = [
        Detection(bbox=BBox(x1=100, y1=50,  x2=170, y2=130), score=0.95),  # A  cx=135
        Detection(bbox=BBox(x1=300, y1=160, x2=370, y2=240), score=0.90),  # B  cx=335
        Detection(bbox=BBox(x1=500, y1=300, x2=580, y2=390), score=0.88),  # C  cx=540
    ]

    sorted_r1 = scanner.sort_faces(faces_r1)
    print("\nRound 1 detections (sorted): "
          + ", ".join(_label(d) for d in sorted_r1))

    # Process A
    tid_a, det_a = scanner.select_next(sorted_r1)
    assert tid_a == 1, f"Expected target #1, got #{tid_a}"
    assert _label(det_a) == "A", f"Expected A, got {_label(det_a)}"
    print(f"  -> #{tid_a} {_label(det_a)}   cursor: {scanner.cursor_position}")

    # Process B
    tid_b, det_b = scanner.select_next(sorted_r1)
    assert tid_b == 2, f"Expected target #2, got #{tid_b}"
    assert _label(det_b) == "B", f"Expected B, got {_label(det_b)}"
    print(f"  -> #{tid_b} {_label(det_b)}   cursor: {scanner.cursor_position}")

    # ── Round 1 continued: person moved, re-detect → [B, C, D] ────
    faces_r2 = [
        Detection(bbox=BBox(x1=300, y1=160, x2=370, y2=240), score=0.90),  # B  cx=335
        Detection(bbox=BBox(x1=500, y1=300, x2=580, y2=390), score=0.88),  # C  cx=540
        Detection(bbox=BBox(x1=800, y1=400, x2=880, y2=500), score=0.85),  # D  cx=840  (new!)
    ]

    sorted_r2 = scanner.sort_faces(faces_r2)
    print(f"\nRe-detect (people moved, A gone, D appeared): "
          + ", ".join(_label(d) for d in sorted_r2))
    print(f"  cursor: {scanner.cursor_position}")

    # Next after B → must be C (NOT re-scan B, NOT skip C)
    tid_c, det_c = scanner.select_next(sorted_r2)
    assert tid_c == 3, f"Expected target #3, got #{tid_c}"
    assert _label(det_c) == "C", (
        f"CORRECT: cursor skipped B (already past), picked C. Got {_label(det_c)}"
    )
    print(f"  -> #{tid_c} {_label(det_c)}   (skipped B=already past)  cursor: {scanner.cursor_position}")

    # Next after C → must be D (the new face)
    tid_d, det_d = scanner.select_next(sorted_r2)
    assert tid_d == 4, f"Expected target #4, got #{tid_d}"
    assert _label(det_d) == "D", f"Expected D, got {_label(det_d)}"
    print(f"  -> #{tid_d} {_label(det_d)}   (new face, picked up)  cursor: {scanner.cursor_position}")

    # No more → None
    tid_e = scanner.select_next(sorted_r2)
    assert tid_e is None, "Expected None after D"
    print(f"  -> None  (no more targets after D)")

    # ── Verify A was never re-scanned ─────────────────────────────
    print("\n--- Verification ---")
    print("  A was NOT re-scanned  ✓  (A gone from detection + cursor already past)")
    print("  B was NOT re-scanned  ✓  (cursor already past B's position)")
    print("  C was picked          ✓  (first face after cursor)")
    print("  D was picked          ✓  (new face after C)")
    print("  Cursor never went backward  ✓")


if __name__ == "__main__":
    main()
    test_dynamic_input()
