"""PTZ alignment test: detect one face -> send one PTZ command -> verify.

Usage:
    python3 scripts/test_ptz_alignment.py                  # live: grab pano + PTZ move
    python3 scripts/test_ptz_alignment.py --mock            # mock: local image, no camera
    python3 scripts/test_ptz_alignment.py --mock photo.jpg  # mock with specific image
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src.config import load_config
from src.ptz_controller import PTZController
from src.video import VideoReader
from src.detector import nms
from src.models import BBox, Detection

MODEL_DET = os.path.join(PROJECT, "models", "facedect", "1280", "best.onnx")
CONFIG_PATH = os.path.join(PROJECT, "config.yaml")
OUTPUT_DIR = os.path.join(PROJECT, "outputs")


def detect_faces(session, frame, score_thresh=0.5, nms_iou=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (1280, 1280), swapRB=True)
    raw = session.run(["output0"], {"images": blob})[0]
    preds = raw[0].T
    results = []
    for row in preds:
        score = float(row[4])
        if score < score_thresh:
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        sx, sy = w / 1280, h / 1280
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        results.append(Detection(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), score=score))
    return nms(results, nms_iou)


def check_centering(detections, img_w, img_h):
    if not detections:
        return False, 0.0, 0.0
    b = detections[0].bbox
    off_x = abs(b.cx - img_w / 2.0) / (img_w / 2.0)
    off_y = abs(b.cy - img_h / 2.0) / (img_h / 2.0)
    return (off_x < 0.20 and off_y < 0.20), off_x, off_y


def run_mock(config, image_path=None):
    import onnxruntime as ort

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    det_session = ort.InferenceSession(MODEL_DET)
    ptz = PTZController(config)

    if image_path:
        img = cv2.imread(image_path)
    else:
        photos = sorted(glob.glob(os.path.join(PROJECT, "data", "photo", "*.jpg")))
        if not photos:
            print("No images found")
            return
        img = cv2.imread(photos[0])

    if img is None:
        print("Cannot read image")
        return

    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    detections = detect_faces(det_session, img)
    print(f"Detected {len(detections)} face(s)")
    if not detections:
        print("No faces detected")
        return

    # Only use first face
    det = detections[0]
    bbox = det.bbox
    print(f"Target: ({bbox.x1},{bbox.y1})-({bbox.x2},{bbox.y2})  "
          f"cx={bbox.cx:.0f} cy={bbox.cy:.0f}  score={det.score:.3f}")

    # Show PTZ coordinates (for reference)
    rect = ptz.calculate_coordinates(bbox, stage_num=1)
    print(f"PTZ rect (0-255): ({rect[0]},{rect[1]},{rect[2]},{rect[3]})")

    # Save pano with target marked
    pano_out = img.copy()
    cv2.rectangle(pano_out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 0, 255), 2)
    cv2.putText(pano_out, "TARGET", (bbox.x1, bbox.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ptz_pano_marked.jpg"), pano_out)
    print(f"Saved: outputs/ptz_pano_marked.jpg")


def run_live(config):
    import onnxruntime as ort

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    det_session = ort.InferenceSession(MODEL_DET)
    ptz = PTZController(config)

    pano_url = config.get("video.panoramic_url")
    ptz_url = config.get("video.ptz_url")
    print(f"Pano: {pano_url}")
    print(f"PTZ:  {ptz_url}")

    # Step 1: Grab panoramic frame
    print("\n[1] Grab panoramic frame")
    pano_reader = VideoReader(pano_url, config.get("reconnect.interval_ms", 5000))
    if not pano_reader.open_with_retry(max_retries=3):
        print("Failed to connect panoramic stream")
        return

    pano_frame = pano_reader.get_latest_frame()
    pano_reader.release()
    if pano_frame is None:
        print("Failed to read panoramic frame")
        return

    pano_h, pano_w = pano_frame.shape[:2]
    print(f"  Frame: {pano_w}x{pano_h}")

    # Save pano frame
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ptz_pano_frame.jpg"), pano_frame)

    # Step 2: Detect faces
    print("\n[2] Detect faces")
    detections = detect_faces(det_session, pano_frame)
    print(f"  Detected {len(detections)} face(s)")
    if not detections:
        print("  No faces detected")
        return

    det = detections[0]
    bbox = det.bbox
    print(f"  First face: ({bbox.x1},{bbox.y1})-({bbox.x2},{bbox.y2})  "
          f"cx={bbox.cx:.0f} cy={bbox.cy:.0f}  score={det.score:.3f}")

    # Save pano with target marked
    pano_marked = pano_frame.copy()
    cv2.rectangle(pano_marked, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 0, 255), 3)
    cv2.putText(pano_marked, "TARGET", (bbox.x1, bbox.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "ptz_pano_marked.jpg"), pano_marked)

    # Step 3: Send ONE PTZ command
    print("\n[3] Send PTZ command")
    print(f"  Pixel bbox: ({bbox.x1},{bbox.y1})-({bbox.x2},{bbox.y2})")

    if not ptz.connect():
        print("  PTZ connect failed")
        return

    ok = ptz.move_to_target(1, bbox, frame_w=pano_w, frame_h=pano_h)
    if not ok:
        print("  PTZ move failed")
        ptz.disconnect()
        return
    print("  PTZ command sent OK")

    # Step 4: Wait for PTZ to stabilize
    print(f"\n[4] Wait {ptz._stable_wait:.1f}s for PTZ to stabilize...")
    ptz.wait_stable()
    ptz.disconnect()
    print("  Done")

    # Step 5: Grab PTZ frame
    print("\n[5] Grab PTZ frame")
    ptz_reader = VideoReader(ptz_url, config.get("reconnect.interval_ms", 5000))
    if not ptz_reader.open_with_retry(max_retries=3):
        print("  Failed to connect PTZ stream")
        return

    # Skip buffered frames to get fresh image
    for _ in range(3):
        time.sleep(0.3)
        ptz_reader.get_latest_frame()

    ptz_frame = ptz_reader.get_latest_frame()
    ptz_reader.release()

    if ptz_frame is None:
        print("  Failed to read PTZ frame")
        return

    ptz_h, ptz_w = ptz_frame.shape[:2]
    print(f"  PTZ frame: {ptz_w}x{ptz_h}")

    # Step 6: Detect on PTZ frame
    print("\n[6] Detect faces on PTZ frame")
    ptz_dets = detect_faces(det_session, ptz_frame)
    print(f"  Detected {len(ptz_dets)} face(s)")
    for i, d in enumerate(ptz_dets):
        b = d.bbox
        print(f"  face{i}: score={d.score:.3f}  cx={b.cx:.0f} cy={b.cy:.0f}")

    # Step 7: Check centering
    print("\n[7] Verify centering")
    centered, off_x, off_y = check_centering(ptz_dets, ptz_w, ptz_h)
    print(f"  Image center: ({ptz_w//2}, {ptz_h//2})")
    if ptz_dets:
        b = ptz_dets[0].bbox
        print(f"  Face center:  ({b.cx:.0f}, {b.cy:.0f})")
    print(f"  Offset: x={off_x:.1%}  y={off_y:.1%}")
    print(f"  Result: {'PASS - CENTERED' if centered else 'FAIL - OFF-CENTER'}")

    # Save result with crosshair
    result = ptz_frame.copy()
    cx, cy = ptz_w // 2, ptz_h // 2
    cv2.line(result, (cx - 40, cy), (cx + 40, cy), (0, 255, 255), 2)
    cv2.line(result, (cx, cy - 40), (cx, cy + 40), (0, 255, 255), 2)
    for d in ptz_dets:
        b = d.bbox
        cv2.rectangle(result, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
        cv2.putText(result, f"{d.score:.2f}", (b.x1, b.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = os.path.join(OUTPUT_DIR, "ptz_result.jpg")
    cv2.imwrite(out_path, result)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PTZ alignment test (single face)")
    parser.add_argument("--mock", action="store_true", help="Mock mode")
    parser.add_argument("image", nargs="?", help="Image path (mock only)")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    if args.mock:
        run_mock(config, args.image)
    else:
        run_live(config)


if __name__ == "__main__":
    main()
