"""FaceTrack PTZ — Minimal main flow: image -> detect -> sort -> identify.

Usage:
    python3 main.py [image_path]

If no image_path given, uses first photo from data/photo/.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.vector_db import VectorDB
from src.scanner import Scanner
from src.detector import nms
from src.models import BBox, Detection
from src.logger import setup_logging

PROJECT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT, "data", "face_db.sqlite")
MODEL_DET = os.path.join(PROJECT, "models", "facedect", "1", "model.onnx")
MODEL_REC = os.path.join(PROJECT, "models", "facerecognize", "1", "model.onnx")
PHOTO_DIR = os.path.join(PROJECT, "data", "photo")


def detect_faces(session, frame, score_thresh=0.5, nms_iou=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True)
    raw = session.run(["output0"], {"images": blob})[0]
    preds = raw[0].T
    results = []
    for row in preds:
        score = float(row[4])
        if score < score_thresh:
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        sx, sy = w / 640, h / 640
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        results.append(Detection(bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), score=score))
    return nms(results, nms_iou)


def extract_embedding(session, frame, bbox):
    x1, y1 = max(0, bbox.x1), max(0, bbox.y1)
    x2, y2 = min(frame.shape[1], bbox.x2), min(frame.shape[0], bbox.y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (112, 112))
    blob = crop.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return session.run(["1333"], {"input.1": blob})[0][0].flatten()


def run(image_path):
    import onnxruntime as ort

    setup_logging("INFO")
    config = Config({
        "scan": {"row_bucket": 80},
        "detection": {"score_threshold": 0.5},
        "recognition": {"top_k": 5},
        "video": {"panoramic_url": "mock"},
        "device": {"address": "mock", "username": "mock", "password": "mock"},
        "triton": {"url": "mock"},
        "vector_db": {"path": DB_PATH},
    })

    db = VectorDB(config)
    db.open()
    print(f"VectorDB: {db.count()} identities")

    det_session = ort.InferenceSession(MODEL_DET)
    rec_session = ort.InferenceSession(MODEL_REC)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return
    print(f"\nImage: {os.path.basename(image_path)} ({img.shape[1]}x{img.shape[0]})")

    scanner = Scanner(config)

    t0 = time.time()
    detections = detect_faces(det_session, img, 0.5)
    det_ms = (time.time() - t0) * 1000
    print(f"Detected {len(detections)} face(s) in {det_ms:.0f}ms")

    if not detections:
        print("No faces detected")
        return

    sorted_faces = scanner.sort_faces(detections)

    print(f"\n{'Tgt':>4}  {'Identity':12s}  {'Sim':>10}  {'Score':>6}  {'BBox'}")
    print("-" * 65)

    tid = 0
    while True:
        target = scanner.select_next(sorted_faces)
        if target is None:
            break
        tid += 1
        _, det = target
        b = det.bbox

        emb = extract_embedding(rec_session, img, b)
        if emb is None:
            print(f"#{tid:>3}  {'<failed>':12s}  {'---':>10}  {det.score:.3f}")
            continue

        results = db.search(emb, top_k=5)
        top1_name, top1_sim = results[0]
        print(f"#{tid:>3}  {top1_name:12s}  {top1_sim:>10.4f}  {det.score:.3f}  "
              f"({b.x1},{b.y1})-({b.x2},{b.y2})")

    db.close()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="FaceTrack PTZ minimal main flow")
    parser.add_argument("image", nargs="?", help="Path to image file")
    args = parser.parse_args()

    if args.image:
        run(args.image)
    else:
        photos = sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg")))
        if photos:
            run(photos[0])
        else:
            print("No images found")


if __name__ == "__main__":
    main()
