"""Initialize face vector database from data/photo/.

Reads all images, detects the largest face, extracts 512-dim embedding,
and stores (name, embedding) in SQLite.

Usage:
    python3 scripts/init_vector_db.py
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

MODEL_DET = os.path.join(PROJECT, "models", "facedect", "1", "model.onnx")
MODEL_REC = os.path.join(PROJECT, "models", "facerecognize", "1", "model.onnx")
PHOTO_DIR = os.path.join(PROJECT, "data", "photo")
DB_PATH = os.path.join(PROJECT, "data", "face_db.sqlite")


def detect_largest_face(img: np.ndarray, session) -> tuple | None:
    """Detect faces, return bbox of the largest one."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (640, 640), swapRB=True)
    raw = session.run(["output0"], {"images": blob})[0]
    preds = raw[0].T

    best = None
    best_area = 0
    for row in preds:
        score = float(row[4])
        if score < 0.5:
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        sx, sy = w / 640, h / 640
        x1 = max(0, int((cx - bw / 2) * sx))
        y1 = max(0, int((cy - bh / 2) * sy))
        x2 = min(w, int((cx + bw / 2) * sx))
        y2 = min(h, int((cy + bh / 2) * sy))
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
    return best


def extract_embedding(img: np.ndarray, bbox: tuple, session) -> np.ndarray | None:
    """Crop face and extract embedding."""
    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    blob = face.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    vec = session.run(["1333"], {"input.1": blob})[0]
    return vec[0]


def main():
    import onnxruntime as ort
    from src.vector_db import VectorDB
    from src.config import Config

    print("Loading models...")
    det_session = ort.InferenceSession(MODEL_DET)
    rec_session = ort.InferenceSession(MODEL_REC)

    config = Config({
        "vector_db": {"path": DB_PATH},
        "recognition": {"top_k": 5},
        "video": {"panoramic_url": "mock"},
        "device": {"address": "mock", "username": "mock", "password": "mock"},
        "triton": {"url": "mock"},
    })

    db = VectorDB(config)
    db.open()
    print(f"VectorDB: {DB_PATH}")

    photos = sorted(glob.glob(os.path.join(PHOTO_DIR, "*")))
    photos = [p for p in photos if p.lower().endswith((".jpg", ".png"))]

    if not photos:
        print("No photos found in data/photo/")
        return

    success = 0
    for path in photos:
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            print(f"  SKIP {name}: cannot read")
            continue

        bbox = detect_largest_face(img, det_session)
        if bbox is None:
            print(f"  SKIP {name}: no face detected")
            continue

        emb = extract_embedding(img, bbox, rec_session)
        if emb is None:
            print(f"  SKIP {name}: embedding failed")
            continue

        db.add_face(name, emb)
        success += 1
        print(f"  OK   {name}  (emb norm={np.linalg.norm(emb):.3f})")

    print(f"\nDone: {success}/{len(photos)} identities stored in {DB_PATH}")


if __name__ == "__main__":
    main()
