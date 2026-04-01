"""End-to-end test: detect → recognize → vector_db search.

Usage:
    python3 scripts/init_vector_db.py       # run this first
    python3 scripts/test_recognition_pipeline.py [image_path]
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
DB_PATH = os.path.join(PROJECT, "data", "face_db.sqlite")
PHOTO_DIR = os.path.join(PROJECT, "data", "photo")


def detect_faces(img, session):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (640, 640), swapRB=True)
    raw = session.run(["output0"], {"images": blob})[0]
    preds = raw[0].T
    results = []
    for row in preds:
        score = float(row[4])
        if score < 0.5:
            continue
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        sx, sy = w / 640, h / 640
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        results.append(((x1, y1, x2, y2), score))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def extract_embedding(img, bbox, session):
    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    blob = face.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return session.run(["1333"], {"input.1": blob})[0][0]


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path is None:
        photos = sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg")))
        if not photos:
            print("No photos found"); return
        image_path = photos[0]

    if not os.path.isfile(DB_PATH):
        print(f"ERROR: vector DB not found at {DB_PATH}")
        print("Run first: python3 scripts/init_vector_db.py")
        return

    import onnxruntime as ort
    from src.vector_db import VectorDB
    from src.config import Config

    config = Config({
        "vector_db": {"path": DB_PATH},
        "recognition": {"top_k": 5},
        "video": {"panoramic_url": "mock"},
        "device": {"address": "mock", "username": "mock", "password": "mock"},
        "triton": {"url": "mock"},
    })

    db = VectorDB(config)
    db.open()
    print(f"VectorDB loaded: {db.count()} identities")

    det_session = ort.InferenceSession(MODEL_DET)
    rec_session = ort.InferenceSession(MODEL_REC)

    img = cv2.imread(image_path)
    print(f"\nImage: {os.path.basename(image_path)} ({img.shape[1]}x{img.shape[0]})")

    faces = detect_faces(img, det_session)
    if not faces:
        print("No faces detected"); return

    for i, (bbox, score) in enumerate(faces):
        emb = extract_embedding(img, bbox, rec_session)
        if emb is None:
            continue

        results = db.search(emb, top_k=5)
        top1_name, top1_sim = results[0]

        print(f"\n  Face #{i+1}  bbox={bbox}  det_score={score:.3f}")
        print(f"  Top1: {top1_name}  similarity={top1_sim:.4f}")
        print(f"  TopK:")
        for rank, (name, sim) in enumerate(results, 1):
            marker = " <-- Top1" if rank == 1 else ""
            print(f"    {rank}. {name:12s}  {sim:.4f}{marker}")

    db.close()


if __name__ == "__main__":
    main()
