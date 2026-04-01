"""Test recognizer model directly via onnxruntime (no Triton needed).

Usage:
    python3 scripts/test_recognizer.py [image_path]

If no image_path given, uses first photo from data/photo/.
"""

from __future__ import annotations

import os
import sys
import glob

import cv2
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

MODEL_DET = os.path.join(PROJECT, "models", "facedect", "1", "model.onnx")
MODEL_REC = os.path.join(PROJECT, "models", "facerecognize", "1", "model.onnx")
PHOTO_DIR = os.path.join(PROJECT, "data", "photo")


def detect_faces(img: np.ndarray, session) -> list[dict]:
    """Run face detection, return list of {bbox, score}."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (640, 640), swapRB=True)
    raw = session.run(["output0"], {"images": blob})[0]  # [1,5,8400]

    preds = raw[0].T  # [8400, 5]
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
        results.append({"bbox": (x1, y1, x2, y2), "score": score})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def extract_embedding(img: np.ndarray, bbox: tuple, session) -> np.ndarray:
    """Crop face and extract 512-dim embedding."""
    x1, y1, x2, y2 = bbox
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    blob = face.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # [1,3,112,112]
    vec = session.run(["1333"], {"input.1": blob})[0]  # [1,512]
    return vec[0]


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if image_path is None:
        photos = sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg")))
        if not photos:
            print("No photos found in data/photo/")
            return
        image_path = photos[0]

    print(f"Image: {os.path.basename(image_path)}")

    # Load models
    import onnxruntime as ort
    det_session = ort.InferenceSession(MODEL_DET)
    rec_session = ort.InferenceSession(MODEL_REC)
    print("Models loaded (onnxruntime)")

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: cannot read {image_path}")
        return
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    # Detect
    faces = detect_faces(img, det_session)
    if not faces:
        print("No faces detected")
        return

    print(f"Detected {len(faces)} face(s), using top-1 (score={faces[0]['score']:.3f})")
    bbox = faces[0]["bbox"]
    print(f"  bbox: {bbox}")

    # Recognize
    emb = extract_embedding(img, bbox, rec_session)
    if emb is None:
        print("ERROR: embedding extraction failed")
        return

    print(f"\n  Embedding dim: {emb.shape[0]}")
    print(f"  Embedding norm: {np.linalg.norm(emb):.4f}")
    print(f"  First 8 values: {emb[:8].tolist()}")
    print(f"\n  SUCCESS: recognizer works!")


if __name__ == "__main__":
    main()
