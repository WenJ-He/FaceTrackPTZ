"""Visualize detection + recognition pipeline result on a composite image.

Usage:
    python3 scripts/test_visualize.py                     # auto composite from data/photo
    python3 scripts/test_visualize.py path/to/image.jpg   # use specific image

The script:
  1. Stitches multiple photos into a panoramic composite (if no image given)
  2. Detects faces via ONNX model
  3. Sorts faces with scanner (left-to-right, top-to-bottom)
  4. Recognizes each face against vector_db
  5. Draws bounding boxes + labels on the image (Chinese supported via PIL)
  6. Saves result to outputs/result.jpg
"""

from __future__ import annotations

import glob
import os
import random
import sys

import cv2
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src.config import Config
from src.vector_db import VectorDB
from src.scanner import Scanner
from src.detector import nms
from src.models import BBox, Detection

MODEL_DET = os.path.join(PROJECT, "models", "facedect", "1", "model.onnx")
MODEL_REC = os.path.join(PROJECT, "models", "facerecognize", "1", "model.onnx")
DB_PATH = os.path.join(PROJECT, "data", "face_db.sqlite")
PHOTO_DIR = os.path.join(PROJECT, "data", "photo")
OUTPUT_DIR = os.path.join(PROJECT, "outputs")

# Chinese font path: prefer Windows simhei, fallback to DroidSans
_FONT_PATHS = [
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def _find_font():
    for p in _FONT_PATHS:
        if os.path.isfile(p):
            return p
    return None


def make_composite(photo_dir, n=5, seed=42):
    """Randomly pick n photos and stitch them side-by-side."""
    exts = ("*.jpg", "*.png")
    photos = []
    for ext in exts:
        photos.extend(glob.glob(os.path.join(photo_dir, ext)))
    if not photos:
        return None

    random.seed(seed)
    chosen = random.sample(photos, min(n, len(photos)))

    target_h = 600
    strips = []
    for path in chosen:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = target_h / h
        img = cv2.resize(img, (int(w * scale), target_h))
        strips.append(img)

    if not strips:
        return None
    return np.hstack(strips)


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

    # Debug: raw detection count
    print(f"  [DEBUG] Raw detections (score >= {score_thresh}): {len(results)}")
    for i, d in enumerate(results):
        b = d.bbox
        print(f"    raw[{i}]  score={d.score:.4f}  bbox=({b.x1},{b.y1})-({b.x2},{b.y2})  "
              f"cx={b.cx:.0f} cy={b.cy:.0f}")

    results = nms(results, nms_iou)
    print(f"  [DEBUG] After NMS (iou={nms_iou}): {len(results)}")
    return results


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


def draw_labels(img, annotations):
    """Draw boxes and Chinese labels using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = _find_font()
    if font_path:
        font = ImageFont.truetype(font_path, 20)
        font_small = ImageFont.truetype(font_path, 14)
    else:
        font = ImageFont.load_default()
        font_small = font

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    colors = [
        (0, 200, 0), (200, 120, 0), (200, 0, 200),
        (0, 180, 200), (200, 0, 50), (120, 200, 0),
    ]

    for ann in annotations:
        tid = ann["tid"]
        x1, y1, x2, y2 = ann["bbox"]
        name = ann["name"]
        sim = ann["sim"]
        score = ann["score"]
        color = colors[(tid - 1) % len(colors)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Label background
        label = f"{tid} {name} {sim:.2f}"
        bbox_text = draw.textbbox((x1, y1 - 24), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1 + 2, y1 - 24), label, fill=(255, 255, 255), font=font)

        # Debug: score below box
        debug_text = f"score={score:.3f} ({x1},{y1})-({x2},{y2})"
        draw.text((x1, y2 + 2), debug_text, fill=color, font=font_small)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    import onnxruntime as ort

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load or compose image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot read: {image_path}")
            return
        print(f"Input: {image_path} ({img.shape[1]}x{img.shape[0]})")
    else:
        img = make_composite(PHOTO_DIR, n=5)
        if img is None:
            print("No photos found")
            return
        print(f"Composite: {img.shape[1]}x{img.shape[0]}")

    config = Config({
        "scan": {"row_bucket": 80},
        "detection": {"score_threshold": 0.3},
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

    # Detect
    print("\n=== Detection ===")
    detections = detect_faces(det_session, img, score_thresh=0.3, nms_iou=0.5)
    print(f"Final detections: {len(detections)}")
    if not detections:
        print("No faces detected")
        # Save detection-only image anyway
        cv2.imwrite(os.path.join(OUTPUT_DIR, "result.jpg"), img)
        return

    # Save detection-only image for debug
    det_img = img.copy()
    for i, d in enumerate(detections):
        b = d.bbox
        cv2.rectangle(det_img, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
        cv2.putText(det_img, f"raw{i}", (b.x1, b.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "detect_only.jpg"), det_img)
    print(f"Detection debug image: {os.path.join(OUTPUT_DIR, 'detect_only.jpg')}")

    # Sort
    print("\n=== Sort ===")
    scanner = Scanner(config)
    sorted_faces = scanner.sort_faces(detections)

    print("  Original (detection order):")
    for i, d in enumerate(detections):
        b = d.bbox
        print(f"    det[{i}]  cx={b.cx:.0f} cy={b.cy:.0f}  score={d.score:.4f}")
    print("  Sorted (scanner order):")
    for i, d in enumerate(sorted_faces):
        b = d.bbox
        row_bkt = int(b.cy // 80)
        print(f"    sort[{i}]  cx={b.cx:.0f} cy={b.cy:.0f}  row_bkt={row_bkt}  score={d.score:.4f}")

    # Recognize in scan order
    print("\n=== Recognize ===")
    annotations = []

    for i, det in enumerate(sorted_faces):
        b = det.bbox

        emb = extract_embedding(rec_session, img, b)
        if emb is None:
            print(f"  sort[{i}]  <failed to extract embedding>")
            continue

        matches = db.search(emb, top_k=5)
        name, sim = matches[0]

        print(f"  sort[{i}]  name={name}  sim={sim:.4f}  "
              f"score={det.score:.4f}  bbox=({b.x1},{b.y1})-({b.x2},{b.y2})")

        # Verify identity is valid Chinese string
        print(f"    [DEBUG] identity repr: {repr(name)}  len={len(name)}")

        annotations.append({
            "tid": i + 1,
            "bbox": (b.x1, b.y1, b.x2, b.y2),
            "name": name,
            "sim": sim,
            "score": det.score,
        })

    # Draw and save
    result = draw_labels(img, annotations)
    out_path = os.path.join(OUTPUT_DIR, "result.jpg")
    cv2.imwrite(out_path, result)
    print(f"\nSaved: {out_path}")

    db.close()


if __name__ == "__main__":
    main()
