"""Grid layout test: verify scanner sorting + recognition on a 3x3 grid.

Constructs a 3x3 grid of face photos, detects faces, sorts via scanner,
recognizes each face against vector_db, and draws results on the image.

Usage:
    python3 scripts/test_grid_layout.py
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
from src.scanner import Scanner
from src.vector_db import VectorDB
from src.detector import nms
from src.models import BBox, Detection

MODEL_DET = os.path.join(PROJECT, "models", "facedect", "1", "model.onnx")
MODEL_REC = os.path.join(PROJECT, "models", "facerecognize", "1", "model.onnx")
DB_PATH = os.path.join(PROJECT, "data", "face_db.sqlite")
PHOTO_DIR = os.path.join(PROJECT, "data", "photo")
OUTPUT_DIR = os.path.join(PROJECT, "outputs")
FONT_PATHS = [
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def find_font():
    for p in FONT_PATHS:
        if os.path.isfile(p):
            return p
    return None


def make_grid(photo_dir, rows=3, cols=3, cell_h=400, seed=7):
    """Stitch photos into a rows x cols grid."""
    exts = ("*.jpg", "*.png")
    photos = []
    for ext in exts:
        photos.extend(glob.glob(os.path.join(photo_dir, ext)))
    random.seed(seed)
    chosen = random.sample(photos, min(rows * cols, len(photos)))

    cell_w = int(cell_h * 0.75)
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    grid[:] = 240

    names = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(chosen):
                break
            img = cv2.imread(chosen[idx])
            if img is None:
                idx += 1
                continue
            h, w = img.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

            y_off = r * cell_h + (cell_h - img.shape[0]) // 2
            x_off = c * cell_w + (cell_w - img.shape[1]) // 2
            y1, y2 = y_off, y_off + img.shape[0]
            x1, x2 = x_off, x_off + img.shape[1]
            grid[y1:y2, x1:x2] = img

            names.append(os.path.splitext(os.path.basename(chosen[idx]))[0])
            idx += 1

    return grid, names


def detect_faces(session, frame, score_thresh=0.3, nms_iou=0.4):
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


def draw_result(img, annotations):
    """Draw boxes, sort numbers, row labels, and identity using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_font()
    font = ImageFont.truetype(font_path, 20) if font_path else ImageFont.load_default()
    font_big = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    colors = [
        (220, 60, 60), (60, 180, 60), (60, 60, 220),
        (200, 160, 0), (180, 0, 200), (0, 180, 200),
        (200, 100, 50), (100, 200, 100), (150, 100, 200),
    ]

    for ann in annotations:
        tid = ann["tid"]
        x1, y1, x2, y2 = ann["bbox"]
        row_bkt = ann["row_bucket"]
        color = colors[(tid - 1) % len(colors)]

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Top label: #N rowX  identity sim
        identity = ann.get("identity", "?")
        sim = ann.get("similarity", 0.0)
        top_label = f"#{tid} row{row_bkt}  {identity} {sim:.2f}"
        tb = draw.textbbox((x1, y1 - 26), top_label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((x1, y1 - 26), top_label, fill=(255, 255, 255), font=font)

        # Center: large sort number
        num_text = str(tid)
        nb = draw.textbbox((0, 0), num_text, font=font_big)
        nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        nx = (x1 + x2) // 2 - nw // 2
        ny = (y1 + y2) // 2 - nh // 2
        draw.rectangle([nx - 4, ny - 2, nx + nw + 4, ny + nh + 2], fill=(0, 0, 0, 160))
        draw.text((nx, ny), num_text, fill=(255, 255, 0), font=font_big)

        # Bottom: debug info
        debug = f"cx={ann['cx']:.0f} cy={ann['cy']:.0f}  score={ann['det_score']:.3f}"
        draw.text((x1, y2 + 2), debug, fill=color, font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    import onnxruntime as ort

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    row_bucket = 80
    rows, cols = 3, 3
    cell_h = 400

    grid, grid_names = make_grid(PHOTO_DIR, rows=rows, cols=cols, cell_h=cell_h)
    print(f"Grid: {rows}x{cols} = {grid.shape[1]}x{grid.shape[0]}")
    print(f"Photos in grid: {grid_names}")

    # Init models
    det_session = ort.InferenceSession(MODEL_DET)
    rec_session = ort.InferenceSession(MODEL_REC)

    config = Config({
        "scan": {"row_bucket": row_bucket},
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

    # Detect
    detections = detect_faces(det_session, grid, score_thresh=0.3, nms_iou=0.4)
    print(f"\nDetected {len(detections)} face(s)")
    if not detections:
        print("No faces detected")
        return

    # Print raw order
    print("\n--- Raw (score desc) ---")
    for i, d in enumerate(sorted(detections, key=lambda d: d.score, reverse=True)):
        b = d.bbox
        print(f"  [{i}] score={d.score:.3f}  ({b.x1},{b.y1})-({b.x2},{b.y2})  "
              f"cx={b.cx:.0f} cy={b.cy:.0f}")

    # Sort
    scanner = Scanner(config)
    sorted_faces = scanner.sort_faces(detections)

    # Recognize each face
    print(f"\n--- Sorted + Recognized (row_bucket={row_bucket}) ---")
    annotations = []
    correct = 0
    total = 0

    for i, det in enumerate(sorted_faces):
        b = det.bbox
        row_bkt = int(b.cy // row_bucket)

        emb = extract_embedding(rec_session, grid, b)
        if emb is None:
            print(f"  #{i+1}  <embedding failed>")
            annotations.append({
                "tid": i + 1, "bbox": (b.x1, b.y1, b.x2, b.y2),
                "row_bucket": row_bkt, "cx": b.cx, "cy": b.cy,
                "identity": "<fail>", "similarity": 0.0, "det_score": det.score,
            })
            continue

        matches = db.search(emb, top_k=5)
        name, sim = matches[0]

        # Check accuracy against grid photo name
        total += 1
        # Find which grid cell this bbox center falls in
        cell_w = int(cell_h * 0.75)
        col = min(int(b.cx / cell_w), cols - 1)
        row = min(int(b.cy / cell_h), rows - 1)
        gt = grid_names[row * cols + col] if row * cols + col < len(grid_names) else "?"
        match = "OK" if name == gt else "MISS"
        if name == gt:
            correct += 1

        print(f"  #{i+1}  identity={name:10s}  sim={sim:.4f}  "
              f"row_bkt={row_bkt}  cx={b.cx:.0f} cy={b.cy:.0f}  "
              f"gt={gt}  [{match}]")

        annotations.append({
            "tid": i + 1,
            "bbox": (b.x1, b.y1, b.x2, b.y2),
            "row_bucket": row_bkt,
            "cx": b.cx,
            "cy": b.cy,
            "identity": name,
            "similarity": sim,
            "det_score": det.score,
        })

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.0f}%)")

    # Draw and save
    result = draw_result(grid, annotations)
    out_path = os.path.join(OUTPUT_DIR, "grid_with_recognition.jpg")
    cv2.imwrite(out_path, result)
    print(f"\nSaved: {out_path}")

    db.close()


if __name__ == "__main__":
    main()
