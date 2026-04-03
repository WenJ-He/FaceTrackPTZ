"""FaceTrack PTZ — Multi-stage PTZ recognition pipeline.

Usage:
    python3 main.py [image_path]

For each detected face:
  Stage 0: Recognize from panoramic detection bbox
  Stage 1..N: Calculate PTZ zoom coordinates, recognize at tighter crop
  Early stop: similarity >= accept_threshold OR gain < min_similarity_gain
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

from src.config import load_config
from src.vector_db import VectorDB
from src.scanner import Scanner
from src.ptz_controller import PTZController
from src.state_machine import StateMachine, State
from src.detector import nms
from src.models import BBox, Detection, TargetRecord, StageRecord, RecognitionResult
from src.logger import setup_logging

PROJECT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT, "config.yaml")
MODEL_DET = os.path.join(PROJECT, "models", "facedect", "1280", "best.onnx")
MODEL_REC = os.path.join(PROJECT, "models", "facerecognize", "1", "model.onnx")
PHOTO_DIR = os.path.join(PROJECT, "data", "photo")


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


def recognize_at_stage(rec_session, img, bbox, db, top_k):
    """Extract embedding, search vector_db, return RecognitionResult or None."""
    emb = extract_embedding(rec_session, img, bbox)
    if emb is None:
        return None, None
    results = db.search(emb, top_k=top_k)
    if not results:
        return None, emb
    top1_name, top1_sim = results[0]
    rec = RecognitionResult(
        identity=top1_name,
        similarity=float(top1_sim),
        top_k=[(n, float(s)) for n, s in results],
        vector=emb.tolist(),
    )
    return rec, emb


def print_target_detail(target_id, trec, ptz_rects):
    """Print per-stage detail for one target."""
    for s in trec.stages:
        rec = s.recognition
        tag = " (panoramic)" if s.stage_num == 0 else ""
        print(f"  Stage {s.stage_num}: identity={rec.identity:12s}  "
              f"sim={rec.similarity:.4f}  "
              f"delta_s0={s.similarity_delta_from_s0:+.4f}  "
              f"delta_prev={s.similarity_delta_from_prev:+.4f}  "
              f"fixed_sim={rec.fixed_identity_sim:.4f}{tag}")
    print(f"  Final: {trec.final_identity}  sim={trec.final_similarity:.4f}  "
          f"stages={len(trec.stages)}  gain={trec.total_gain:+.4f}")


def print_summary(all_targets):
    """Print summary table for all targets."""
    print(f"\n{'='*90}")
    print("=== Summary ===")
    header = (f"{'Target':>8}  {'Stages':>7}  "
              f"{'Final Identity':16s}  {'Final Sim':>10}  "
              f"{'Best Sim':>10}  {'Gain':>8}  {'Fixed Identity':16s}")
    print(header)
    print("-" * 90)
    for t in all_targets:
        print(f"#{t.target_id:<7}  {len(t.stages):>7}  "
              f"{(t.final_identity or '---'):16s}  {t.final_similarity:>10.4f}  "
              f"{t.best_similarity:>10.4f}  {t.total_gain:>+8.4f}  "
              f"{(t.fixed_identity or '---'):16s}")


def run(image_path):
    import onnxruntime as ort

    config = load_config(CONFIG_PATH)
    setup_logging(config.get("logging.level", "INFO"))

    max_stages = config.get("scan.max_zoom_stages", 5)
    accept_threshold = config.get("recognition.recognition_accept_threshold", 0.6)
    min_gain = config.get("recognition.min_similarity_gain", 0.02)
    top_k = config.get("recognition.top_k", 5)

    sm = StateMachine()
    sm.transition(State.CONNECTING_STREAM)

    db = VectorDB(config)
    db.open()
    ptz = PTZController(config)
    det_session = ort.InferenceSession(MODEL_DET)
    rec_session = ort.InferenceSession(MODEL_REC)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    print(f"Image: {os.path.basename(image_path)} ({img.shape[1]}x{img.shape[0]})")
    print(f"VectorDB: {db.count()} identities")
    print(f"Config: max_stages={max_stages}  accept_threshold={accept_threshold}  "
          f"min_gain={min_gain}  top_k={top_k}")

    # DETECT
    sm.transition(State.DETECTING)
    t0 = time.time()
    detections = detect_faces(det_session, img, 0.5)
    det_ms = (time.time() - t0) * 1000
    print(f"\nDetected {len(detections)} face(s) in {det_ms:.0f}ms")

    if not detections:
        print("No faces detected")
        db.close()
        return

    # SORT
    sm.transition(State.SORTING)
    scanner = Scanner(config)
    sorted_faces = scanner.sort_faces(detections)

    # MULTI-STAGE RECOGNITION LOOP
    print(f"\n=== Multi-Stage PTZ Recognition ===")
    all_targets = []

    while True:
        target = scanner.select_next(sorted_faces)
        if target is None:
            break

        target_id, det = target
        bbox = det.bbox

        trec = TargetRecord(
            target_id=target_id,
            sort_index=target_id,
            start_time=time.time(),
        )

        print(f"\n--- Target #{target_id} "
              f"bbox=({bbox.x1},{bbox.y1})-({bbox.x2},{bbox.y2}) "
              f"det_score={det.score:.3f} ---")

        # Stage 0: panoramic recognition
        sm.transition(State.MOVING)
        sm.transition(State.RECOGNIZING)
        rec0, emb0 = recognize_at_stage(rec_session, img, bbox, db, top_k)

        if rec0 is None:
            print("  Stage 0: <embedding failed>")
            sm.transition(State.HOLDING)
            trec.finalize()
            all_targets.append(trec)
            sm.transition(State.NEXT_TARGET)
            continue

        stage0 = StageRecord(
            stage_num=0,
            timestamp=time.time(),
            bbox=bbox,
            recognition=rec0,
        )
        trec.add_stage(stage0)
        print(f"  Stage 0: identity={rec0.identity:12s}  sim={rec0.similarity:.4f}  "
              f"delta_s0={stage0.similarity_delta_from_s0:+.4f}  "
              f"delta_prev={stage0.similarity_delta_from_prev:+.4f}  "
              f"fixed_sim={rec0.fixed_identity_sim:.4f} (panoramic)")

        # Check early stop at stage 0
        if rec0.similarity >= accept_threshold:
            print(f"  >> ACCEPTED at stage 0 (sim={rec0.similarity:.4f} >= {accept_threshold})")
        else:
            # Stages 1..N
            for stage_num in range(1, max_stages):
                sm.transition(State.MOVING)
                rect = ptz.calculate_coordinates(bbox, stage_num)
                print(f"  PTZ stage {stage_num}: rect=({rect[0]:3d},{rect[1]:3d},"
                      f"{rect[2]:3d},{rect[3]:3d})")

                sm.transition(State.RECOGNIZING)
                rec_s, _ = recognize_at_stage(rec_session, img, bbox, db, top_k)
                if rec_s is None:
                    print(f"  Stage {stage_num}: <embedding failed>")
                    break

                stage_s = StageRecord(
                    stage_num=stage_num,
                    timestamp=time.time(),
                    bbox=bbox,
                    recognition=rec_s,
                )
                trec.add_stage(stage_s)

                delta_prev = stage_s.similarity_delta_from_prev
                print(f"  Stage {stage_num}: identity={rec_s.identity:12s}  "
                      f"sim={rec_s.similarity:.4f}  "
                      f"delta_s0={stage_s.similarity_delta_from_s0:+.4f}  "
                      f"delta_prev={delta_prev:+.4f}  "
                      f"fixed_sim={rec_s.fixed_identity_sim:.4f}")

                if rec_s.similarity >= accept_threshold:
                    print(f"  >> ACCEPTED at stage {stage_num} "
                          f"(sim={rec_s.similarity:.4f} >= {accept_threshold})")
                    break
                if delta_prev < min_gain:
                    print(f"  >> STOP: gain={delta_prev:+.4f} < {min_gain}")
                    break

        # Finalize target
        sm.transition(State.HOLDING)
        trec.finalize()
        print(f"  Final: {trec.final_identity}  sim={trec.final_similarity:.4f}  "
              f"stages={len(trec.stages)}  gain={trec.total_gain:+.4f}")
        all_targets.append(trec)
        sm.transition(State.NEXT_TARGET)

    # Summary
    print_summary(all_targets)
    db.close()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="FaceTrack PTZ multi-stage recognition")
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
