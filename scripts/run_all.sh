#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== FaceTrack PTZ: Init + Test ==="
echo ""

# Step 1: init vector_db
echo "--- Step 1: Initialize face vector database ---"
python3 "$SCRIPT_DIR/init_vector_db.py"
echo ""

# Step 2: test recognizer
echo "--- Step 2: Test recognizer model ---"
python3 "$SCRIPT_DIR/test_recognizer.py"
echo ""

# Step 3: test full pipeline
echo "--- Step 3: Test recognition pipeline (detect → recognize → search) ---"
python3 "$SCRIPT_DIR/test_recognition_pipeline.py"
echo ""

echo "=== All done ==="
