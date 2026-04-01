#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Starting Triton Inference Server..."
cd "$PROJECT_DIR"

# Check docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: docker not installed."
    echo "Install Docker first: https://docs.docker.com/engine/install/"
    echo ""
    echo "Alternative: use onnxruntime directly (no Docker needed):"
    echo "  python3 scripts/test_recognizer.py"
    exit 1
fi

# Pull image if not present
if ! docker image inspect nvcr.io/nvidia/tritonserver:24.01-py3 &>/dev/null; then
    echo "Pulling Triton image (first time only, ~4GB)..."
    docker pull nvcr.io/nvidia/tritonserver:24.01-py3
fi

docker compose up -d

echo ""
echo "Waiting for Triton to become ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/v2/health/ready 2>/dev/null | grep -q "ready"; then
        echo "Triton is ready!"
        echo "  HTTP:  localhost:8000"
        echo "  gRPC:  localhost:8001"
        echo "  Metrics: localhost:8002"
        exit 0
    fi
    sleep 2
done

echo "WARNING: Triton not ready after 60s. Check: docker compose logs triton"
