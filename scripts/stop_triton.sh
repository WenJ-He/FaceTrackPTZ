#!/usr/bin/env bash
# Stop Triton Inference Server
set -e
cd "$(dirname "$0")/.."
docker compose down
echo "Triton stopped."
