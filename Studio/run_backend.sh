#!/bin/bash
# OmniRoam Backend Startup Script

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}


echo "=========================================="
echo "    OmniRoam Interactive System"
echo "=========================================="
echo "Device: cuda:${CUDA_VISIBLE_DEVICES}"
echo "Host: ${HOST}:${PORT}"
echo "=========================================="

cd "$(dirname "$0")"

python main.py



