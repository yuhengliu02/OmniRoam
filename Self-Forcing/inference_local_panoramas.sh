#!/bin/bash
set -e

# ==========================================
# Configuration - Modify these paths as needed
# ==========================================

# Model and config paths
CONFIG_PATH="./configs/self_forcing_dmd_omniroam.yaml"
CHECKPOINT_PATH="../models/OmniRoam/Self-Forcing/self-forcing.pt"  # default checkpoint path from the main OmniRoam setup

# Input data
LOCAL_FOLDER="../vis_images"

# Output settings
OUTPUT_FOLDER="./inference_output"
NUM_SAMPLES=5

# Trajectory settings
TRAJ_PRESET="forward"  # Options: forward, backward, left, right
TRAJ_STEP_M=1.0        # Step size in meters
RE_SCALE_POSE="fixed:1.0"  # Trajectory rescale: none | unit_median | fixed:<float>

# Video generation settings
HEIGHT=480
WIDTH=960
NUM_FRAMES=81
PROMPT="panoramic video"
SPEED_SCALAR=1.0

# Other settings
SEED=2026
USE_EMA=""  # Add --use_ema to use EMA parameters

# ==========================================
# Run Inference
# ==========================================

echo "=========================================="
echo "OmniRoam Self-Forcing - Local Inference"
echo "=========================================="
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Input Folder: ${LOCAL_FOLDER}"
echo "Output Folder: ${OUTPUT_FOLDER}"
echo "Trajectory: ${TRAJ_PRESET} (step=${TRAJ_STEP_M}m)"
echo "Resolution: ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "=========================================="
echo ""

# Check if input folder exists
if [ ! -d "${LOCAL_FOLDER}" ]; then
    echo "Error: Input folder does not exist: ${LOCAL_FOLDER}"
    echo "Please create the folder and add panorama images (.jpg, .png, .bmp)"
    exit 1
fi

# Check if config exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config file does not exist: ${CONFIG_PATH}"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint file does not exist: ${CHECKPOINT_PATH}"
    exit 1
fi

# Run inference
python custom_inference.py \
    --config_path "${CONFIG_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --local_folder "${LOCAL_FOLDER}" \
    --traj_preset "${TRAJ_PRESET}" \
    --traj_step_m ${TRAJ_STEP_M} \
    --re_scale_pose "${RE_SCALE_POSE}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num_frames ${NUM_FRAMES} \
    --prompt "${PROMPT}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --num_samples ${NUM_SAMPLES} \
    --speed_scalar ${SPEED_SCALAR} \
    --seed ${SEED} \
    ${USE_EMA}

echo ""
echo "=========================================="
echo "Inference Complete!"
echo "Results saved to: ${OUTPUT_FOLDER}"
echo "=========================================="
