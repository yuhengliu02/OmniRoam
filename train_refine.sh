#!/usr/bin/env bash
set -e
umask 007

# ============================================
# Auto-detect or use environment variables
# ============================================

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "[ERROR] nvidia-smi not found. Is CUDA available?"
    exit 1
fi

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}
JOB_ID=${JOB_ID:-"OmniRoam_$(date +%Y%m%d_%H%M%S)"}

# ============================================
# Network and NCCL settings
# ============================================
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print configuration
echo ""
echo "============================================"
echo "OmniRoam Refine Training Configuration"
echo "============================================"
echo "GPUs per node:    ${NUM_GPUS}"
echo "Number of nodes:  ${WORLD_SIZE}"
echo "Node rank:        ${RANK}"
echo "Master address:   ${MASTER_ADDR}"
echo "Master port:      ${MASTER_PORT}"
echo "Job ID:           ${JOB_ID}"
echo "============================================"
echo ""

# ============================================
# Data and Model Paths
# ============================================

# Dataset paths
SPLIT_JSON="configs/train_test_files_test.json"
DATA_ROOT="data/InteriorGS-360video"
FRAMES_SUBDIR="pano_camera0"
MAX_FRAMES=800
FRAME_EXT="png"

# Model paths
TEXT_ENCODER_PATH="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
VAE_PATH="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
DIT_PATH="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"

# Output path
OUTPUT_PATH="./output/refine_training"

# W&B settings
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_NAME="refine_$(date +%Y%m%d_%H%M%S)"
WANDB_API_KEY_FILE="configs/wandb.txt"

# ============================================
# Launch Training
# ============================================

echo "[INFO] Starting training..."
echo ""

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes=${WORLD_SIZE} \
  --node_rank=${RANK} \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv-id=${JOB_ID} \
  train_omniroam.py \
  --task refine \
  --train_test_split_json ${SPLIT_JSON} \
  --text_encoder_path ${TEXT_ENCODER_PATH} \
  --vae_path ${VAE_PATH} \
  --dit_path ${DIT_PATH} \
  --output_path ${OUTPUT_PATH} \
  --learning_rate 1e-4 \
  --use_gradient_checkpointing \
  --steps_per_epoch 8000 \
  --max_epochs 200 \
  --accumulate_grad_batches 1 \
  --checkpoint_every_n_epochs 1 \
  --dataloader_num_workers 4 \
  --height 720 \
  --width 1440 \
  --refine_speed_min 1.1 \
  --refine_speed_max 8.0 \
  --refine_window_policy random \
  --refine_degrade_down_h 480 \
  --refine_degrade_down_w 960 \
  --refine_use_global_context \
  --interiorgs_data_root ${DATA_ROOT} \
  --interiorgs_frames_subdir ${FRAMES_SUBDIR} \
  --interiorgs_max_frames ${MAX_FRAMES} \
  --interiorgs_frame_ext ${FRAME_EXT} \
  --num_nodes ${WORLD_SIZE} \
  --use_wandb \
  --wandb_run_name ${WANDB_RUN_NAME} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_entity ${WANDB_ENTITY} \
  --wandb_api_key_file ${WANDB_API_KEY_FILE}
  # --resume_ckpt_path path/to/your/checkpoint.ckpt

echo ""
echo "[INFO] Training complete!"
