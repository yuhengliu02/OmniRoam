#!/usr/bin/env bash
set -e

# ============================================
# OmniRoam Refine Stage Inference Script
# ============================================

python infer_omniroam.py \
  --enable_refine \
  --refine_local_dir path/to/generated/preview/videos \
  --refine_num_segments 8 \
  --refine_degrade_down_h 480 \
  --refine_degrade_down_w 960 \
  --refine_use_crossfade \
  --refine_crossfade_alpha 0.5 \
  --height 720 \
  --width 1440 \
  --num_frames 81 \
  --ckpt_path models/OmniRoam/Refine/refine.ckpt \
  --output_dir ./refined \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7

