#!/usr/bin/env bash
set -e

# ============================================
# OmniRoam Preview Stage Inference Script
# ============================================

python infer_omniroam.py \
  --local_images_dir vis_images \
  --height 480 \
  --width 960 \
  --num_frames 81 \
  --ckpt_path models/OmniRoam/Preview/preview.ckpt \
  --enable_speed_control \
  --speed_fixed 1.0 \
  --use_cam_traj \
  --traj_mode fixed \
  --traj_preset forward \
  --re_scale_pose fixed:1.0 \
  --traj_s_curve_amp_m 1.4 \
  --traj_loop_radius_m 1.5 \
  --cfg_scale 5.0 \
  --num_inference_steps 50 \
  --output_dir ./vis_ours_480p_speed_1_forward \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7



