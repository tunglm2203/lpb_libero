#!/bin/bash

GPU=0

PROJECT="GBC-LIBERO-PBRL-2026"
ENTITY="tunglm"

export MUJOCO_GL="glx"

VAL_RATIO_ALL=(
#  0.02
#  0.6
#  0.8
  0.9
)

for VAL_RATIO in "${VAL_RATIO_ALL[@]}"; do

  TRAIN_RATIO=$(echo "1 - $VAL_RATIO" | bc -l)
  EXP_NAME="dp_cnn_ERate${TRAIN_RATIO}"

  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --config-dir=. \
    --config-name=image_libero_diffusion_policy_cnn.yaml \
    training.seed=42 \
    name="${EXP_NAME}" \
    logging.project="${PROJECT}" +logging.entity="${ENTITY}" \
    logging.mode="online" \
    task.dataset.val_ratio=${VAL_RATIO} \
    hydra.run.dir='logs/reproduce/${task_name}/${logging.group}/${logging.name}'
done
