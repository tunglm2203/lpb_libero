#!/bin/bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/miniconda3/envs/lpb/bin:$PATH

PROJECT="GBC-PBRL-2026"
ENTITY="Robotics_VLA"

GPU=0

ALL_CONFIGS=(
  "bc_aloha_diffusion_policy_cnn.yaml"
  # "bc_libero_diffusion_policy_cnn.yaml"
)
VAL_RATIO_ALL=(
  0.1
  # 0.3
#  0.8
#  0.9
)

for VAL_RATIO in "${VAL_RATIO_ALL[@]}"; do
  TRAIN_RATIO=$(echo "1 - $VAL_RATIO" | bc -l)
  EXP_NAME="mixedbc_dp_cnn_ERate${TRAIN_RATIO}"

  for CONFIG_NAME in "${ALL_CONFIGS[@]}"; do

    DATASET_PATH="/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/data/aloha/fold_shirt/fold_shirt_demo.hdf5"
    ROLLOUT_DATA="/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/data/aloha/fold_shirt/short_folding_rollout.hdf5"
    for SEED in 42; do
      HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --config-dir=diffusion_policy/config/bc_dp \
        --config-name=$CONFIG_NAME \
        checkpoint_dir='/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/reproduce/aloha_image/None/2026.06.02_03.50.28_train_diffusion_unet_hybrid_aloha_image/checkpoints/40.ckpt' \
        training.resume=False \
        name="${EXP_NAME}" \
        logging.project="${PROJECT}" +logging.entity="${ENTITY}" \
        logging.name="${EXP_NAME}" \
        training.rollout_every=5 training.checkpoint_every=10 \
        task.dataset_path=${DATASET_PATH} \
        task.env_runner.dataset_path=${DATASET_PATH} \
        task.dataset.dataset_path=${DATASET_PATH} \
        task.dataset.val_ratio=${VAL_RATIO} \
        task.rollout_dataset.mixed_bc=True task.rollout_dataset.filtered_bc=False \
        task.rollout_dataset.dataset_path=${ROLLOUT_DATA} \
        training.seed=${SEED} \
        training.num_epochs=500 \
        checkpoint.topk.k=20 \
        logging.mode="online" \
        hydra.run.dir='logs/mixedbc/${task_name}/${logging.group}/${logging.name}'
    done
  done
done
