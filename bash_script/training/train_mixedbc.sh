#!/bin/bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/miniconda3/envs/lpb/bin:$PATH

PROJECT="GBC-PBRL-2026"
ENTITY="Robotics_VLA"

GPU=0

ALL_CONFIGS=(
  "bc_libero_diffusion_policy_cnn.yaml"
)
VAL_RATIO_ALL=(
#  0.02
  0.3
#  0.8
#  0.9
)

for VAL_RATIO in "${VAL_RATIO_ALL[@]}"; do
  TRAIN_RATIO=$(echo "1 - $VAL_RATIO" | bc -l)
  EXP_NAME="mixedbc_dp_transformer_ERate${TRAIN_RATIO}"

  for CONFIG_NAME in "${ALL_CONFIGS[@]}"; do

    DATASET_PATH="data/libero_10/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo"
    ROLLOUT_DATA="logs/collect_data_200eps/libero_10/datacollect_diffusion_unet_libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo"
    for SEED in 42; do
      HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --config-dir=diffusion_policy/config/bc_dp \
        --config-name=$CONFIG_NAME \
        training.resume=True \
        name="${EXP_NAME}" \
        logging.project="${PROJECT}" +logging.entity="${ENTITY}" \
        training.rollout_every=50 training.checkpoint_every=100000 \
        task.dataset_path=${DATASET_PATH} \
        task.env_runner.dataset_path=${DATASET_PATH} \
        task.dataset.dataset_path=${DATASET_PATH} \
        task.dataset.val_ratio=${VAL_RATIO} \
        task.rollout_dataset.mixed_bc=True task.rollout_dataset.filtered_bc=False \
        task.rollout_dataset.dataset_path=${ROLLOUT_DATA} \
        training.seed=${SEED} \
        training.num_epochs=1000 \
        checkpoint.topk.k=1 \
        logging.mode="online" \
        hydra.run.dir='logs/pbrl/${task_name}/${logging.group}/${logging.name}'
    done
  done
done
