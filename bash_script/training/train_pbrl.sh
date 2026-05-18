#!/bin/bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/miniconda3/envs/lpb/bin:$PATH


PROJECT="GBC-PBRL-2026"
ENTITY="Robotics_VLA"

GPU=0

ALL_CONFIGS=(
  "pbrl_libero_diffusion_policy_cnn.yaml"
)

for VAL_RATIO in 0.1; do
  TRAIN_RATIO=$(echo "1 - $VAL_RATIO" | bc -l)
  for CONFIG_NAME in "${ALL_CONFIGS[@]}"; do
    
    for CPL_LOSS_TYPE in "sft"; do # "sft", "cpl", "cpl_kl"
      for BIAS_REG in 0.25; do      # 0.25 0.5 0.75 1
        USE_EXPERT_DATA=False
        DENSE_REWARD=True
        NUM_QUERIES=1000
        USE_BC=False
        BC_COEF=0
        IGNORE_EQUAL_PREF=False
        SEG_SIZE=150

        DATASET_PATH='data/libero_10/libero_10'   # ${task_name} will be replaced during run-time
        DATASET_1="logs/collect_data/libero_10/datacollect_diffusion_unet_libero_10"
        DATASET_2="logs/collect_data/libero_10/datacollect_diffusion_unet_libero_10"

        EXP_NAME="${CPL_LOSS_TYPE}_dpT_EData${USE_EXPERT_DATA}_Dense${DENSE_REWARD}_N${NUM_QUERIES}_L${SEG_SIZE}_bc${USE_BC}_${BC_COEF}_bias${BIAS_REG}_Equal${IGNORE_EQUAL_PREF}_1E2C_1ERate${TRAIN_RATIO}"

        for SEED in 42; do
          HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} python train.py \
            --config-dir=diffusion_policy/config/pbrl_dp \
            --config-name=$CONFIG_NAME \
            checkpoint_dir='/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/offficial/checkpoints/160.pth' \
            training.resume=True \
            name="${EXP_NAME}" \
            logging.project="${PROJECT}" +logging.entity="${ENTITY}" \
            training.rollout_every=10 training.checkpoint_every=50 \
            training.cpl_loss_type="${CPL_LOSS_TYPE}" \
            training.use_bc=${USE_BC} \
            training.use_expert_data=${USE_EXPERT_DATA} \
            training.dataset_1_dir=${DATASET_1} training.dataset_2_dir=${DATASET_2} \
            policy.bc_coef=${BC_COEF} \
            policy.bias_reg=${BIAS_REG} \
            policy.ignore_equal_pref=${IGNORE_EQUAL_PREF} \
            task.dataset_path=${DATASET_PATH} \
            task.env_runner.dataset_path=${DATASET_PATH} \
            task.dataset.dataset_path=${DATASET_PATH} \
            task.dense_reward=${DENSE_REWARD} \
            task.pref_dataset.num_queries=${NUM_QUERIES} \
            task.pref_dataset.sequence_length=${SEG_SIZE} \
            task.dataset.val_ratio=${VAL_RATIO} \
            training.seed=${SEED} \
            training.num_epochs=1000 \
            logging.mode="online" \
            hydra.run.dir='logs/pbrl/${task_name}/${logging.group}/${logging.name}'
        done
      done
    done
  done
done
