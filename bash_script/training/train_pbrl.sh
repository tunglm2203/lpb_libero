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
    
    for CPL_TYPE in "cplkl"; do # "sft", "cplkl"
      for BIAS_REG in 0.25; do      # 0.25 0.5 0.75 1

        SEG_SIZE=100

        USE_EXP_DATA_1=1   # To sample left segments
        USE_EXP_DATA_2=0   # To sample right segments
        DENSE_REWARD=1
        N_QUERIES=2
        IGNORE_EQUAL_PREF=0
        EQUAL_THRESHOLD=0.0
        N_EPOCH_SFT=100
        SFT_TYPE="pos"   # positive, both
        STRIDE=1
        CPL_BETA=0.03
        CLIP_MARGIN=0.3
        SEG_MARGIN=0.0      # Segment must beat the other by 60% coverage to win
        MIN_PROGRESS=0      # At least one segment must achieve 2% coverage
        N_DEMOS_FOR_PREF=30
        UNCLIP_WIN=1
        SMOOTH_LABEL=0.1


        DATASET_PATH='data/libero_10/libero_10'   # ${task_name} will be replaced during run-time
        DATASET_1="logs/collect_data_200eps/libero_10/datacollect_diffusion_unet_libero_10"
        DATASET_2="logs/collect_data_200eps/libero_10/datacollect_diffusion_unet_libero_10"

        EXP_NAME="${CPL_TYPE}_pseu_dpT_ExpD${USE_EXP_DATA_1}${USE_EXP_DATA_2}_Rew${DENSE_REWARD}_N${N_QUERIES}_L${SEG_SIZE}_1ER${TRAIN_RATIO}_SFT${SFT_TYPE}${N_EPOCH_SFT}_segM${SEG_MARGIN}_nD${N_DEMOS_FOR_PREF}_beta${CPL_BETA}_clip${CLIP_MARGIN}_unclipwin${UNCLIP_WIN}_smooth${SMOOTH_LABEL}"

        for SEED in 42; do
          HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} python train.py \
            --config-dir=diffusion_policy/config/pbrl_dp \
            --config-name=$CONFIG_NAME \
            checkpoint_dir='/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/offficial/checkpoints/160.pth' \
            training.resume=True \
            name="${EXP_NAME}" \
            logging.project="${PROJECT}" +logging.entity="${ENTITY}" \
            training.rollout_every=50 training.checkpoint_every=50 \
            training.cpl_loss_type="${CPL_TYPE}" \
            training.use_expert_data_1=${USE_EXP_DATA_1} training.use_expert_data_2=${USE_EXP_DATA_2} \
            training.dataset_1_dir=${DATASET_1} training.dataset_2_dir=${DATASET_2} \
            task.dataset_path=${DATASET_PATH} task.dense_reward=${DENSE_REWARD} \
            task.pref_dataset.num_queries=${N_QUERIES} task.pref_dataset.sequence_length=${SEG_SIZE} \
            task.pref_dataset.val_ratio_data1=${VAL_RATIO} \
            policy.bias_reg=${BIAS_REG} \
            policy.ignore_equal_pref=${IGNORE_EQUAL_PREF} \
            policy.beta=${CPL_BETA} \
            policy.clip_margin=${CLIP_MARGIN} \
            policy.unclip_win=${UNCLIP_WIN} \
            policy.smooth_label=${SMOOTH_LABEL} \
            training.preference_learning.equal_threshold=${EQUAL_THRESHOLD} \
            training.n_epoch_sft=${N_EPOCH_SFT} training.sft_type=${SFT_TYPE} \
            training.stride_ratio=${STRIDE} \
            training.seed=${SEED} \
            training.num_epochs=800 \
            training.pseudo_preference=True \
            task.pref_dataset.n_demos_for_preference=${N_DEMOS_FOR_PREF} \
            task.pref_dataset.seg_margin=${SEG_MARGIN} task.pref_dataset.min_progress=${MIN_PROGRESS} \
            logging.mode="online" \
            hydra.run.dir='logs/pbrl/${task_name}/${logging.group}/${logging.name}'
        done
      done
    done
  done
done
