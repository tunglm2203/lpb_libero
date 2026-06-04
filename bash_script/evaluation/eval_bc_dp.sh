#!/bin/bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/miniconda3/envs/lpb/bin:$PATH
export PYTHONPATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero:$PYTHONPATH
GPU=0

# export MUJOCO_GL="glx"

# OUTDIR="eval_logs/LR6/cplkl_pseu_dpT_ExpD10_Rew1_N20000_L100_bias0.25_Eq0_thr0.05_1ER_SFTpos0_strid1_segM0.2_0_nD10_beta0.01_clip0.3_unclipwin1/cplkl_pseu_dpT_ExpD10_Rew1_N20000_L100_bias0.25_Eq0_thr0.05_1ER_SFTpos0_strid1_segM0.2_0_nD10_beta0.01_clip0.3_unclipwin1_42/checkpoints/latest.ckpt"

NOISE_SCHEDULER="ddpm"
NUM_INFERENCE_STEPS=100

ALL_DATASETS=(
  # "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5"
  "transport_ph_demo_v141_20_perc.hdf5"
)


PROJECT_DIR=/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero
POLICY_CKPT="${PROJECT_DIR}/logs/pbrl/transport_image/None/transport_2026.06.04_04.58.08_cplkl_pseu_dpT_ExpD10_Rew0_N20000_L300_1ER_SFTpos0_segM0.2_nD35_beta0.01_clip0.3_unclipwin1_smooth0.1/checkpoints/epoch_0010.ckpt"

OUTDIR="${PROJECT_DIR}/eval_logs/transport/pbrl_transport_N20000_segM0.2/checkpoints/epoch_0010.ckpt"

for DATASET in "${ALL_DATASETS[@]}"; do
  TASK=${DATASET%.hdf5}
  for SEED in 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPU} python eval_libero.py \
      --checkpoint ${POLICY_CKPT} \
      --output_dir "${OUTDIR}" \
      --noise_scheduler ${NOISE_SCHEDULER} --num_inference_steps ${NUM_INFERENCE_STEPS} \
      --dataset_name ${DATASET} \
      --ntest 10 \
      --seed ${SEED} \
      --max_steps 700
  done
done