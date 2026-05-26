#!/bin/bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/miniconda3/envs/lpb/bin:$PATH
export PYTHONPATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero:$PYTHONPATH
GPU=0

# export MUJOCO_GL="glx"

OUTDIR="eval_logs/LR6/cplkl_pseu_dpT_ExpD10_Rew1_N20000_L100_bias0.25_Eq0_thr0.05_1ER_SFTpos0_strid1_segM0.2_0_nD10_beta0.01_clip0.3_unclipwin1/cplkl_pseu_dpT_ExpD10_Rew1_N20000_L100_bias0.25_Eq0_thr0.05_1ER_SFTpos0_strid1_segM0.2_0_nD10_beta0.01_clip0.3_unclipwin1_42/checkpoints/latest.ckpt"

NOISE_SCHEDULER="ddpm"
NUM_INFERENCE_STEPS=100

ALL_DATASETS=(
  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5"
)


POLICY_CKPT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/pbrl/libero_image/cplkl_pseu_dpT_ExpD10_Rew1_N20000_L100_bias0.25_Eq0_thr0.05_1ER_SFTpos0_strid1_segM0.2_0_nD10_beta0.01_clip0.3_unclipwin1/cplkl_pseu_dpT_ExpD10_Rew1_N20000_L100_bias0.25_Eq0_thr0.05_1ER_SFTpos0_strid1_segM0.2_0_nD10_beta0.01_clip0.3_unclipwin1_42/checkpoints/latest.ckpt"

for DATASET in "${ALL_DATASETS[@]}"; do
  TASK=${DATASET%.hdf5}
  for SEED in 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPU} python eval_libero.py \
      --checkpoint ${POLICY_CKPT} \
      --output_dir "${OUTDIR}" \
      --noise_scheduler ${NOISE_SCHEDULER} --num_inference_steps ${NUM_INFERENCE_STEPS} \
      --dataset_name ${DATASET} \
      --ntest 50 \
      --seed ${SEED} \
      --max_steps 500
  done
done