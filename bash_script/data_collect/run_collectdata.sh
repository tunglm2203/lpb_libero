#!/bin/bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/miniconda3/envs/lpb/bin:$PATH

GPU=0

NUM_EPISODES=100

ALL_CONFIGS=(
  "datacollect_libero_10.yaml"
  # "datacollect_transport.yaml"
)

checkpoint=/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/libero_base_policy/checkpoints/160.pth
# checkpoint=/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/transport_base_policy/checkpoints/270.ckpt

for CONFIG_NAME in "${ALL_CONFIGS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --config-dir=diffusion_policy/config/data_collect/ \
    --config-name=$CONFIG_NAME \
    name='datacollect_diffusion_unet' \
    collecting.num_episodes=${NUM_EPISODES} \
    collecting.render_image=True \
    hydra.run.dir=logs/libero_collect_data_${NUM_EPISODES}eps/${task_name}/${name}_${task_name} \
    checkpoint_dir=${checkpoint} \
    task.env_runner.max_steps=500 \
    task.env_runner.collect_data=true
done
