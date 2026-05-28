#!/bin/bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/miniconda3/envs/lpb/bin:$PATH

GPU=0

NUM_EPISODES=10
c
ALL_CONFIGS=(
  "datacollect_libero_10.yaml"
)

for CONFIG_NAME in "${ALL_CONFIGS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --config-dir=diffusion_policy/config/data_collect/ \
    --config-name=$CONFIG_NAME \
    name='datacollect_diffusion_unet' \
    collecting.num_episodes=${NUM_EPISODES} \
    collecting.render_image=True \
    hydra.run.dir='logs/collect_data_200eps/${task_name}/${name}_${task_name}' \
    checkpoint_dir='/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/offficial/checkpoints/160.pth' \
    task.env_runner.max_steps=15 \
    task.env_runner.collect_data=true
done
