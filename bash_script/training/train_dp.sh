GPU=0

export MUJOCO_GL="glx"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} python train.py \
  --config-dir=. \
  --config-name=image_libero_diffusion_policy_cnn.yaml \
  training.seed=42 training.device=cuda:0 \
  logging.mode="disabled" \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'