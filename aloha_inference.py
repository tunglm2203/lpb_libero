
import os
import numpy as np
import hydra
import torch
import dill
from omegaconf import OmegaConf, open_dict
from termcolor import cprint
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.common.normalizer import LinearNormalizer



def setup_noise_schedule(cfg, noise_scheduler, num_inference_steps=None):
    if noise_scheduler.lower() == "ddpm":
        if cfg.policy.noise_scheduler['_target_'] == 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler':
            cprint(f"Skip setup scheduler: Pretrained model already used DDPM scheduler", "green")
        else:
            with open_dict(cfg.policy.noise_scheduler):
                cfg.policy.noise_scheduler = OmegaConf.create({
                    '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                    'num_train_timesteps': 100,
                    'beta_start': 0.0001,
                    'beta_end': 0.02,
                    'beta_schedule': 'squaredcos_cap_v2',
                    'variance_type': 'fixed_small',
                    'clip_sample': True,
                    'prediction_type': 'epsilon'
                })
    elif noise_scheduler.lower() == "ddim":
        if cfg.policy.noise_scheduler['_target_'] == 'diffusers.schedulers.scheduling_ddim.DDIMScheduler':
            cprint(f"Skip setup scheduler: Pretrained model already used DDIM scheduler", "green")
        else:
            with open_dict(cfg.policy.noise_scheduler):
                cfg.policy.noise_scheduler = OmegaConf.create({
                    '_target_': 'diffusers.schedulers.scheduling_ddim.DDIMScheduler',
                    'num_train_timesteps': 100,
                    'beta_start': 0.0001,
                    'beta_end': 0.02,
                    'beta_schedule': 'squaredcos_cap_v2',
                    'clip_sample': True,
                    'set_alpha_to_one': True,
                    'steps_offset': 0,
                    'prediction_type': 'epsilon'
                })
    else:
        raise ValueError(f"Unknown noise scheduler {noise_scheduler}")

    if num_inference_steps is not None:
        cfg.policy.num_inference_steps = num_inference_steps
    cprint(f"Noise Scheduler: {noise_scheduler.upper()}-{cfg.policy.num_inference_steps}", "green")
    print(OmegaConf.to_yaml(cfg.policy.noise_scheduler))
    return cfg

def load_normalizer(checkpoint_path: str, policy):
    normalizer_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "normalizer.pth")
    print("loading normalizer from", normalizer_path)
    state_dict = torch.load(normalizer_path, map_location="cpu")
    policy.normalizer = LinearNormalizer()
    policy.normalizer.load_state_dict(state_dict)
    policy.normalizer.to(policy.device)
    return policy


if __name__ == "__main__":
    checkpoint = '/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/reproduce/aloha_image/None/2026.06.02_03.50.28_train_diffusion_unet_hybrid_aloha_image/checkpoints/0.ckpt'
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

    ## Load payload
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    cfg = setup_noise_schedule(cfg, noise_scheduler='ddpm', num_inference_steps=100)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    exclude_keys = ['optimizer']
    workspace.load_payload(payload, exclude_keys=exclude_keys, include_keys=None)

    ## get policy from workspace
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    ## Load normalizer
    policy = load_normalizer(checkpoint, policy)

    ## Inference
    policy.reset()
    obs = {
        'high_images': np.zeros((1, 2, 3, 480, 640)),
        'wrist_left_images': np.zeros((1, 2, 3, 480, 640)),
        'wrist_right_images': np.zeros((1, 2, 3, 480, 640)),
        'states': np.zeros((1, 2, 14))
    }
    action_dict = policy.predict_action(obs)
    action = action_dict['action'].detach().to('cpu').numpy()
    print(action.shape)
