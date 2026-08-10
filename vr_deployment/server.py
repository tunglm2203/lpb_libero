import argparse
# from lerobot.configs.policies import PreTrainedConfig
# from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
# from lerobot.common.policies.rtc.modeling_rtc import RTCConfig, RTCAttentionSchedule
# from lerobot.common.policies.repaint.modeling_repaint import RepaintConfig
import os
import sys 

sys.path.append("/home/binhng/Workspace/spcorl/lpb_libero")

from diffusion_policy.eval.robot import RobotInferenceServer

CHECKPOINT_PATH = '/home/binhng/Workspace/spcorl/cpkt/dp_aloha_40epoch/checkpoints/40.ckpt'

def parse_args():
    parser = argparse.ArgumentParser(description="PI0 Robot Inference Server")
    # parser.add_argument(
    #     "--smooth-option",
    #     type=str,
    #     default="",
    #     help="",
    # )
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--execution-horizon", type=int, default=10)
    return parser.parse_args()


# def build_config(args) -> PreTrainedConfig:
#     print("[Config] Loading config from pretrained!")
#     config = PreTrainedConfig.from_pretrained(args.checkpoint)
#     if args.smooth_option == "rtc":
#         print("[Config] Build RTCConfig")
#         config.rtc_config = RTCConfig(
#             enabled=True,
#             execution_horizon=args.execution_horizon,
#             max_guidance_weight=10,
#             prefix_attention_schedule=RTCAttentionSchedule.EXP,
#             debug=False,
#         )
#     elif args.smooth_option == "repaint":
#         print("[Config] Build RepaintConfig")
#         config.repaint_config = RepaintConfig(
#             enabled=True,
#             execution_horizon=args.execution_horizon,
#             prefix_attention_schedule=RTCAttentionSchedule.EXP,
#             debug=False,
#         )
#     return config


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


def main():
    args = parse_args()
    # print(f"Smooth: {args.smooth_option}")

    # config = build_config(args)
    # model = PI0Policy.from_pretrained(args.checkpoint, config=config)
    # model = model.cuda().eval()

    checkpoint = '/home/binhng/Workspace/spcorl/cpkt/placing_drawer_100ep/100.ckpt'
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

    ## Load payload
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    # cfg = setup_noise_schedule(cfg, noise_scheduler='ddpm', num_inference_steps=100)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    exclude_keys = ['optimizer']
    workspace.load_payload(payload, exclude_keys=exclude_keys, include_keys=None)

    ## get policy from workspace
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.test_denoising_steps = 8

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    ## Load normalizer
    policy.set_normalizer(policy.normalizer)  # need to set it again to set min, max range

    ## Inference
    policy.reset()
    
    robot = RobotInferenceServer(policy, port=args.port)
    robot.run()


if __name__ == "__main__":
    main()