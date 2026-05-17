"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import argparse
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import OmegaConf, open_dict
from termcolor import cprint
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.load_env import load_env_runner, env_rollout

def str2bool(v):
    # used for parsing boolean arguments
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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

def main(args):
    checkpoint = args.checkpoint
    if os.path.isfile(checkpoint):
        exp_name = checkpoint.split("/")[-3]
    else:
        raise NotImplementedError
    output_dir = os.path.join(args.output_dir, exp_name, args.dataset_name.replace(".hdf5", ""))
    seed = args.seed
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    cfg = setup_noise_schedule(cfg, args.noise_scheduler, args.num_inference_steps)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    normalizer_dir = os.path.dirname(os.path.dirname(checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    # Setup eval runner
    cfg.task.env_runner['n_train_vis'] = 0
    cfg.task.env_runner['n_test_vis'] = 0
    cfg.task.env_runner['n_train'] = 0
    cfg.task.env_runner['n_test'] = args.ntest
    cfg.task.env_runner['n_envs'] = 1
    cfg.task.env_runner['test_start_seed'] = 20000 + 10000 * seed

    cprint(f"Checkpoint: {checkpoint}", 'yellow', attrs=['bold'])
    cprint(f"Evaluation setting:", 'yellow', attrs=['bold'])
    cprint(f"    Env:    Ta={cfg.task.env_runner.n_action_steps}, To={cfg.task.env_runner.n_obs_steps}", 'yellow', attrs=['bold'])
    cprint(f"    Policy: Ta={policy.n_action_steps}, To={policy.n_obs_steps}, Tp={policy.horizon}", 'yellow', attrs=['bold'])

    # run eval
    cfg.task.env_runner._target_ = "diffusion_policy.env_runner.libero_image_sequential_runner.SequentialLiberoImageRunner"
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        task_dir=os.path.join(cfg.task.env_runner.dataset_path, args.dataset_name)
    )
    runner_log = env_runner.run(policy)
    results = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            results[key] = value._path
        else:
            results[key] = value

    results_path = os.path.join(output_dir, f"eval_results_{args.seed}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ntest', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default=None)

    # Diffusion policy setting
    parser.add_argument('--noise_scheduler', type=str, default='ddpm')
    parser.add_argument('--num_inference_steps', type=int, default=100)
    args = parser.parse_args()
    main(args)