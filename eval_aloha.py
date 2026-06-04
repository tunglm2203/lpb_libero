
import os
import numpy as np
import hydra
import torch
import dill
import h5py
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

import numpy as np
import matplotlib.pyplot as plt


def evaluate_trajectory(
    policy,
    batched_obs,
    gt_actions,
    action_horizon=10,
):
    """
    Returns
    -------
    all_gt : (N, action_dim)
    all_pred : (N, action_dim)
    inference_points : (N,)
    """

    length = len(gt_actions)

    all_gt = []
    all_pred = []
    inference_points = []

    for i in range(1, length):

        # không đủ future action
        if i + action_horizon > length:
            break

        cur_obs = {
            k: v[:, i - 1:i + 1]
            for k, v in batched_obs.items()
        }

        action_dict = policy.predict_action(cur_obs)

        pred_action = (
            action_dict["action"]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )[:action_horizon]

        gt_action = gt_actions[i:i + action_horizon]

        # kiểm tra orientation
        if pred_action.shape != gt_action.shape:
            if pred_action.T.shape == gt_action.shape:
                pred_action = pred_action.T
            else:
                raise ValueError(
                    f"Shape mismatch: pred={pred_action.shape}, gt={gt_action.shape}"
                )

        # lấy bước đầu tiên của horizon
        all_gt.append(gt_action[0])
        all_pred.append(pred_action[0])

        inference_points.append(i)

    all_gt = np.asarray(all_gt)
    all_pred = np.asarray(all_pred)
    inference_points = np.asarray(inference_points)

    return all_gt, all_pred, inference_points


def visualize_action_dimensions(
    gt_actions,
    pred_actions,
    inference_points=None,
    save_path=None,
    marker_stride=16,
):
    """
    gt_actions   : (T, action_dim)
    pred_actions : (T, action_dim)
    """

    action_dim = gt_actions.shape[1]

    fig, axes = plt.subplots(
        action_dim,
        1,
        figsize=(15, 3.5 * action_dim),
        squeeze=False
    )

    for dim in range(action_dim):

        ax = axes[dim, 0]

        ax.plot(
            gt_actions[:, dim],
            label="gt action",
            linewidth=2,
        )

        ax.plot(
            pred_actions[:, dim],
            label="pred action",
            linewidth=2,
        )

        if inference_points is not None:

            marker_idx = np.arange(
                0,
                len(inference_points),
                marker_stride
            )

            ax.scatter(
                marker_idx,
                pred_actions[marker_idx, dim],
                color="red",
                s=40,
                label="inference point" if dim == 0 else None,
                zorder=10,
            )

        ax.set_title(
            f"Action Dimension {dim}",
            fontsize=18,
            fontweight="bold",
        )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        if dim == 0:
            ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    # plt.show()

if __name__ == "__main__":
    checkpoint = '/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/pbrl/aloha_image/None/aloha_2026.06.04_06.26.50_cplkl_pseu_dpT_ExpD10_N30000_L250_1ER_SFTpos0_segM0.4_nD40_beta0.01_clip0.3_unclipwin1_smooth0.1/checkpoints/epoch_0060.ckpt'
    action_horizon = 4

    ## Load payload
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
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
    file = '/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/data/aloha/short_folding_debug/fold_shirt_debug.hdf5'
    data = h5py.File(file, 'r')['data']

    results = []

    for i, key in enumerate(data.keys()):
        if i > 2:
            break
        # After each trajectory, reset
        policy.reset()
        obs = data[key]['obs']

        batched_obs = {}
        for okey in obs:
            batched_obs[okey] = np.array(obs[okey][()])[None, ...]
            if 'image' in okey:
                batched_obs[okey] = np.transpose(batched_obs[okey], (0, 1, 4, 2, 3))

        all_gt, all_pred, inference_points = evaluate_trajectory(
            policy=policy,
            batched_obs=batched_obs,
            gt_actions=data[key]['actions'][()],
            action_horizon=action_horizon,
        )

        print("GT shape:", all_gt.shape)
        print("Pred shape:", all_pred.shape)
        # MSE toàn bộ tensor        
        mse = np.mean((all_gt - all_pred) ** 2)
        results.append(mse)
        print(f"Overall MSE: {mse:.6f}")

        visualize_action_dimensions(
            gt_actions=all_gt,
            pred_actions=all_pred,
            inference_points=inference_points,
            save_path=f"action_comparison_{i}.png",
            marker_stride=16,
        )
    print(checkpoint)
    print("Average MSE:", np.mean(results))
    print("Std MSE:", np.std(results))