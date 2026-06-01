if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import json
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import numpy as np
import random
import dill
import h5py
from tqdm import tqdm
from termcolor import colored
from hydra.core.hydra_config import HydraConfig
from diffusion_policy.env_runner.load_env import env_rollout, load_libero_env_runner
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class DatacollectDiffusionWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Load payload from checkpoint
        if cfg.checkpoint_dir is None:
            checkpoint_dir_dict = {
                'libero_10': {
                    'datacollect_diffusion_unet': '',
                    'datacollect_diffusion_transformer': '',
                },
            }

            checkpoint_dir = checkpoint_dir_dict[cfg.task_name][cfg.name]
        else:
            checkpoint_dir = cfg.checkpoint_dir


        ckpt_file = pathlib.Path(checkpoint_dir)
        assert ckpt_file.is_file()
        print(colored(f"Collecting from: {ckpt_file}", "green", attrs=["bold"]))
        payload = torch.load(ckpt_file.open('rb'), pickle_module=dill)
        self.pretrained_cfg = payload['cfg']

        # set seed
        seed = cfg.collecting.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy
        self.model = hydra.utils.instantiate(self.pretrained_cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if self.pretrained_cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Load weights from pretrained models
        exclude_keys = ['optimizer']
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=None)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        run_dir = HydraConfig.get().run.dir
        cfg.task.env_runner['n_train_vis'] = 0
        cfg.task.env_runner['n_test_vis'] = 0
        cfg.task.env_runner['n_train'] = 0
        cfg.task.env_runner['n_test'] = cfg.collecting.num_episodes
        cfg.task.env_runner['n_envs'] = min(100, cfg.collecting.num_episodes)


        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir,
        )
        # assert isinstance(env_runner, BaseLowdimRunner)
        assert env_runner.collect_data, "Wrong configs in collect mode"

        # device transfer
        device = torch.device(cfg.collecting.device)
        policy = self.model
        if self.ema_model is not None:
            policy = self.ema_model
        policy.to(device)

        # Collect data
        policy.eval()
        runner_log, all_episodes = env_runner.run(policy)

        # Writing data to h5 file
        rollout_num_episodes = len(all_episodes['observations'])
        data_collect_file = os.path.join(run_dir, f"collect_{cfg.task_name}.hdf5")
        data_writer = h5py.File(data_collect_file, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0
        all_successes = []
        for i in range(rollout_num_episodes):
            states = []
            successes = []
            for t in range(len(all_episodes['infos'][i])):
                states.append(all_episodes['infos'][i][t]['states'])
                successes.append(all_episodes['infos'][i][t]['success'])

            if np.sum(successes) > 0:
                first_succ_idx = np.argmax(successes) # No need to +1 here since we have success flag at reset
            else:
                first_succ_idx = len(all_episodes['actions'][i])
            states = np.array(states)
            successes = np.array(successes)
            all_successes.append(np.max(successes))

            ep_data_grp = data_grp.create_group(f"demo_{i}")

            def stack_obs(obs_list):
                keys = obs_list[0].keys()
                out = {}
                for k in keys:
                    out[k] = np.stack([obs[k] for obs in obs_list], axis=0)
                    if 'image' in k:
                        out[k] = out[k].transpose(0,2,3,1).astype('uint8')
                return out


            obs_grp = ep_data_grp.create_group("obs")
            obs_seq = all_episodes['observations'][i][:first_succ_idx]
            obs_dict = stack_obs(obs_seq)
            save_nested_dict(obs_grp, obs_dict)

            next_obs_grp = ep_data_grp.create_group("next_obs")
            next_obs_seq = all_episodes['observations'][i][1:first_succ_idx+1]
            next_obs_dict = stack_obs(next_obs_seq)
            save_nested_dict(next_obs_grp, next_obs_dict)

            # ep_data_grp.create_dataset("obs", data=np.array(all_episodes['observations'][i][:first_succ_idx]))
            # ep_data_grp.create_dataset("next_obs", data=np.array(all_episodes['observations'][i][1:first_succ_idx + 1]))
            ep_data_grp.create_dataset("actions", data=np.array(all_episodes['actions'][i][:first_succ_idx]))
            ep_data_grp.create_dataset("abs_actions", data=np.array(all_episodes['actions'][i][:first_succ_idx]))
            ep_data_grp.create_dataset("rewards", data=np.array(all_episodes['rewards'][i][:first_succ_idx]))
            ep_data_grp.create_dataset("dones", data=np.array(all_episodes['terminals'][i][:first_succ_idx])) # this may not contain any done
            ep_data_grp.create_dataset("states", data=states[:first_succ_idx + 1])
            ep_data_grp.create_dataset("successes", data=successes[1:first_succ_idx + 1])

            ep_data_grp.attrs["model_file"] = all_episodes['infos'][i][0]['model'] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = len(all_episodes['actions'][i]) # number of transitions in this episode

            total_samples += len(all_episodes['actions'][i])

        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env_runner.env_meta, indent=4)
        data_writer.close()

        json_log = dict()
        for key, value in runner_log.items():
            if 'video' not in key:
                json_log[key] = float(value)
        json.dump(json_log, open(os.path.join(run_dir, f"collect_{cfg.task_name}.json"), 'w'), indent=2, sort_keys=True)

        print(colored(f"Avg. Performance: {np.mean(all_successes):.4f}", "green", attrs=['bold']))
        print(colored(f"Dumped to: {run_dir}\n", 'green'))

        if cfg.collecting.render_image:
            del env_runner
            print(f"Rendering video from collected data...")
            replay_collected_data(data_collect_file, run_dir)


def replay_collected_data(dataset_path, run_dir, cam_width=140, cam_height=140):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=['agentview'],
        camera_height=cam_height,
        camera_width=cam_width,
        reward_shaping=True,
    )

    # Read data from offline dataset
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
    demos = [demos[i] for i in inds]

    video_recoder = VideoRecorder.create_h264(
        fps=10,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=22,
        thread_type='FRAME',
        thread_count=1
    )
    video_path = os.path.join(run_dir, "videos")
    os.makedirs(video_path, exist_ok=True)
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]

        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        obs = env.reset_to(initial_state)

        # Reset video writer
        video_recoder.stop()
        video_recoder.start(f"{video_path}/episode_{ind}.mp4")
        video_recoder.write_frame(obs['agentview_image'])  # Write initial state

        traj_len = states.shape[0]
        assert video_recoder.is_ready()
        for t in tqdm(range(1, traj_len), leave=False):
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states": states[t]})
            video_recoder.write_frame(next_obs['agentview_image'])

def save_nested_dict(h5_group, data):
    """
    Save a nested dict/list/ndarray into hdf5 recursively.
    """

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                subgrp = h5_group.create_group(k)
                save_nested_dict(subgrp, v)

            else:
                h5_group.create_dataset(k, data=np.asarray(v))

    else:
        raise TypeError(f"Unsupported type: {type(data)}")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = DatacollectDiffusionLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

