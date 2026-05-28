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
from diffusion_policy.env_runner.libero_image_runner import LiberoImageRunner
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


        # configure env runner
        # env_runner: LiberoImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.env_runner,
        #     output_dir=self.output_dir,
        #     collect_data=True,
        #     task_dir=cfg.env_runner.dataset_path,
        #     dataset_path=cfg.env_runner.dataset_path,
        # )
        env_runners = load_libero_env_runner(cfg, self.output_dir, tasks_name=['LIVING_ROOM_SCENE6'])
        # assert isinstance(env_runners, list[tuple[str, LiberoImageRunner]])

        # device transfer
        device = torch.device(cfg.collecting.device)
        policy = self.model
        if self.ema_model is not None:
            policy = self.ema_model
        policy.to(device)

        # Collect data
        policy.eval()
        for task_name, env_runner in env_runners:
            runner_log, all_episodes = env_runner.run(policy)

            # print(all_episodes['actions'].shape) all_episodes['actions'][0][0].shape
            # Writing data to h5 file
            rollout_num_episodes = len(all_episodes['observations'])
            data_collect_file = os.path.join(run_dir, f"collect_{task_name}.hdf5")
            data_writer = h5py.File(data_collect_file, "w")
            data_grp = data_writer.create_group("data")
            total_samples = 0
            all_successes = []
            for i in range(rollout_num_episodes):
                successes = []
                eef_pos_states = []
                eef_quat_states = []
                joint_pos_states = []
                agentview_images = []
                next_agentview_images = []
                for t in range(len(all_episodes['successes'][i])):
                    successes.append(all_episodes['successes'][i][t])
                if np.sum(successes) > 0:
                    first_succ_idx = np.argmax(successes) # No need to +1 here since we have success flag at reset
                else:
                    first_succ_idx = len(all_episodes['actions'][i])

                successes = np.array(successes)
                all_successes.append(np.max(successes))

                for obs_idx in range(len(all_episodes['observations'][i][:first_succ_idx])):
                    eef_pos = all_episodes['observations'][i][obs_idx]['robot0_eef_pos']
                    eef_quat = all_episodes['observations'][i][obs_idx]['robot0_eef_quat']
                    joint_pos = all_episodes['observations'][i][obs_idx]['robot0_joint_pos']
                    eef_pos_states.append(eef_pos)
                    eef_quat_states.append(eef_quat)
                    joint_pos_states.append(joint_pos)
                    agentview_images.append(all_episodes['observations'][i][obs_idx]['agentview_image'])
                    if obs_idx < len(all_episodes['observations'][i][:first_succ_idx]) - 1:
                        next_agentview_images.append(all_episodes['observations'][i][obs_idx + 1]['agentview_image'])

                ep_data_grp = data_grp.create_group(f"episode_{i}")
                ep_data_grp.create_dataset("agentview_image", data=np.array(agentview_images)*255)
                ep_data_grp.create_dataset("next_agentview_image", data=np.array(next_agentview_images)*255)
                ep_data_grp.create_dataset("actions", data=np.array(all_episodes['actions'][i][:first_succ_idx]))
                ep_data_grp.create_dataset("rewards", data=np.array(all_episodes['rewards'][i][:first_succ_idx]))
                ep_data_grp.create_dataset("dones", data=np.array(all_episodes['terminals'][i][:first_succ_idx])) # this may not contain any done
                ep_data_grp.create_dataset("eef_pos_states", data=np.array(eef_pos_states))
                ep_data_grp.create_dataset("eef_quat_states", data=np.array(eef_quat_states))
                ep_data_grp.create_dataset("joint_pos_states", data=np.array(joint_pos_states))
                ep_data_grp.create_dataset("successes", data=successes[1:first_succ_idx + 1])

                # ep_data_grp.attrs["model_file"] = all_episodes['infos'][i][0]['model'] # model xml for this episode
                ep_data_grp.attrs["num_samples"] = len(all_episodes['actions'][i]) # number of transitions in this episode

                total_samples += len(all_episodes['actions'][i])

            data_grp.attrs["total"] = total_samples
            data_grp.attrs["env_args"] = json.dumps(env_runner.env_meta, indent=4)
            data_writer.close()

            json_log = dict()
            for key, value in runner_log.items():
                if 'video' not in key:
                    json_log[key] = float(value)
            json.dump(json_log, open(os.path.join(run_dir, f"collect_{task_name}.json"), 'w'), indent=2, sort_keys=True)

            print(colored(f"Avg. Performance: {np.mean(all_successes):.4f}", "green", attrs=['bold']))
            print(colored(f"Dumped to: {run_dir}\n", 'green'))

            if cfg.collecting.render_image:
                del env_runner
                print(f"Rendering video from collected data...")
                replay_collected_data(data_collect_file, run_dir, task_name)


def replay_collected_data(dataset_path, run_dir, task_name):
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
    video_path = os.path.join(run_dir, f"videos_{task_name}")
    os.makedirs(video_path, exist_ok=True)

    for ind in tqdm(range(len(demos))):
        ep = demos[ind]

        agentview_image = f["data/{}/agentview_image".format(ep)][()]
        agentview_image = (agentview_image * 255).clip(0, 255).astype(np.uint8)
        agentview_image = agentview_image.transpose(0,1,3,4,2)
        # Reset video writer
        video_recoder.stop()
        video_recoder.start(f"{video_path}/episode_{ind}.mp4")
        video_recoder.write_frame(agentview_image[0][-1])  # Write initial state

        traj_len = agentview_image.shape[0]
        print(f"Trajectory length: {traj_len}")
        assert video_recoder.is_ready(), "Video recorder is not ready"
        assert traj_len > 1, f"Trajectory length is not greater than 1, got {traj_len}"
        for t in tqdm(range(1, traj_len), leave=False):
            video_recoder.write_frame(agentview_image[t][-1])


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = DatacollectDiffusionLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
