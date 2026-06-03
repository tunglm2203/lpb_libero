import sys
# sys.path.append('/mnt/workspace/vla_bench/diffusion_policy')
import os
import h5py
import numpy as np
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder


def main():
    root_repo = "/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero"
    IMG_SIZE = 224
    RENDER = True

    # ALL_TASKS = ["lift_mh", "can_mh", "square_mh", "transport_mh"]
    ALL_TASKS = ["transport_ph"]
    for task_name in ALL_TASKS:
        # This is imitation learning dataset
        if task_name == "tool_hang_ph":
            task, type = "tool_hang", "ph"
        else:
            task, type = task_name.split("_")
        dataset_path = f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/data/transport/transport_ph_demo_v141_20_perc.hdf5"

        dataset_path = os.path.join(root_repo, dataset_path)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
        env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=['agentview'] if RENDER else [],
            camera_height=IMG_SIZE if RENDER else 0,
            camera_width=IMG_SIZE if RENDER else 0,
            reward_shaping=True,
        )


        # Read data from offline dataset
        with h5py.File(dataset_path, "r") as f:

            demos = list(f["data"].keys())
            inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
            demos = [demos[i] for i in inds]

            breakpoint()

            if RENDER:
                video_recoder = VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                )
            video_path = os.path.join(root_repo, f"/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/data/transport/videos")
            os.makedirs(video_path, exist_ok=True)
            for ind in tqdm(range(len(demos))):
                ep = demos[ind]

                demo = f['data'][ep]
                # --- 1. copy rewards → success ---
                rewards = demo['rewards'][()]  # load as numpy

                # if 'success' not in demo:
                #     demo.create_dataset('success', data=rewards)
                # else:
                #     print(f"{ep}: success already exists, skipping")

                # prepare initial state to reload from
                states = f["data/{}/states".format(ep)][()]

                initial_state = dict(states=states[0])
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

                env.reset()
                obs = env.reset_to(initial_state)

                # Reset video writer
                if RENDER:
                    video_recoder.stop()
                    video_recoder.start(f"{video_path}/episode_{ind}.mp4")
                    video_recoder.write_frame(obs['agentview_image'])   # Write initial state
                    assert video_recoder.is_ready()

                traj_len = states.shape[0]
                new_rewards = []
                for t in range(1, traj_len):
                    # reset to simulator state to get observation
                    next_obs = env.reset_to({"states": states[t]})
                    if RENDER:
                        video_recoder.write_frame(next_obs['agentview_image'])
                    reward = env.get_reward()
                    new_rewards.append(reward)

                    if t == traj_len - 1:
                        # Perform final action
                        final_action = f["data/{}/actions".format(ep)][()][t]
                        next_obs, _, _, _ = env.step(final_action)
                        reward = env.get_reward()
                        new_rewards.append(reward)

                # overwrite in-place
                # demo['rewards'][...] = np.array(new_rewards)




if __name__ == "__main__":
    main()