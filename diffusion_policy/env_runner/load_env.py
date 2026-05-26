import glob
import hydra
import numpy as np
from termcolor import cprint
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import os

def load_libero_env_runner(cfg, output_dir, tasks_name=None):
    # hdf5_files = glob.glob(cfg.env_runner.dataset_path + "/*.hdf5")

    hdf5_files = [
        os.path.join(cfg.task.env_runner.dataset_path, name) for name in os.listdir(cfg.task.env_runner.dataset_path)
        if os.path.isdir(os.path.join(cfg.task.env_runner.dataset_path, name))
    ]
    
    # sort files
    hdf5_files.sort()

    # filter by tasks_name
    if tasks_name:
        hdf5_files = [file for file in hdf5_files if any(task_name in file for task_name in tasks_name)]
    

    env_runners = []
    for idx, file in enumerate(hdf5_files):
        task_name = file.split("/")[-1].split(".")[0]

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, task_dir=file + f'/{task_name}.hdf5', output_dir=output_dir)
        assert isinstance(env_runner, BaseImageRunner)
        env_runners.append((task_name, env_runner))

    return env_runners

def load_env_runner(cfg, output_dir):
    if "libero" in cfg.task.name:
        hdf5_files = glob.glob(cfg.task.dataset.dataset_path + "/*.hdf5")

        env_runners = []
        for file in hdf5_files:
            # configure env
            env_runner: BaseImageRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner, task_dir=file, output_dir=output_dir
            )
            assert isinstance(env_runner, BaseImageRunner)
            env_runners.append(env_runner)

            if cfg.training.debug:
                break
            # break
        return env_runners

    else:
        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
        assert isinstance(env_runner, BaseImageRunner)
        return env_runner


def env_rollout(cfg, env_runners, policy):
    step_log = {}
    if "libero" in cfg.task.name:
        cprint(f"Evaluating in LIBERO: {len(env_runners)}", "green", attrs=["bold"])
        if isinstance(env_runners[0], tuple):
            for task_name, env_runner in env_runners:
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)
        else:
            for env_runner in env_runners:
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)

        if cfg.checkpoint.topk.monitor_key == "test_mean_score":
            assert "test_mean_score" not in step_log
            all_test_mean_score = {
                k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
            }
            if len(all_test_mean_score) > 0:
                step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

            all_train_mean_score = {
                k: v
                for k, v in step_log.items()
                if "train/" in k and "_mean_score" in k
            }
            if len(all_train_mean_score) > 0:
                step_log["train_mean_score"] = np.mean(list(all_train_mean_score.values()))
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)
        step_log.update(runner_log)

        step_log["train_mean_score"] = runner_log["train/mean_score"]
        step_log["test_mean_score"] = runner_log["test/mean_score"]
    return step_log