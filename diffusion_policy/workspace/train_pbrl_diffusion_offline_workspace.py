if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import ConcatDataset, DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import scipy.stats as stats
from termcolor import colored

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.env_runner.load_env import load_libero_env_runner
from diffusion_policy.env_runner.libero_image_runner import LiberoImageRunner
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
from diffusion_policy.model.common.normalizer import LinearNormalizer


OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class PbrlDiffusionWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())


        self.global_step = 0
        self.epoch = 0

    def prepare_preference_dataset(self, cfg):

        tasks_name = [name for name in os.listdir(cfg.task.dataset_path) if os.path.isdir(os.path.join(cfg.task.dataset_path, name))][-1:]
        tasks_name = ['LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo']
        all_pref_datasets = {}

        for task_name in tasks_name:
            print(f"Processing task: {task_name}")
            # configure dataset
            dataset_1: BaseImageDataset
            dataset_1 = hydra.utils.instantiate(cfg.task.dataset, dataset_path=os.path.join(cfg.task.dataset_path, task_name))

            # if cfg.training.use_expert_data:
            #     dataset_1 = hydra.utils.instantiate(cfg.task.dataset, include_reward=True)
            # else:
            #     dataset_1 = hydra.utils.instantiate(cfg.task.dataset_1)
            assert isinstance(dataset_1, BaseImageDataset)

            # configure dataset
            dataset_2: BaseImageDataset
            if cfg.training.use_expert_data:
                breakpoint()
                dataset_2 = hydra.utils.instantiate(cfg.task.dataset, dataset_path=os.path.join(cfg.task.dataset_path, task_name))
            else:
                dataset_2 = hydra.utils.instantiate(cfg.task.dataset_2, shape_meta=cfg.task.dataset.shape_meta, dataset_path=os.path.join(cfg.task.dataset_2.dataset_path, task_name))
            assert isinstance(dataset_2, BaseImageDataset)


            pref_dataset: BaseImageDataset
            pref_dataset = hydra.utils.instantiate(
                cfg.task.pref_dataset, replay_buffer_1=dataset_1.replay_buffer, replay_buffer_2=dataset_2.replay_buffer,
                max_episodes_dataset_1=int(dataset_1.replay_buffer.n_episodes * (1 - cfg.task.dataset.val_ratio)),
                task_name=task_name,
            )


            # cut online groups
            votes_1, votes_2 = pref_dataset.pref_replay_buffer.meta['votes'], pref_dataset.pref_replay_buffer.meta['votes_2']

            all_votes_1 = np.array([votes_1 for _ in range(cfg.training.preference_learning.num_rounds)])
            all_votes_2 = np.array([votes_2 for _ in range(cfg.training.preference_learning.num_rounds)])

            # add noise to votes
            if cfg.training.preference_learning.reverse_rate > 0:
                # select uncertain samples
                var = (votes_1 * votes_2) / (((votes_1 + votes_2 + 1e-6) ** 2) * (votes_1 + votes_2 + 1))
                mask = (votes_1 + votes_2) != 0
                var_masked = var[mask]
                var_flat = var_masked.flatten()
                count = int(len(var_flat) * cfg.training.preference_learning.reverse_ratio)
                threshold = np.partition(var_flat, -count)[-count]
                masked_indices = np.where(var_flat >= threshold)[0]
                original_indices = np.where(mask.flatten())[0]
                indices = original_indices[masked_indices]

                for local_epoch_idx in range(cfg.training.preference_learning.num_rounds):
                    if local_epoch_idx % cfg.training.preference_learning.reverse_freq == 0:
                        X = stats.truncnorm(-3, 3, loc=cfg.training.preference_learning.reverse_rate, scale=cfg.training.preference_learning.reverse_rate / 3)
                        noise_ratio = X.rvs(all_votes_1.shape[1])
                        noise_ratio = noise_ratio.reshape(-1, 1)

                        all_votes_1[local_epoch_idx][indices] = all_votes_1[local_epoch_idx][indices] + np.round(
                            (all_votes_2[local_epoch_idx][indices] - all_votes_1[local_epoch_idx][indices]) * noise_ratio[indices])
                        all_votes_2[local_epoch_idx][indices] = all_votes_2[local_epoch_idx][indices] + np.round(
                            (all_votes_1[local_epoch_idx][indices] - all_votes_2[local_epoch_idx][indices]) * noise_ratio[indices])

                        all_votes_1[local_epoch_idx] = np.maximum(all_votes_1[local_epoch_idx], 0)
                        all_votes_2[local_epoch_idx] = np.maximum(all_votes_2[local_epoch_idx], 0)
                        
            all_pref_datasets[task_name] = [pref_dataset, all_votes_1, all_votes_2]
        return all_pref_datasets, tasks_name

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            ckpt_path = pathlib.Path(cfg.checkpoint_dir)
            assert ckpt_path.is_file()
            print(colored(f"Resuming from checkpoint {ckpt_path}", "green", attrs=['bold']))
            self.load_checkpoint(path=ckpt_path)
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())
            self.global_step = 0
            self.epoch = 0
        else:
            print(colored(f"Do not train from scratch", "red", attrs=['bold']))
            raise NotImplementedError

        device = torch.device(cfg.training.device)
        ref_policy = copy.deepcopy(self.model)
        ref_policy.train()  #.eval()
        for param in ref_policy.parameters():
            param.requires_grad = False
        ref_policy.to(device)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        ### Load normalizer
        normalizer_path = os.path.join(os.path.dirname(os.path.dirname(cfg.checkpoint_dir)), "normalizer.pth")
        print("loading normalizer from", normalizer_path)
        state_dict = torch.load(normalizer_path, map_location="cpu")
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(state_dict)
        self.model.set_normalizer(normalizer)
        ###

        all_pref_datasets, tasks_name = self.prepare_preference_dataset(cfg)

        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runners = load_libero_env_runner(cfg, self.output_dir, tasks_name)


        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path)
        json_logger.start()
        for round_idx in range(cfg.training.preference_learning.num_rounds):
            print(f"Round {round_idx + 1} of {cfg.training.preference_learning.num_rounds} for online training")

            all_pref_datasets_local_votes = []

            for k,v in all_pref_datasets.items():
                pref_dataset, all_votes_1, all_votes_2 = v

                local_votes_1 = np.array(all_votes_1[round_idx].T, dtype=np.float32).reshape(-1, 1)
                local_votes_2 = np.array(all_votes_2[round_idx].T, dtype=np.float32).reshape(-1, 1)

                pref_dataset.pref_replay_buffer.meta['votes'] = local_votes_1
                pref_dataset.pref_replay_buffer.meta['votes_2'] = local_votes_2
                pref_dataset.pref_replay_buffer.root['meta']['votes'] = local_votes_1
                pref_dataset.pref_replay_buffer.root['meta']['votes_2'] = local_votes_2

                all_pref_datasets_local_votes.append(pref_dataset)
            
            combined_dataset = ConcatDataset(all_pref_datasets_local_votes)

            train_dataloader = DataLoader(combined_dataset, **cfg.dataloader)
            # self.optimizer = self.model.get_optimizer(**cfg.optimizer)
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())

            # Place lr_scheduler here to reset in each round
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    len(train_dataloader) * cfg.training.num_epochs) \
                        // cfg.training.gradient_accumulate_every,
                last_epoch=-1,
            )

            equal_pref_threshold = cfg.training.preference_learning.equal_threshold * pref_dataset.sequence_length   # following LiRE
            # tung: for Tp=10 (transformer), stride=5 seems more stable for training compared to stride=10, still cannot
            # figure out why. Theoretically, the policy should output a single action, but we work in chunk-mode, so
            # stride=5 means overlap between slices are 5 seems increasing training stability.
            stride = int(np.round(self.model.horizon * 0.5))
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        if cfg.training.cpl_loss_type == 'cpl_kl':
                            raw_loss, loss_metrics = self.model.compute_loss_cpl_kl(batch, ref_model=ref_policy.model,
                                                                                    stride=stride,
                                                                                    use_bc=cfg.training.use_bc,
                                                                                    equal_pref_threshold=equal_pref_threshold)
                        elif cfg.training.cpl_loss_type == 'cpl':
                            raw_loss, loss_metrics = self.model.compute_loss_cpl(batch,
                                                                                 stride=stride,
                                                                                 use_bc=cfg.training.use_bc,
                                                                                 equal_pref_threshold=equal_pref_threshold)
                        elif cfg.training.cpl_loss_type == 'sft':
                            raw_loss, loss_metrics = self.model.compute_loss_sft(batch,
                                                                                 stride=stride,
                                                                                 equal_pref_threshold=equal_pref_threshold)
                        else:
                            raise NotImplementedError
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update(loss_metrics)

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 or self.epoch == cfg.training.num_epochs - 1:
                    for task_name, env_runner in env_runners:
                        runner_log = env_runner.run(policy)
                        # log all
                        step_log.update(runner_log)

                # run diffusion sampling on a training batch
                # if (self.epoch % cfg.training.sample_every) == 0:
                #     with torch.no_grad():
                #         # sample trajectory from training set, and evaluate difference
                #         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))

                #         start_idx = np.random.randint(0, batch['action'].shape[1] - self.model.horizon + 1)
                #         end_idx = start_idx + self.model.horizon

                #         start_idx_2 = np.random.randint(0, batch[f'action_2'].shape[1] - self.model.horizon + 1)
                #         end_idx_2 = start_idx_2 + self.model.horizon

                #         obs_dict = {'obs': {}}
                #         obs_dict_2 = {'obs': {}}

                #         for key in ['obs', 'language', 'ee_ori', 'ee_pos', 'joint_states']:
                #             get_obs, get_obs_2 = batch[key][:, start_idx:end_idx, :], batch[f'{key}_2'][:, start_idx_2:end_idx_2, :]
                #             obs_dict['obs'][key] = get_obs
                #             obs_dict_2['obs'][key] = get_obs_2

                #         gt_action, gt_action_2 = batch['action'][:, start_idx:end_idx, :], batch['action_2'][:, start_idx_2:end_idx_2, :]

                #         result = policy.predict_action(obs_dict)
                #         result_2 = policy.predict_action(obs_dict_2)
                #         if cfg.pred_action_steps_only:
                #             pred_action = result['action']
                #             pred_action_2 = result_2['action']
                #             start = cfg.n_obs_steps - 1
                #             end = start + cfg.n_action_steps
                #             gt_action = gt_action[:,start:end]
                #             gt_action_2 = gt_action_2[:,start:end]
                #         else:
                #             pred_action = result['action_pred']
                #             pred_action_2 = result_2['action_pred']
                #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                #         mse_2 = torch.nn.functional.mse_loss(pred_action_2, gt_action_2)
                #         # log
                #         step_log['train_action_mse_error'] = mse.item()
                #         step_log['train_action_mse_2_error'] = mse_2.item()
                #         # release RAM
                #         del batch
                #         del obs_dict
                #         del gt_action
                #         del result
                #         del pred_action
                #         del mse
                #         del obs_dict_2
                #         del gt_action_2
                #         del result_2
                #         del pred_action_2
                #         del mse_2

                # checkpoint
                if self.epoch != 0 and ((self.epoch % cfg.training.checkpoint_every) == 0 or self.epoch == cfg.training.num_epochs - 1):
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        json_logger.stop()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = PbrlDiffusionTransformerLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
