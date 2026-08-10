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
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.shortcutflow_mlp_lowdim_policy import ShortcutFlowMlpLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import CosineAnnealingWarmupRestarts
from diffusion_policy.model.diffusion.ema_model import EMAFlow

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainShortcutFlowMlpLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.epoch_start_ema = cfg.training.epoch_start_ema
        self.update_ema_freq = cfg.training.update_ema_freq
        self.test_denoising_steps = cfg.test_denoising_steps
        self.test_model_type = cfg.test_model_type
        self.test_clip_intermediate_actions = cfg.test_clip_intermediate_actions

        # configure model
        self.model: ShortcutFlowMlpLowdimPolicy
        self.model = hydra.utils.instantiate(
            cfg.policy,
            test_denoising_steps=self.test_denoising_steps,
            test_clip_intermediate_actions=self.test_clip_intermediate_actions
        )

        self.ema_model: ShortcutFlowMlpLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = None

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema = EMAFlow(cfg.ema)
            self.ema_model.set_normalizer(normalizer)

        # optimizer and lr scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )

        # configure lr scheduler
        lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.training.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.optimizer.lr,
            min_lr=cfg.training.lr_scheduler.min_lr,
            warmup_steps=cfg.training.lr_scheduler.warmup_steps,
            gamma=1.0
        )

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

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
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path)
        json_logger.start()
        for local_epoch_idx in range(cfg.training.num_epochs):
            self.model.train()
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    # compute loss
                    self.optimizer.zero_grad()
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss
                    loss.backward()
                    self.optimizer.step()

                    # update ema
                    if self.global_step % self.update_ema_freq == 0:
                        self.step_ema()

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_lr()[0]
                    }

                    is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                    if batch_idx == len(train_dataloader) - 2:
                        n_samples_last_batch = len(dataset) - (len(train_dataloader) - 1) * train_dataloader.batch_size
                        if int(n_samples_last_batch * self.model.self_consistency_k) == 0:
                            is_last_batch = True
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                    if ((cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1)) or is_last_batch:
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if self.test_model_type == "ema":
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 or self.epoch == cfg.training.num_epochs - 1:
                runner_log = env_runner.run(policy)
                # log all
                step_log.update(runner_log)

            lr_scheduler.step() # schedule learning rate after each epoch following ReinFlow

            # run validation
            if (self.epoch % cfg.training.val_every) == 0:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if batch_idx == len(val_dataloader) - 2:
                                n_samples_last_batch = len(val_dataset) - (len(val_dataloader) - 1) * val_dataloader.batch_size
                                if int(n_samples_last_batch * self.model.self_consistency_k) == 0:
                                    break
                            if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

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

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.epoch < self.epoch_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainShortcutFlowMlpLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
