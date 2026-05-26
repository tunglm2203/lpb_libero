import os
from typing import Dict

import yaml
from diffusion_policy.common.language_models import extract_text_features, get_text_model
import hydra
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
# from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.common.slice import slice_episode


def boundary_penalty(action, lower_bound=-1.0, upper_bound=1.0):
    penalty = torch.relu(action - upper_bound) + torch.relu(lower_bound - action)
    return penalty.sum()


class PbrlDiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # pbrl
            obs_as_cond=False,
            pred_action_steps_only=False,
            beta=1.0,
            bias_reg=1.0,
            bc_coef=1.0,
            ignore_equal_pref=False,
            gamma=0.999,
            clip_margin=None,
            smooth_label=0.0,
            confidence_weight=False,
            unclip_win=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        if pred_action_steps_only:
            assert obs_as_cond

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            if key == 'language':
                continue
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            if 'language' in shape_meta['obs']:
                global_cond_dim += 32

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.dynamics_model_normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.gamma = gamma
        self.clip_margin = clip_margin
        self.smooth_label = smooth_label    # 0 = disabled
        self.confidence_weight = confidence_weight
        self.unclip_win = unclip_win
        # assert self.unclip_win == 1 and self.smooth_label == 0.1
        self.kwargs = kwargs

        # Parameters for preference learning
        self.beta = beta
        self.bias_reg = bias_reg
        self.bc_coef = bc_coef
        self.ignore_equal_pref = ignore_equal_pref

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.correct_num = 0

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        ## =========================== load language model ===========================
        if 'language' in shape_meta['obs']:
            self.text_model, self.tokenizer, self.max_length = get_text_model(
                'libero_10', 'clip'
            )

    def initialize_planner(self,
                           planner_target,
                           demo_dataset_config,
                           dynamics_model_ckpt,
                           action_step,
                           output_dir,
                           guidance_start_timestep,
                           guidance_scale,
                           threshold,
                           demo_dataset_path=None):
        planner_cls = hydra.utils.get_class(planner_target)
        self.planner = planner_cls(demo_dataset_config, dynamics_model_ckpt, action_step, output_dir, demo_dataset_path)
        self.guidance_start_timestep = guidance_start_timestep
        self.guidance_scale = guidance_scale
        self.planner.set_policy_action_normalizer(self.normalizer['action'])
        self.threshold = threshold
        
    # ========= inference  ============
    def guided_conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            classifier_guidance=False,
            current_obs=None,
            text_latents=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        # print('variant2')
        if text_latents is not None:
            current_obs['language'] = text_latents

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        if classifier_guidance:
            current_cost = -1 * self.planner.compute_current_reward(current_obs)
            current_cost = current_cost.item()
            if current_cost >= self.threshold:
                self.correct_num += 1
        
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            trajectory = trajectory.detach().requires_grad_()

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            if classifier_guidance and t < self.guidance_start_timestep and current_cost > self.threshold:
                trajectory0 = scheduler.step(model_output, t, trajectory).pred_original_sample
                loss = self.planner.compute_loss(trajectory0, current_obs)
                cond_grad = -torch.autograd.grad(loss, trajectory)[0]
                guidance_scale = self.guidance_scale
                grad_scale = guidance_scale * (1 - scheduler.alphas_cumprod[t]).sqrt()
                trajectory = trajectory.detach() + grad_scale * cond_grad

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], language_goal=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        # print(obs_dict.keys())
        # breakpoint()
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        text_latents = None
        if language_goal is not None:
            text_tokens = self.tokenizer(
                language_goal,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            text_latents = extract_text_features(
                self.text_model,
                text_tokens,
                language_emb_model='clip',
            )

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        with torch.no_grad():
            nsample = self.guided_conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                current_obs=dict_apply(obs_dict, lambda x: x[:, -1:, ...]),
                **self.kwargs)
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, return_pred=False, return_cond=False):


        # print(batch.keys())
        # print(batch['obs']['language'].shape)
        # print(batch['obs']['agentview_rgb'].shape)
        # print(batch['action'].shape)
        # breakpoint()

        # language: torch.Size([128, 32, 2, 30])
        # agentview_rgb: torch.Size([128, 32, 3, 128, 128])
        # ee_ori torch.Size([128, 32, 4])
        # ee_pos torch.Size([128, 32, 3])
        # joint_states torch.Size([128, 32, 7])
        # action: torch.Size([128, 32, 10])
        # normalize input
        assert 'valid_mask' not in batch
        text_latents = None
        if 'language' in batch['obs']:
            if "language" in batch["obs"]:
                language_goal = batch["obs"]["language"]
                del batch["obs"]["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    self.text_model,
                    text_tokens,
                    language_emb_model='clip',
                )
            elif "language_latents" in batch:
                text_latents = batch["language_latents"]

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory


        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        if return_cond:
            return global_cond, trajectory

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
    

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        if return_pred:
            return pred, target

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


    def compute_loss_sft(self, batch, stride=1, equal_pref_threshold=0.05):

        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        batch['length'] = batch['length'].detach()
        batch['length_2'] = batch['length_2'].detach()
    

        diff = torch.abs(batch["votes"] - batch["votes_2"])

        # Convert to all left segments are preferred, i.e., actions_1 is preferred over actions_2
        condition_2 = (batch["votes"] < batch["votes_2"]) & (diff > equal_pref_threshold)  # votes_1 < votes_2 and diff >= threshold
        mask_pref_right = condition_2.squeeze(-1)
    
        for key in ["obs", "action", "votes", "length"]:
            batch[key][mask_pref_right], batch[f"{key}_2"][mask_pref_right] = batch[f"{key}_2"][mask_pref_right], batch[key][mask_pref_right]

        # Slice to make it compatible with action chunking
        keys_to_slice = ['obs', 'action', 'ee_ori', 'ee_pos', 'joint_states', 'language']
        sliced_batch = {key: slice_episode(batch[key], horizon=self.horizon, stride=stride) for key in keys_to_slice}
        assert not self.pred_action_steps_only and self.obs_as_global_cond and self.noise_scheduler.config.prediction_type == 'epsilon'

        bsz = sliced_batch['obs'][0].shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()

        # Behavior cloning loss for only preferred segments (i.e., segment 1 (left))
        idx = torch.randint(0, len(sliced_batch['obs']), (bsz,), device=self.device)  # Sampling chunk from preferred segment (similar to BC)
        batch_idx = torch.arange(bsz, device=self.device)
        sampled_batch = {key: value[idx, batch_idx] for key, value in sliced_batch.items()}


        new_batch = {}
        new_batch.update(obs={
            'agentview_rgb': sampled_batch['obs'].permute(0,1,4,2,3),
            'ee_ori': sampled_batch['ee_ori'],
            'ee_pos': sampled_batch['ee_pos'],
            'joint_states': sampled_batch['joint_states'],
            'language': sampled_batch['language'],
        })
        new_batch.update(action=sampled_batch['action'])

        loss = self.compute_loss(new_batch)

        loss_metrics = {
            'bc_loss': loss.item(),
        }
        return loss, loss_metrics

    def encode_condition(self, obs_batch):
        text_latents = None

        if 'language' in obs_batch:
            language_goal = obs_batch["language"]

            text_tokens = {
                "input_ids": language_goal[:, 0].long()[:, 0],
                "attention_mask": language_goal[:, 0].long()[:, 1],
            }

            text_latents = extract_text_features(
                self.text_model,
                text_tokens,
                language_emb_model='clip',
            )

        nobs = self.normalizer.normalize(obs_batch)

        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )

        nobs_features = self.obs_encoder(this_nobs)

        B = next(iter(nobs.values())).shape[0]

        global_cond = nobs_features.reshape(B, -1)

        if text_latents is not None:
            global_cond = torch.cat([global_cond, text_latents], dim=-1)

        return global_cond

    def compute_loss_cpl_kl(
            self, batch, epoch, ref_model, n_epoch_sft=0, sft_type="pos", stride=10, equal_pref_threshold=0.05
    ):
        assert sft_type in ["pos", "both"]
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        batch['length'] = batch['length'].detach()
        batch['length_2'] = batch['length_2'].detach()

        diff = torch.abs(batch["votes"] - batch["votes_2"])
        mask_not_equal_pref = torch.squeeze(diff > equal_pref_threshold, dim=-1).type(torch.float32)

        if self.confidence_weight:
            temperature = 0.03
            confidence_weight = torch.sigmoid((diff - equal_pref_threshold) / temperature)

        # Swap so segment 1 is always the preferred/winner trajectory
        mask_pref_right = ((batch["votes"] < batch["votes_2"]) & (diff > equal_pref_threshold)).squeeze(-1)
        for key in ["obs", "action", "votes", "length", 'ee_ori', 'ee_pos', 'joint_states', 'language']:
            batch[key][mask_pref_right], batch[f"{key}_2"][mask_pref_right] = batch[f"{key}_2"][mask_pref_right], batch[key][mask_pref_right]

        # Slice to make it compatible with action chunking
        keys_to_slice = ['obs', 'action', 'ee_ori', 'ee_pos', 'joint_states', 'language']
        sliced_batch = {key: slice_episode(batch[key], horizon=self.horizon, stride=stride) for key in keys_to_slice}
        sliced_batch_2 = {key: slice_episode(batch[f"{key}_2"], horizon=self.horizon, stride=stride) for key in keys_to_slice}
        assert (len(sliced_batch['obs']) == len(sliced_batch_2['obs'])) and (len(sliced_batch['action']) == len(sliced_batch_2['action']))
        assert not self.pred_action_steps_only and self.obs_as_global_cond and self.noise_scheduler.config.prediction_type == 'epsilon'


        bsz = sliced_batch['obs'][0].shape[0]
        n_train_denoise_timesteps = self.noise_scheduler.config.num_train_timesteps
        use_bc = True if epoch < n_epoch_sft else False

        valid_count_1 = torch.zeros(bsz, device=self.device)
        valid_count_2 = torch.zeros(bsz, device=self.device)
        segment_loss_1, segment_loss_2, imitation_loss = 0.0, 0.0, 0.0
        for i in range(len(sliced_batch['obs'])):
            timesteps = torch.randint(0, n_train_denoise_timesteps, (bsz,), device=self.device).long()
            timesteps_1 = timesteps
            timesteps_2 = timesteps

            sample_1 = {key: sliced_batch[key][i] for key in keys_to_slice}
            sample_2 = {key: sliced_batch_2[key][i] for key in keys_to_slice}

            #### Encode condition

            obs_dict_1 = {}
            obs_dict_1.update(obs={
                'agentview_rgb': sample_1['obs'].permute(0,1,4,2,3),
                'ee_ori': sample_1['ee_ori'],
                'ee_pos': sample_1['ee_pos'],
                'joint_states': sample_1['joint_states'],
                'language': sample_1['language'],
            })
            obs_dict_1.update(action=sample_1['action'])

            obs_dict_2 = {}
            obs_dict_2.update(obs={
                'agentview_rgb': sample_2['obs'].permute(0,1,4,2,3),
                'ee_ori': sample_2['ee_ori'],
                'ee_pos': sample_2['ee_pos'],
                'joint_states': sample_2['joint_states'],
                'language': sample_2['language'],
            })
            obs_dict_2.update(action=sample_2['action'])

            global_cond_1, trajectory_1 = self.compute_loss(obs_dict_1, return_cond=True)
            global_cond_2, trajectory_2 = self.compute_loss(obs_dict_2, return_cond=True)

            #### Masks
            actual_timesteps = i * stride + torch.arange(self.horizon, device=self.device)
            step_mask_1 = (actual_timesteps.unsqueeze(0) < batch['length'].view(-1,1)).float()
            step_mask_2 = (actual_timesteps.unsqueeze(0) < batch['length_2'].view(-1,1)).float()
            valid_count_1 += step_mask_1.sum(dim=-1)  # Accumulate total valid timesteps evaluated
            valid_count_2 += step_mask_2.sum(dim=-1)  # Accumulate total valid timesteps evaluated
            discounts = (self.gamma ** actual_timesteps).unsqueeze(0)  # Compute gamma discounts (shape of [1, horizon])
            weights_1 = discounts * step_mask_1
            weights_2 = discounts * step_mask_2

            #### Diffusion forward
            condition_mask = self.mask_generator(trajectory_1.shape)
            loss_mask = (~condition_mask).float()

            ## segment 1
            noise_1 = torch.randn_like(trajectory_1)
            noisy_trajectory_1 = self.noise_scheduler.add_noise(trajectory_1, noise_1, timesteps_1)
            noisy_trajectory_1[condition_mask] = trajectory_1[condition_mask]
            pred_1 = self.model(noisy_trajectory_1, timesteps_1, global_cond=global_cond_1)

            ## segment 2
            if (not use_bc) or (use_bc and sft_type == "both"):
                noise_2 = torch.randn_like(trajectory_2)
                noisy_trajectory_2 = self.noise_scheduler.add_noise(trajectory_2, noise_2, timesteps_2)
                noisy_trajectory_2[condition_mask] = trajectory_2[condition_mask]
                pred_2 = self.model(noisy_trajectory_2, timesteps_2, global_cond=global_cond_2)
            else:
                noise_2 = noisy_trajectory_2 = pred_2 = None

            if use_bc:
                if sft_type == "pos":
                    imitation_loss_1 = torch.norm((pred_1 - noise_1) * loss_mask, dim=-1) ** 2
                    imitation_loss += torch.sum(imitation_loss_1 * step_mask_1, dim=-1)
                elif sft_type == "both":
                    imitation_loss_1 = torch.norm((pred_1 - noise_1) * loss_mask, dim=-1) ** 2
                    imitation_loss_2 = torch.norm((pred_2 - noise_2) * loss_mask, dim=-1) ** 2
                    imitation_loss += (torch.sum(imitation_loss_1 * step_mask_1, dim=-1) + torch.sum(imitation_loss_2 * step_mask_2, dim=-1))
                else:
                    raise NotImplementedError    

            else:
                with torch.no_grad():
                    ref_pred_1 = ref_model(noisy_trajectory_1, timesteps_1, global_cond=global_cond_1)
                    ref_pred_2 = ref_model(noisy_trajectory_2, timesteps_2, global_cond=global_cond_2)

                slice_loss_1 = (torch.norm((pred_1 - noise_1) * loss_mask, dim=-1) ** 2 - torch.norm((ref_pred_1 - noise_1) * loss_mask, dim=-1) ** 2)
                slice_loss_2 = (torch.norm((pred_2 - noise_2) * loss_mask, dim=-1) ** 2 - torch.norm((ref_pred_2 - noise_2) * loss_mask, dim=-1) ** 2)


                if self.clip_margin is not None:
                    # TODO: Test this Soft Clip later to avoid abruptly cut the gradient
                    # slice_loss_1 = self.clip_margin * torch.tanh(slice_loss_1 / self.clip_margin)
                    # slice_loss_2 = self.clip_margin * torch.tanh(slice_loss_2 / self.clip_margin)
                    if not self.unclip_win:
                        slice_loss_1 = torch.clamp(slice_loss_1, min=-self.clip_margin, max=self.clip_margin)
                    slice_loss_2 = torch.clamp(slice_loss_2, min=-self.clip_margin, max=self.clip_margin)

                if self.ignore_equal_pref:
                    segment_loss_1 += torch.sum(slice_loss_1 * weights_1, dim=-1) * mask_not_equal_pref
                    segment_loss_2 += torch.sum(slice_loss_2 * weights_2, dim=-1) * mask_not_equal_pref
                else:
                    segment_loss_1 += torch.sum(slice_loss_1 * weights_1, dim=-1)
                    segment_loss_2 += torch.sum(slice_loss_2 * weights_2, dim=-1)

        if use_bc:
            if sft_type == "pos":
                norm_factor = torch.clamp(valid_count_1, min=1.0)
            else:   # both
                norm_factor = (torch.clamp(valid_count_1, min=1.0) + torch.clamp(valid_count_2, min=1.0))

            imitation_loss = imitation_loss / norm_factor
            loss_total = torch.mean(imitation_loss)
            mle_loss_1, accuracy = 0.0, 0.0
        else:
            norm_factor_1 = torch.clamp(valid_count_1 / self.horizon, min=1.0)  # num of chunk that calculated
            norm_factor_2 = torch.clamp(valid_count_2 / self.horizon, min=1.0)  # num of chunk that calculated

            segment_loss_1 = -self.beta * n_train_denoise_timesteps * segment_loss_1 / norm_factor_1
            segment_loss_2 = -self.beta * n_train_denoise_timesteps * segment_loss_2 / norm_factor_2

            z = segment_loss_1 - self.bias_reg * segment_loss_2

            epsilon_smooth = self.smooth_label
            if epsilon_smooth == 0:
                # Standard CPL
                mle_loss_1 = -F.logsigmoid(z)
            else:
                # Conservative CPL blends the forward and reversed preferences
                mle_loss_1 = -(1 - epsilon_smooth) * F.logsigmoid(z) - epsilon_smooth * F.logsigmoid(-z)

            if self.confidence_weight:
                # Squeeze confidence weight to match mle_loss_1 shape (B,)
                cw = confidence_weight.squeeze(-1)

            # mle_loss_1 = -F.logsigmoid(segment_loss_1 - self.bias_reg * segment_loss_2)
            if self.ignore_equal_pref:
                # Average ONLY pairs that have unequal preferences
                valid_pairs = torch.clamp(mask_not_equal_pref.sum(), min=1.0)
                if self.confidence_weight:
                    # Apply hard mask AND soft confidence weight
                    weighted_loss = mle_loss_1 * mask_not_equal_pref * cw
                    loss_total = weighted_loss.sum() / valid_pairs
                else:
                    loss_total = (mle_loss_1 * mask_not_equal_pref).sum() / valid_pairs
                # Ignore tied pairs so they don't count as incorrect
                with torch.no_grad():
                    correct_preds = (segment_loss_1.detach() > segment_loss_2.detach()).float()
                    accuracy = ((correct_preds * mask_not_equal_pref).sum() / valid_pairs).item()
            else:
                if self.confidence_weight:
                    # Apply soft confidence weight to ALL pairs
                    weighted_loss = mle_loss_1 * cw
                    # Use weighted mean to maintain stable gradient magnitudes
                    loss_total = weighted_loss.sum() / torch.clamp(cw.sum(), min=1.0)
                else:
                    loss_total = torch.mean(mle_loss_1)
                with torch.no_grad():
                    accuracy = (segment_loss_1.detach() > segment_loss_2.detach()).float().mean().item()

        loss_metrics = {
            'mle_loss_1': mle_loss_1.mean().item() if isinstance(mle_loss_1, torch.Tensor) else mle_loss_1,
            'segment_loss_1': segment_loss_1.mean().item() if isinstance(segment_loss_1, torch.Tensor) else segment_loss_1,
            'segment_loss_2': segment_loss_2.mean().item() if isinstance(segment_loss_2, torch.Tensor) else segment_loss_2,
            'bc_loss': imitation_loss.mean().item() if isinstance(imitation_loss, torch.Tensor) else imitation_loss,
            'accuracy': accuracy
        }

        if isinstance(segment_loss_1, torch.Tensor) and isinstance(segment_loss_2, torch.Tensor):
            scale = self.beta * n_train_denoise_timesteps
            # Argument to logsigmoid: |reward_logit| >> 5 means sigmoid is saturated -> gradients vanish
            reward_logit = (segment_loss_1 - self.bias_reg * segment_loss_2).mean().item()
            # Raw log-ratios (before beta scaling): < 0 = model improved vs ref, > 0 = drifted away
            # log_ratio_win should be ≤ 0 (improving on preferred); log_ratio_lose >> 0 = collapse
            log_ratio_win  = (-segment_loss_1 / scale).mean().item()
            log_ratio_lose = (-segment_loss_2 / scale).mean().item()
            loss_metrics.update({
                'reward_logit': reward_logit,
                'log_ratio_win': log_ratio_win,
                'log_ratio_lose': log_ratio_lose,
            })

        if self.ignore_equal_pref:
            loss_metrics.update({'total_mask_not_equal': mask_not_equal_pref.sum()})
        return loss_total, loss_metrics

