from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
if hasattr(rmbn, 'CropRandomizer'):
    RobomimicCropRandomizer = rmbn.CropRandomizer
else:
    # robomimic >= 0.3 moved the randomizers out of base_nets
    from robomimic.models.obs_core import CropRandomizer as RobomimicCropRandomizer
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

Sample = namedtuple("Sample", "trajectories chains")


class ReFlowTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            # rectified flow
            sample_t_type: str = 'uniform',
            test_denoising_steps=20,
            test_clip_intermediate_actions=True):
        super().__init__()

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
                predicate=lambda x: isinstance(x, RobomimicCropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create velocity field model
        # rectified flow only supports obs as condition, the trajectory being
        # transported is the action sequence alone.
        obs_feature_dim = obs_encoder.output_shape()[0]
        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.data_shape = (self.horizon, self.action_dim)
        self.sample_t_type = sample_t_type
        self.test_denoising_steps = test_denoising_steps
        self.test_clip_intermediate_actions = test_clip_intermediate_actions
        self.act_range = (None, None)
        assert self.n_action_steps <= self.horizon, f"To={self.n_obs_steps}, Ta={self.n_action_steps}, Tp={self.horizon} are incompatible"

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: str -> B,To,*
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps

        # encode the first To observation steps into the condition
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(B, To, -1)

        samples = self.sample(
            cond,
            inference_steps=self.test_denoising_steps,
            record_intermediate=False,
            clip_intermediate_actions=self.test_clip_intermediate_actions
        )

        # unnormalize prediction
        naction_pred = samples.trajectories
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        # the sampled trajectory already starts at the current step (see
        # compute_loss, which trains on action[:, To-1:]), so no extra offset.
        action = action_pred[:, :self.n_action_steps]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        # clip range for the *normalized* action space
        act_stats = self.normalizer['action'].get_output_stats()
        self.act_range = (act_stats['min'].min().item(), act_stats['max'].max().item())

    def get_optimizer(
            self,
            transformer_weight_decay: float,
            obs_encoder_weight_decay: float,
            learning_rate: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        To = self.n_obs_steps

        # reshape B, To, ... to B*To and encode
        this_nobs = dict_apply(nobs,
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        cond = nobs_features.reshape(batch_size, To, -1)

        # the predicted trajectory starts at the current step
        x1 = nactions[:, To - 1:, :]
        assert x1.shape[1] == self.horizon, \
            f"dataloader horizon {nactions.shape[1]} with To={To} yields {x1.shape[1]} steps, expected {self.horizon}"

        (xt, t), v = self.generate_target(x1)
        v_hat = self.model(xt, t, cond)
        loss = F.mse_loss(input=v_hat, target=v)
        return loss

    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """Generate rectified flow trajectory xt = t * x1 + (1 - t) * x0.

        Args:
            x1: Target data tensor of shape (batch_size, horizon, action_dim).
            x0: Initial noise tensor of shape (batch_size, horizon, action_dim).
            t: Time step tensor of shape (batch_size,).

        Returns:
            Tensor: Interpolated trajectory xt of shape (batch_size, horizon, action_dim).
        """
        t_ = t.view(-1, 1, 1).to(dtype=x1.dtype, device=x1.device)
        xt = t_ * x1 + (1 - t_) * x0
        return xt

    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
        """Sample time steps from a specified distribution in [0, 1).

        Args:
            batch_size: Number of time samples to generate.
            time_sample_type: Type of distribution ('uniform', 'logitnormal', 'beta').
            **kwargs: Additional parameters for non-uniform distributions.

        Returns:
            Tensor: Time samples of shape (batch_size,).

        Raises:
            ValueError: If time_sample_type is not supported.
        """
        supported_time_sample_type = ['uniform', 'logitnormal', 'beta']
        if time_sample_type == 'uniform':
            return torch.rand(batch_size, device=self.device)
        elif time_sample_type == 'logitnormal':
            m = kwargs.get("m", 0)  # Default mean
            s = kwargs.get("s", 1)  # Default standard deviation
            normal_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=self.device)
            logit_normal_samples = torch.sigmoid(normal_samples)
            return logit_normal_samples
        elif time_sample_type == 'beta':
            alpha = kwargs.get("alpha", 1.5)  # Default alpha
            beta = kwargs.get("beta", 1.0)  # Default beta
            s = kwargs.get("s", 0.999)  # Default cutoff
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((batch_size,)).to(self.device)
            tau = s * (1 - beta_sample)
            return tau
        else:
            raise ValueError(
                f'Unknown time_sample_type = {time_sample_type}. Supported types: {supported_time_sample_type}')

    def generate_target(self, x1: Tensor) -> tuple:
        """Generate training targets for the velocity field.

        Args:
            x1: Real data tensor of shape (batch_size, horizon, action_dim).

        Returns:
            tuple: Contains (xt, t) and v where:
                - xt: Corrupted data tensor of shape (batch_size, horizon, action_dim).
                - t: Time step tensor of shape (batch_size,).
                - v: Target velocity tensor of shape (batch_size, horizon, action_dim).
        """
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        x0 = torch.randn(x1.shape, dtype=x1.dtype, device=self.device)
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0
        return (xt, t), v

    @torch.no_grad()
    def sample(
            self,
            cond: Tensor,
            inference_steps: int,
            record_intermediate: bool = False,
            clip_intermediate_actions: bool = True,
            z: torch.Tensor = None
    ) -> Sample:
        """Sample trajectories using the learned velocity field.

        Args:
            cond: Encoded observation of shape (batch_size, n_obs_steps, obs_feature_dim).
            inference_steps: Number of denoising steps.
            record_intermediate: Whether to return intermediate predictions.
            clip_intermediate_actions: Whether to clip actions to act_range.
            z: Optional initial noise of shape (batch_size,) + data_shape.

        Returns:
            Sample: Named tuple with 'trajectories' (and 'chains' if record_intermediate).
        """
        B = cond.shape[0]
        x_hat_list = None
        if record_intermediate:
            x_hat_list = torch.zeros(
                (inference_steps, B) + self.data_shape,
                dtype=cond.dtype, device=self.device)
        x_hat = z if z is not None else torch.randn(
            (B,) + self.data_shape, dtype=cond.dtype, device=self.device)
        dt = 1.0 / inference_steps
        steps = torch.linspace(0, 1 - dt, inference_steps, device=self.device)
        for i in range(inference_steps):
            t = steps[i].expand(B)
            vt = self.model(x_hat, t, cond)
            x_hat = x_hat + vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat
        return Sample(trajectories=x_hat, chains=x_hat_list)
