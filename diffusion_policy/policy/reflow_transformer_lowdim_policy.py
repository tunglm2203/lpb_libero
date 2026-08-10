from typing import Dict
import torch
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

Sample = namedtuple("Sample", "trajectories chains")


class ReFlowTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(
            self,
            model: TransformerForDiffusion,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            sample_t_type: str = 'uniform',
            test_denoising_steps=20,
            test_clip_intermediate_actions=True
    ):
        super().__init__()
        self.model = model

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.data_shape = (self.horizon, self.action_dim)
        self.sample_t_type = sample_t_type
        self.test_denoising_steps = test_denoising_steps
        self.test_clip_intermediate_actions = test_clip_intermediate_actions
        self.act_range = (None, None)
        assert self.n_action_steps <= self.horizon, f"To={self.n_obs_steps}, Ta={self.n_action_steps}, Tp={self.horizon} are incompatible"

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        cond = nobs[:, :To]

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
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        act_min = self.normalizer['action'].params_dict.input_stats.min.min().item()
        act_max = self.normalizer['action'].params_dict.input_stats.max.max().item()
        self.act_range = (act_min, act_max)

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        cond = nbatch['obs']
        x1 = nbatch['action']

        # Extract condition and action from batch (similar to Diffusion Policy)
        cond = cond[:, :self.n_obs_steps, :]
        x1 = x1[:, self.n_obs_steps - 1:, :]

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
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(
            self.device)  # ReinFlow Authors revised on 04/23/2025
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
            logit_normal_samples = (1 / (1 + torch.exp(-normal_samples))).to(self.device)
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
            tuple: Contains (xt, t, obs) and v where:
                - xt: Corrupted data tensor of shape (batch_size, horizon, action_dim).
                - t: Time step tensor of shape (batch_size,).
                - v: Target velocity tensor of shape (batch_size, horizon, action_dim).
        """
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0
        return (xt, t), v

    @torch.no_grad()
    def sample(
            self,
            cond: dict,
            inference_steps: int,
            record_intermediate: bool = False,
            clip_intermediate_actions: bool = True,
            z: torch.Tensor = None
    ) -> Sample:
        """Sample trajectories using the learned velocity field.

                Args:
                    cond: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
                    inference_steps: Number of denoising steps.
                    record_intermediate: Whether to return intermediate predictions.
                    clip_intermediate_actions: Whether to clip actions to act_range.

                Returns:
                    Sample: Named tuple with 'trajectories' (and 'chains' if record_intermediate).
                """
        B = cond.shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=self.device)
        steps = torch.linspace(0, 1 - 1 / inference_steps, inference_steps, device=self.device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.model(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:  # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)