from typing import Dict
import torch
import torch.nn.functional as F
from collections import namedtuple

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.flow.mlp_shortcut import ShortCutFlowMLP
Sample = namedtuple("Sample", "trajectories chains")


class ShortcutFlowMlpLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ShortCutFlowMLP,
            horizon_steps,
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            max_denoising_steps=20,
            self_consistency_k=0.25,
            delta: float = 1e-5,
            sample_t_type: str = 'uniform',
            test_denoising_steps=20,
            test_clip_intermediate_actions=True,
            ):
        super().__init__()
        self.model = model

        self.normalizer = LinearNormalizer()
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.data_shape = (self.horizon_steps, self.action_dim)
        self.max_denoising_steps = max_denoising_steps
        self.self_consistency_k = self_consistency_k
        self.delta = delta
        self.sample_t_type = sample_t_type
        self.test_denoising_steps = test_denoising_steps
        self.test_clip_intermediate_actions = test_clip_intermediate_actions
        self.act_range = (None, None)
        assert self.n_action_steps <= self.horizon_steps, f"To={self.n_obs_steps}, Ta={self.n_action_steps}, Tp={self.horizon_steps} are incompatible"


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

        # build input
        cond = {'state': nobs[:, :To]}

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
        device = x1.device

        B = x1.shape[0]
        k_num = int(B * self.self_consistency_k)

        # Extract condition and action from batch (similar to Diffusion Policy)
        cond = {'state': cond[:, :self.n_obs_steps, :]}
        x1 = x1[:, self.n_obs_steps - 1:, :]

        # Self-consistency part. `_sc` means self-consistency. 
        x_sc_1 = x1[:k_num]
        d_base_sc = torch.randint(0, self.max_denoising_steps - 1, (k_num,), device=device)
        d_sc = 1.0 / (2 ** d_base_sc.float())
        d_boostrap_sc = d_sc / 2
        dt_sections_sc = 2 ** d_base_sc
        # print(f"d_base_sc={d_base_sc}, dt_sections_sc={dt_sections_sc}")
        
        t_idx_sc = torch.cat([torch.randint(0, int(dt_sections_sc[i]), (1,), device=device) for i in range(k_num)], dim=-1)
        t_sc = t_idx_sc.float() / dt_sections_sc
        x0_sc = torch.randn_like(x_sc_1, device=device)
        t_full_sc = t_sc[:, None, None].expand(-1, self.horizon_steps, self.action_dim)
        x_t_sc = (1 - (1 - self.delta) * t_full_sc) * x0_sc + t_full_sc * x_sc_1
        cond_sc = {key:value[:k_num] for key, value in cond.items()}
        with torch.no_grad():
            v_t_sc = self.model.forward(x_t_sc, t_sc, d_boostrap_sc, cond_sc)
            d_boostrap_full = d_boostrap_sc[:, None, None].expand(-1, self.horizon_steps, self.action_dim)
            x_t_sc2 = x_t_sc + v_t_sc * d_boostrap_full
            t2_sc = t_sc + d_boostrap_sc
            v_t_sc2 = self.model.forward(x_t_sc2, t2_sc, d_boostrap_sc, cond_sc)
            v_target_sc = (v_t_sc + v_t_sc2) / 2

        # Flow-matching part
        x_fm_1 = x1[k_num:]
        t_fm = torch.rand((B - k_num,), device=device)
        x0_fm = torch.randn_like(x_fm_1, device=device)
        t_full_fm = t_fm[:, None, None].expand(-1, self.horizon_steps, self.action_dim)
        x_t_fm = (1 - (1 - self.delta) * t_full_fm) * x0_fm + t_full_fm * x_fm_1
        v_target_fm = x_fm_1 - (1 - self.delta) * x0_fm
        d_fm = torch.zeros((B - k_num,), device=device)
        
        # Combine to a whole batch
        x_t = torch.cat([x_t_sc, x_t_fm], dim=0)
        t = torch.cat([t_sc, t_fm], dim=0)
        d = torch.cat([d_sc, d_fm], dim=0)
        v_target = torch.cat([v_target_sc, v_target_fm], dim=0)

        # Predict and compute loss
        v_pred = self.model.forward(x_t, t, d, cond)
        loss   = F.mse_loss(v_pred, v_target)
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        inference_steps: int,
        record_intermediate: bool = False,
        clip_intermediate_actions: bool = True
    ) -> Sample:
        """
        Sample action trajectories using the learned shortcut velocity field. We will use Euler integrator.

        Args:
            cond: dict with 'state' - Observations (B, To, Do)
            inference_steps: Number of denoising steps
            record_intermediate: Whether to record intermediate trajectories
            clip_intermediate_actions: Whether to clip actions to act_range

        Returns:
            Sample: Named tuple with 'trajectories' and optional 'chains'
        """
        B = cond['state'].shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        x_hat = torch.randn((B,) + self.data_shape, device=self.device)
        dt = 1.0 / inference_steps
        t = torch.linspace(0, 1 - dt, inference_steps, device=self.device)
        d = torch.full((B,), dt, device=self.device)

        for i in range(inference_steps):
            t_i = t[i] * torch.ones(B, device=self.device)
            vt = self.model.forward(x_hat, t_i, d, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps-1: # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)