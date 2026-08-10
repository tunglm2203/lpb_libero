# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict
import torch
import numpy as np
# from lerobot.eval.service import BaseInferenceClient, BaseInferenceServer


import os
import hydra
import dill
from omegaconf import OmegaConf, open_dict
from termcolor import cprint
# from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.eval.service import BaseInferenceClient, BaseInferenceServer 
import time


class TemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    # def update(self, actions: Tensor) -> Tensor:
    def update(self, actions):
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action






class PolicyWrapper:
    """Wrapper to make policy compatible with the service interface"""
    
    def __init__(self, policy):
        self.policy = policy
        self.temporal_ensembler = TemporalEnsembler(0.0, 8)
        
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Convert observations to tensors, call policy, and return action as dict"""
        # Convert numpy arrays to tensors and ensure proper device placement
        # batch = {}
        # for key, value in observations.items():
        #     if isinstance(value, np.ndarray):
        #         batch[key] = torch.from_numpy(value).to(device=self.policy.config.device)
        #     elif isinstance(value, (list, tuple)) and key == "task":
        #         # Task is typically a list of strings, keep as is
        #         batch[key] = value
        #     else:
        #         batch[key] = value
        
        # Call the policy's select_action method
        # action = self.policy.select_action(batch)
        # actions = self.policy.predict_action(observations)
        
        # # Return as dict format expected by the service
        # for k in actions.keys():
        #     if isinstance(actions[k], torch.Tensor):
        #         actions[k] = actions[k].cpu().numpy()
        #     elif isinstance(actions[k], np.ndarray):
        #         pass  # Already numpy
        #     else:
        #         actions[k] = np.array(actions[k])
            
        #     if isinstance(actions[k], torch.Tensor):
        #         actions[k] = actions[k].cpu().numpy()
        #     elif isinstance(actions[k], np.ndarray):
        #         pass  # Already numpy
        #     else:
        #         actions[k] = np.array(actions[k])
        
        start = time.time()
        action = self.policy.predict_action(observations)['action']
        action = self.temporal_ensembler.update(action)
        end = time.time()
        
        print("inference time:", end - start)
        
        # Return as dict format expected by the service
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            pass  # Already numpy
        else:
            action = np.array(action)
            
        return {
            "action": action,
            "success": True
        }
            
        return {
            "action": actions,
            "success": True
        }

    def predict_action_chunk(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Convert observations to tensors, call policy, and return action as dict"""
        # Convert numpy arrays to tensors and ensure proper device placement
        # batch = {}
        # for key, value in observations.items():
        #     if isinstance(value, np.ndarray):
        #         batch[key] = torch.from_numpy(value).to(device=self.policy.config.device)
        #     elif isinstance(value, (list, tuple)) and key == "task":
        #         # Task is typically a list of strings, keep as is
        #         batch[key] = value
        #     else:
        #         batch[key] = value
        
        # # Call the policy's select_action method
        # action = self.policy.predict_action_chunk(batch)
        
        # actions = self.policy.
        
        # # Return as dict format expected by the service
        # for k in action.keys():
        #     if isinstance(action[k], torch.Tensor):
        #         action[k] = action[k].cpu().numpy()
        #     elif isinstance(action[k], np.ndarray):
        #         pass  # Already numpy
        #     else:
        #         action[k] = np.array(action[k])
            
        # return {
        #     "action": action,
        #     "success": True
        # }
        start = time.time()
        action = self.policy.predict_action(observations)['action']
        end = time.time()
        
        print("inference time:", end - start)
        # Return as dict format expected by the service
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        elif isinstance(action, np.ndarray):
            pass  # Already numpy
        else:
            action = np.array(action)
            
        return {
            "action": action,
            "success": True
        }

    def reset(self):
        self.policy.reset()


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "*", port: int = 5555, api_token: str = None):
        super().__init__(host, port, api_token)
        self.wrapped_model = PolicyWrapper(model)
        self.register_endpoint("get_action", self.wrapped_model.get_action)
        self.register_endpoint("predict_action_chunk", self.wrapped_model.predict_action_chunk)
        self.register_endpoint("reset", self.wrapped_model.reset, requires_input=False)

    @staticmethod
    def start_server(policy, port: int, api_token: str = None):
        server = RobotInferenceServer(policy, port=port, api_token=api_token)
        server.run()


class RobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None):
        super().__init__(host=host, port=port, api_token=api_token)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def predict_action_chunk(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("predict_action_chunk", observations)

    def reset(self):
        return self.call_endpoint("reset", requires_input=False)
    
    
###################################################################################


def setup_noise_schedule(cfg, noise_scheduler, num_inference_steps=None):
    if noise_scheduler.lower() == "ddpm":
        if cfg.policy.noise_scheduler['_target_'] == 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler':
            cprint(f"Skip setup scheduler: Pretrained model already used DDPM scheduler", "green")
        else:
            with open_dict(cfg.policy.noise_scheduler):
                cfg.policy.noise_scheduler = OmegaConf.create({
                    '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                    'num_train_timesteps': 100,
                    'beta_start': 0.0001,
                    'beta_end': 0.02,
                    'beta_schedule': 'squaredcos_cap_v2',
                    'variance_type': 'fixed_small',
                    'clip_sample': True,
                    'prediction_type': 'epsilon'
                })
    elif noise_scheduler.lower() == "ddim":
        if cfg.policy.noise_scheduler['_target_'] == 'diffusers.schedulers.scheduling_ddim.DDIMScheduler':
            cprint(f"Skip setup scheduler: Pretrained model already used DDIM scheduler", "green")
        else:
            with open_dict(cfg.policy.noise_scheduler):
                cfg.policy.noise_scheduler = OmegaConf.create({
                    '_target_': 'diffusers.schedulers.scheduling_ddim.DDIMScheduler',
                    'num_train_timesteps': 100,
                    'beta_start': 0.0001,
                    'beta_end': 0.02,
                    'beta_schedule': 'squaredcos_cap_v2',
                    'clip_sample': True,
                    'set_alpha_to_one': True,
                    'steps_offset': 0,
                    'prediction_type': 'epsilon'
                })
    else:
        raise ValueError(f"Unknown noise scheduler {noise_scheduler}")

    if num_inference_steps is not None:
        cfg.policy.num_inference_steps = num_inference_steps
    cprint(f"Noise Scheduler: {noise_scheduler.upper()}-{cfg.policy.num_inference_steps}", "green")
    print(OmegaConf.to_yaml(cfg.policy.noise_scheduler))
    return cfg

def load_normalizer(checkpoint_path: str, policy):
    normalizer_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "normalizer.pth")
    print("loading normalizer from", normalizer_path)
    state_dict = torch.load(normalizer_path, map_location="cpu")
    policy.normalizer = LinearNormalizer()
    policy.normalizer.load_state_dict(state_dict)
    policy.normalizer.to(policy.device)
    return policy