import os
import torch
import numpy as np
import copy
import random
import math
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pref_replay_buffer import PrefReplayBuffer
from diffusion_policy.common.pref_sampler import PrefSequenceSampler
from typing import Optional, Dict
from diffusion_policy.common.prior_utils_confidence import BetaNetwork
from tqdm import tqdm


def _load_or_create_indices(
    path: str,
    num_queries: int,
    num_episodes_1: int,
    num_episodes_2: int,
    episode_ends_1: np.ndarray,
    episode_ends_2: np.ndarray,
    sequence_length: int,
    seed: int,
) -> np.ndarray:
    #Load pair indices from *path* if it already exists; otherwise sample them randomly, save them, and return.

    # Index layout — array shape ``(num_queries, 4)``:
    # [ep_idx_1, timestep_idx_1, ep_idx_2, timestep_idx_2]

    if os.path.isfile(path):
        print(f"=====================> PbrlDataset: Loading pair indices from {path}")
        data = np.load(path)
        indices = data["indices"]
        assert indices.shape == (num_queries, 4), (
            f"Loaded indices shape {indices.shape} does not match "
        )
        return indices

    # Generate fresh indices
    print(f"=====================> PbrlDataset: Generating new pair indices → {path}")
    rng = random.Random(seed)

    # Pre-compute per-episode lengths from cumulative episode_ends.
    def episode_length(episode_ends: np.ndarray, idx: int) -> int:
        start = episode_ends[idx - 1] if idx > 0 else 0
        return int(episode_ends[idx]) - int(start)

    indices = np.zeros((num_queries, 4), dtype=np.int64)
    for i in range(num_queries):
        ep_idx_1 = rng.randrange(num_episodes_1)
        ep_idx_2 = rng.randrange(num_episodes_2)

        len_ep1 = episode_length(episode_ends_1, ep_idx_1)
        len_ep2 = episode_length(episode_ends_2, ep_idx_2)

        # Random start index; fall back to 0 when the episode is shorter than
        # the requested sequence length (the caller will pad in that case).
        max_start_1 = max(len_ep1 - sequence_length, 0)
        max_start_2 = max(len_ep2 - sequence_length, 0)

        ts_idx_1 = rng.randint(0, max_start_1)  # inclusive on both ends
        ts_idx_2 = rng.randint(0, max_start_2)

        indices[i] = [ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, indices=indices)
    print(f"=====================> PbrlDataset: Saved pair indices to {path}")
    return indices

class PbrlDataset(BaseImageDataset):
    def __init__(self,
                 replay_buffer_1: ReplayBuffer,
                 replay_buffer_2: ReplayBuffer,
                 abs_action=True,
                 sequence_length=1,
                 gamma=0.9999,
                 num_queries=1,
                 seed=42,
                 val_ratio=0.0,
                 gpu_device='cuda:0',
                 dense_reward=False,
                 max_episodes_dataset_1=None,
                 max_episodes_dataset_2=None,
                 task_name="can_lowdim",
                 ):
        super().__init__()
        assert abs_action is True, "Only absolute action is supported"

        episode_ends_1 = replay_buffer_1.episode_ends
        num_episodes_1 = len(episode_ends_1) if max_episodes_dataset_1 is None else max_episodes_dataset_1
        episode_ends_2 = replay_buffer_2.episode_ends
        num_episodes_2 = len(episode_ends_2) if max_episodes_dataset_2 is None else max_episodes_dataset_2
        self.pref_replay_buffer = PrefReplayBuffer.create_empty_numpy()
        print(f"=====================> PbrlDataset: Num episodes (dataset_1): {num_episodes_1}")
        print(f"=====================> PbrlDataset: Num episodes (dataset_2): {num_episodes_2}")


        random.seed(seed)

        # NOTE: 18/05 tri load npz or sample-and-save
        idx_path = f"logs/pbrl_indices_10eps/{num_queries}_{sequence_length}/pair_{task_name}_{num_episodes_1}_{num_episodes_2}.npz"
        pair_indices = _load_or_create_indices(
            path=idx_path,
            num_queries=num_queries,
            num_episodes_1=num_episodes_1,
            num_episodes_2=num_episodes_2,
            episode_ends_1=episode_ends_1,
            episode_ends_2=episode_ends_2,
            sequence_length=sequence_length,
            seed=seed,
        )


        # check if saved indices existing
        for i in tqdm(range(num_queries), desc="Processing queries"):
            # print(f"=====================> PbrlDataset: Processing query {i}")
            ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2 = pair_indices[i]

            episode_1 = replay_buffer_1.get_episode(int(ep_idx_1), copy=False)
            episode_2 = replay_buffer_2.get_episode(int(ep_idx_2), copy=False)


            # Equal length processing for episode 1
            episode_1_len = len(episode_1['agentview_rgb'])
            if episode_1_len >= sequence_length:
                start_1 = int(ts_idx_1)
                length = sequence_length
                for key in episode_1.keys():
                    episode_1[key] = episode_1[key][start_1:start_1 + sequence_length]
            else:
                length = episode_1_len
                for key in episode_1.keys():
                    episode_1[key] = np.pad(episode_1[key],
                                        ((0, sequence_length - episode_1_len),) + ((0, 0),) * (episode_1[key].ndim - 1),
                                        mode='edge')

            # Equal length processing for episode 2
            episode_2_len = len(episode_2['agentview_rgb'])
            if episode_2_len >= sequence_length:
                start_2 = int(ts_idx_2)
                length_2 = sequence_length
                for key in episode_2.keys():
                    episode_2[key] = episode_2[key][start_2:start_2 + sequence_length]
            else:
                length_2 = episode_2_len
                for key in episode_2.keys():
                    episode_2[key] = np.pad(episode_2[key],
                                        ((0, sequence_length - episode_2_len),) + ((0, 0),) * (episode_2[key].ndim - 1),
                                        mode='edge')

            # Set up votes and metadata based on the presence of 'reward' in episode1
            if dense_reward:
                votes = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_1['rewards'])])
                votes_2 = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_2['rewards'])])
            else:
                votes = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_1['rewards'])])
                votes_2 = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_2['rewards'])])

            # Add preferred episode to the replay buffer

            self.pref_replay_buffer.add_pref_episode(
                data={
                    'obs': episode_1['agentview_rgb'],          # First trajectory observations (shape T, obs_dim)
                    'action': episode_1['action'],     # First trajectory actions (shape T, action_dim)
                    'obs_2': episode_2['agentview_rgb'],         # Second trajectory observations
                    'action_2': episode_2['action'],    # Second trajectory actions
                    'language': episode_1['language'],  # Language description
                    'language_2': episode_2['language'],  # Language description
                    'ee_pos': episode_1['ee_pos'],  # End-effector position
                    'ee_pos_2': episode_2['ee_pos'],  # End-effector position
                    'ee_ori': episode_1['ee_ori'],  # End-effector orientation
                    'ee_ori_2': episode_2['ee_ori'],  # End-effector orientation
                    'joint_states': episode_1['joint_states'],  # Joint states
                    'joint_states_2': episode_2['joint_states'],  # Joint states
                },
                meta_data={
                    'votes': votes,                   # Vote for the first trajectory
                    'votes_2': votes_2,               # Vote for the second trajectory
                    'length': np.array([length]),     # Length of the first trajectory
                    'length_2': np.array([length_2]), # Length of the second trajectory
                    'beta_priori': np.ones([2]),
                    'beta_priori_2': np.ones([2]),
                }
            )

        val_mask = get_val_mask(
            n_episodes=num_queries,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        self.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer,
            sequence_length=sequence_length,
            episode_mask=train_mask,
        )

        self.gpu_device = gpu_device
        self.length = num_queries
        self.train_mask = train_mask
        self.sequence_length = sequence_length
        self.beta_model: Optional[BetaNetwork] = None
        self.dense_reward = dense_reward

    def construct_pref_data(self):
        data = self.pref_replay_buffer.data
        pref_data = data.copy()
        meta = self.pref_replay_buffer.meta
        pref_data.update(meta)
        if 'episode_ends' in pref_data.keys():
            del pref_data['episode_ends']

        return pref_data

    def set_beta_priori(self, data_size=100):
        pref_data = self.construct_pref_data()
        self.beta_model = BetaNetwork(data=pref_data,
                                 device=self.gpu_device,
                                 data_size=data_size)

    def update_beta_priori(self, batch_size=3):

        def scale_to_range(x, min_val, max_val, target_min=1, target_max=10):

            if min_val == max_val:
                raise ValueError("min_val and max_val must be different to avoid division by zero.")
            return target_min + (x - min_val) * (target_max - target_min) / (max_val - min_val)

        # Define unified scaling logic
        def scale_tensor(x, global_min, global_max, target_min=1, target_max=10):

            # Ensure the tensor is a floating-point tensor
            if not torch.is_floating_point(x):
                x = x.float()

            local_min, local_max = torch.min(x), torch.max(x)

            # Handle division by zero for local range
            if local_min == local_max:
                return torch.full_like(x, target_min)  # Return tensor filled with target_min

            # Handle division by zero for global range
            # if global_min < 1:
            #     global_min = 1  # Replace 0 with a small positive value to avoid division by zero
            if global_min == global_max:
                raise ValueError("global_min and global_max must be different to avoid division by zero.")

            # Compute scaled local range
            scaled_min = (local_min / global_min) * target_min
            scaled_max = (local_max / global_max) * target_max

            # Apply scaling
            return scale_to_range(x, local_min, local_max, scaled_min, scaled_max)

        obs_1 = self.pref_replay_buffer.data['obs']
        obs_2 = self.pref_replay_buffer.data['obs_2']
        action_1 = self.pref_replay_buffer.data['action']
        action_2 = self.pref_replay_buffer.data['action_2']
        s_a_1 = np.concatenate([obs_1, action_1], axis=-1)
        s_a_2 = np.concatenate([obs_2, action_2], axis=-1)

        with torch.no_grad():
            interval = math.ceil(s_a_1.shape[0] / batch_size)
            alpha, beta = [], []
            alpha_2, beta_2 = [], []
            for i in range(interval):
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, s_a_1.shape[0])
                batch_s_a_1 = s_a_1[start_pt:end_pt, ...]
                batch_s_a_2 = s_a_2[start_pt:end_pt, ...]

                batch_alpha, batch_beta = self.beta_model.get_alpha_beta(torch.from_numpy(batch_s_a_1).float().to(self.beta_model.device))
                batch_alpha_2, batch_beta_2 = self.beta_model.get_alpha_beta(torch.from_numpy(batch_s_a_2).float().to(self.beta_model.device))

                alpha.append(batch_alpha)
                beta.append(batch_beta)
                alpha_2.append(batch_alpha_2)
                beta_2.append(batch_beta_2)

            alpha = torch.cat(alpha, dim=0)+1
            beta = torch.cat(beta, dim=0)+1
            alpha_2 = torch.cat(alpha_2, dim=0)+1
            beta_2 = torch.cat(beta_2, dim=0)+1

            mean_value = torch.mean(torch.cat([alpha, beta, alpha_2, beta_2]))
            std_value = torch.std(torch.cat([alpha, beta, alpha_2, beta_2]))

            alpha = torch.clamp(alpha, max=mean_value+3*std_value)
            beta = torch.clamp(beta, max=mean_value+3*std_value)
            alpha_2 = torch.clamp(alpha_2, max=mean_value+3*std_value)
            beta_2 = torch.clamp(beta_2, max=mean_value+3*std_value)

            max_value = torch.max(torch.cat([alpha, beta, alpha_2, beta_2]))
            min_value = torch.min(torch.cat([alpha, beta, alpha_2, beta_2]))

            target_min, target_max = 1, 3

            alpha = scale_tensor(alpha, min_value, max_value, target_min, target_max)
            beta = scale_tensor(beta, min_value, max_value, target_min, target_max)
            alpha_2 = scale_tensor(alpha_2, min_value, max_value, target_min, target_max)
            beta_2 = scale_tensor(beta_2, min_value, max_value, target_min, target_max)

            self.pref_replay_buffer.meta['beta_priori'] = np.array([alpha.cpu().numpy(), beta.cpu().numpy()]).T
            self.pref_replay_buffer.meta['beta_priori_2'] = np.array([alpha_2.cpu().numpy(), beta_2.cpu().numpy()]).T
            self.pref_replay_buffer.root['meta']['beta_priori'] = np.array([alpha.cpu().numpy(), beta.cpu().numpy()]).T
            self.pref_replay_buffer.root['meta']['beta_priori_2'] = np.array([alpha_2.cpu().numpy(), beta_2.cpu().numpy()]).T

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer, 
            sequence_length=self.sequence_length,
            episode_mask=~self.train_mask,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_all_actions(self) -> torch.Tensor:
        actions = np.concatenate(self.pref_replay_buffer.data['action'], self.pref_replay_buffer.data['action_2'], dim = 0)
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        return self.sampler.__len__()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = self.sampler.sample_sequence(idx)
        return torch_data