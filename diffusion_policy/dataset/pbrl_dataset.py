import os
import torch
import torch.nn as nn
import numpy as np
import copy
import random
import time
from termcolor import cprint
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.pref_replay_buffer import PrefReplayBuffer
from diffusion_policy.common.pref_sampler import PrefSequenceSampler
from diffusion_policy.preference_labeling.preference_labeling import (
    load_or_create_indices,
    load_or_compute_feats,
    precompute_pair_rewards,
    get_context_observations,
    extract_segment_pseudo_reward,
    ResNet, R3M, LIV, VIP
)
from typing import Dict


class PbrlDataset(BaseImageDataset):
    def __init__(
            self,
            replay_buffer_1: ReplayBuffer,
            replay_buffer_2: ReplayBuffer,
            abs_action=True,
            sequence_length=1,
            gamma=0.999,
            num_queries=1,
            seed=42,
            gpu_device='cuda:0',
            dense_reward=False,
            val_ratio_data1=None,
            val_ratio_data2=None,
            dataset_1_path=None,
            dataset_2_path=None,
            task_name="can_lowdim",
            pseudo_preference=False,
            replay_buffer_expert=None,
            dataset_expert_path=None,
            feature_extractor="r3m_resnet18",
            context_num=3,
            seg_margin=0.6,
            min_progress=0.0,
            n_demos_for_preference=10,
    ):
        super().__init__()
        assert abs_action is True, "Only absolute action is supported"
        assert feature_extractor in ["imagenet_resnet18", "r3m_resnet18", "liv_resnet50", "vip_resnet50"]
        self.pseudo_preference = pseudo_preference

        # Hyperparameters for pseudo-labeling
        self.seg_margin = seg_margin        # Segment must beat the other by `seg_margin` % coverage to win
        self.min_progress = min_progress    # At least one segment must achieve 'min_progress' % coverage
        self.context_num = context_num        # context window for computing ORCA
        self.n_demos_for_preference = n_demos_for_preference

        episode_ends_1 = replay_buffer_1.episode_ends
        episode_ends_2 = replay_buffer_2.episode_ends
        num_episodes_1 = int(len(episode_ends_1) * (1 - val_ratio_data1))
        num_episodes_2 = int(len(episode_ends_2) * (1 - val_ratio_data2))
        self.pref_replay_buffer = PrefReplayBuffer.create_empty_numpy()
        random.seed(seed)
        print(f"=====================> PbrlLowdimDataset: Num episodes (dataset_1): {num_episodes_1}, "
              f"min_len={replay_buffer_1.episode_lengths.min()}, max_len={replay_buffer_1.episode_lengths.max()}")
        print(f"=====================> PbrlLowdimDataset: Num episodes (dataset_2): {num_episodes_2},"
              f"min_len={replay_buffer_2.episode_lengths.min()}, max_len={replay_buffer_2.episode_lengths.max()}")


        # NOTE: 18/05 tri load npz or sample-and-save
        idx_path = f"logs/pbrl_indices/{task_name}/pair_{task_name}_nQ{num_queries}_L{sequence_length}_{num_episodes_1}_{num_episodes_2}.npz"
        pair_indices = load_or_create_indices(
            path=idx_path,
            num_queries=num_queries,
            num_episodes_1=num_episodes_1,
            num_episodes_2=num_episodes_2,
            episode_ends_1=episode_ends_1,
            episode_ends_2=episode_ends_2,
            sequence_length=sequence_length,
            seed=seed,
            use_cached=False,
            save_cached=True
        )

        self.selected_pair_indices = []
        self.replay_buffer_1 = replay_buffer_1
        self.replay_buffer_2 = replay_buffer_2

        if self.pseudo_preference:
            if feature_extractor == "imagenet_resnet18":
                encoder = ResNet().to(gpu_device).eval()
            elif feature_extractor == "r3m_resnet18":
                encoder = R3M().to(gpu_device).eval()
            elif feature_extractor == "liv_resnet50":
                encoder = LIV().to(gpu_device).eval()
            elif feature_extractor == "vip_resnet50":
                encoder = VIP().to(gpu_device).eval()
            else:
                raise ValueError(f"Unknown feature extractor: {feature_extractor}")

            base_1 = os.path.join(os.path.dirname(dataset_1_path), "videos")
            base_2 = os.path.join(os.path.dirname(dataset_2_path), "videos")
            video_paths_1 = [f"{base_1}/episode_{i}.mp4" for i in range(num_episodes_1)]
            video_paths_2 = [f"{base_2}/episode_{i}.mp4" for i in range(num_episodes_2)]


            # Get visual features for trajectories in dataset_1 and dataset_2
            os.makedirs("cache", exist_ok=True)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            start = time.time()
            feats_1 = load_or_compute_feats(
                f"cache/dataset_1_{task_name.replace('_lowdim', '')}_{feature_extractor}_len{len(video_paths_1)}.npz",
                video_paths_1, encoder, device, drop_last='collect' in video_paths_1,
                use_cached=True, save_cached=True) # len = 45, [0].shape = numframes,512
            feats_2 = load_or_compute_feats(
                f"cache/dataset_2_{task_name.replace('_lowdim', '')}_{feature_extractor}_len{len(video_paths_2)}.npz",
                video_paths_2, encoder, device, drop_last='collect' in video_paths_2,
                use_cached=True, save_cached=True) # len = 200, [0].shape = numframes,512
            print(f"Total time to load/encode {len(video_paths_1) + len(video_paths_2)} videos: {time.time() - start:.2f}s")
            
            if replay_buffer_expert is None:
                replay_buffer_expert = replay_buffer_1
                dataset_expert_path = dataset_1_path

            top_k = np.argpartition(replay_buffer_expert.episode_lengths, self.n_demos_for_preference)[:self.n_demos_for_preference]
            base_3 = os.path.join(os.path.dirname(dataset_expert_path), "videos")
            expert_paths = [f"{base_3}/episode_{i}.mp4" for i in top_k]
            expert_feats = load_or_compute_feats(f"cache/experts_{task_name}_nD{self.n_demos_for_preference}_{feature_extractor}.npz", expert_paths, encoder, device, use_cached=False, save_cached=False)
            expert_ctx = [get_context_observations(f, context_num=self.context_num) for f in expert_feats] # [0].shape = context_num, numframes, 512


            # ----- Precompute per-(traj, expert) rewards (the big win) -----
            print("precomputing trajectory's rewards for dataset 1...")
            dataset_rewards_1 = precompute_pair_rewards(feats_1, expert_ctx, context_num=self.context_num)
            print("precomputing trajectory's rewards for dataset 2...")
            dataset_rewards_2 = precompute_pair_rewards(feats_2, expert_ctx, context_num=self.context_num)

            assert len(dataset_rewards_1) == num_episodes_1 and len(dataset_rewards_2) == num_episodes_2

        # check if saved indices existing
        orca_match = 0
        retained_pairs = 0
        for i in tqdm(range(num_queries), desc="Processing queries"):
            ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2 = pair_indices[i]
            ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2 = int(ep_idx_1), int(ts_idx_1), int(ep_idx_2), int(ts_idx_2)

            # episode_1 = replay_buffer_1.get_episode(ep_idx_1, copy=False)
            # episode_2 = replay_buffer_2.get_episode(ep_idx_2, copy=False)

            episode_1 = replay_buffer_1.get_episode(ep_idx_1, keys=['action', 'rewards'], copy=False)
            episode_2 = replay_buffer_2.get_episode(ep_idx_2, keys=['action', 'rewards'], copy=False)

            # Equal length processing for episode 1
            episode_1_len = len(episode_1['action'])
            if episode_1_len >= sequence_length:
                start_1 = ts_idx_1
                length = sequence_length
                for key in episode_1.keys():
                    episode_1[key] = episode_1[key][start_1:start_1 + sequence_length]
            else:
                length = episode_1_len
                for key in episode_1.keys():
                    episode_1[key] = np.pad(episode_1[key], ((0, sequence_length - episode_1_len),) + ((0, 0),) * (episode_1[key].ndim - 1), mode='edge')

            # Equal length processing for episode 2
            episode_2_len = len(episode_2['action'])
            if episode_2_len >= sequence_length:
                start_2 = ts_idx_2
                length_2 = sequence_length
                for key in episode_2.keys():
                    episode_2[key] = episode_2[key][start_2:start_2 + sequence_length]
            else:
                length_2 = episode_2_len
                for key in episode_2.keys():
                    episode_2[key] = np.pad(episode_2[key], ((0, sequence_length - episode_2_len),) + ((0, 0),) * (episode_2[key].ndim - 1), mode='edge')

            # Set up votes and metadata based on the presence of 'reward' in episode1
            votes = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_1['rewards'])])
            votes_2 = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_2['rewards'])])

            if self.pseudo_preference:
                gt = 1 if votes_2 > votes else 0

                orca_1_scores, orca_1_scores_all = extract_segment_pseudo_reward(dataset_rewards_1[ep_idx_1], ts_idx_1, sequence_length)
                orca_2_scores, orca_2_scores_all = extract_segment_pseudo_reward(dataset_rewards_2[ep_idx_2], ts_idx_2, sequence_length)
                # Max-pool over expert's trajectories
                score_1 = orca_1_scores.max()
                score_2 = orca_2_scores.max()

                # Apply Threshold & Margin
                if max(score_1, score_2) < self.min_progress:
                    pref_label = -1  # Discard: Neither segment did anything useful
                elif score_2 - score_1 > self.seg_margin:
                    pref_label = 1  # Right wins cleanly
                elif score_1 - score_2 > self.seg_margin:
                    pref_label = 0  # Left wins cleanly
                else:
                    pref_label = -1  # Discard: Difference is too small (noise)

                if pref_label != -1:
                    retained_pairs += 1
                    orca_match += (pref_label == gt)

                    votes, votes_2 = score_1, score_2
                    # Add preferred episode to the replay buffer

                    data_pref = {}
                    for key in episode_1.keys():
                        data_pref[key] = episode_1[key]
                        data_pref[key + '_2'] = episode_2[key]
                    
                    # Add indices for seg_1 and seg_2
                    self.selected_pair_indices.append([ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2, sequence_length])

                    self.pref_replay_buffer.add_pref_episode(
                        data=data_pref,
                        meta_data={
                            'votes': votes,                   # Vote for the first trajectory
                            'votes_2': votes_2,               # Vote for the second trajectory
                            'length': np.array([length]),     # Length of the first trajectory
                            'length_2': np.array([length_2]), # Length of the second trajectory
                            'beta_priori': np.ones([2]),
                            'beta_priori_2': np.ones([2]),
                        }
                    )

                # Calculate metrics
                retention_rate = (retained_pairs / num_queries) * 100
                accuracy = (orca_match / retained_pairs * 100) if retained_pairs > 0 else 0.0
                # end
            else:
                # Add preferred episode to the replay buffer
                data_pref = {}
                for key in episode_1.keys():
                    data_pref[key] = episode_1[key]
                    data_pref[key + '_2'] = episode_2[key]

                self.pref_replay_buffer.add_pref_episode(
                    data=data_pref,
                    meta_data={
                        'votes': votes,                   # Vote for the first trajectory
                        'votes_2': votes_2,               # Vote for the second trajectory
                        'length': np.array([length]),     # Length of the first trajectory
                        'length_2': np.array([length_2]), # Length of the second trajectory
                        'beta_priori': np.ones([2]),
                        'beta_priori_2': np.ones([2]),
                    }
                )

        if self.pseudo_preference:
            assert retained_pairs > 0, f"Margin ({self.seg_margin}) is too strict! 0 pairs retained out of {num_queries}."
            cprint(f"Task={task_name.upper()}: n_expert={self.n_demos_for_preference}, n_queries={num_queries}, seq_len={sequence_length}, feat={feature_extractor}, min_progress={min_progress}, margin={seg_margin}", "green", attrs=["bold"])
            cprint(f"   -> Pairs Retained: {retained_pairs} ({retention_rate:.1f}%)", "cyan")
            cprint(f"   -> ORCA Accuracy (on retained): {accuracy:.1f}%", "green", attrs=["bold"])
            train_mask = np.ones(retained_pairs, dtype=bool)
            self.retained_pairs = retained_pairs
            self.accuracy = accuracy
            self.retention_rate = retention_rate
        else:
            train_mask = np.ones(num_queries, dtype=bool)
            self.accuracy = 0.0
            self.retained_pairs = num_queries
            self.retention_rate = 100

        self.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer,
            sequence_length=sequence_length,
            episode_mask=train_mask,
        )

        self.gpu_device = gpu_device
        self.train_mask = train_mask
        self.sequence_length = sequence_length
        self.dense_reward = dense_reward

    def construct_pref_data(self):
        data = self.pref_replay_buffer.data
        pref_data = data.copy()
        meta = self.pref_replay_buffer.meta
        pref_data.update(meta)
        if 'episode_ends' in pref_data.keys():
            del pref_data['episode_ends']

        return pref_data

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


        ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2, seg_size = self.selected_pair_indices[idx]
        ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2 = int(ep_idx_1), int(ts_idx_1), int(ep_idx_2), int(ts_idx_2)
        
        # load full trajectories
        episode_1 = self.replay_buffer_1.get_episode(ep_idx_1, copy=False)  # keys=['action', 'rewards']
        episode_2 = self.replay_buffer_2.get_episode(ep_idx_2, copy=False)

        # Equal length processing for episode 1
        episode_1_len = len(episode_1['action'])
        if episode_1_len >= self.sequence_length:
            start_1 = ts_idx_1
            for key in episode_1.keys():
                episode_1[key] = episode_1[key][start_1:start_1 + self.sequence_length]
        else:
            for key in episode_1.keys():
                episode_1[key] = np.pad(episode_1[key], ((0, self.sequence_length - episode_1_len),) + ((0, 0),) * (episode_1[key].ndim - 1), mode='edge')

        # Equal length processing for episode 2
        episode_2_len = len(episode_2['action'])
        if episode_2_len >= self.sequence_length:
            start_2 = ts_idx_2
            for key in episode_2.keys():
                episode_2[key] = episode_2[key][start_2:start_2 + self.sequence_length]
        else:
            for key in episode_2.keys():
                episode_2[key] = np.pad(episode_2[key], ((0, self.sequence_length - episode_2_len),) + ((0, 0),) * (episode_2[key].ndim - 1), mode='edge')


        assert (episode_1['action'] - torch_data['action'].cpu().numpy()).sum() == 0
        assert (episode_2['action'] - torch_data['action_2'].cpu().numpy()).sum() == 0

        # Convert to torch for segment_1
        for key in episode_1.keys():
            if key in ['abs_action', 'action', 'rewards']:
                continue
            value = episode_1[key]
            if 'image' in key or 'rgb' in key:
                if value.shape[-1] == 3:
                    value = value.transpose(0, 3, 1, 2)
            if isinstance(value, np.ndarray):
                torch_data[key] = torch.from_numpy(value)
            elif isinstance(value, (np.float32, np.float64, float, int)):
                torch_data[key] = torch.tensor(value, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported type {type(value)} for key '{key}'")

        # Convert to torch for segment_2
        for key in episode_2.keys():
            if key in ['abs_action', 'action', 'rewards']:
                continue
            value = episode_2[key]

            if 'image' in key or 'rgb' in key:
                if value.shape[-1] == 3:
                    value = value.transpose(0, 3, 1, 2)
            if isinstance(value, np.ndarray):
                torch_data[key + "_2"] = torch.from_numpy(value)
            elif isinstance(value, (np.float32, np.float64, float, int)):
                torch_data[key + "_2"] = torch.tensor(value, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported type {type(value)} for key '{key}'")

        return torch_data