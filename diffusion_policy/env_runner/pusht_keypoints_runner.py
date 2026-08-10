import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

from termcolor import colored
from diffusion_policy.sampler.single import coherence_sampler, ema_sampler, ac_sampler, sgac_sampler
from diffusion_policy.sampler.multi import contrastive_sampler, bidirectional_sampler
from diffusion_policy.sampler.condition import NoiseGenerator

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(
            self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            perturb_level=0.0,
            return_intermediate_state=False,
            use_oracle_ac=False,
            oracle_ac_config=None,
            collect_data=False,
        ):
        super().__init__(output_dir)
        self.return_intermediate_state = return_intermediate_state
        self.use_oracle_ac = use_oracle_ac
        self.oracle_ac_config = oracle_ac_config
        self.collect_data = collect_data

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        self.env_n_action_steps = n_action_steps
        _env_n_action_steps = 1 if self.return_intermediate_state else self.env_n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        kp_kwargs['perturb_level'] = perturb_level

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=_env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    if self.collect_data:
                        filename = pathlib.Path(output_dir).joinpath('media', f"episode_{seed - test_start_seed}.mp4")
                    else:
                        filename = pathlib.Path(output_dir).joinpath('media', f"{seed}_" + wv.util.generate_id() + ".mp4")
                    print(filename)
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.sampler = None
        self.n_samples = 0
        self.nmode = 0
        self.weak = None
        self.decay = 1.0
        self.noise = 0.0
        self.disruptor = None

    def set_sampler(self, sampler, nsample=1, nmode=1, noise=0.0, decay=1.0, tau=0.99):
        self.sampler = sampler
        self.n_samples = nsample
        self.nmode = nmode
        self.noise = noise
        self.decay = decay
        self.tau = tau
        if noise > 0:
            self.disruptor = NoiseGenerator(self.noise)
        print(colored(f'Set sampler: {sampler} {nsample}/{nmode}', 'yellow'))

    def set_reference(self, weak):
        self.weak = weak

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_steps_until_done = [None] * n_inits
        all_calls_until_done = np.ones((n_inits,), dtype=int)  # default for querying at least one

        if self.collect_data:
            collect_observations = [[] for _ in range(n_inits)]
            collect_actions = [[] for _ in range(n_inits)]
            collect_rewards = [[] for _ in range(n_inits)]
            collect_terminals = [[] for _ in range(n_inits)]
        else:
            collect_observations = collect_actions = collect_rewards = collect_terminals = None

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            if self.use_oracle_ac:
                raise NotImplementedError
            else:
                oracle_ac = None
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", leave=False)
            done = False
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,-policy.n_obs_steps:,:Do].astype(np.float32),
                    'obs_mask': obs[...,-policy.n_obs_steps:,Do:] > 0.5
                }

                # previous conditional (ot-1)
                if self.sampler in ['sg', 'sgac']:
                    prev_obs_dict = {
                        # handle n_latency_steps by discarding the last n_latency_steps
                        'obs': obs[..., -policy.n_obs_steps-1:-1, :Do].astype(np.float32),
                        'obs_mask': obs[..., -policy.n_obs_steps-1:-1, Do:] > 0.5
                    }

                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[:,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                # run policy
                with torch.no_grad():
                    if self.sampler == 'random':
                        action_dict = policy.predict_action(obs_dict)
                    elif self.sampler == 'ema':
                        if 'action_prior' not in locals():
                            action_prior = None
                        action_dict = ema_sampler(policy, action_prior, obs_dict, self.decay)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                    elif self.sampler == 'contrast':
                        action_dict = contrastive_sampler(policy, self.weak, obs_dict, self.n_samples, self.nmode, self.sampler)
                    elif self.sampler == 'coherence':
                        if 'action_prior' not in locals():
                            action_prior = None
                        action_dict = coherence_sampler(policy, action_prior, obs_dict, self.n_samples, self.decay)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                    elif self.sampler == 'bid':
                        if 'action_prior' not in locals():
                            action_prior = None
                        action_dict = bidirectional_sampler(policy, self.weak, obs_dict, action_prior, self.n_samples, self.decay, self.nmode)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                    elif self.sampler == 'sg':
                        action_dict = policy.predict_action(obs_dict, prev_obs_dict)
                    elif self.sampler == 'ac':
                        if 'action_prior' not in locals():
                            action_prior = None
                        action_dict = ac_sampler(policy, action_prior, obs_dict, self.tau)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                    elif self.sampler == 'sgac':
                        if 'action_prior' not in locals():
                            action_prior = None
                            action_dict = sgac_sampler(policy, action_prior, obs_dict, obs_dict, self.tau)
                        else:
                            action_dict = sgac_sampler(policy, action_prior, obs_dict, prev_obs_dict, self.tau)
                        action_prior = action_dict['action_pred'][:, self.n_action_steps:]
                    else:
                        action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                # noise
                if self.noise > 0.0:
                    noise_cum = self.disruptor.step(np_action_dict['action_pred'])
                    action += noise_cum[:, :action.shape[1]]

                # step env
                if self.return_intermediate_state:  # Expose intermediate states while executing sequence of actions
                    if self.use_oracle_ac:
                        # At this point, always need to update action queue
                        if oracle_ac.first_time:
                            oracle_ac.update_action_chunk(action, replanning_mask=None)  # fill action for all envs at reset
                        else:
                            oracle_ac.update_action_chunk(action, replanning_mask=replanning_mask)

                        total_executed_steps = 0
                        while True:
                            single_step_action = oracle_ac.get_action()
                            obs, reward, done, info = env.step(single_step_action)
                            total_executed_steps += 1
                            replanning_mask = oracle_ac.compute_mask_to_replan(obs, reward, info, done, config=self.oracle_ac_config)
                            if replanning_mask.any():
                                break

                        query_mask = 1 - done  # 1 means query, 0 means no query
                        all_calls_until_done[start:end] = all_calls_until_done[start:end] + replanning_mask.astype(int)[0:end - start] * query_mask[0:end - start]
                        done = np.all(done)
                        past_action = action
                        # update pbar
                        pbar.update(total_executed_steps)

                    else:
                        for a_idx in range(self.n_action_steps):
                            single_step_action = action[:, a_idx:a_idx + 1, :]
                            obs, reward, done, info = env.step(single_step_action)

                            # Record data if in collect_data mode
                            if self.collect_data:
                                for i in range(n_envs):
                                    collect_observations[chunk_idx * n_envs + i].append(obs[i, 0, :Do])
                                    collect_actions[chunk_idx * n_envs + i].append(single_step_action[i, 0, ...])
                                    # collect_rewards[chunk_idx * n_envs + i].append(reward[i]) # This per-step reward is not correct.
                                    collect_terminals[chunk_idx * n_envs + i].append(done[i])

                        query_mask = 1 - done  # 1 means query, 0 means no query
                        all_calls_until_done[start:end] = all_calls_until_done[start:end] + query_mask[0:end - start]
                        done = np.all(done)
                        past_action = action
                        # update pbar
                        pbar.update(action.shape[1])

                else:
                    obs, reward, done, info = env.step(action)
                    query_mask = 1 - done  # 1 means query, 0 means no query
                    all_calls_until_done[start:end] = all_calls_until_done[start:end] + query_mask[0:end - start]
                    done = np.all(done)
                    past_action = action

                    # update pbar
                    pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_steps_until_done[this_global_slice] = env.call('get_attr', 'step_elapsed')[this_local_slice]
            if self.collect_data:
                for i in range(n_envs):
                    episode_reward = np.array(all_rewards[chunk_idx * n_envs + i])
                    collect_rewards[chunk_idx * n_envs + i].extend(episode_reward)

        # log
        max_rewards = collections.defaultdict(list)
        successes = collections.defaultdict(list)
        env_step_till_max_reward = collections.defaultdict(list)
        env_step_till_done = collections.defaultdict(list)
        policy_step_till_done = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            success = float(max_reward == 1.0)

            max_rewards[prefix].append(max_reward)
            successes[prefix].append(success)
            env_step_till_max_reward[prefix].append(np.argmax(all_rewards[i]))
            env_step_till_done[prefix].append(all_steps_until_done[i])
            policy_step_till_done[prefix].append(all_calls_until_done[i])

            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix + f'sim_success_{seed}'] = success
            log_data[prefix + f'sim_step_to_max_reward_{seed}'] = float(np.argmax(all_rewards[i]))
            log_data[prefix + f'sim_step_to_success_{seed}'] = float(all_steps_until_done[i])
            log_data[prefix + f'sim_policy_call_to_success_{seed}'] = float(all_calls_until_done[i])

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in successes.items():
            name = prefix + 'mean_success'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in env_step_till_max_reward.items():
            name = prefix + 'mean_env_step_till_max_reward'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in env_step_till_done.items():
            name = prefix + 'mean_env_step_till_done'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in policy_step_till_done.items():
            name = prefix + 'mean_policy_step_till_done'
            value = np.mean(value)
            log_data[name] = value

        if self.collect_data:
            final_observations, final_actions, final_rewards, final_terminals = [], [], [], []

            for i in range(n_inits):
                idx = np.argmax(collect_terminals[i]) + 1   # Find that first done
                final_observations.extend(collect_observations[i][:idx])
                final_actions.extend(collect_actions[i][:idx])
                final_rewards.extend(collect_rewards[i][:idx])
                final_terminals.extend(collect_terminals[i][:idx])

            episode_data = {
                'observations': np.array(final_observations),
                'actions': np.array(final_actions),
                'rewards': np.array(final_rewards),
                'terminals': np.array(final_terminals),
            }
            return log_data, episode_data
        else:
            return log_data
