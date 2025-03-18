from functools import partial

import numpy as np

from algorithm.components.episode_buffer import EpisodeBatch
from train_test_wildfire import create_env, wrap_env
from gym_multigrid.utils.misc import save_frames_as_gif
import torch as th
from algorithm.utils.logging import Logger
from wandb import Video
import os


class EpisodeRunner:
    def __init__(self, args, logger: Logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = create_env(
            args,
            inference=args.evaluate
        )

        # if not args.evaluate:
        #     self.env = wrap_env(self.env, args)
        self.episode_limit = self.env.max_steps
        self.t = 0

        self.t_env = 0
        self.win_treshold = 0.5
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.frames = []

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()
    
    def start_rec(self):
        self.env.start_rec()

    def stop_rec(self):
        self.env.stop_rec()

    def save_replay(self):
        path = os.path.join(self.args.log,self.args.run_id+'-'+str(self.t)+".gif")
        self.env.save_replay(self.args.log, self.args.run_id, self.t_env)
        if self.args.wandb['enabled']:
            try:
                self.logger.wandb.log({"video": Video(path, format="gif")})
            except(FileNotFoundError):
                print("file not found:", path)
        # self.frames = []

    def close_env(self):
        self.stop_rec()
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        # self.frames = []
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        truncated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.env.num_agents)
        self.mac.init_hidden(batch_size=self.batch_size)
        log_prefix = "eval/" if test_mode else "train/"
        last_burnt = 0.
        last_unburnt = 1.
        while not terminated:
            s = [self.env.get_state()]
            # print(self.env.step_count)
            s_np = np.array([s for _ in range(self.env.num_agents)])
            o_np = np.array([obs for _, obs in self.env._get_obs().items()])
            pre_transition_data = {
                "state": th.tensor(s_np).flatten().unsqueeze(0),
                "avail_actions": th.tensor(self.env.get_avail_actions()),
                "obs": th.tensor(o_np).flatten().unsqueeze(0),
            }
            # for k, v in pre_transition_data.items():
            #     print(f"{k} shape: ", v.shape)
            self.batch.max_seq_length
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            outputs, actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            _, reward, terminated, truncated, env_info = self.env.step(actions.squeeze().cpu().numpy())
            terminated = terminated[0] or truncated[0]
            if self.args.save_replay:
                self.env.render()
            if test_mode:
                # burnt trees and unburnt trees
                for k, v in env_info[0].items():
                    self.logger.log_stat(
                        log_prefix + str(k), v, self.t_env
                    )
            
            post_transition_data = {
                "actions": actions.unsqueeze(0),
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                episode_return += reward
                # post_transition_data["reward"] = th.tensor(reward)
            else:
                # Collapsing reward into one single
                reward = tuple(reward.values())
                episode_return += np.mean(reward)
            post_transition_data["reward"] = th.tensor([reward]).unsqueeze(0)

            self.batch.update(post_transition_data, ts=self.t)
            last_burnt = env_info[0].get('burnt trees')
            last_unburnt = env_info[0].get('unburnt trees')
            self.t += 1
        s = self.env.get_state()
        s_np = np.array([s for _ in range(self.env.num_agents)])
        o_np = np.array([obs for _, obs in self.env._get_obs().items()])
        last_data = {
                "state": th.tensor(s_np).flatten().unsqueeze(0),
                "avail_actions": th.tensor(self.env.get_avail_actions()),
                "obs": th.tensor(o_np).flatten().unsqueeze(0),
            }
        if test_mode and self.args.render:
            print(f"Episode reward: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        outputs, actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions.unsqueeze(0)}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        # TODO: Insert reward, metric stats
        # cur_stats.update(
        #     {
        #         k: cur_stats.get(k, 0) + env_info.get(k, 0)
        #         for k in set(cur_stats)
        #     }
        # )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["burnt trees"] = last_burnt + cur_stats.get("burnt trees", 0)
        cur_stats['unburnt trees'] = last_unburnt + cur_stats.get("unburnt trees", 0)
        # cur_stats["win"] = cur_stats.get("win", []).append(last_burnt <= self.win_treshold)
        # cur_stats["win_rate"] = np.mean(cur_stats['win']) if 'win' in cur_stats.keys() else 0
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)


        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "mean_reward", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "std_of_mean_reward", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_reward",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_std_reward",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_mean_reward", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "std_of_total_mean_reward", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + "mean_" + k, v / stats["n_episodes"], self.t_env
                )
        # self.logger.log_stat(
        #             prefix + "win_rate", np.mean(stats['win']), self.t_env
        #         )
        
        stats.clear()
