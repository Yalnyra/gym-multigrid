import os
import time
from copy import deepcopy
import hydra.core
import hydra
import numpy as np
import matplotlib.pyplot as plt
from pz_multigrid.envs import WildfireEnv
from gym_multigrid.utils.misc import save_frames_as_gif
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError
import wandb

from functools import partial

import numpy as np

from components.episode_buffer import EpisodeBatch
from train_test_wildfire import create_env

class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = create_env(
            args.env,
            inference=args.evaluate
        )
        self.episode_limit = self.env.max_steps
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

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

    # def save_replay(self):
    #     self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.actions()],
                "obs": [self.env.observe()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
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
            self.logger.log_stat(prefix + "reward", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "reward_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_reward",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_episode_reward", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_reward_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

# PyMARL algorithms
# SEE LICENSE & NOTICE

from algorithm.controllers import REGISTRY as mac_REGISTRY
from algorithm.components.episode_buffer import ReplayBuffer
from algorithm.components.transforms import OneHot
from algorithm.learners import REGISTRY as le_REGISTRY
from algorithm.runners import REGISTRY as r_REGISTRY
from algorithm.utils.general_reward_support import test_alg_config_supports_reward
from algorithm.utils.logging import Logger
from algorithm.utils.timehelper import time_left, time_str


# AgileRL HPO & algorithms
from train_multi_agent import train_multi_agent
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
# from agilerl.hpo.mutation import Mutations
# from agilerl.hpo.tournament import TournamentSelection
# from agilerl.utils.utils import create_population

# Async Env parameter sharing
# from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1


def load_model(model_path: str, name:str, env=None):
    print(f"Attempting to load {name} from path: {model_path}")
    # does it need env?
    # model_path = os.path.abspath(model_path)
    print("Current dir: ", os.path.abspath(os.curdir))
    define = None
    match name:
        case 'ppo':
            define = hydra.utils.get_class('sbx.PPO')
        case 'matd3':
            define = hydra.utils.get_class('agilerl.algorithms.matd3.MATD3')
            model_path += '.pt'
        case 'maddpg':
            define = hydra.utils.get_class('agilerl.algorithms.maddpg.MADDPG')
            model_path += '.pt'
        case _:
            raise KeyError(name)
    if define is None:
        raise ImportError(f"Could not import class definition of algorithm: {name}")
    
    if env is not None:
        return define.load(model_path, env=env)
    
    return define.load(model_path)


def test_model(env, config:DictConfig, model=None):
    frames = []
    for ep in range(config["episodes"]):
        steps = 0
        terminated, truncated = {0: False}, {0: False}
        frac_burned = -0.1
        (obs, info) = env.reset()
        ep_reward = 0.
        while not terminated[0]:
            agents = list(range(config['env']['agents']))
            actions = {agent: env.action_space().sample() for agent in agents}
            inference_dt = time.time()
            # Sample actual actions
            if model is not None:
                # actions, states = model.predict(obs, deterministic=True)
                match config['training_type']:
                    case 'sb3':
                        actions = {agent: model.predict(obs[agent], deterministic=False)[0] for agent in agents}
                    case 'agile_rl':
                        _, actions = model.get_action(
                            obs, training=False, infos=info
                        )
                        
                print("using model, actions are: ", actions)

            inference_dt = 1.0 / (time.time() - inference_dt + 1e-6)
            
            obs, reward, terminated, truncated, infos = env.step(actions)
            # Show next step on screen and add to the video frames list
            frames.append(env.render())
            mean_reward_per_agent = np.mean(list(reward.values()))
            ep_reward += float(mean_reward_per_agent)
            frac_burned = infos[0]['burnt trees']
            frac_unburned = infos[0]['unburnt trees']
            print(
                f"Reward: {reward}, \n burnt trees %: {frac_burned}, \n terminated: {terminated}, truncated: {truncated}, Frames length: {len(frames)}"
                # , Inference time in ms: {inference_dt}"
            )
            
            steps += 1
            if config['wandb']['enabled']:
            # Log reward at each step to wandb
                wandb.log(
                    {
                        "eval/reward": mean_reward_per_agent,
                        "eval/burnt trees": frac_burned,
                        "eval/unburnt trees": frac_unburned,
                        "Inference FPS": inference_dt,
                    }
                )
        if config['wandb']['enabled']:
        # Total episode reward
            wandb.log(
                {
                    "eval/mean_reward": ep_reward,
                }
        )
    save_frames_as_gif(frames=frames, path=config['log'], filename=f"{config['run_id']}-{config['job_type']}", ep=config['train_epochs'], fps=5)
    if config['wandb']['enabled']:
        wandb.log({"video": wandb.Video(f"{config['log']}/{config['run_id']}-{config['job_type']}-{config['train_epochs']}.gif", format="gif")})
        


def train(config: DictConfig):
    # Create and wrap the training environment
    model_path = f"{config['model_save_path']}/{config['run_id']}"
    # Save configuration for eval purposes
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(model_path + "/config.yaml", "w") as f:
        OmegaConf.save(config, f)
    
    train_env = create_env(config['env'], render_mode=None, inference=False)
    print("Original observation space shape:", train_env.observation_space().shape)
    train_env = wrap_env(train_env, config)
    # print("Wrapped observation space shape:", train_env.observation_space().shape)
    # Create and wrap the evaluation environment
    eval_env = create_env(config['env'], render_mode="rgb_array", inference=False)
    eval_env = wrap_env(eval_env, config)
    # Create the model from model_construct func
    model = hydra.utils.call(
        config['algorithm'],
        train_env, 
        config,
        _recursive_=False)
    if not config["from_scratch"]:
        model = load_model(model_path, train_env, config['name'])
        # if config['wandb']['enabled']:

    train_env.reset()

    # Train the model
    match config['training_type']:
        case 'sb3':
            hydra.utils.call(
                config['sb3'],
                model, 
                eval_env,
                config,
                _recursive_=False
                )
        case 'agile_rl':
            # Initialise separate copies of the agent policy algorithm
            pop = [deepcopy(model) for _ in range(config['env']['agents'])]
            # Configure the multi-agent replay buffer
            field_names = ["state", "action", "reward", "next_state", "done"]
            memory = MultiAgentReplayBuffer(
                config["memory_size"],
                field_names=field_names,
                agent_ids=train_env.unwrapped.agents,
                device=config['device'],
            )
            trained_pop, pop_fitnesses = train_multi_agent(
                env=train_env,  # Pettingzoo-style environment
                env_name=config['run_id'],  # Environment name
                algo=config['name'],  # Algorithm
                pop=pop,  # Population of agents
                memory=memory,  # Replay buffer
                INIT_HP=config['init_hp'],  # IINIT_HP dictionary
                # net_config=config['encoder_config'],  # Network configuration
                max_steps=config['train_epochs'],  # Max number of training steps
                evo_steps=config['valid_interval'],  # Evolution frequency
                eval_steps=100,  # Number of steps in evaluation episode
                eval_loop = config['episodes'],
                learning_delay=1000,  # Steps before starting learning
                target=0.,  # Target score for early stopping
                checkpoint=config['valid_interval'],
                # checkpoint_path=config['run_id'],
                overwrite_checkpoints=True,
                checkpoint_path=f"{config['model_save_path']}{config['run_id']}",
                sum_scores=True,
                wb=config['wandb']['enabled'],  # Weights and Biases tracking
                config=config
            )
            last_fitness = pop_fitnesses[-1]
            elite_idx = last_fitness.index(max(last_fitness))
            suffix = f'{config["run_id"]}_0_{config["train_epochs"]}.pt'
            best_model = trained_pop[int(elite_idx)]
            best_model.save_checkpoint(f"{config['model_save_path']}{suffix}")
    eval_env.close()


def test(config:DictConfig):
    test_env = create_env(config['env'], render_mode="rgb_array", inference=True)
    # Load the trained model from model_class constructor
    model=None
    ext = '.zip' if config['training_type'] == 'sb3' else '.pt'
    if config['name'] in ['ppo', 'matd3', 'maddpg']:
        path = "{}{}_0_{}".format(config['model_save_path'],config['run_id'],config['train_epochs'])
        model = load_model(path, config['name'])
        

    # Test the trained model
    test_model(test_env, config, model=model)

    # Close the environment
    test_env.close()
    if config['wandb']['enabled']:
            wandb.log_model(path=f"{config['run_id']}_0_{config['train_epochs']}.{ext}",
                            name=f"{config['run_id']}_0_{config['train_epochs']}")
