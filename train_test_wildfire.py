import os
import time
from copy import deepcopy
import hydra.core
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LRScheduler
import gymnasium as gym
import pettingzoo as pz
from pz_multigrid.envs import WildfireEnv
from gym_multigrid.utils.misc import save_frames_as_gif
# Stable baselines 3
from sbx import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from tests.tensorboard.log_callback import TensorboardCallback
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
from wandb.integration.sb3 import WandbCallback
# AgileRL HPO & algorithms
from agilerl.algorithms import maddpg, matd3, dqn_rainbow
from agilerl.algorithms.ppo import PPO as agile_ppo
from train_multi_agent import train_multi_agent
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population

# Async Env parameter sharing
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1



# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def create_env(config:DictConfig, render_mode=None):
    """Function to test the environment's functionality. Runs episodes with random agents in the Wildfire environment and save episode renders as GIFs."""
    start_pos = np.random.uniform(low=1, high=config['world_size']-1, size=(config['n_env'], 2))
    env = WildfireEnv(
        render_mode=render_mode,
        agent_representation_mode=config['obs_type'],
        reward_type=config['reward_type'],
        alpha=config["alpha_transition"],
        beta=config["beta_transition"],
        delta_beta=config['agent_beta_impact'],
        # max_episode_steps=1,
        num_agents=config['n_env'],
        # tuple(np.random.uniform(low=1, high=16, size=(3, 2)))
        # agent_start_positions=((4, 6), (10, 12), (8, 6), (2, 1)),
        agent_start_positions=np.astype(start_pos, np.int32),
        size=config["world_size"],
        initial_fire_size=3,
        cooperative_reward=True,
        render_selfish_region_boundaries=False,
        log_selfish_region_metrics=False,
    )
    return env


# Could replace DummyVecEnv w/ SubProcEnv if using SAC model
def wrap_env(env, config:DictConfig):
    # Add zero observation for agents upon death, useful with dynamic agents
    # env = black_death_v3(env)
    match config['training_type']:
        case 'sb3':
            env = pettingzoo_env_to_vec_env_v1(env)
            env = concat_vec_envs_v1(env, config['n_env'], num_cpus=1, base_class="stable_baselines3")
        case 'agile_rl':
            # env = pettingzoo_env_to_vec_env_v1(env)
            # env = concat_vec_envs_v1(env, config['n_env'], num_cpus=1, base_class="stable_baselines3")
            # env = AsyncPettingZooVecEnv([lambda: env for _ in range(config['n_env'])])
            env = AsyncPettingZooVecEnv([lambda:env])
    return env


def model_PPO(env, config:DictConfig):
    nn_t = (128, 128, 128)
    policy_kwargs = dict(net_arch=dict(pi=nn_t, vf=nn_t))
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.1,
        tensorboard_log=config["tensorboard"]
    )

def model_DQN(env, config:DictConfig):
    observation_spaces = [env.single_observation_space(agent).shape for agent in env.agents]
    action_spaces = [env.single_action_space(agent).n for agent in env.agents]
    return dqn_rainbow.RainbowDQN(
        observation_spaces,
        action_spaces,
        False,
        config['encoder_config'],
        **config['INIT_HP'],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

def model_MADDPG(env, config:DictConfig):
    observation_spaces = [env.observation_space(agent).shape[1:] for agent in env.agents]
    action_spaces = [env.single_action_space(agent).n for agent in env.agents]
    # max_action = [space.n for space in action_spaces]
    # min_action = [space.start for space in action_spaces]
    return maddpg.MADDPG(
        observation_spaces,
        action_spaces,
        False,
        env.num_agents,
        env.agents,
        net_config=config['encoder_config'],
        discrete_actions=True,
        max_action=None,
        min_action=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **config['INIT_HP'],
    )

def model_MATD3(env: AsyncPettingZooVecEnv, config:DictConfig):
    observation_spaces = [env.observation_space(agent).shape[1:] for agent in env.agents]
    action_spaces = [env.single_action_space(agent).n for agent in env.agents]
    # max_action = [space.n for space in action_spaces]
    # min_action = [space.start for space in action_spaces]

    # return AgileRL policy
    return matd3.MATD3(
        observation_spaces,
        action_spaces,
        False,
        env.num_agents,
        env.agents,
        net_config=config['encoder_config'],
        discrete_actions=True,
        max_action=None,
        min_action=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **config['INIT_HP'],
    )

def setup_callbacks(eval_env, config:DictConfig):
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f".{config['model_save_path']}/checkpoints/",
        name_prefix=f"{config['run_id']}",
    )
    tensorboard_callback = TensorboardCallback()
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=tensorboard_callback,
        best_model_save_path="./logs/",
        deterministic=True,
        log_path="./logs/",
        eval_freq=5000,
        verbose=2,
    )
    log_callback = WandbCallback(
        model_save_freq=10000, model_save_path=f"{config['model_save_path']}"
    )
    # log_callback = WandbCallback(model_save_freq=10000, model_save_path=wandb.run.dir)
    return [checkpoint_callback, eval_callback, tensorboard_callback, log_callback]


def train_sb3(model, callbacks, config:DictConfig):
    model.learn(
        total_timesteps=config['train_epochs'],
        tb_log_name=f"{config['run_id']}",
        callback=callbacks,
        progress_bar=True,
    )
    model.save(f"{config['model_save_path']}")


def load_model(model_path: str, env=None, definition=PPO):
    # does it need env?
    return definition.load(model_path, env=env)


def test_model(env, config:DictConfig, model=None):
    frames = []
    for ep in range(config["n_episodes"]):
        steps = 0
        terminated, truncated = {0: False}, {0: False}
        frac_burned = -0.1
        (obs, info) = env.reset()
        ep_reward = 0.
        while not terminated[0]:
            agents = list(range(config['n_env']))
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
            frac_burned = infos[0]['burnt trees'] / ((config["world_size"] - 2) ** 2)
            frac_unburned = infos[0]['unburnt trees'] / ((config["world_size"] - 2) ** 2)
            print(
                f"Reward: {reward}, \n burnt trees %: {frac_burned}, \n terminated: {terminated}, truncated: {truncated}, Frames length: {len(frames)}"
                # , Inference time in ms: {inference_dt}"
            )
            
            steps += 1
            # Log reward at each step to wandb
            wandb.log(
                {
                    "reward": mean_reward_per_agent,
                    "burnt trees": frac_burned,
                    "unburnt trees": frac_unburned,
                    "Inference FPS": inference_dt,
                }
            )
        # Total episode reward
        wandb.log(
            {
                "total_episode_reward": ep_reward,
            }
        )
    save_frames_as_gif(frames=frames, path=config['tensorboard'], filename=f"{config['run_id']}-{config['job_type']}", ep=config['train_epochs'], fps=5)
    wandb.log({"video": wandb.Video(f"{config['tensorboard']}/{config['run_id']}-{config['job_type']}-{config['train_epochs']}.gif", format="gif")})
        


def train(model_class, model_construct: callable, config: DictConfig):
    # Create and wrap the training environment
    model_path = f"{config['model_save_path']}/{config['run_id']}"
    train_env = create_env(config, render_mode=None)
    print("Original observation space shape:", train_env.observation_space().shape)
    train_env = wrap_env(train_env, config)
    # print("Wrapped observation space shape:", train_env.observation_space().shape)
    # Create and wrap the evaluation environment
    eval_env = create_env(config, render_mode="rgb_array")
    eval_env = wrap_env(eval_env, config)

    # Create the model from model_construct func
    model = model_construct(env=train_env, config=config)
    if not config["from_scratch"]:
        model = load_model(model_path, train_env, model_class)
    
    train_env.reset()

    # Train the model
    match config['training_type']:
        case 'sb3':
            callbacks = setup_callbacks(eval_env, config=config)
            train_sb3(model, callbacks, config=config)
        case 'agile_rl':
            # Initialise separate copies of the agent policy algorithm
            pop = [deepcopy(model) for _ in range(config['n_env'])]
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
                algo=config['algorithm'],  # Algorithm
                pop=pop,  # Population of agents
                memory=memory,  # Replay buffer
                INIT_HP=config['INIT_HP'],  # IINIT_HP dictionary
                net_config=config['encoder_config'],  # Network configuration
                max_steps=config['train_epochs'],  # Max number of training steps
                evo_steps=10000,  # Evolution frequency
                eval_steps=500,  # Number of steps in evaluation episode
                eval_loop=config['train_epochs'] // 5000,  # Number of evaluation episodes
                learning_delay=1000,  # Steps before starting learning
                target=200.,  # Target score for early stopping
                checkpoint=10000,
                checkpoint_path=config['model_save_path'],
                wb=True,  # Weights and Biases tracking
            )
            plt.figure()
            plt.plot(pop_fitnesses)
            plt.title("Score History - Mutations")
            plt.xlabel("Steps")
            plt.ylim(bottom=-400)
            plt.show()
            elite_idx = pop_fitnesses.index(max(pop_fitnesses[-1]))
            suffix = f'_{elite_idx}_{config["train_epochs"]}.pt'
            best_model = trained_pop[int(elite_idx)]
            best_model.save_checkpoint(f"{config['model_save_path']}{suffix}")
    eval_env.close()


def test(model_class, config:DictConfig):
    test_env = create_env(config, render_mode="rgb_array")
    # Load the trained model from model_class constructor
    model=None
    if config['algorithm'] != "Random":
        model_suffix = f'_0_{config["train_epochs"]}.pt'
        path = f"{config['model_save_path']}{model_suffix}"
        model = model_class.load(path)
        wandb.log_model(path=path,name=f"{config['run_id']}{model_suffix}")

    # Test the trained model
    test_model(test_env, config, model=model)

    # Close the environment
    test_env.close()


##############################################
# MAKE CHANGES IN /$ROOT/config files
##############################################
@hydra.main(config_path="configs/", config_name="default", version_base="1.3")
def run(cfg: DictConfig):

    agents = list(range(cfg['n_env']))
    test_env = create_env(cfg, render_mode="rgb_array")
    obs, _ = test_env.reset()
    print(f"---------------First observation---------------- \n{obs[0].shape} \n")
    # agents = list(range(config['n_env']))
    actions = {agent: test_env.action_space().sample() for agent in agents}        
    obs, reward, terminated, truncated, infos = test_env.step(actions)
    while not terminated[0]:
        print(f"\n \n \n After reset \n {obs[0].shape}")
        obs, reward, terminated, truncated, infos = test_env.step(actions)
# Close the environment
    test_env.close()
    
    wandb.config = wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )


    match cfg['algorithm']:
        case 'PPO':
            algo_cls = PPO
            algo_constructor = model_PPO
        case 'MATD3':
            algo_cls = matd3.MATD3
            algo_constructor = model_MATD3
        case 'MADDPG':
            algo_cls = maddpg.MADDPG
            algo_constructor = model_MADDPG
        case 'Rainbow':
            raise PendingDeprecationWarning("Not tested implementation")
            algo_cls = dqn_rainbow.RainbowDQN
            algo_constructor = model_DQN
        case _:
            algo_cls = None
            algo_constructor = None
        

    with wandb.init(
        project=cfg['wandb']['project'],
        name=cfg["run_id"],
        sync_tensorboard=True,
        job_type=cfg['job_type'],
        resume="allow",
        # python -c "import wandb; print(wandb.util.generate_id())"
        # id="8up8c0w8",
    ):
        if cfg['job_type'] == "train":
            train(algo_cls,algo_constructor, cfg)
        test(algo_cls, cfg)

if __name__ == "__main__":
    run()