import os
import time
import sys
import numpy as np
import gymnasium as gym
import pettingzoo as pz
from pettingzoo.utils.conversions import parallel_to_aec
import gym_multigrid
from pz_multigrid.envs import WildfireEnv
from sbx import DQN, TQC, CrossQ, TD3
from sbx import PPO
# from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO, ARS
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from tests.tensorboard.log_callback import TensorboardCallback
from gymnasium.wrappers import RecordVideo
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
import wandb
from wandb.integration.sb3 import WandbCallback


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_env(render_mode=None, **kwaargs):
    """Function to test the environment's functionality. Runs episodes with random agents in the Wildfire environment and save episode renders as GIFs."""
    env = WildfireEnv(
        render_mode=render_mode,
        alpha=0.3,
        beta=0.8,
        # max_episode_steps=1,
        num_agents=4,
        # tuple(np.random.uniform(low=1, high=16, size=(3, 2)))
        agent_start_positions=((4, 6), (10, 12), (8, 6), (2, 1)),
        size=config["world_size"],
        initial_fire_size=3,
        cooperative_reward=True,
        render_selfish_region_boundaries=True,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[2, 2, 2, 2],
        selfish_region_xmax=[8, 8, 8, 8],
        selfish_region_ymin=[2, 2, 2, 2],
        selfish_region_ymax=[8, 8, 8, 8],
    )
    return env


# Could replace DummyVecEnv w/ SubProcEnv if using SAC model
def wrap_env(env, **kwaargs):
    # if env.render_mode == "rgb_array":
    #     env = RecordVideo(
    #         env,
    #         video_folder=config["run_id"],
    #         video_length=30000,
    #         episode_trigger=lambda x: x % 100 == 0,
    #     )
    # Add zero observation for agents upon death, useful with dynamic agents
    # env = black_death_v3(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, config['n_env'], num_cpus=1, base_class="stable_baselines3")
    return env


def model_PPO(env, **PPO_kwaargs):
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
        clip_range=0.2,
        # ent_coef="auto_0.05",
        tensorboard_log=config["tensorboard"]
    )


def model_ARS(env, **ARS_kwaargs):
    nn_t = [128, 128]
    policy_kwargs = dict(net_arch=nn_t)
    return ARS(
        "MlpPolicy",
        env,
        n_delta=20,
        n_top=5,
        learning_rate=3e-4,
        delta_std=0.05,
        zero_policy=True,
        alive_bonus_offset=0,
        n_eval_episodes=1000,
        policy_kwargs=policy_kwargs,
        stats_window_size=100,
        tensorboard_log=f"{config['tensorboard']}",
    )


def model_CrossQ(env, **CrossQ_kwaargs):
    return CrossQ(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        batch_size=128,
        gamma=0.99,
        use_sde=True,
        ent_coef="auto_0.05",
        tensorboard_log=f"{config['tensorboard']}",
    )


def model_TQC(env, **TQC_kwaargs):
    return TQC(
        "MlpPolicy",
        env,
        verbose=0,
        # learning_rate=1e-4,
        batch_size=128,
        tau=0.05,
        gamma=0.99,
        top_quantiles_to_drop_per_net=2,
        use_sde=True,
        ent_coef="auto_0.05",
        tensorboard_log=f"{config['tensorboard']}",
    )

def model_DQN(env, **DQN_kwaargs):
    return DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        batch_size=256,
        tau=0.05,
        gamma=0.99,
        # exploration_initial_eps=0.05,
        tensorboard_log=config["tensorboard"],
    )


def model_RPPO(env, **RPPO_kwaargs):
    nn_t = (128, 128)
    policy_kwargs = dict(net_arch=dict(pi=nn_t, vf=nn_t))
    return RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=0,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        tensorboard_log=config["tensorboard"],
    )

def model_TD3(env, **kwaargs):
    return TD3(
        "MlpPolicy",
        env=env,
        tensorboard_log=config["tensorboard"])


def setup_callbacks(eval_env, **kwaargs):
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix=f"{config['algorithm']}_model",
    )
    cutoff_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=30
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


def train_model(model, callbacks, **kwaargs):
    timesteps = 3_000_000
    model.learn(
        total_timesteps=timesteps,
        tb_log_name=f"{config['run_id']}",
        callback=callbacks,
        progress_bar=True,
    )
    model.save(f"{config['model_save_path']}")


def load_model(env, model_path: str, definition=PPO):
    # does it need env?
    return definition.load(model_path)


def test_model(env, model, **kwaargs):
    (obs, _) = env.reset()
    steps = 0
    terminated, truncated = False, False
    video_path = f".{config['model_save_path']}/rl-video-episode-0.mp4"
    terminated = {0: False}
    frac_burned = -0.1
    for ep in range(config["n_episodes"]):
        ep_reward = 0.
        # for _ in range(50):
        while not terminated[0]:
            agents = list(range(wandb.config['n_env']))
            
            inference_dt = time.time()
            if config['algorithm'] != "Random":
                
                # actions, states = model.predict(obs, deterministic=True)
                actions = {agent: model.predict(obs[agent], deterministic=True)[0] for agent in agents}
            else:
                actions = {agent: env.action_space().sample() for agent in agents}
            inference_dt = 1.0 / (time.time() - inference_dt + 1e-6)
            
            obs, reward, terminated, truncated, infos = env.step(actions)
            mean_reward_per_agent = sum(reward.values()) / len(reward.values())
            ep_reward += float(mean_reward_per_agent)
            frac_burned = infos[0]['burnt trees'] / (wandb.config["world_size"] ** 2)
            print(
                f"Observation: {obs}, \n Reward: {mean_reward_per_agent}, \n burnt trees %: {frac_burned}, \n terminated: {terminated}, truncated: {truncated}"
                # , Inference time in ms: {inference_dt}"
            )

            steps += 1
            # Log reward at each step to wandb
            wandb.log(
                {
                    "reward": mean_reward_per_agent,
                    "burnt trees": frac_burned,
                    "Inference FPS": inference_dt,
                }
            )

            # if not all(terminated.values()) and not all(truncated.values()):
            #     print("--------------EPISODE-END------------\n")
            # if old_frac_burned == frac_burned:
            #     print("--------------EPISODE-END------------\n")
            #     break
        # Total episode reward
        # wandb.log({"video": wandb.Video(video_path, format="mp4")})
        wandb.log(
            {
                "total_episode_reward": ep_reward,
            }
        )


def train(model_class, model_construct: callable, config: dict):
    # Create and wrap the training environment
    model_path = f"{config['model_save_path']}{config['run_id']}"
    train_env = create_env(render_mode=None)
    print("Original observation space shape:", train_env.observation_space().shape)
    train_env = wrap_env(train_env)
    # print("Wrapped observation space shape:", train_env.observation_space().shape)
    # Create and wrap the evaluation environment
    eval_env = create_env(render_mode="rgb_array")
    eval_env = wrap_env(eval_env)

    # Create the model from model_construct func
    model = model_construct(env=train_env, config=config)
    if not config["from_scratch"]:
        model = load_model(train_env, model_path, model_class)
    # Setup callbacks
    callbacks = setup_callbacks(eval_env, config=config)
    # Train the model
    train_model(model, callbacks, config=config)
    eval_env.close()


def test(model_class, **kwaargs):
    test_env = create_env(render_mode="rgb_array", config=config)
    # Load the trained model from model_class constructor
    model = model_class.load(f'{config["model_save_path"]}/model.zip')
    wandb.log_model(path=f'./{config["model_save_path"]}',name=f'model.zip')

    # Test the trained model
    test_model(test_env, model, config=config)

    # Close the environment
    test_env.close()


##############################################
# MAKE CHANGES HERE
##############################################
# Change the algorithm name here
if __name__ == "__main__":
    config = {
        "algorithm": "PPO",
        "training_type":"central",
        "run_id": "ppo_sbx_4",
        "from_scratch": True,
        "job_type": "test",
        "n_episodes": 100,
        "n_env": 4,
        "world_size": 17,
        "tensorboard": "./out/logs/wildfire/",
        "model_save_path":"./out/models/ppo_sbx",
    }
    with wandb.init(
        project="Wildfire",
        name=config["run_id"],
        config=config,
        sync_tensorboard=True,
        job_type=config["job_type"],
        resume="allow",
        # python -c "import wandb; print(wandb.util.generate_id())"
        # id="8up8c0w8",
    ):
        # train(PPO,model_PPO, config)
        test(PPO, config = config)
