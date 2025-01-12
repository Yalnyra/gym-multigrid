import os
import time
import sys
import numpy as np
import gymnasium as gym
import gym_multigrid
from sbx import DQN, TQC, CrossQ
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO, ARS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from tests.tensorboard.log_callback import TensorboardCallback
from gymnasium.wrappers import RescaleAction, RecordVideo
import wandb
from wandb.integration.sb3 import WandbCallback


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_env(render_mode=None, **kwaargs):
    """Function to test the environment's functionality. Runs episodes with random agents in the Wildfire environment and save episode renders as GIFs."""
    env = gym.make(
        "wildfire-v0",
        render_mode=render_mode,
        alpha=0.3,
        beta=0.8,
        max_episode_steps=1,
        num_agents=3,
        # tuple(np.random.uniform(low=1, high=16, size=(3, 2)))
        agent_start_positions=((4, 6), (10, 12), (8, 6)),
        size=wandb.config["world_size"],
        initial_fire_size=3,
        cooperative_reward=True,
        render_selfish_region_boundaries=True,
        log_selfish_region_metrics=True,
        selfish_region_xmin=[2, 2, 2],
        selfish_region_xmax=[8, 8, 8],
        selfish_region_ymin=[2, 2, 2],
        selfish_region_ymax=[8, 8, 8],
    )
    return env


# Could replace DummyVecEnv w/ SubProcEnv if using SAC model
def wrap_env(env, **kwaargs):

    # Scale action wrapper to gaussian bounds
    # env =  RescaleAction(env, min_action=low, max_action=high)
    if env.render_mode == "rgb_array":
        env = RecordVideo(
            env,
            video_folder=config["run_id"],
            video_length=30000,
            episode_trigger=lambda x: x % 100 == 0,
        )
    # env = DummyVecEnv([lambda: env])
    # return VecNormalize(
    #     env, training=config["job_type"], norm_obs=True, norm_reward=True, epsilon=1e-7
    # )
    return env


def model_PPO(env, **PPO_kwaargs):
    nn_t = (128, 128, 128)
    policy_kwargs = dict(net_arch=dict(pi=nn_t, vf=nn_t))
    return PPO(
        "MultiInputPolicy",
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
        # tensorboard_log="./out/logs/wildfire/",
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
        tensorboard_log="./wildfire_tensorboard/",
    )


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
        callback_after_eval=[cutoff_callback, tensorboard_callback],
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
    timesteps = 300_000
    model.learn(
        total_timesteps=timesteps,
        tb_log_name=f"{config['run_id']}",
        callback=callbacks,
        progress_bar=True,
    )
    model.save(f"{config['model_save_path']}")


def load_model(env, model_path: str, definition=PPO):
    return definition.load(model_path, env=env)


def test_model(env, model, **kwaargs):
    obs, _ = env.reset()
    steps = 0
    terminated, truncated = False, False
    video_path = f".{config['model_save_path']}/rl-video-episode-0.mp4"
    for ep in range(config["n_episodes"]):
        ep_reward = 0.
        while not terminated or not truncated:
            inference_dt = time.time()
            # actions = env.action_space.sample()
            actions, states = model.predict(obs, deterministic=True)
            inference_dt = 1.0 / (time.time() - inference_dt + 1e-6)
            obs, reward, terminated, truncated, _ = env.step(actions)
            ep_reward += float(reward)
            frac_burned = env.unwrapped.burnt_trees / (wandb.config["world_size"] ** 2)
            print(
                f"Observation: {obs}, Max action: {np.max(actions)} Reward: {reward}, burnt trees %: {frac_burned}"
                # , Inference time in ms: {inference_dt}"
            )

            steps += 1
            # Log reward at each step to wandb
            wandb.log(
                {
                    "reward": reward,
                    "burnt trees": frac_burned,
                    "Inference FPS": inference_dt,
                }
            )
        # Total episode reward
        wandb.log({"video": wandb.Video(video_path, format="mp4")})
        wandb.log(
            {
                "total_episode_reward": ep_reward,
            }
        )
            # frames.append(env.render())



def train(model_class, model_construct: callable, config: dict):
    # Create and wrap the training environment
    model_path = f"{config['model_save_path']}{config['run_id']}"
    train_env = create_env(render_mode=None)
    print("Original observation space shape:", train_env.observation_space.shape)
    train_env = wrap_env(train_env)
    print("Wrapped observation space shape:", train_env.observation_space.shape)

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
    test_env = wrap_env(test_env)

    # Load the trained model from model_class constructor

    model = load_model(test_env, f'{config["model_save_path"]}/model.zip', model_class)
    wandb.log_model(path=f'./{config["run_id"]}',name=f'wandb_model.zip')

    # Test the trained model
    test_model(test_env, model, config=config)
    # wandb.log_artifact(artifact, name=config['run_id'], tags=[config['algorithm'], config['job_type']])
    # wandb.log(
    #         {
    #             "video": wandb.Video(
    #                 data_or_path=f"./{config['run_id']}/rl-video-episode-{step}.mp4"
    #             )
    #         }

    # Close the environment
    test_env.close()


##############################################
# MAKE CHANGES HERE
##############################################
# Change the algorithm name here
if __name__ == "__main__":
    config = {
        "algorithm": "PPO",
        "run_id": "ppo_test_v0",
        "from_scratch": True,
        "job_type": "test",
        "n_episodes": 100,
        "n_env": 1,
        "world_size": 17,
        "tensorboard": "./out/logs/wildfire/",
        "model_save_path":"./out/models/wildfire/",
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
        # test_random(PPO, config=config)
