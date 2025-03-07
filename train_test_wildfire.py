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

# AgileRL HPO & algorithms
from train_multi_agent import train_multi_agent
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
# from agilerl.hpo.mutation import Mutations
# from agilerl.hpo.tournament import TournamentSelection
# from agilerl.utils.utils import create_population

# Async Env parameter sharing
# from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1



# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def create_env(config:DictConfig, render_mode=None, inference=False):
    # Resize team size based on if run in inference or test
    num_agents = config['agents'] if not inference else config['agents_inference']
    """Function to test the environment's functionality. Runs episodes with random agents in the Wildfire environment and save episode renders as GIFs."""
    start_pos = np.random.uniform(low=1, high=config['world_size']-1, size=(num_agents, 2))
    start_pos = tuple((int(pos[0]), int(pos[1])) for pos in start_pos)
    env = WildfireEnv(
        render_mode=render_mode,
        agent_representation_mode=config['obs_type'],
        reward_type=config['reward_type'],
        alpha=config["alpha_transition"],
        beta=config["beta_transition"],
        delta_beta=config['agent_beta_impact'],
        # max_episode_steps=1,
        num_agents=num_agents,
        max_steps=25,
        # unsupported in numpy 1.26
        # agent_start_positions=np.astype(start_pos, np.int32),
        agent_start_positions=start_pos,
        size=config["world_size"],
        initial_fire_size=config['flashpoints'],
        cooperative_reward=config['cooperative_reward'],
        partial_obs=config['partial_obs'],
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
            env = concat_vec_envs_v1(env, config['env']['agents'], num_cpus=1, base_class="stable_baselines3")
        case 'agile_rl':
            # env = AsyncPettingZooVecEnv([lambda:env])
            pass
    return env


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
            plt.figure()
            plt.plot(pop_fitnesses)
            plt.title("Score History - Mutations")
            plt.xlabel("Steps")
            plt.ylim(bottom=-400)
            plt.show()
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

OmegaConf.register_new_resolver(
    "random",
    lambda x: os.urandom(x).hex(),
)

##############################################
# MAKE CHANGES IN /$ROOT/config files
##############################################
@hydra.main(config_path="configs/", config_name="default", version_base="1.3")
def main(cfg: DictConfig):

    match cfg['name']:
        case 'ppo':
            cfg['training_type'] = 'sb3'
        case 'matd3':
            cfg['training_type'] = 'agile_rl'
        case 'maddpg':
            cfg['training_type'] = 'agile_rl'

    # torch.set_num_threads(1)

    # if cfg.seed is not None:
    #     torch.manual_seed(cfg.seed)
    #     np.random.seed(cfg.seed)
    
    # TODO - return to True once env will be pre-initialized 
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True #noqa
    )

    print("whole config:", OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    ))
    # exit()
    if cfg['wandb']['enabled']:
        wandb.init(
        project=cfg['wandb']['project'],
        name=cfg["run_id"],
        sync_tensorboard=True,
        job_type=cfg['job_type'],
        resume="allow",
        reinit=True,
        monitor_gym=True,
        # python -c "import wandb; print(wandb.util.generate_id())"
        # id="8up8c0w8",
        )


    if cfg['job_type'] == "train":
        train(cfg)
    test(cfg)

    wandb.finish(0)    



if __name__ == "__main__":
    main()