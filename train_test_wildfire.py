import os
import time
import hydra.core
import hydra
import numpy as np
from pz_multigrid.envs import WildfireEnv
from gym_multigrid.utils.misc import save_frames_as_gif
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError
import wandb

# PyMARL algorithms
# SEE LICENSE & NOTICE
import shutil
import torch as th
from types import SimpleNamespace as SN

from algorithm.controllers import REGISTRY as mac_REGISTRY
from algorithm.components.episode_buffer import ReplayBuffer
from algorithm.components.transforms import OneHot
from algorithm.learners import REGISTRY as le_REGISTRY
from algorithm.runners import REGISTRY as r_REGISTRY
from algorithm.utils.general_reward_support import test_alg_config_supports_reward
from algorithm.utils.logging import Logger, get_logger
from algorithm.utils.timehelper import time_left, time_str

console_logger = get_logger()


# Async Env parameter sharing
# from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1



# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def create_env(total_conf:DictConfig, render_mode=None, inference=False):
    # Resize team size based on if run in inference or test
    config = total_conf.env
    num_agents = config['agents'] if not inference else config['agents_inference']
    seed = total_conf.seed if not inference else total_conf.eval_seed
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
        size=config["world_size"],
        initial_fire_size=config['flashpoints'],
        cooperative_reward=config['cooperative_reward'],
        common_reward=total_conf.common_reward,
        reward_scalarisation=total_conf.reward_scalarisation,
        partial_obs=config['partial_obs'],
        render_selfish_region_boundaries=False,
        log_selfish_region_metrics=False,
        seed=total_conf.seed,
    )
    return env


# Could replace DummyVecEnv w/ SubProcEnv if using SAC model
def wrap_env(env, config:DictConfig):
    # Add zero observation for agents upon death, useful with dynamic agents
    # env = black_death_v3(env)
    match config['training_type']:
        case 'sb3':
            from stable_baselines3.common.monitor import Monitor
            # env = Monitor(env,filename=config['log'],info_keywords=('eval/burnt trees', 'eval/unburnt trees', 'mean_reward'))
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
        seed = config.get("seed", None)
        (obs, info) = env.reset(seed)
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
    save_frames_as_gif(frames=frames, path=config['log'], filename=f"{config['run_id']}", ep=config['train_epochs'], fps=5)
    if config['wandb']['enabled']:
        wandb.log({"video": wandb.Video(f"{config['log']}/{config['run_id']}-{config['train_epochs']}.gif", format="gif")})
        


def train(config: DictConfig, logger: Logger):
    # Create and wrap the training environment
    model_path = f"{config['model_save_path']}/{config['run_id']}"

    train_env = create_env(config, render_mode=None, inference=False)
    print("Original observation space shape:", train_env.observation_space().shape)
    train_env = wrap_env(train_env, config)
    # print("Wrapped observation space shape:", train_env.observation_space().shape)
    # Create and wrap the evaluation environment
    eval_env = create_env(config, render_mode="rgb_array", inference=False)
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
            hydra.utils.call(
                config['agilerl'],
                model, 
                train_env,
                config,
                _recursive_=False
                )
    eval_env.close()


def test(config:DictConfig):
    test_env = create_env(config, render_mode="rgb_array", inference=True)
    # Load the trained model from model_class constructor
    model=None
    ext = '.zip' if config['training_type'] == 'sb3' else '.pt'
    if config['name'] in ['ppo', 'matd3', 'maddpg']:
        path = os.path.join(config['model_save_path'],config['run_id']+"_0_"+str(config['train_epochs']))
        model = load_model(path, config['name'])
        

    # Test the trained model
    test_model(test_env, config, model=model)

    # Close the environment
    test_env.close()
    if config['wandb']['enabled']:
            wandb.log_model(path=f"{config['run_id']}_0_{config['train_epochs']}.{ext}",
                            name=f"{config['run_id']}_0_{config['train_epochs']}")


def evaluate_sequential(args, runner):
    for _ in range(args.episodes):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(config: dict, logger):
    # Init runner so we can get env info
    args = SN(**config)
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_shape = env_info['obs_shape']
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    # TODO - replace (n,) tuple
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], 
                # "group": "agents"
                },
        "actions": {
                    "vshape": (env_info['n_agents'],),
                    # "vshape": (env_info['n_agents'], env_info['n_actions']),
                    # "group": "agents", 
                    "dtype": th.long},
        # "actor_hidden_states": {"vshape": (args.hidden_dim,), 
        #                         # "group": "agents"
        #                         },
        "avail_actions": {
            "vshape": (env_info["n_agents"],env_info["n_actions"],),
            # "vshape": (env_info["n_actions"],),
            # "group": "agents",
            "dtype": th.int,
        },
        # "reward": {"vshape": (env_info["n_agents"],)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # For individual rewards in gymmai reward is of shape (1, n_agents)
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (env_info['n_agents'],)}
    groups = {"agents": (1,args.n_agents)}
    # apreprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    preprocess = {}
    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.model_save_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.model_save_path)
            )
            return

        # # Go through all files in args.checkpoint_path
        for file in os.listdir(args.model_save_path):
            full_name = os.path.join(args.model_save_path, file)
            # Check if they are dirs the names of which are numbers
            name = str.split(file)[-1]
            if os.path.isfile(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.train_epochs == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.check))

        model_path = os.path.join(args.model_save_path,args.run_id+"_0_"+str(args.train_epochs))


        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.valid_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(args.model_save_path,args.run_id+"_0_"+str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.wandb["enabled"]:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", logger.config_hash, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    logger.console_logger.info("Finished Training, time: {}".format(time_str(time.time() - start_time)))
    runner.close_env()

def args_sanity_check(config):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        print(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["episodes"] < config["batch_size_run"]:
        config["episodes"] = config["batch_size_run"]
    else:
        config["episodes"] = (
            config["episodes"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config


##############################################
# MAKE CHANGES IN /$ROOT/config files
##############################################
@hydra.main(config_path="configs/", config_name="default", version_base="1.3")
def main(cfg: DictConfig):

    cfg = args_sanity_check(cfg)



    if cfg['name'] == 'ppo':
        cfg['training_type'] = 'sb3'
    elif cfg['name'] in ['matd3', 'maddpg']:
        cfg['training_type'] = 'agile_rl'
    elif cfg['name'] != 'heuristic':
        cfg['training_type'] = 'marl'

    # console_logger.info("whole config:", OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # ))

    # torch.set_num_threads(1)

    if cfg.seed is not None:
        th.manual_seed(cfg.seed)
    
    logger = Logger(console_logger)
    logger.setup_tb(cfg['log']+cfg['run_id'])
    # exit()
    if cfg['wandb']['enabled']:
        # TODO - return to True once env will be pre-initialized 
        config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True #noqa
        )
        # wandb.tensorboard.patch(root_logdir=cfg['log'])
        
        # config_hash = sha256(
        #     json.dumps(
        #         {k: v for k, v in config.items() if k not in non_hash_keys},
        #         sort_keys=True,
        #     ).encode("utf8")
        # ).hexdigest()[-10:]

        # group_name = "_".join([cfg.name, config_hash])

        # wb = wandb.init(
        #     entity=cfg['wandb']['entity'],
        #     project='Wildfire',
        #     config=config,
        #     # group=group_name,
        #     mode=cfg['wandb']['mode'],
        #     name=cfg["run_id"],
        #     sync_tensorboard=True,
        #     job_type="test" if cfg['evaluate'] else "train",
        #     resume="allow",
        #     reinit=True,
        #     monitor_gym=True,
        # )
        logger.setup_wandb(config,cfg['wandb']['entity'],cfg['wandb']['project'],cfg['wandb']['mode'])
    print(OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True #noqa
        ))
    
    model_path = f"{cfg['model_save_path']}/{cfg['run_id']}"


    # Save configuration for eval purposes
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(model_path + "/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
        if cfg['wandb']['enabled']:
            logger.wandb.log_artifact(model_path+'/config.yaml')
    
    if cfg["training_type"] == 'marl':
        args = OmegaConf.to_container(cfg,
                                      resolve=True,
                                      throw_on_missing=True,)
        run_sequential(args, logger)
        exit()
    if not cfg['evaluate']:
        train(cfg, logger)
    test(cfg)
    logger.save(model_path)
    wandb.finish(0)    



if __name__ == "__main__":
    main()