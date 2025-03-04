from agilerl.algorithms import maddpg, matd3
from omegaconf import DictConfig
from omegaconf.errors import ConfigKeyError

def model_MADDPG(env, config:DictConfig):
    nn_t = None
    try:
        nn_t = {"encoder_config": config['encoder_config']}
        print(f"Loaded encoder: {nn_t}")
    except(ConfigKeyError):
        print("Couldn't load encoder")
        pass
    observation_spaces = [env.observation_space(agent) for agent in env.agents]
    action_spaces = [env.action_space(agent) for agent in env.agents]
    # max_action = [space.n for space in action_spaces]
    # min_action = [space.start for space in action_spaces]
    return maddpg.MADDPG(
        observation_spaces,
        action_spaces,
        env.agents,
        net_config=nn_t,
        device=config['device'],
        **config['init_hp'],
    )

# TODO: add passing agents & critics networks on inference
def model_MATD3(env, config:DictConfig):
    nn_t = None
    try:
        nn_t = {"encoder_config": config['encoder_config']}
        print(f"Loaded encoder: {nn_t}")
    except(ConfigKeyError):
        print("Couldn't load encoder")
        pass
    observation_spaces = [env.observation_space(agent) for agent in env.agents]
    action_spaces = [env.action_space(agent) for agent in env.agents]
    # max_action = [space.n for space in action_spaces]
    # min_action = [space.start for space in action_spaces]

    # return AgileRL policy
    return matd3.MATD3(
        observation_spaces,
        action_spaces,
        env.agents,
        net_config=nn_t,
        
        device=config['device'],
        **config['init_hp'],
    )