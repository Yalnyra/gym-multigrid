import pz_multigrid.envs.wildfire
from train_test_wildfire import create_env

def test_pz_wildfire():
    cfg = {
        'n_env': 2,
        # TODO
    }

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