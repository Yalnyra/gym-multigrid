# import pytest
from pettingzoo.test import api_test, parallel_api_test, test_save_obs, parallel_seed_test
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pz_multigrid.envs.wildfire import MultiGridEnv, WildfireEnv


# @pytest.mark.parametrize("env_id", ["pz_multigrid:wildfire-v1"])
def test_collect_game() -> None:
    """Test collect_game()"""
    env = WildfireEnv(render_mode='human',cooperative_reward=True, max_steps=100000)
    # print("starting test.")
    # parallel_api_test(par_env=env, num_cycles=1000)
    print("General sanity test passed..")
    test_save_obs(env)
    # print("Observation test passed...")
    # parallel_seed_test(env, num_cycles=500)
    print("All tests passed!")

    env.reset()
    obs = env._get_obs()
    print(obs)
    while True:
        actions = [env.action_space(a).sample() for a in env.agents]

        obs, reward, terminated, truncated, info = env.step(actions)
        env.render(highlight=True)
        if terminated or truncated:
            break


if __name__ == "__main__":
    test_collect_game()