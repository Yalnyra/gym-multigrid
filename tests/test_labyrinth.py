import pytest

import imageio

from gym_multigrid.envs.labyrinth import LabyrinthEnv


def test_labyrinth() -> None:

    animation_path: str = "tests/out/animations/labyrinth.gif"

    env = LabyrinthEnv()

    obs, _ = env.reset()
    frames = [env.render()]

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        if terminated or truncated:
            break

    imageio.mimsave(animation_path, frames, loop=10)
