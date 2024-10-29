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


def test_check_door_0() -> None:

    animation_path: str = "tests/out/animations/labyrinth_door_0.gif"

    init_pos: tuple[tuple[int, int], ...] = ((3, 1), (3, 2), (3, 3))

    env = LabyrinthEnv(init_pos=init_pos)
    action = [env.actions.right, env.actions.right, env.actions.right]
    obs, _ = env.reset()
    frames = [env.render()]

    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        print(f"reward: {reward}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")
        if terminated or truncated:
            break

    imageio.mimsave(animation_path, frames, loop=10)
