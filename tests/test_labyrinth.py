import pytest

import imageio

from gym_multigrid.envs.labyrinth import GoalGroupConfig, LabyrinthEnv


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


def test_check_door_1() -> None:

    animation_path: str = "tests/out/animations/labyrinth_door_1.gif"

    init_pos: tuple[tuple[int, int], ...] = ((2, 5), (2, 6), (2, 7))

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


def test_check_door_2() -> None:

    animation_path: str = "tests/out/animations/labyrinth_door_2.gif"

    init_pos: tuple[tuple[int, int], ...] = ((5, 3), (5, 4), (5, 5))

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


def test_find_first_goal_groups() -> None:

    goal_group_config: list[GoalGroupConfig] = [
        {
            "group_index": 0,
            "pos": ((4, 1), (4, 2), (4, 3)),
            "valid_agent_indices": (0, 1, 2),
            "called_actions": ["open"],
            "action_obj_type": "block",
            "action_obj_group": 0,
            "next_goal": 2,
        },
        {
            "group_index": 1,
            "pos": ((3, 5), (3, 6), (3, 7)),
            "valid_agent_indices": (0, 1, 2),
            "called_actions": ["open"],
            "action_obj_type": "block",
            "action_obj_group": 1,
            "next_goal": 2,
        },
        {
            "group_index": 2,
            "pos": ((6, 3), (6, 4), (6, 5)),
            "valid_agent_indices": (0, 1, 2),
            "called_actions": ["open"],
            "action_obj_type": "block",
            "action_obj_group": 2,
            "next_goal": 3,
        },
        {
            "group_index": 3,
            "pos": ((8, 3), (8, 4), (8, 5)),
            "valid_agent_indices": (0, 1, 2),
            "called_actions": ["open"],
            "action_obj_type": "block",
            "action_obj_group": -1,
            "next_goal": "terminal",
        },
    ]
    env = LabyrinthEnv(goal_group_config=goal_group_config)
    first_goal_groups: list[int] = env._find_first_goal_groups()

    assert first_goal_groups == [0, 1]
