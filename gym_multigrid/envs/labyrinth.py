from typing import Any, Literal, Type, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray
from gymnasium.spaces import Box, MultiDiscrete

from gym_multigrid.core.agent import NavigationActions, ActionsT, Agent, NAV_DIR_TO_VEC
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import AgentGoal, Block, WorldObjT, Zone
from gym_multigrid.core.world import WorldT, LabyrinthWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position


class GoalGroupConfig(TypedDict):
    group_index: int
    pos: tuple[tuple[int, int], ...]
    valid_agent_indices: tuple[int, ...]
    called_actions: list[str]
    action_obj_type: str
    action_obj_group: int
    next_goal: int | Literal["terminal"]


class ObjectGroupConfig(TypedDict):
    obj_type: str
    group_index: int
    pos: tuple[tuple[int, int], ...]
    obj_args: dict[str, Any]
    group_args: dict[str, Any]


class RewardConfig(TypedDict):
    movement_reward: float
    agent_on_goal_reward: float
    agent_move_away_from_goal_reward: float
    all_agents_on_goal_reward: float


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
        "action_obj_group": 3,
        "next_goal": "terminal",
    },
]

obj_group_config: list[ObjectGroupConfig] = [
    {
        "obj_type": "block",
        "group_index": 0,
        "pos": ((5, 1), (5, 2), (5, 3)),
        "obj_args": {},
        "group_args": {},
    },
    {
        "obj_type": "block",
        "group_index": 1,
        "pos": ((4, 5), (4, 6), (4, 7)),
        "obj_args": {},
        "group_args": {},
    },
    {
        "obj_type": "block",
        "group_index": 2,
        "pos": ((7, 2), (7, 3), (7, 4), (7, 5), (7, 6)),
        "obj_args": {},
        "group_args": {},
    },
    {
        "obj_type": "zone",
        "group_index": 0,
        "pos": ((2, 1), (2, 2), (2, 3)),
        "obj_args": {"color": "blue"},
        "group_args": {"visual_detect_prob": 0.005},
    },
    {
        "obj_type": "zone",
        "group_index": 1,
        "pos": ((2, 5), (2, 6), (2, 7)),
        "obj_args": {"color": "red"},
        "group_args": {
            "visual_detect_prob": 0.005,
            "radio_detect_prob": 0.06,
        },
    },
]

reward_config: RewardConfig = {
    "movement_reward": -0.02,
    "agent_on_goal_reward": 0.2,
    "agent_move_away_from_goal_reward": -0.3,
    "all_agents_on_goal_reward": 1.0,
}


class ObjectGroup:
    def __init__(
        self, obj_type: str, group_index: int, pos: tuple[tuple[int, int], ...]
    ) -> None:
        self.obj_type: str = obj_type
        self.group_index: int = group_index
        self.pos: tuple[tuple[int, int], ...] = pos

    def call_action(self, action: str, args: dict[str, Any] = {}) -> Any:
        return getattr(self, action)(**args)


ObjectGroupT = TypeVar("ObjectGroupT", bound="ObjectGroup")


class GoalGroup(ObjectGroup):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int], ...],
        valid_agent_indices: tuple[int, ...],
        called_actions: list[str],
        action_obj_type: str,
        action_obj_group: int,
        next_goal: int | Literal["terminal"],
    ) -> None:
        super().__init__(obj_type, group_index, pos)
        self.valid_agent_indices: tuple[int, ...] = valid_agent_indices
        self.called_actions: list[str] = called_actions
        self.action_obj_type: str = action_obj_type
        self.action_obj_group: int = action_obj_group
        self.next_goal: int | Literal["terminal"] = next_goal

    def agents_on_goals(self, agents: list[Agent]) -> bool:
        for pos, agent_index in zip(self.pos, self.valid_agent_indices):
            if (
                agents[agent_index].pos[0] != pos[0]
                or agents[agent_index].pos[1] != pos[1]
            ):
                return False
            else:
                pass

        return True

    def open(
        self, obj_group_dict: dict[str, dict[int, ObjectGroupT]], grid: Grid
    ) -> None:
        obj_group_dict[self.action_obj_type][self.action_obj_group].call_action(
            "open", {"grid": grid}
        )


class BlockGroup(ObjectGroup):
    def __init__(
        self, obj_type: str, group_index: int, pos: tuple[tuple[int, int], ...]
    ) -> None:
        super().__init__(obj_type, group_index, pos)

    def open(self, grid: Grid) -> None:
        for pos in self.pos:
            grid[pos[0]][pos[1]].unlock()


class ZoneGroup(ObjectGroup):
    def __init__(
        self,
        obj_type: str,
        group_index: int,
        pos: tuple[tuple[int, int], ...],
        color: str,
        visual_detect_prob: float = 0.0,
        radio_detect_prob: float = 0.0,
    ) -> None:
        super().__init__(obj_type, group_index, pos)
        self.color: str = color
        self.visual_detect_prob: float = visual_detect_prob
        self.radio_detect_prob: float = radio_detect_prob

    def detect_agents(
        self, agents: list[Agent], random_generator: np.random.Generator
    ) -> bool:
        for agent in agents:
            if self.detect_agent(agent, random_generator):
                return True
            else:
                pass

        return False

    def detect_agent(self, agent: Agent, random_generator: np.random.Generator) -> bool:
        if (agent.pos[0], agent.pos[1]) in self.pos:
            visual_detect: bool = random_generator.uniform() < self.visual_detect_prob
            radio_detect: bool = random_generator.uniform() < self.radio_detect_prob
            return visual_detect or radio_detect
        else:
            return False


class LabyrinthEnv(MultiGridEnv):
    """
    Multi-agent labyrinth env with multiple tasks.
    """

    def __init__(
        self,
        num_agents: int = 3,
        init_pos: list[tuple[int, int]] = [(1, 3), (1, 4), (1, 5)],
        goal_group_config: list[GoalGroupConfig] = goal_group_config,
        obj_group_config: list[ObjectGroupConfig] = obj_group_config,
        reward_config: RewardConfig = reward_config,
        observation_option: Literal["final_goal", "all_goals"] = "final_goal",
        width: int | None = 9,
        height: int | None = 10,
        max_steps: int = 100,
        actions_set: type[ActionsT] = NavigationActions,
        agent_dir_to_vec: list[NDArray[np.int_]] = NAV_DIR_TO_VEC,
        world: WorldT = LabyrinthWorld,
        render_mode: Literal["human"] | Literal["rgb_array"] = "rgb_array",
    ) -> None:
        self.num_agents: int = num_agents
        self.goal_group_config: list[GoalGroupConfig] = goal_group_config
        self.obj_group_config: list[ObjectGroupConfig] = obj_group_config
        self.reward_config: RewardConfig = reward_config
        self.observation_option: Literal["final_goal", "all_goals"] = observation_option
        self.init_pos: list[tuple[int, int], ...] = init_pos

        agent_view_size: int = 7
        agents: list[Agent] = [
            Agent(world, i, agent_view_size, actions_set, agent_dir_to_vec)
            for i in range(num_agents)
        ]

        uncached_object_types: list[str] = (["agent"],)

        super().__init__(
            agents=agents,
            width=width,
            height=height,
            max_steps=max_steps,
            actions_set=actions_set,
            world=world,
            render_mode=render_mode,
            uncached_object_types=uncached_object_types,
        )

        self.action_space = MultiDiscrete(
            [len(self.actions) for _ in range(self.num_agents)]
        )

        self.final_goal: tuple[tuple[int, int], ...] = ()
        self.goals: list[tuple[tuple[int, int], ...]] = []

        for goal_config in self.goal_group_config:
            self.goals.append(goal_config["pos"])
            if goal_config["next_goal"] == "terminal":
                self.final_goal = goal_config["pos"]

    def _set_observation_space(self) -> Box:
        max_x: int = self.width - 1
        max_y: int = self.height - 1

        match self.observation_option:
            case "final_goal":
                num_elements: int = self.num_agents + self.num_agents
                observation_space = Box(
                    low=np.zeros(num_elements),
                    high=np.array([max_x, max_y] * self.num_agents).flatten(),
                    dtype=np.int_,
                )
            case "all_goals":
                num_elements: int = self.num_agents + self.num_agents * len(
                    self.goal_group_config
                )
                observation_space = Box(
                    low=np.zeros(num_elements),
                    high=np.array([max_x, max_y] * self.num_agents).flatten(),
                    dtype=np.int_,
                )
            case _:
                raise ValueError(
                    f"Invalid observation option: {self.observation_option}"
                )

        return observation_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[NDArray[np.int_], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs = self._get_obs()
        info: dict[str, Any] = self._get_info()

        return obs, info

    def _gen_grid(self, width, height) -> None:
        self.grid = Grid(width, height, self.world)

        # Generate walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect_filled(7, 1, 2, 1)
        self.grid.wall_rect_filled(7, 7, 2, 1)
        self.grid.wall_rect_filled(3, 4, 2, 1)

        # Place the agents
        assert len(self.agents) == len(self.init_pos)
        for agent, pos in zip(self.agents, self.init_pos):
            self.place_agent(agent, pos)

        obj_group_dict: dict[str, dict[int, ObjectGroupT]] = {}

        # Place the goal objects
        for goal_config in self.goal_group_config:
            positions: tuple[tuple[int, int], ...] = goal_config["pos"]
            valid_agent_indices: tuple[int, ...] = goal_config["valid_agent_indices"]
            group_index: int = goal_config["group_index"]
            obj_type: str = goal_config["action_obj_type"]
            action_obj_group: int = goal_config["action_obj_group"]
            next_goal: int | Literal["terminal"] = goal_config["next_goal"]
            called_actions: list[str] = goal_config["called_actions"]

            obj_group_dict[obj_type] = {
                group_index: GoalGroup(
                    obj_type,
                    group_index,
                    positions,
                    valid_agent_indices,
                    called_actions,
                    obj_type,
                    action_obj_group,
                    next_goal,
                )
            }

            for pos, agent_index in zip(positions, valid_agent_indices):
                goal = AgentGoal(self.world, agent_index)
                self.put_obj(goal, *pos)

        # Place blocks
        for obj_group_config in self.obj_group_config:
            obj_type: str = obj_group_config["obj_type"]
            group_index: int = obj_group_config["group_index"]
            positions: tuple[tuple[int, int], ...] = obj_group_config["pos"]

            match obj_type:
                case "block":
                    obj_group_dict[obj_type] = {
                        group_index: BlockGroup(obj_type, group_index, positions)
                    }

                    for pos in positions:
                        block = Block(self.world)
                        self.put_obj(block, *pos)

                case "zone":
                    color: str = obj_group_config["obj_args"]["color"]
                    args: dict[str, Any] = (
                        obj_group_config["group_args"] | obj_group_config["obj_args"]
                    )

                    obj_group_dict[obj_type] = {
                        group_index: ZoneGroup(
                            obj_type,
                            group_index,
                            positions,
                            color,
                            **args,
                        )
                    }

                    for pos in positions:
                        zone = Zone(self.world, color)
                        self.put_obj(zone, *pos)

                case _:
                    raise ValueError(f"Invalid object type: {obj_type}")

        self.obj_group_dict = obj_group_dict

    def _get_obs(self) -> NDArray[np.int_]:
        obs: list[tuple[int, int]] = []

        for agent in self.agents:
            obs.extend(agent.pos)

        match self.observation_option:
            case "final_goal":
                obs.extend(self.final_goal)
            case "all_goals":
                for goal in self.goals:
                    obs.extend(goal)

        return np.array(obs).flatten()

    def step(
        self,
        actions: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self._move_agents(actions)
        obs = self._get_obs()
        reward: float = self.compute_reward(actions)
        terminated: bool = self._agents_detected()
        truncated: bool = self.step_count >= self.max_steps
        info: dict[str, Any] = self._get_info()

        return obs, reward, terminated, truncated, info

    def _move_agents(self, actions: list[int]) -> None:
        # Randomly generate the order of the agents by indices using self.np_random.
        agent_indices: list[int] = list(range(self.num_agents))
        self.np_random.shuffle(agent_indices)
        for i in agent_indices:
            self._move_agent(actions[i], self.agents[i])

    def _move_agent(self, action: int, agent: Agent) -> None:
        next_pos: Position

        assert agent.pos is not None

        next_pos = agent.pos + agent.dir_vec[action]

        if (
            next_pos[0] < 0
            or next_pos[1] < 0
            or next_pos[0] >= self.width
            or next_pos[1] >= self.height
        ):
            pass
        else:
            next_cell: WorldObjT | None = self.grid.get(*next_pos)

            if self._is_agent_on_goal(next_pos):
                bg_color: str = "yellow"
            elif self._is_agent_on_unlocked_block(next_pos):
                bg_color: str = "light_grey"
            else:
                bg_color = "white"

            if next_cell is None:
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            elif next_cell.can_overlap():
                agent.move(next_pos, self.grid, self.init_grid, bg_color=bg_color)
            else:
                pass

    def _is_agent_on_goal(self, pos: Position) -> bool:
        for goal in self.goals:
            if pos[0] == goal[0] and pos[1] == goal[1]:
                return True

        return False

    def _is_agent_on_assigned_goal(self, pos: Position, agent_index: int) -> bool:
        cell = self.grid.get(*pos)
        if isinstance(cell, AgentGoal) and cell.agent_index == agent_index:
            return True
        else:
            return False

    def _is_agent_on_unlocked_block(self, pos: Position) -> bool:
        cell: WorldObjT | None = self.grid.get(*pos)

        if isinstance(cell, Block) and not cell.locked:
            return True
        else:
            return False

    def compute_reward(self, actions: NDArray[np.int_]) -> float:
        reward: float = 0

        # 1. Movement penalty for each agent if an action is not "stay"
        reward += self.reward_config["movement_reward"] * np.sum(actions != 0)

        # 2. Reward for each agent if it is on its assigned goal
        all_agents_on_assigned_goals: bool = True
        for agent in self.agents:
            if not self._is_agent_on_assigned_goal(agent.pos, agent.index):
                all_agents_on_assigned_goals = False
            else:
                reward += self.reward_config["agent_on_goal_reward"]

        # 3. Penalty for each agent if it moves away from its assigned goal though it was on it
        for agent, action in zip(self.agents, actions):
            prev_pos: Position = self._get_previous_agent_pos(action, agent)
            if self._is_agent_on_assigned_goal(
                prev_pos, agent.index
            ) and not self._is_agent_on_assigned_goal(agent.pos, agent.index):
                reward += self.reward_config["agent_move_away_from_goal_reward"]
            else:
                pass

        # 4. Reward for all agents if they are on their assigned goals
        if all_agents_on_assigned_goals:
            reward += self.reward_config["all_agents_on_goal_reward"]
        else:
            pass

        return reward

    def _get_previous_agent_pos(self, action: int, agent: Agent) -> Position:
        previous_pos: Position

        assert agent.pos is not None

        previous_pos = agent.pos - agent.dir_vec[action]

        return previous_pos

    def _agents_detected(self) -> bool:
        for zone in self.obj_group_dict["zone"].values():
            if zone.call_action(
                "detect_agents",
                {"agents": self.agents, "random_generator": self.np_random},
            ):
                return True
            else:
                pass
