# pylint: disable=line-too-long, dangerous-default-value
"""Defines the WildfireEnv class, which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire
"""
import functools
from itertools import combinations
from typing import TypeVar, Callable, NamedTuple, Literal
from collections import Counter, deque
import gym
from gym.spaces import flatdim
import gym.spaces
# from heapq import heapify, heappop, heappush
from gym_multigrid.typing import Position
from gym_multigrid.core.world import WorldT
from gym_multigrid.core.object import WorldObj, WorldObjT
import random
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
import numpy as np
from numpy.typing import NDArray
from pz_multigrid.multigrid import MultiGridEnv
from pz_multigrid.utils.utils import manhattan_distance
from gym_multigrid.core.world import WildfireWorld
from gym_multigrid.core.agent import WildfireActions, Agent
from gym_multigrid.core.object import Tree
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.constants import (
    STATE_TO_IDX_WILDFIRE,
    TILE_PIXELS,
    STATE_IDX_TO_COLOR_WILDFIRE,
    COLORS,
    AGENT_TYPES_WILDFIRE,
)
from gym_multigrid.utils.window import Window
from gym_multigrid.utils.misc import (
    render_agent_tiles,
    get_initial_fire_coordinates,
    save_frames_as_gif
)

AgentID = TypeVar("AgentID", int, str)




def aec_env(env:MultiGridEnv):
    return parallel_to_aec_wrapper(env)

class WildfireEnv(MultiGridEnv):
    """Grid environment which simulates dynamics of unmanned aerial vehicles (UAVs) fighting a spreading wildfire"""

    def __init__(
        self,
        alpha=0.05,
        beta=0.99,
        delta_beta=0.8,
        size=17,
        num_agents=2,
        agent_colors=("red", "blue"),
        agent_groups=None,
        agent_types=None,
        agent_view_size=10,
        initial_fire_size=1,
        max_steps=100,
        partial_obs=False,
        actions_set=WildfireActions,
        render_mode="rgb_array",
        # Mode avialable: one_hot, typed_one_hot 
        agent_representation_mode="one_hot",
        reward_type: Literal[
            "default", 
            "islands_size",
            "agent_above",
            "islands_size",
            "fire_islands_size",
            "fire_entropy",
            "fire_adjacent",
            "agent_adjacent",
            "agent_healthy_adjacent",
            ] = "default",
        render_selfish_region_boundaries=False,
        cooperative_reward=False,
        common_reward=False,
        reward_scalarisation: Literal["sum", "mean"] = "sum",
        log_selfish_region_metrics=False,
        selfish_region_xmin=None,
        selfish_region_xmax=None,
        selfish_region_ymin=None,
        selfish_region_ymax=None,
        seed=-1
    ):
        """Create a WildfireEnv environment

        Parameters
        ----------
        alpha : float, optional
            parameter for the wildfire dynamics model, by default 0.05
        beta : float, optional
            parameter for the wildfire dynamics model, by default 0.99
        delta_beta : float, optional
            parameter for the wildfire dynamics model, by default 0
        size : int, optional
            side of the square gridworld, by default 17
        num_agents : int, optional
            number of UAV agents, by default 2
        agent_start_positions : tuple[tuple[int,int]], optional
            tuple of tuples containing the start positions of the agents, in order of agent index. By default ((1, 1), (15, 15))
        agent_colors : tuple[str,str], optional
            tuple of strings of color names of all agents (or groups if agents are grouped) in order of increasing index. All agents in a group have the same color. The strings should be keys in the COLORS dictionary in constants.py. Only applicable if cooperative_reward is False. Fully cooperative agents have light_blue color by default. By default self-interested agents have red and blue colors
        agent_groups : tuple[tuple], optional
            tuple of tuples containing the indices (in ascending order) of agents in each group. Only applicable if cooperative_reward is False. By default None
        agent_view_size : int, optional
            side of the square region visible to an agent with partial observability, by default 10. Only applicable if partial_obs is True
        initial_fire_size : int, optional
            side of the square shaped initial fire region, by default 1
        max_steps : int, optional
            maximum number of steps in an episode, by default 100
        partial_obs : bool, optional
            whether agents have partial observability, by default False
        actions_set : WildfireActions, optional
            action space of the agents. All agents have the same action space. By default WildfireActions.
        render_mode : str, optional
            mode of rendering the environment, by default "rgb_array"
        render_selfish_region_boundaries : bool, optional
            whether to render boundaries of selfish regions, by default False
        cooperative_reward : bool, optional
            whether the agents use a cooperative reward, by default False. If True, the agents are fully cooperative and receive the same reward.
        log_selfish_region_metrics : bool, optional
            whether to log metrics related to trees in selfish regions, by default False
        selfish_region_xmin : list, optional
            list containing x-coordinates of the left boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_xmax : list, optional
            list containing x-coordinates of the right boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_ymin : list, optional
            list containing y-coordinates of the top boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        selfish_region_ymax : list, optional
            list containing y-coordinates of the bottom boundaries of the regions of selfish interest for the agents (or groups if the agents are grouped. All agents in a group have same region of selfish interest). Regions of selfish interest are rectangular. List elements are in order of agent (or group) indices. Only applicable if cooperative_reward is False. By default None.
        """
        self.alpha = alpha
        self.beta = beta
        self.delta_beta = delta_beta
        self.agent_colors = agent_colors
        self.agent_groups = agent_groups
        # one-hot encoding of rival non-cooperative groups of agents
        self.num_groups = 0
        if agent_groups:
            self.idx_to_group = {}
            self.num_groups += len(agent_groups)
            for i, group in enumerate(agent_groups):
                for agent_index in group:
                    self.idx_to_group[agent_index] = i
        # one-hot encoding of logically separate types of agents
        self.possible_agent_types = (0)
        if agent_types:
            self.idx_to_type = {}
            for i, a_type in enumerate(agent_types):
                self.possible_agent_types += i
                for agent_index in a_type:
                    self.idx_to_type[agent_index] = i
        # observation vector of each agent is concatenation of obs_depth number of one-hot encodings, see paper for details. len(STATE_IDX_TO_COLOR_WILDFIRE) = the number of tree states
        self.agent_representation_mode = agent_representation_mode
        match agent_representation_mode:
            case "one_hot":
                self.obs_depth = num_agents + len(STATE_IDX_TO_COLOR_WILDFIRE)
        # len(AGENT_TYPES_WILDFIRE) = the constant number of possible agent types, for now only using one generic 0
        # num_groups represents the number of teams above 1, so 3 teams means num_groups = 2
            case "typed_one_hot":
        # TODO - add one-hot encoding for different teams
                # self.obs_depth = len(AGENT_TYPES_WILDFIRE) + len(STATE_IDX_TO_COLOR_WILDFIRE) + self.num_groups
                self.obs_depth = len(AGENT_TYPES_WILDFIRE) + len(STATE_IDX_TO_COLOR_WILDFIRE)
            case _:
                raise ValueError(f"Allowed representation types are: one_hot; typed_one_hot, was given: {agent_representation_mode}")
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.world = WildfireWorld
        self.grid_size = size
        self.grid_size_without_walls = size - 2
        self.initial_fire_size = initial_fire_size
        self.burnt_trees = 0
        self.unburnt_trees = []
        self.trees_on_fire = []
        self.cooperative_reward = cooperative_reward
        self.render_selfish_region_boundaries = render_selfish_region_boundaries
        self.log_selfish_region_metrics = log_selfish_region_metrics
        if self.log_selfish_region_metrics:
            # initialize attributes for logging metrics related to trees in selfish regions
            self.selfish_xmin = np.array(selfish_region_xmin)
            self.selfish_xmax = np.array(selfish_region_xmax)
            self.selfish_ymin = np.array(selfish_region_ymin)
            self.selfish_ymax = np.array(selfish_region_ymax)
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))
            self.selfish_region_size = (
                self.selfish_xmax
                - self.selfish_xmin
                + np.ones(len(selfish_region_xmin))
            ) * (
                self.selfish_ymax
                - self.selfish_ymin
                + np.ones(len(selfish_region_ymin))
            )
        # TODO - Add a agent type property to an agent
        if self.cooperative_reward:
            # initialize cooperative agents
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color="light_blue",
                )
                for i in range(num_agents)
            ]
        elif self.agent_groups:
            # initialize self-interested agents with different colors
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color=self.agent_colors[self.idx_to_group[i]],
                )
                for i in range(num_agents)
            ]
        else:
            # initialize self-interested agents with different colors
            agents = [
                Agent(
                    world=self.world,
                    index=i,
                    view_size=agent_view_size,
                    actions=actions_set,
                    color=self.agent_colors[i],
                )
                for i in range(num_agents)
            ]

        super().__init__(
            agents=agents,
            grid_size=size,
            max_steps=max_steps,
            partial_obs=partial_obs,
            agent_view_size=agent_view_size,
            actions_set=actions_set,
            world=self.world,
            render_mode=render_mode,
            
            np_random_seed=seed,
        )
        self.helper_grid = None
        self._np_random = np.random.default_rng(seed)
        self._np_random_seed = seed
        start_pos = self._np_random.uniform(low=1, high=size-1, size=(num_agents, 2))
        agent_start_positions = tuple((int(pos[0]), int(pos[1])) for pos in start_pos)
        self.agent_start_positions = agent_start_positions
        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                ) 
        self.frames = []
        self.recording = False
        self.step_count = 0
        # self.observation_space: Box | Dict = self.observation_space()
        # self.action_space = Discrete(n=len(self.actions), start=0)

    def seed(self):
        return self._np_random_seed
    
    def start_rec(self):
        self.frames = []
        self.recording = True

    def stop_rec(self):
        self.recording = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent=None) -> Box:
        """Set the observation space for each agent in the environment. All agents possess the same observation space

        Returns
        -------
        observation_space : Box
            the observation space for that agent
        """
        # observation vector of agent is the concatenation of obs_depth number of one-hot encodings where each encoding has grid_size_without_walls number of elements valued either 0 or 1. Additionally, the observation vector contains the normalized time step at the end
        low = np.full(self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 0)
        high = np.full(
            self.obs_depth * ((self.grid_size_without_walls + 1) ** 2) + 1, 1
        )
        observation_space = Box(
                    low=low,
                    high=high,
                    dtype=np.int32,
                )
                

        return observation_space
    

    def get_state_size(self):
        """ Returns the shape of the state"""
        return int(self.observation_space().shape[0])
    
    def get_avail_actions(self):
        avail_actions = []
        for agent_id in self.agents:
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1] * len(self.actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return len(self.actions) 

    def get_obs_shape(self):
        return  int(self.observation_space().shape[0])
    
    def get_env_info(self):
        env_info = {"state_shape": self.num_agents * self.get_obs_shape(),
                    "obs_shape": self.num_agents * self.get_obs_shape(),
                    "n_actions": int(len(self.actions)),
                    "avail_actions": self.get_total_actions(),
                    "n_agents": self.num_agents,
                    "episode_limit": self.max_steps}
        return env_info

    def save_replay(self, path, f, ep):
        save_frames_as_gif(frames=self.frames, path=path, filename=f, ep=ep, fps=5)


    def get_stats(self):
        info = self.info[0]
        return {
                "burnt trees": info['burnt trees'],
                "unburnt trees": info['unburnt trees'],
                "episode_limit": self.max_steps,
                "episode_length": self.step_count,
                }

    def observe(self, agent:AgentID):
        """
        Returns
        -------
        obs: numpy.NDArray
          masked grid observation of agent with correspoding AgentID
        """
        return self.obs[agent]

    def _gen_grid(self, width, height, state=None):
        """Generate the grid for the environment

        Parameters
        ----------
        width : int
            width of the grid
        height : int
            height of the grid
        state : ndarray, optional
            specifies the initial state of the environment, by default None.
            If none, it is chosen uniformly at random from the assumed initial state distribution
        """
        self.grid = Grid(width, height, self.world)

        # generate the walls of the grid
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        agent_start_pos = []
        if state is not None:
            # store positions of agents and trees on fire as per the specified initial state.
            state = state[:-1].reshape(
                (
                    self.obs_depth + 1,
                    self.grid_size,
                    self.grid_size,
                ),
            )
            initial_fire = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if state[1, j, i]:
                        # i and j are swapped because the y-coordinate specifies the row, while the x-coordinate specifies the column.
                        initial_fire.append((i, j))
                    for o in self.agents:
                        index = 0
                        if self.agent_representation_mode == "one_hot":
                            index = o
                        if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + index, j, i]:
                            # i and j are swapped because the y-coordinate specifies the row, while the x-coordinate specifies the column.
                            agent_start_pos.append((i, j))

        else:
            # choose location of initial fire uniformly at random
            if self.initial_fire_size % 2 == 0:
                # for even sized initial fires, choose location of top left corner of fire region uniformly at random
                top_left_corner = (
                    self._np_random.integers(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                    self._np_random.integers(
                        1,
                        self.grid_size_without_walls - (self.initial_fire_size),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *top_left_corner,
                    self.grid_size_without_walls,
                    self.initial_fire_size,
                )
            else:
                # for odd sized initial fires, choose location of center of fire region uniformly at random
                fire_square_center = (
                    self._np_random.integers(
                       low = 1 + ((self.initial_fire_size - 1) / 2),

                       high = self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) / 2),
                    ),
                    self._np_random.integers(
                       low= 1 + ((self.initial_fire_size - 1) / 2),
                       
                       high= self.grid_size_without_walls
                        - ((self.initial_fire_size - 1) / 2),
                    ),
                )
                initial_fire = get_initial_fire_coordinates(
                    *fire_square_center,
                    self.grid_size_without_walls,
                    self.initial_fire_size,
                )
            # agent_start_pos is specified during environment initialization
            agent_start_pos = self.agent_start_positions

        for pos in initial_fire:
            region = "common"
            if self.log_selfish_region_metrics:
                # update count of trees on fire in selfish regions
                if self.agent_groups:
                    for i, _ in enumerate(self.agent_groups):
                        if self.in_selfish_region(pos[0], pos[1], i):
                            # selfish region is identified by the lowest index among indices of the corresponding group of selfish agents
                            region = f"{i}"
                            self.selfish_region_trees_on_fire[i] += 1
                            break
                else:
                    for a in self.agents:
                        if self.in_selfish_region(pos[0], pos[1], a):
                            # selfish region is identified by the lowest index among indices of the corresponding selfish agent
                            region = f"{a}"
                            self.selfish_region_trees_on_fire[a] += 1
                            break
            # insert tree on fire in grid
            self.put_obj(
                Tree(self.world, STATE_TO_IDX_WILDFIRE["on fire"], region=region),
                int(pos[0]),
                int(pos[1]),
            )


        # update counts of healthy trees
        num_healthy_trees = self.grid_size_without_walls**2 - len(initial_fire)
        # insert healthy tree in grid
        for _ in range(num_healthy_trees):
            tree_obj = Tree(self.world, STATE_TO_IDX_WILDFIRE["healthy"])
            self.place_obj(tree_obj)
            if self.log_selfish_region_metrics:
                # check if tree is in a selfish region, and update region attribute of tree if it is
                if self.agent_groups:
                    for i, _ in enumerate(self.agent_groups):
                        if self.in_selfish_region(
                            *(tree_obj.pos), i  # pylint: disable=not-an-iterable
                        ):
                            tree_obj.region = f"{i}"
                            break
                else:
                    for a in self.agents:
                        if self.in_selfish_region(
                            *(tree_obj.pos), a  # pylint: disable=not-an-iterable
                        ):
                            tree_obj.region = f"{a}"
                            break

        # helper grid is a work around for grid being unable to store multiple objects at a single cell. It does not contain agents
        self.helper_grid = self.grid.copy()

        # create list of unburnt trees. initial state does not have burnt trees.
        for c in self.helper_grid.grid:
            if c is not None and c.type == "tree":
                self.unburnt_trees.append(c)
        # Add all starting burning trees 
        for pos in initial_fire:
            self.trees_on_fire.append(self.helper_grid.get(*pos))
        # insert agents in grid
        for i, a in enumerate(self.agents_storage):
            self.place_agent(a, pos=agent_start_pos[i])
            self.helper_grid.get(*agent_start_pos[i]).agent_above = True

    def _get_obs(self) -> dict[AgentID, np.typing.NDArray]:
        """Get observation vectors of all agents in the environment.

        Returns
        -------
        agent_obs: dict(int: ndarray)
            dict of agent observations where the element at i^th index is the observation vector for the agent with index i.
        """
        # initialize list of observation vector of each agent
        agent_obs = {
            i: np.zeros(
                (
                    self.obs_depth,
                    self.grid_size_without_walls + 1,
                    self.grid_size_without_walls + 1,
                ),
                dtype=np.int32,
            )
            for i in range(self.num_agents)
        }

        # update walls and tree states in agent observations
        for obj in self.helper_grid.grid:
            if obj is None:
                continue
            i, j = obj.pos
            for a in self.agents_storage:
                # convert to agent centered coordinates
                nc = [i - a.pos[0], j - a.pos[1]]
                # wrap around to get agent centered toroidal coordinates
                if nc[0] < 0:
                    nc[0] += self.grid_size_without_walls + 1
                if nc[1] < 0:
                    nc[1] += self.grid_size_without_walls + 1
                # update agent's observation. # switch x and y coordinates because the y-coordinate specifies the row, while the x-coordinate specifies the column
                if obj.type == "tree":
                    agent_obs[a.index][obj.state, nc[1], nc[0]] = 1
                elif obj.type == "wall":
                    agent_obs[a.index][
                        len(STATE_IDX_TO_COLOR_WILDFIRE), nc[1], nc[0]
                    ] = 1

        # for each agent, update other agents' positions in agent observations
        for a in self.agents_storage:
            for o in self.agents_storage:
                if o.index != a.index:
                    # convert to agent centered coordinates
                    nc = [
                        o.pos[0] - a.pos[0],
                        o.pos[1] - a.pos[1],
                    ]
                    # wrap around to get agent centered toroidal coordinates
                    if nc[0] < 0:
                        nc[0] += self.grid_size_without_walls + 1
                    if nc[1] < 0:
                        nc[1] += self.grid_size_without_walls + 1
                    if self.agent_representation_mode == "one_hot":
                        idx = o.index - int(np.heaviside(o.index - a.index, 0))
                        agent_obs[a.index][
                            len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + idx,
                            nc[1],
                            nc[0],
                        ] = 1
                    # Add a mapping to an agent type to an index to pad from starting "common" agent position
                    else:
                        agent_obs[a.index][
                            len(STATE_IDX_TO_COLOR_WILDFIRE) + 1,
                            nc[1],
                            nc[0],
                        ] = 1

        # flatten, and append normalized time step at the end of, each agent observation
        for a in self.agents:
            agent_obs[a] = np.append(
                agent_obs[a].flatten(),
                np.array(self.step_count / self.max_steps, dtype=np.float32),
            )
            # agent_obs[a] = agent_obs[a].flatten()
        return agent_obs

    def get_state(self):
        """Get the state representation of the environment.

        Returns
        -------
        ndarray
            state representation of the environment
        """
        # initialize array to store state vector
        s = np.zeros(
                (
                    self.obs_depth,
                    self.grid_size_without_walls + 1,
                    self.grid_size_without_walls + 1,
                ),
                dtype=np.int32,
            )

        # update tree states and walls in state representation
        # TODO: fix shape of the grid to have (N, M - 1, M - 1)
        # Where M is full size with walls, N obs depth
        for o in self.helper_grid.grid:
            # switch x and y coordinates because the y-coordinate specifies the row, while the x-coordinate specifies the column
            if o.type == "tree":
                s[o.state, o.pos[1] - 1, o.pos[0] - 1] = 1
            if o.type == "wall":
                continue
                s[len(STATE_IDX_TO_COLOR_WILDFIRE), o.pos[1], o.pos[0]] = 1

        # update agent positions in state representation
        if self.agent_representation_mode == "one_hot":
            for a in self.agents_storage:
                s[
                    len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + a.index,
                    a.pos[1],
                    a.pos[0],
                ] = 1
        else:
            for a in self.agents_storage:
                s[
                    len(STATE_IDX_TO_COLOR_WILDFIRE) + 1,
                    a.pos[1],
                    a.pos[0],
                ] = 1

        # flatten, and append normalized time step at the end of, state representation
        s = np.append(
            s.flatten(),
            np.array(self.step_count / self.max_steps, dtype=np.float32),
        )
        # s = s.flatten()
        return s

    def get_state_interpretation(self, state, print_interpretation=True):
        """Get human readable interpretation of the state of the environment

        Parameters
        ----------
        state : ndarray
            state representation of the environment
        print_interpretation : bool, optional
           whether to print the interpretation of the state, by default True. Human readable interpretation refers to printing the positions of trees on fire, agents, and the time step.

        Returns
        -------
        on_fire_trees : list[tuple[int,int]]
            list of tuples containing position coordinates (x,y) of trees on fire. Coordinates are in grid without wall coordinates
        time_step : float
            normalized time step of the episode at which time the state was recorded
        """
        time_step = state[-1]
        state = state[:-1].reshape(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
        )
        if print_interpretation:
            print("-------------------------------------------------------------")
            print("State interpretation:")
        on_fire_trees = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if state[1, j, i] == 1:
                    if print_interpretation:
                        print(f"Tree at position {(i,j)} is on fire.")
                    on_fire_trees.append((i, j))
                if self.agent_representation_mode == "one_hot":
                    for o in self.agents_storage:
                        index = o.index
                        if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + index, j, i] == 1:
                            if print_interpretation:
                                print(f"Agent {o.index} is at position {(i,j)}.")
                else:
                    if state[len(STATE_IDX_TO_COLOR_WILDFIRE) + 1, j, i] == 1:
                            if print_interpretation:
                                print(f"Agent of type common is at position {(i,j)}.")
        if print_interpretation:
            print(f"Time step: {time_step}")
            print("-------------------------------------------------------------")
        return on_fire_trees, time_step

    def construct_state(self, trees_on_fire, agent_pos, time_step: int):
        """Construct the state representation vector of the environment for given positions of trees on fire and agents

        Parameters
        ----------
        trees_on_fire : list
            list of tuples containing position coordinates (x,y) of trees on fire. Coordinates should be in grid without wall coordinates
        agent_pos : list
            list of tuples containing position coordinates (x,y) of agents, in order of agent index. Coordinates should be in grid without wall coordinates
        time_step : int
            normalized time step of the episode at which time the state was recorded

        Returns
        -------
        state : ndarray
            state representation of the environment
        """
        state = np.zeros(
            (
                self.obs_depth + 1,
                self.grid_size,
                self.grid_size,
            ),
            dtype=np.float32,
        )
        # update tree states and walls in state representation. There are no burnt trees in the state
        state[0, :, :] = 1
        for pos in trees_on_fire:
            state[1, pos[1], pos[0]] = 1
            state[0, pos[1], pos[0]] = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (
                    i == 0
                    or i == self.grid_size - 1
                    or j == 0
                    or j == self.grid_size - 1
                ):
                    state[len(STATE_IDX_TO_COLOR_WILDFIRE), j, i] = 1
                    state[0, j, i] = 0

        # update agent positions in state representation
        if self.agent_representation_mode == "one_hot":
            for i, pos in enumerate(agent_pos):
                state[
                    len(STATE_IDX_TO_COLOR_WILDFIRE) + 1 + i,
                    pos[1],
                    pos[0],
                ] = 1
        else:
            for pos in agent_pos:
                state[
                    len(STATE_IDX_TO_COLOR_WILDFIRE) + 1,
                    pos[1],
                    pos[0],
                ] = 1
        # flatten, and append normalized time step at the end of, state representation
        state = np.append(
            state.flatten(),
            np.array(time_step / self.max_steps, dtype=np.float32),
        )
        # state = state.flatten()
        return state

    def reset(self, seed: int | None = None, options=None):
        """Reset the state of the environment

        Parameters
        ----------
        seed : int, optional
            seed for random number generator, by default None
        state : ndarray, optional
            specifies the initial state of the environment upon reset, by default None.
            If none, initial state is chosen uniformly at random from initial state distribution.

        Returns
        -------
        obs : OrderedDict
            dictionary where each key is the agent index and the value is the observation vector for that agent.
        info : dict
            dictionary containing additional information about the environment.
            Here, it contains the number of burnt trees in the environment after reset
        """
        # reset environment attributes
        if seed is None:
            seed = self._np_random_seed
        self.burnt_trees = 0
        self.trees_on_fire = []
        self.unburnt_trees = []
        if self.log_selfish_region_metrics:
            self.selfish_region_trees_on_fire = np.zeros(len(self.selfish_xmin))
            self.selfish_region_burnt_trees = np.zeros(len(self.selfish_xmin))

        # # reset the grid
        if options is not None and options.get('state') is not None:
             super().reset(seed=seed, options=options['state'])
        else:
             super().reset(seed=seed)
        assert seed is None or seed > 0
        self._np_random = np.random.default_rng(seed)
        self._np_random_seed = seed
        start_pos = self._np_random.uniform(low=1, high=self.grid_size-1, size=(self.num_agents, 2))
        self.agent_start_positions = tuple((int(pos[0]), int(pos[1])) for pos in start_pos)
        # get agent observations
        self.obs = self._get_obs()
        # obs = {a: agent_obs[a] for a in self.agents}
        match self.reward_type, self.cooperative_reward:
            case "islands_size", True:
                self.fire_board = self.connected_component()
            case "fire_islands_size", True:
                self.fire_board = self.connected_component(avoided_conditions= [lambda obj: obj.type == "tree" and obj.state==2])
            case "fire_entropy", True:
                self.fire_board = self.connected_component()[:][:][0]
                self.fire_board = self.fire_board[self.fire_board > -1]
        # create info dictionary
        self.info = {"burnt trees": self.burnt_trees, 
                     "unburnt trees": len(self.unburnt_trees), }
                    #  "env_defined_observations": self.observation_space(), }
                    #  "env_defined_actions": np.arange(len(self.actions)),}
                    # "blocking_actions": 0}
                # in AECEnv/ParallelEnv, both obs and info must contain agents set as a subset 
        self.info = {a: self.info for a in self.agents}
        return self.obs, self.info

    def move_agent(self, i, next_pos):
        """Move agent to a new position in the grid

        Parameters
        ----------
        i : int
            index of agent to be moved
        next_pos : tuple[int, int]
            coordinates of new position
        """
        # add agent to grid in new position
        self.grid.set(*next_pos, self.agents_storage[i])

        # get tree in agent's old position from helper grid and add tree to grid
        tree = self.helper_grid.get(*self.agents_storage[i].pos)
        tree.agent_above = False
        self.grid.set(*self.agents_storage[i].pos, tree)

        # update attributes
        next_tree = self.helper_grid.get(*next_pos)
        next_tree.agent_above = True
        self.agents_storage[i].pos = next_pos

    def neighbors_on_fire(self, tree_pos, state="on fire") -> int:
        """Get the number of neighboring trees on fire for a given tree.
           Neighbors are adjacent trees in the cardinal directions. A tree can have at most 4 neighbors

        Parameters
        ----------
        tree_pos : tuple[int, int]
            position coordinates of the tree whose neighbors are to be checked

        Returns
        -------
        num : int
            the number of neighboring trees on fire
        """
        idx = STATE_TO_IDX_WILDFIRE[state]
        assert idx is not None, f"Invalid state: {state}" 
        num = 0
        for r in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            neighbor_pos = tree_pos + r
            if neighbor_pos[0] >= 0 and neighbor_pos[0] < self.helper_grid.width:
                if neighbor_pos[1] >= 0 and neighbor_pos[1] < self.helper_grid.height:
                    o = self.helper_grid.get(*neighbor_pos)
                    if o is not None and o.type == "tree":
                        if o.state == idx:
                            num += 1
        return num

    def in_selfish_region(self, i: int, j: int, region_index: int) -> bool:
        """Check if given tree is in region of selfish interest with given index

        Parameters
        ----------
        i : int
            x-coordinate of tree position
        j : int
            y-coordinate of tree position
        region_index : int
            index of selfish region. Same as index of corresponding selfish agent (or group of selfish agents).

        Returns
        -------
        bool
            True, if tree is in the region of selfish interest. Otherwise, False
        """
        return (
            i >= self.selfish_xmin[region_index]
            and i <= self.selfish_xmax[region_index]
            and j >= self.selfish_ymin[region_index]
            and j <= self.selfish_ymax[region_index]
        )
    
    def grid_bfs(
        self,
        queue: deque[tuple], 
        board: NDArray, 
        color: int = 0,
        avoided_objects:list[str] = ["wall"],
        avoided_conditions: list[Callable[[WorldObj], bool]] = [lambda x: False],
        ):
        """
        TODO: fix
        WARNING: unexpected behavior from initialising deque with more than one instance
        Possible race condition of splitting a connected graph with different colors
         from different fire starts being present in the same queue at once
        queue: deque
            starting deque of (Initialised objects, x) to start exploring from
        board: NDArray 
            3-tuple (m, n, 2) numpy array
            m and n are coordinates of objects
            last index corresponds for (x, y) connectivity island information list
            \n x is the closest distance to the fire in that region (unexplored is '-1')
            \n y is The region id (unexplored is '-1')

            NOT INCLUDED
            \n z is the amount of fires adjacent to that region (unexplored is '0')
        avoided_objects: list[WorldObj] 
            tile types that block exploration
        avoided_conditions: list[Callable[[WorldObj], bool]]
            Extra functions that filter objects based not just on their type

        \n Returns: bool
            Whether has revisited the same cluster in grid
        """
        revisiting = False
        while len(queue) > 0:
            obj:WorldObj
            obj, s = queue.popleft()
            board[obj.pos[0]][obj.pos[1]][0] = s
            board[obj.pos[0]][obj.pos[1]][1] = color

            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                next_obj:WorldObj
                next_obj = self.helper_grid.get(obj.pos[0] + dx, obj.pos[1] + dy)
                if next_obj is None:
                    continue

                if next_obj.pos[0] < 0 or next_obj.pos[0] >= self.helper_grid.width:
                    continue

                if next_obj.pos[1] < 0 or next_obj.pos[1] >= self.helper_grid.height:
                    continue
                
                # Hit an avoidance condition, skip
                if next_obj.type in avoided_objects:
                    continue

                # If an object passes on any condition of a more function that filters objects based not just their type, skip
                if any(condition(next_obj) for condition in avoided_conditions):
                    continue        
                
                # Already present
                if board[next_obj.pos[0]][next_obj.pos[1]][0] > -1:
                    revisiting = True
                    if board[next_obj.pos[0]][next_obj.pos[1]][0] <= s + 1:
                        continue
                    
                # person arrives at the same time to the end of the board, special case
                # if not is_fire and nr == m - 1 and nc == n - 1 and fire_board[nr][nc] == s + 1:
                #     queue.append((nr, nc, s+1))
                #     continue

                # # person arrives to a place fire has been there first or at the same time, can't proceed
                # if not is_fire and fire_board[nr][nc] > -1 and fire_board[nr][nc] <= s + 1:
                #     continue

                queue.append((next_obj, s+1))

        return revisiting

    def connected_component(self,
        avoided_objects:list[str] = ["wall"],
        avoided_conditions: list[Callable[[WorldObj], bool]] = [lambda obj: obj.type == "tree" and obj.state!=0]
        ):
        board = np.full(shape=(self.grid_size,self.grid_size,2), fill_value=-1)
        color = 0
        for fire in self.trees_on_fire:
            if self.grid_bfs(
                # deque([(fire, 0) for tree in self.trees_on_fire]), 
                deque([(fire, 0)]),
                board, 
                color=color,
                avoided_objects=avoided_objects,
                avoided_conditions=avoided_conditions):
                # Start coloring to the next island cluster if this one was already explored
                color += 1
        # print(f"Total clusters grid: {board}")
        return board
    
    def fire_spread_sizes(self, graph_bag, min_area) -> list[int]:
        """
        Find sizes of all unburnt trees components adjacent to burning fire
        graph_bag: Indexed list of all unburnt tree tiles, each containing a bag of healthy connections fire could spread through \n
        Shall be initialised through BFS of all healthy tiles
        min_area: Min component size to be included in the returned list

        returns: Sorted list of sizes of all components where there is minimal fire present
        """

        ### Create combined array w/o burned trees, and count distance from burning trees once using BFS



        n = len(graph_bag)
        parents = range(n)
        def find(x):
            if x != parents[x]:
                parents[x] = find(parents[x])
        def union(x, y):
            parents[find(x)] = find(y)
        
        for i in range(n):
            for j in range(i + 1, n):
                # If connection in connectivity matrix; change to a bagged component
                if graph[i][j] == 1:
                    union(i, j)
        # Find areas of touching fires
        area = Counter(find(i) for i in range(n))
        fire_spread = Counter(find(i) for i in self.trees_on_fire)
        unburnt_sizes = []
        # unburnt_size is the size of the area
        # res is the initial fire index in the graph
        unburnt_size, res = 0, min(self.trees_on_fire)
        for i in self.trees_on_fire:
            if fire_spread[find(i)] >= 1:
                # Find the original 
                parent = find(i)
                if area[parent] > unburnt_size:
                    if area[parent] >= min_area:
                        unburnt_sizes.append(unburnt_size)
                    unburnt_size, res = area[parent], i
                elif area[parent] == unburnt_size:
                    res = min(res, i)
        unburnt_sizes.sort(reverse=True)
        return unburnt_sizes
    


    def _reward(self, current_agent, rewards, reward=1):
        raise NotImplementedError("Reward is computed in step")

    def step(self, actions):
        """Take a step in the environment. Wildfire dynamics are propagated by one time step, and agents move according to their actions.

        Parameters
        ----------
        actions : dict
            dict in the Discrete (n actions) space containing actions for each agent. Corresponding agent index key contains the action value.

        Returns
        -------
        next_obs : dict
            dictionary where each key is the agent index and the value is the new observation vector (ndarray) for that agent after the environment step.
        rewards : dict
            dictionary where each key is the agent index and the value is the reward given to that agent after the environment step.
        terminated : dict[bool]
            True, if the episode is done, otherwise, False. Episode is done if maximum number of steps is reached or there are zero trees on fire.
        info : dict
            dictionary where each key is the agent index and the value is an info dictionary containing additional information about the environment. Here, each agent's info dictionary contains the same information, viz., the number of burnt trees.
        """
        self.step_count += 1
        terminated = np.zeros(len(self.agents))
        truncated = np.zeros(len(self.agents))
        # Move agents sequentially, in random order
        assert self._np_random is not None
        order = self._np_random.permutation(len(self.agents))
        blocking_agent_index = []
        for i in order:
            next_pos = self.agents_storage[i].pos
            action = actions[i]
            match action:
                case self.actions.STILL:
                        continue
                case self.actions.NORTH:
                    next_pos = self.agents_storage[i].north_pos()
                case self.actions.SOUTH:
                    next_pos = self.agents_storage[i].south_pos()
                case self.actions.EAST:
                    next_pos = self.agents_storage[i].east_pos()
                case self.actions.WEST:
                    next_pos = self.agents_storage[i].west_pos()
            next_cell = self.grid.get(*next_pos)
            if next_cell is None or next_cell.can_overlap():
                self.move_agent(i, next_pos)
            elif next_cell.type == 'agent':
                blocking_agent_index.append(self.agents_storage.index(next_cell))
            elif next_cell.type == 'wall':
                blocking_agent_index.append(i)

        # propagate wildfire dynamics by one time step
        # initialize lists to store trees transitioning to on fire and burnt state in the current time step
        trees_to_fire_state = []
        if self.log_selfish_region_metrics:
            num_trees_to_fire_state_sr = {
                f"{i}": 0 for i in self.agents
            }
        trees_to_burnt_state = []
        # loop over all unburnt trees. Burnt trees remain burnt
        for c in self.unburnt_trees:
            if c.state == 0:
                pos = np.array(c.pos)
                # transition from healthy to on fire with probability 1 - (1 - alpha)^n
                if self._np_random.random() < 1 - (1 - self.alpha) ** self.neighbors_on_fire(
                    pos
                ):
                    # update relevant attributes and lists
                    trees_to_fire_state.append(c)
                    self.trees_on_fire.append(c)
                    if self.log_selfish_region_metrics:
                        if c.region != "common":
                            self.selfish_region_trees_on_fire[int(c.region)] += 1
                            num_trees_to_fire_state_sr[c.region] += 1
            if c.state == 1:
                # transition from on fire to burnt with probability 1 - beta + delta_beta * agent_above
                if self._np_random.random() < 1 - self.beta + c.agent_above * self.delta_beta:
                    # update relevant attributes and list
                    trees_to_burnt_state.append(c)
                    self.burnt_trees += 1
                    self.trees_on_fire.remove(c)
                    if self.log_selfish_region_metrics:
                        if c.region != "common":
                            self.selfish_region_burnt_trees[int(c.region)] += 1
                            self.selfish_region_trees_on_fire[int(c.region)] -= 1

        # update tree objects in helper grid and grid. This update is done after the loop to avoid affecting
        # the transition probabilities of trees later in the loop due to trees that have already transitioned earlier in the loop.
        for c in trees_to_fire_state:
            c.state = 1
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            # update tree in grid
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 1
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]
        for c in trees_to_burnt_state:
            self.unburnt_trees.remove(c)
            c.state = 2
            c.color = STATE_IDX_TO_COLOR_WILDFIRE[c.state]
            # update tree in grid
            o = self.grid.get(c.pos[0], c.pos[1])
            if o.type == "tree":
                o.state = 2
                o.color = STATE_IDX_TO_COLOR_WILDFIRE[o.state]
        # check if episode is done
        if len(self.trees_on_fire) == 0:
            terminated = np.ones(len(self.agents))
            term_reward = len(self.unburnt_trees) / self.grid_size_without_walls ** 2
            rewards = term_reward if self.common_reward else {a: term_reward for a in self.agents}
            # self.agents = []
            # self.agents_storage = []
        elif self.step_count >= self.max_steps:
            truncated = np.ones(len(self.agents))
            trunc_reward = -self.burnt_trees / self.grid_size_without_walls ** 2
            rewards = trunc_reward if self.common_reward else {a: trunc_reward for a in self.agents}
            # self.agents = []
            # self.agents_storage = []
        else:
            # compute agent rewards
            
            agent_rewards = np.zeros(self.num_agents)
            # Add reward for each unburnt tree
            # alpha reward coeff = 1
            agent_rewards += len(self.unburnt_trees) / self.grid_size_without_walls ** 2
            # Only used for *_adjacent rewards
            exst_trees_pos = np.array([[*tree.pos] for tree in trees_to_burnt_state if tree.agent_above])
            # exst_trees_pos = np.array([[*tree.pos] for tree in self.agents_storage])

            for tree_pos in exst_trees_pos:
            # Add reward to agent index which extinguished the fire
            # beta reward coeff = 3
                a = self.grid.get(*tree_pos).index
                agent_rewards[a] += 3.
            # Reward accounts for each action blocking other agents (including itself)
            # Moving/Congesting agent coeff = 0.5
            for a in blocking_agent_index:
                agent_rewards[a] -= 0.5
            match self.reward_type, self.cooperative_reward:
                case _, False:
                    if self.agent_groups:
                        for a in self.agents:
                            agent_rewards[a] -= 0.5 * num_trees_to_fire_state_sr[
                                f"{self.idx_to_group[a]}"
                            ] + 0.1 * (
                                len(trees_to_fire_state)
                                - num_trees_to_fire_state_sr[
                                    f"{self.idx_to_group[a]}"
                                ]
                            )
                    else:
                        for a in self.agents:
                            agent_rewards[a] -= 0.5 * num_trees_to_fire_state_sr[
                                f"{a}"
                            ] + 0.1 * (
                                len(trees_to_fire_state)
                                - num_trees_to_fire_state_sr[f"{a}"]
                            )
                case "agent_above", True:
                # Delta reward coeff = 0.5
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
                    for a in self.agents:
                        tree = self.helper_grid.get(*self.agents_storage[a].pos)
                # Beta+ reward coeff = 1
                        agent_rewards[a] += 1 if tree.type == "tree" and tree.state == 1 else -1
                case "islands_size", True:
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
                    next_fire_board = self.connected_component()
                    clusters, counts = np.unique(self.fire_board[:][:][2], return_counts=True)
                    # Remove the -1 unexplored cluster
                    if clusters[0] == -1:
                        clusters, counts = clusters[1:], counts[1:]
                    # print(f"Current clusters at step: {self.step_count}: \n {len(clusters), len(counts)}")
                    assert len(clusters) == len(counts)
                    next_clusters, next_counts = np.unique(next_fire_board[:][:][2], return_counts=True)
                    if next_clusters[0] == -1:
                        next_clusters, next_counts = next_clusters[1:], next_counts[1:]
                    # print(f"Next clusters at step: {self.step_count}: \n {len(next_clusters), len(next_counts)}")
                    assert len(next_clusters) == len(next_counts)
                    # Discard small clusters
                    clusters = clusters[counts > 2]
                    next_clusters = next_clusters[next_counts > 2]
                    # Find the difference in clusters
                    if len(next_clusters) < len(clusters):
                        agent_rewards += 50. * (np.sum(counts) - np.sum(next_counts)) / self.grid_size_without_walls ** 2
                    self.fire_board = next_fire_board
                case "fire_islands_size", True:
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
                    next_fire_board = self.connected_component(avoided_conditions= [lambda obj: obj.type == "tree" and obj.state==2])
                    clusters, counts = np.unique(self.fire_board[:][:][2], return_counts=True)
                    # Remove the -1 unexplored cluster
                    if clusters[0] == -1:
                        clusters, counts = clusters[1:], counts[1:]
                    next_clusters, next_counts = np.unique(next_fire_board[:][:][2], return_counts=True)
                    if next_clusters[0] == -1:
                        next_clusters, next_counts = next_clusters[1:], next_counts[1:]
                    # Discard small clusters
                    clusters = clusters[counts > 2]
                    next_clusters = next_clusters[next_counts > 2]
                    # Find the difference in clusters
                    if len(next_clusters) < len(clusters):
                        agent_rewards += 50. * (np.sum(counts) - np.sum(next_counts)) / self.grid_size_without_walls ** 2
                    self.fire_board = next_fire_board
                case "fire_entropy", True:
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
                    next_fire_board = self.connected_component()[:][:][0]
                    next_fire_board = next_fire_board[next_fire_board > -1]
                    next_dt = np.sum(next_fire_board)
                    dt = np.sum(self.fire_board)
                    agent_rewards += 2. * dt > next_dt + max(dt - next_dt, 0)
                    self.fire_board = next_fire_board
                case "fire_adjacent", True:
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
                    for tree_pos in exst_trees_pos:
                        a = self.grid.get(*tree_pos).index
                        # Reward for adjacent trees to the extinguished fire
                        agent_rewards[a] += 2. ** self.neighbors_on_fire(tree_pos, state="healthy")
                case "agent_adjacent", True:
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
                    # Unique pairs of trees that were extinguished
                    # Broadcasting to get all possible combinations of tree coordinates
                    # combinations = np.transpose([np.tile(exst_trees, len(exst_trees)), np.repeat(exst_trees, len(exst_trees))])
                    for pos_i, pos_j in combinations(exst_trees_pos, 2):
                        agent_i = self.grid.get(*pos_i).index
                        agent_j = self.grid.get(*pos_j).index
                        tree_dt = manhattan_distance(pos_i, pos_j)
                        if tree_dt <= self.grid_size_without_walls // 2:
                            agent_rewards[agent_i] += 2.**(3-tree_dt)
                            agent_rewards[agent_j] += 2.**(3-tree_dt)
                        if tree_dt <= 2:
                            agent_rewards[agent_i] += 10
                            agent_rewards[agent_j] += 10
                case "agent_healthy_adjacent", True:
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
                    # Unique pairs of trees that were extinguished
                    # Broadcasting to get all possible combinations of tree coordinates
                    # combinations = np.transpose([np.tile(exst_trees, len(exst_trees)), np.repeat(exst_trees, len(exst_trees))])
                    for pos_i, pos_j in combinations(exst_trees_pos, 2):
                        agent_i = self.grid.get(*pos_i).index
                        agent_j = self.grid.get(*pos_j).index
# TODO: USE A DIFFERENT FUNC IN DIAGONALS BECOME ADJACENT
                        tree_dt = manhattan_distance(pos_i, pos_j)
                        agent_rewards[agent_i] += 2.**(3-tree_dt) 
                        agent_rewards[agent_j] += 2.**(3-tree_dt)
# TODO: INCREASE IF DIAGONALS BECOME ADJACENT
                        if tree_dt > 2:
                            continue
# Selections of all possible adjacencies
                        adj_trees = 0
                        for i, r in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
# Two agents can't be in the same tile
                            for s in [(1, 0), (0, 1), (-1, 0), (0, -1)].pop(i):
                                neighbor_pos_i = pos_i + r
                                neighbor_pos_j = pos_j + s
                                # If positions don't match, skip
                                if neighbor_pos_i != neighbor_pos_j:
                                    continue
                                # Find if position is in grid and has a healthy tree
                                neighbor_pos = neighbor_pos_i
                                if neighbor_pos[0] >= 0 and neighbor_pos[0] < self.helper_grid.width:
                                    if neighbor_pos[1] >= 0 and neighbor_pos[1] < self.helper_grid.height:
                                        o = self.helper_grid.get(*neighbor_pos)
                                        if o is not None and o.type == "tree":
                                            if o.state == STATE_TO_IDX_WILDFIRE["healthy"]:
                                                adj_trees += 1

                        agent_rewards[agent_i] += 5 + adj_trees
                        agent_rewards[agent_j] += 5 + adj_trees
                case _, True:   
                    agent_rewards -= 0.5 * len(trees_to_fire_state)
            
            agent_rewards /= 16.
        # agent rewards dictionary
            if self.common_reward:
                rewards = float(self.reward_agg_fn(agent_rewards))
            else:
                rewards = {a: agent_rewards[a] for a in self.agents}

        # get agent observations after the environment step
        self.obs = self._get_obs()
        # print("Is done: ",self.step_count, terminated[0])
        # info dictionary
        burnt_frac = self.burnt_trees / (self.grid_size_without_walls ** 2)
        unburnt_frac = len(self.unburnt_trees) / (self.grid_size_without_walls ** 2)

        self.info = {"burnt trees": burnt_frac, "unburnt trees": unburnt_frac, }
        # self.info = {a: self.info.update({"blocked actions": blocking_agent_index.count(a)}) for a in self.agents}
        self.info = {a: self.info for a in self.agents}
        terminated = {idx: True if val==1. else False for idx, val in enumerate(terminated)}
        truncated = {idx: True if val==1. else False for idx, val in enumerate(truncated)}
        print(self.step_count)
        return self.obs, rewards, terminated, truncated, self.info

    def render(self, close=False, highlight=False, tile_size=TILE_PIXELS):
        """Render the whole-grid human view

        Parameters
        ----------
        close : bool, optional
            close the rendering window, by default False. Only applicable if render_mode is "human"
        highlight : bool, optional
            highlight the cells visible to the agent, by default False
        tile_size : int, optional
            size of each tile in pixels, by default TILE_PIXELS (defined in constants.py)

        Returns
        -------
        img : ndarray
            image of the grid
        """
        if close:
            if self.window:
                self.window.close()
            return

        if self.render_mode == "human" and not self.window:
            self.window = Window("gym_multigrid")
            self.window.show(block=False)

        if highlight:
            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {
                (i, j): [] for i in range(self.width) for j in range(self.height)
            }

            for i, a in enumerate(self.agents_storage):
                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = (
                    a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)
                )

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the grid
        if self.render_selfish_region_boundaries:
            # include selfish region boundaries in the render
            colors = [COLORS[color] for color in self.agent_colors]
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncached_object_types,
                x_min=self.selfish_xmin,
                y_min=self.selfish_ymin,
                x_max=self.selfish_xmax,
                y_max=self.selfish_ymax,
                colors=colors,
            )
        else:
            img = self.grid.render(
                tile_size,
                highlight_masks=highlight_masks if highlight else None,
                uncached_object_types=self.uncached_object_types,
            )

        # Re-render the tiles containing agents to include trees below agent
        if self.render_selfish_region_boundaries:
            # include selfish region boundaries in the render
            for a in self.agents_storage:
                img = render_agent_tiles(
                    img,
                    a,
                    self.helper_grid,
                    self.world,
                    x_min=self.selfish_xmin,
                    y_min=self.selfish_ymin,
                    x_max=self.selfish_xmax,
                    y_max=self.selfish_ymax,
                    colors=colors,
                )
        else:
            for a in self.agents_storage:
                img = render_agent_tiles(img, a, self.helper_grid, self.world)

        if self.render_mode == "human":
            self.window.show_img(img)
        
        if self.recording:
            self.frames.append(img)

        return img
