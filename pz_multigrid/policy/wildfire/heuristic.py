from typing import TypedDict, Type
import numpy as np
from numpy.typing import NDArray
from gym_multigrid.core.agent import ActionsT, WildfireActions
from gym_multigrid.core.world import WorldT, WildfireWorld
from gym_multigrid.policy.ctf.typing import ObservationDict
from gym_multigrid.typing import Position
from gym_multigrid.policy.base import BaseAgentPolicy
from gym_multigrid.policy.ctf.heuristic import CtfPolicyT, DestinationPolicy



class RwPolicy(BaseAgentPolicy):
    """
    Random walk policy
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "rw"

    def act(self, observation: TypedDict, actions: ActionsT) -> int:
        return np.random.choice(list(actions))
    

class AdjFirePolicy(DestinationPolicy):
    """
    Random adjacent fire policy
    """

    def __init__(
        self,
        fire_range: int,
        field_map: NDArray | None = None,
        action_set: ActionsT = WildfireActions,
        random_generator: np.random.Generator | None = None,
        randomness: float = 0.25,
        world: WorldT = WildfireWorld,
        avoided_objects: list[str] = ["wall"],
    ) -> None:
        """
        Initialize the policy.

        Parameters
        ----------
        field_map : numpy.typing.NDArray | None = None
            Field map of the environment.
            Make sure to set it to the field map of the environment.
        actions : gym_multigrid.core.agent.ActionsT = WildfireActions
            Actions available to the agent.
        random_generator : numpy.random.Generator | None = None
            Random number generator. Replace it with the environment's random number generator if needed.
        randomness : float
            Probability of taking an random action instead of an optimal action.
        world : gym_multigrid.core.world.WorldT = CtfWorld
            World object where the policy is applied, and it should be set to the environment's world object.
        avoided_objects : list[str] = ["obstacle", "red_agent", "blue_agent"]
            List of objects to avoid in the path.
            The object names should match with those in the environment's world object.
        """
        super().__init__(
                         field_map,
                         action_set, 
                         random_generator,
                         randomness,
                         world,
                         avoided_objects,
                         )
        self.fire_range = fire_range
        
    def act_randomly(self):
        return np.random.choice(list(self.action_set))
        

    def get_target(self, observation: ObservationDict, curr_pos: Position) -> Position:
        """
        Get the target position of the agent.

        Parameters
        ----------
        observation : ObservationDict
            Observation dictionary (dict from the env).

        Returns
        -------
        Position
            Target position of the agent.
        """
        
        return (
            closest_area_pos(curr_pos, fire_pos)
            if is_fire_in_ego_territory
            else super().get_target(observation, curr_pos)
        )
        


HEURISTIC_POLICIES: dict[str, Type[BaseAgentPolicy]] = {
    "rw": RwPolicy,
    "adjfire": AdjFirePolicy,
}