from gymnasium.envs.registration import register
from pz_multigrid.envs.wildfire import WildfireEnv
from .utils.utils import *
# Wildfire environment
# ----------------------------------------
register(
    id="wildfire-v1",
    entry_point="pz_multigrid.envs:WildfireEnv",
)


