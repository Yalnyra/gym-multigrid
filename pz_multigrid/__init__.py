from gymnasium.envs.registration import register
from pz_multigrid.envs.wildfire import WildfireEnv
# Wildfire environment
# ----------------------------------------
register(
    id="wildfire-v",
    entry_point="pz_multigrid.envs:WildfireEnv",
)


