import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        
        # Convert to VecEnv for consistency
        # if not isinstance(env, VecEnv):
        #     env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
        # self.training_env = env
        self.dt_step_target = 0
        self.total_targets = 0
        self.dt_array_target = []

    def _on_step(self) -> bool:
        # obs, reward, done, info = env.step(action)
        # dt_step_target += 1
        frac_burned = self.training_env.get_attr('burnt_trees') / (wandb.config["world_size"] ** 2)
        self.logger.record('train/burnt trees', frac_burned)
        return True
    
    def _on_training_end(self):
        pass
        #self.logger.record
