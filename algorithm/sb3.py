# Stable baselines 3
from sbx import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from wandb.integration.sb3 import WandbCallback
from omegaconf import DictConfig
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
        info = self.locals.get('info')
        if info is not None:
            frac_burned = info[0]['burnt_trees']
            frac_unburned = info[0]['unburnt_trees']
            self.logger.record('train/burnt trees', frac_burned)
            self.logger.record('train/unburnt trees', frac_unburned)
        return True
    
    def _on_training_end(self):
        pass
        #self.logger.record

def model_PPO(env, config:DictConfig):
    nn_t = config['encoder_config']['hidden_size']
    policy_kwargs = dict(net_arch=dict(pi=nn_t, vf=nn_t))
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        policy_kwargs=policy_kwargs,
        **config['init_hp']
    )

def setup_callbacks(eval_env, config:DictConfig):
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f".{config['model_save_path']}/checkpoints/",
        name_prefix=f"{config['run_id']}",
    )
    tensorboard_callback = TensorboardCallback()
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=tensorboard_callback,
        best_model_save_path="./logs/",
        deterministic=True,
        log_path="./logs/",
        eval_freq=5000,
        verbose=2,
    )
    callbacks = [checkpoint_callback, eval_callback, tensorboard_callback]
    if config['wandb']['enabled']:
        callbacks.append(WandbCallback(
        model_save_freq=10000, model_save_path=f"{config['model_save_path']}"
    ))
    return callbacks

def train_sb3(model, eval_env, config:DictConfig):

    model.learn(
        total_timesteps=config['train_epochs'],
        tb_log_name=f"{config['run_id']}",
        callback=setup_callbacks(eval_env, config),
        progress_bar=True,
    )
    model_suffix = f"_0_{config['train_epochs']}"
    path = f"{config['model_save_path']}{model_suffix}"
    model.save(path)