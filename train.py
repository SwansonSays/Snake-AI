<<<<<<< HEAD
from stable_baselines3 import PPO
import os
from snake_env import SnakeEnv
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_food_count = 0
        self.total_attempts = 0
        self.cur_food_count = 0
        self.cur_attempts = 0

    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self):
        self.total_attempts = sum(vec_env.get_attr("attempts"))
        self.total_food_count = sum(vec_env.get_attr("total_food_count"))
        self.cur_attempts = self.total_attempts - self.cur_attempts
        self.cur_food_count = self.total_food_count - self.cur_food_count
        
        if self.total_attempts > 0:
            self.food_count_mean = self.cur_food_count / self.cur_attempts
            self.total_food_count_mean = self.total_food_count / self.total_attempts
            self.logger.record("food_count_mean", self.food_count_mean)
            self.logger.record("total_food_count_mean", self.total_food_count_mean)

            print("Total Attempts:", self.total_attempts, "| Total Food Count:", self.total_food_count, "| Mean:", self.total_food_count_mean)
            print("Rollout Attempts:", self.cur_attempts, "| Rollout Food Count:", self.cur_food_count, "| Mean:", self.food_count_mean)

        self.cur_attempts = self.total_attempts
        self.cur_food_count = self.total_food_count

TIMESTEPS = 10000
iters = 0
n_envs = 10

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": TIMESTEPS*n_envs,
    "env_name": "SnakeEnv-v0",
}

run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,
    save_code=True,
)

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


vec_env = make_vec_env(SnakeEnv, n_envs=n_envs)

model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=logdir)
#model = PPO.load("models/1704595409/15310000.zip", vec_env, tensorboard_log=logdir)

while True:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS*n_envs, 
        reset_num_timesteps=False, 
        tb_log_name=f"PPO", 
        callback=[
            TensorboardCallback(), 
            WandbCallback(
                verbose=2, 
                gradient_save_freq=100,
                model_save_path=models_dir,
                model_save_freq=100,
            )
        ]
    )
    print("Learn ended: Saving Model")
    model.save(f"{models_dir}/{TIMESTEPS*iters*n_envs}")
=======
from stable_baselines3 import PPO
import os
from snake_env import SnakeEnv
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_food_count = 0
        self.total_attempts = 0
        self.cur_food_count = 0
        self.cur_attempts = 0

    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self):
        self.total_attempts = sum(vec_env.get_attr("attempts"))
        self.total_food_count = sum(vec_env.get_attr("total_food_count"))
        self.cur_attempts = self.total_attempts - self.cur_attempts
        self.cur_food_count = self.total_food_count - self.cur_food_count
        
        if self.total_attempts > 0:
            self.food_count_mean = self.cur_food_count / self.cur_attempts
            self.total_food_count_mean = self.total_food_count / self.total_attempts
            self.logger.record("food_count_mean", self.food_count_mean)
            self.logger.record("total_food_count_mean", self.total_food_count_mean)

            print("Total Attempts:", self.total_attempts, "| Total Food Count:", self.total_food_count, "| Mean:", self.total_food_count_mean)
            print("Rollout Attempts:", self.cur_attempts, "| Rollout Food Count:", self.cur_food_count, "| Mean:", self.food_count_mean)

        self.cur_attempts = self.total_attempts
        self.cur_food_count = self.total_food_count

TIMESTEPS = 10000
iters = 0
n_envs = 10

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": TIMESTEPS*n_envs,
    "env_name": "SnakeEnv-v0",
}

run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,
    save_code=True,
)

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


vec_env = make_vec_env(SnakeEnv, n_envs=n_envs)

model = PPO('MlpPolicy', vec_env, verbose=1, tensorboard_log=logdir)
#model = PPO.load("models/1704595409/15310000.zip", vec_env, tensorboard_log=logdir)

while True:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS*n_envs, 
        reset_num_timesteps=False, 
        tb_log_name=f"PPO", 
        callback=[
            TensorboardCallback(), 
            WandbCallback(
                verbose=2, 
                gradient_save_freq=100,
                model_save_path=models_dir,
                model_save_freq=100,
            )
        ]
    )
    print("Learn ended: Saving Model")
    model.save(f"{models_dir}/{TIMESTEPS*iters*n_envs}")
>>>>>>> 83517401dc00beacb07bef29d54f504856276924
