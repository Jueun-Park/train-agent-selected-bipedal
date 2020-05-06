import argparse
from pathlib import Path
import tensorflow as tf
import os
import gym
import gym_selected_bipedal

from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecEnvWrapper
from vec_monitor import VecMonitor
# from stable_baselines.common import make_vec_env
from stable_baselines.bench import Monitor

from info import *
from save_callback import SaveOnBestTrainingRewardCallback
from make_env import make_env


parser = argparse.ArgumentParser()
parser.add_argument("--base-index", type=int)
args = parser.parse_args()
base_index = args.base_index

algo = "ACKTR"
if base_index not in mode_dict.keys():
    log_dir = f"{str(algo)}/bipedal-hardcore-{expr_nickname}"
else:
    log_dir = f"{str(algo)}/{mode_dict[base_index]}"
Path(log_dir).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    if base_index not in mode_dict.keys():
        env_id = "BipedalWalkerHardcore-v3"
    else:
        env_id = f"selected-bipedal-{mode_dict[base_index]}-v0"
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    callback_log_dir = os.path.join(log_dir, f"schedule-{lr_schedule}-initlr-{init_lr}/")
    Path(callback_log_dir).mkdir(parents=True, exist_ok=True)

    env = VecMonitor(env, filename=callback_log_dir)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, eval_freq=eval_freq, log_dir=callback_log_dir)

    model = ACKTR.load("ACKTR/bipedal-hardcore/model_207.zip")  # 207, 228
    model.set_env(env)

    model.learning_rate = init_lr
    model.lr_schedule = lr_schedule

    model.tensorboard_log = f"tensorboard_ACKTR_{expr_nickname}"

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        )

    model.save(log_dir)
