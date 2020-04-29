import argparse
from pathlib import Path
import tensorflow as tf
import gym
import gym_selected_bipedal

from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecEnvWrapper
from vec_monitor import VecMonitor
# from stable_baselines.common import make_vec_env
from stable_baselines.bench import Monitor

from info import mode_dict, total_timesteps, num_cpu
from save_callback import SaveOnBestTrainingRewardCallback
from make_env import make_env


parser = argparse.ArgumentParser()
parser.add_argument("--base-index", type=int)
args = parser.parse_args()
base_index = args.base_index

algo = "ACKTR"
if base_index not in mode_dict.keys():
    log_dir = f"{str(algo)}/bipedal-hardcore-load-train"
else:
    log_dir = f"{str(algo)}/{mode_dict[base_index]}"
Path(log_dir).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    if base_index not in mode_dict.keys():
        env_id = "BipedalWalkerHardcore-v3"
    else:
        env_id = f"selected-bipedal-{mode_dict[base_index]}-v0"
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecMonitor(env, filename=log_dir)

    # callback = SaveOnBestTrainingRewardCallback(
        # check_freq=1000, log_dir=log_dir)

    model = ACKTR.load("ACKTR/bipedal-hardcore/model_207.zip")
    model.set_env(env)
    model.tensorboard_log = "tensorboard_load_ACKTR"

    model.learn(
        total_timesteps=total_timesteps,
        # callback=callback,
        )

    model.save(log_dir)
