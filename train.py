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
    log_dir = f"{str(algo)}/bipedal-hardcore-save-lr"
else:
    log_dir = f"{str(algo)}/{mode_dict[base_index]}"
Path(log_dir).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    if base_index not in list(range(4)):
        env_id = "BipedalWalkerHardcore-v3"
    else:
        env_id = f"selected-bipedal-{mode_dict[base_index]}-v0"
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecMonitor(env, filename=log_dir)

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=log_dir)

    model = ACKTR(
        MlpPolicy,
        env,
        verbose=1,
        tensorboard_log=f"./tensorboard_{algo}_save_lr/",
        policy_kwargs={
            'layers': [64, 256, 256, 64],
            'act_fun': tf.nn.relu
            },
        lr_schedule="double_linear_con",
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        )

    model.save(log_dir)
