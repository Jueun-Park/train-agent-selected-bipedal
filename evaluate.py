import time
from stable_baselines import ACKTR
from stable_baselines.common.evaluation import evaluate_policy
import gym
import gym_selected_bipedal

from info import mode_dict


index = 2
subenv_name = mode_dict[index]
suffix = 271
model_path = f"ACKTR/{subenv_name}/model_{suffix}"
# model_path = "ACKTR/grass.zip"
env = gym.make(f"selected-bipedal-{subenv_name}-v0")
model = ACKTR.load(model_path)
start_time = time.time()
mean, std = evaluate_policy(model, env, n_eval_episodes=100)
print(time.time() - start_time)
print(mean, std)
