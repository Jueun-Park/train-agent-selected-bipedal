import csv
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.evaluation import evaluate_policy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, eval_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'model')
        self.best_mean_reward = -np.inf
        self.eval_freq = eval_freq
        self.eval_trigger = self.eval_freq

    def _init_callback(self) -> None:
        # Create folder if needed
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{self.log_dir}/eval_log.csv", "w", newline="") as f:
            pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                    self.best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}_{self.best_mean_reward:.0f}_{x[-1]}step")
                self.model.save(f"{self.save_path}_{self.best_mean_reward:.0f}_{x[-1]}step")

        return True

    def _on_rollout_end(self):
        """
        This event is triggered before updating the policy.
        """
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) == 0: return
        if x[-1] > self.eval_trigger:
            env = gym.make("BipedalWalkerHardcore-v3")
            mean, std = evaluate_policy(self.model, env, n_eval_episodes=100)
            with open(f"{self.log_dir}/eval_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([x[-1], mean, std])
            self.eval_trigger += self.eval_freq
