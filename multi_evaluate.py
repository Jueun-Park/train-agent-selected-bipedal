import time
from stable_baselines import ACKTR
from stable_baselines.common.evaluation import evaluate_policy
import gym
import gym_selected_bipedal
from multiprocessing import Pool

from info import mode_dict


def evaluate_agent(env_index, suffix):
    subenv_name = mode_dict[env_index]
    model_path = f"ACKTR/{subenv_name}/model_{suffix}"
    if env_index == 4:
        env = gym.make("BipedalWalkerHardcore-v3")
    else:
        env = gym.make(f"selected-bipedal-{subenv_name}-v0")
    model = ACKTR.load(model_path)
    mean, std = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"env_index {env_index}, suffix {suffix} done")
    return mean, std


if __name__ == "__main__":
    # input_list = [
    #     [0, 88],
    #     [0, 154],
    #     [0, 190],
    #     [0, 224],
    #     [1, 205],
    #     [1, 208],
    #     [1, 212],
    #     [1, 220],
    #     [2, 158],
    #     [2, 165],
    #     [2, 192],
    #     [2, 224],
    #     [3, 190],
    #     [3, 228],
    #     [3, 237],
    #     [3, 256],
    #     ]
    # input_list = [
    #     [0, 181],
    #     [0, 189],
    #     [0, 198],
    #     [2, 156],
    #     [3, 260],
    #     [3, 261],
    #     [3, 274],
    #     [3, 278]
    # ]
    input_list = [
        [4, 196],
        [4, 207],
        [4, 216],
        [4, 228]
    ]
    start_time = time.time()
    with Pool(None) as pool:
        output = pool.starmap(evaluate_agent, input_list)
    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Time processed: {minutes:.0f}m {seconds:.0f}s")
    
    with open("log.txt", "a") as f:
        for i, args in enumerate(input_list):
            print(f"{args}: {output[i]}")
            f.write(f"{args}: {output[i]}\n")
        f.write("\n")
