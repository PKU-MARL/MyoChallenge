""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

"""
This is a launcher script for launching mjrl training using hydra
"""

import os
import time as timer
import hydra
import gym
from stable_baselines3 import PPO
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import myosuite

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="hydra_npg_config", config_path="config")
def configure_jobs(job_data):
    print("========================================")
    print("Job Configuration")
    print("========================================")

    assert 'algorithm' in job_data.keys()

    print(OmegaConf.to_yaml(job_data, resolve=True))

    # env = make_vec_env(job_data.env, n_envs=job_data.num_cpu)
    env = SubprocVecEnv([make_env(job_data.env, i) for i in range(job_data.num_cpu)])

    model = None
    if job_data.algorithm == 'PPO':
        model = PPO("MlpPolicy", env, n_steps=job_data.n_steps, verbose=1)

    model.learn(total_timesteps=128000,  reset_num_timesteps=True)
    while True :
        model.learn(total_timesteps=128000,  reset_num_timesteps=False)
        path = os.path.join("outputs", job_data.job_name)
        print("saving to {}".format(path))
        # model.save(path)

        # eval
        obs = env.reset()
        average_solve = 0
        last_solve = 0
        for t in range(job_data.n_steps) :
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            solved =[x['solved'] for x in info]
            average_solve += sum(solved)/len(solved)
            last_solve = solved
        average_solve = average_solve/job_data.n_steps
        last_solve = sum(last_solve)/len(last_solve)
        print("last_solve", last_solve, "average_solve", average_solve)

if __name__ == "__main__":
    configure_jobs()
