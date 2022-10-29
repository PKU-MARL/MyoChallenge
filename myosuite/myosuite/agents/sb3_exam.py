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
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
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
@hydra.main(config_name="hydra_myo_config_local", config_path="config")
def configure_jobs(job_data):
    print("========================================")
    print("Job Configuration")
    print("========================================")

    assert 'algorithm' in job_data.keys()

    print(OmegaConf.to_yaml(job_data, resolve=True))

    env = gym.make(job_data.env)
    # env = make_vec_env(job_data.env, 1)

    model = None
    if job_data.algorithm == 'PPO':
        model = PPO.load(job_data.model_path)
    elif job_data.algorithm == 'RecurrentPPO':
        model = RecurrentPPO.load(job_data.model_path)
    elif job_data.algorithm == 'SAC':
        model = SAC.load(job_data.model_path)
    # for key, val in model.get_parameters().items() :
    #    print(val)
    # exit()

    # env = model.get_env()
    # env.mujoco_render_frames = True
    # cell and hidden state of the LSTM
    # lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)
    for i in range(10) :
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            # action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            env.mj_render()
            episode_starts = dones
            if dones :
                break
        # env.render()

if __name__ == "__main__":
    configure_jobs()
