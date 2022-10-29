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
import numpy as np
import gym
from stable_baselines3 import PPO, SAC
from torch.utils.tensorboard import SummaryWriter
from sb3_contrib import RecurrentPPO
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecMonitor
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

def curriculum_manager(env, steps, curriculum) :

    remain_steps = steps
    for cur in curriculum :
        remain_steps -= cur["total_steps"]
        if remain_steps < 0 :
            for key, value in cur["dr_parameters"].items() :
                if isinstance(value, str) :
                    value = eval(value)
                cur["dr_parameters"][key] = value
            env.env_method("update_dr", **cur["dr_parameters"])
            return cur["dr_parameters"]["max_episode_steps"]

    exit("All curriculum finished")

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

    # env = make_vec_env(job_data.env, n_envs=job_data.num_cpu)
    env = SubprocVecEnv([make_env(job_data.env, i) for i in range(job_data.num_cpu)])
    env = VecMonitor(env)

    model = None
    if job_data.algorithm == 'PPO':
        model = PPO("MlpPolicy", env, n_steps=job_data.n_steps, verbose=1)
    elif job_data.algorithm == 'RecurrentPPO':
        model = RecurrentPPO(job_data.policy, env, n_steps=job_data.n_steps, verbose=1, policy_kwargs=eval(job_data.policy_kwargs))
    elif job_data.algorithm == 'SAC' :
        model = SAC(
            job_data.policy,
            env,
            train_freq=job_data.train_freq,
            learning_rate=job_data.learning_rate,
            gradient_steps=job_data.gradient_steps,
            batch_size=job_data.batch_size,
            verbose=1,
            policy_kwargs=eval(job_data.policy_kwargs)
        )


    writer = SummaryWriter()

    curriculum = job_data.curriculum
    curriculum_manager(env, 0, curriculum)
    model.learn(total_timesteps=32000,  reset_num_timesteps=True)
    i = 0
    while True :
        i += 32000

        curriculum_manager(env, i, curriculum)

        # eval
        obs = env.reset()
        solves = np.zeros((job_data.num_cpu,))
        sum_reward = 0
        for t in range(job_data.n_steps) :
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            solved =[x['solved'] for x in info]
            solves += np.array(solved)
            sum_reward += sum(rewards)/len(rewards)
        average_reward = sum_reward/job_data.n_steps
        ge5_solve = sum(solves >= 5)/job_data.num_cpu
        avg_solve = sum(solves)/job_data.num_cpu/job_data.n_steps
        print("average_reward", average_reward, "[>=5]:", ge5_solve, "avg:", avg_solve)
        writer.add_scalar('Eval/Average_Reward', average_reward, i)
        writer.add_scalar('Eval/Success_Reorientation', ge5_solve, i)
        writer.add_scalar('Eval/Success_Baoding', avg_solve, i)

        path = os.path.join("outputs", job_data.job_name, str(i))
        print("saving to {}".format(path))
        model.save(path)

        model.learn(total_timesteps=32000,  reset_num_timesteps=False)

if __name__ == "__main__":
    configure_jobs()
