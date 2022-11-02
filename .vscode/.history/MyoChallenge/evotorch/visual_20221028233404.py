# env_name = "myosuite:myoChallengeBaodingP2-v1"
env_name = "myosuite:myoChallengeDieReorientP2-v0"
policy_path = 'agent/policies/learned_policy_reorient_saved.pkl'
import torch
policy = torch.jit.load(policy_path)
import myosuite
import gym

# env_config = {
#         'weighted_reward_keys' : {
#             'pos_dist_1':.0,
#             'pos_dist_2':.0,
#             'solved':1.0,
#         },
# }
episode_length = 72
env_config = {
        'weighted_reward_keys' : {
            "pos_dist": 3.0,
            "rot_dist": 1.0,
            "obj_vel": 1.0,
            "drop": 2.0,
            "tip_err": 0.3,
            "solved":0.0
        }
    }
env = gym.make(
    env_name,
    **env_config,
)
# env.sim.render(mode='window')

obs_space = env.observation_space
act_space = env.action_space
print('Environment has observation space', obs_space, 'action space', act_space)
# from evotorch.neuroevolution.net.layers import reset_module_state
from evotorch.neuroevolution.net.rl import reset_env
import numpy as np
# List to track success rates
success_rates = []
# 10 episodes
for _ in range(100):
    # Reset the environment and policy
    obs = reset_env(env)
    policy = torch.jit.load(policy_path)
    # Reset the observed number of successes
    n_successes = 0
    total_reward = 0.
    last_success = 0
    length = 0
    done = False
    # Run episode to termination
    while not done:
        # Get next action
        with torch.no_grad():
            act = policy(torch.as_tensor(obs, dtype=torch.float32, device="cpu")).numpy()
        # Apply action to environment
        obs, re, done, info, = env.step(act)
        if info['solved'] :
            n_successes += 1
            if episode_length-length <= 5 :
                last_success = 1
        # Render the environment
        # env.sim.render(mode='window')
        length += 1
    print('success rate:', n_successes/episode_length, 'last success:', last_success)
    print('Episode length', length)
    success_rates.append(n_successes / episode_length)
env.close()

print('Mean success rate', np.mean(success_rates))
import myosuite
import gym
exit()
env_config = {
        'weighted_reward_keys' : {
            'pos_dist_1':.0,
            'pos_dist_2':.0,
            'act_reg': -1.,
        },
}

env = gym.make(
    env_name,
    **env_config,
)
# env.sim.render(mode='window')
from evotorch.neuroevolution.net.layers import reset_module_state
from evotorch.neuroevolution.net.rl import reset_env
import numpy as np

# List to track effort
effort = []
# 10 episodes
for _ in range(30):
    # Reset the environment and policy
    obs = reset_env(env)
    policy = torch.jit.load(policy_path)
    # Reset the observed total_effort
    total_effort = 0.
    n_steps = 0

    done = False
    # Run episode to termination
    while not done:
        # Get next action
        with torch.no_grad():
            act = policy(torch.as_tensor(obs, dtype=torch.float32, device="cpu")).numpy()
        # Apply action to environment
        obs, re, done, _, = env.step(act)
        # Render the environment
        # env.sim.render(mode='window')
        total_effort += re
        n_steps += 1
    print('Observed', total_effort, 'over', n_steps, 'steps giving average effort', total_effort/n_steps)
    effort.append(total_effort/n_steps)
env.close()

print('Mean effort', np.mean(effort))
