from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.neuroevolution import GymNE
from evotorch.logging import StdOutLogger, PandasLogger
import torch

baoding_config ={
    "env_name": "myosuite:myoChallengeBaodingP2-v1",
    "load_policy_path": 'agent/policies/learned_policy_boading_10.pkl',
    "save_policy_path": 'agent/policies/learned_policy_boading_saved.pkl',
    "env_config": {
        'weighted_reward_keys' : {
            'dist_max':5.0,
            'solved': 2,
            'act_reg': 0.1,
            'constant': 1
            # 'vel': 0.1,
            # 'close': 10
        }
    }
}

reorient_config ={
    "env_name": "myosuite:myoChallengeDieReorientP2-v",
    "load_policy_path": 'agent/policies/learned_policy_boading_10.pkl',
    "save_policy_path": 'agent/policies/learned_policy_boading_saved.pkl',
    "env_config": {
        'weighted_reward_keys' : {
            'dist_max':5.0,
            'solved': 2,
            'act_reg': 0.1,
            'constant': 1
            # 'vel': 0.1,
            # 'close': 10
        }
    }
}

env_name = "myosuite:myoChallengeBaodingP2-v1"
load_policy_path = 'agent/policies/learned_policy_boading_10.pkl'
save_policy_path = 'agent/policies/learned_policy_boading_saved.pkl'

env_config = {
    'weighted_reward_keys' : {
        'dist_max':5.0,
        'solved': 2,
        'act_reg': 0.1,
        'constant': 1
        # 'vel': 0.1,
        # 'close': 10
    }
}

CLIPUP_MAX_SPEED = 0.15
CLIPUP_ALPHA = CLIPUP_MAX_SPEED * 0.75
RADIUS = CLIPUP_MAX_SPEED * 15

STDEV_LR = 0.1
STDEV_MAX_CHANGE = 0.2

POPSIZE = 16000
POPSIZE_MAX = POPSIZE * 8
NUM_INTERACTIONS = int(POPSIZE * 200)
NUM_GENERATIONS = 2000

from policy import Policy

problem = GymNE(
    env_name=env_name,
    env_config = env_config,
    network=Policy,
    network_args = {
        'hidden_dim': 64,
    },
    observation_normalization=True,
    num_actors='max',
)
print('Solution length is', problem.solution_length)
loaded_solution = None
if load_policy_path != None :
    loaded_policy = torch.jit.load(load_policy_path)
    loaded_solution = []
    for p in loaded_policy.parameters():
        d = p.data.view(-1)
        loaded_solution.append(d)
    loaded_solution = torch.cat(loaded_solution)
    print("Solution Loaded From:", load_policy_path)
else :
    loaded_policy = None

searcher = PGPE(
    problem,
    center_learning_rate=CLIPUP_ALPHA,
    optimizer="clipup",
    optimizer_config={"max_speed": CLIPUP_MAX_SPEED},
    radius_init=RADIUS,
    stdev_learning_rate=STDEV_LR,
    stdev_max_change=STDEV_MAX_CHANGE,
    popsize=POPSIZE,
    center_init=loaded_solution,
    popsize_max=POPSIZE_MAX,
    num_interactions=NUM_INTERACTIONS,
    distributed = True,
)

import torch

def save_policy():
    global problem, searcher, save_policy_path
    policy = problem.to_policy(searcher.status['center'])
    scripted_module = torch.jit.script(policy)
    torch.jit.save(scripted_module, save_policy_path)

searcher.after_step_hook.append(save_policy)

_ = StdOutLogger(searcher)
pandas_logger = PandasLogger(searcher)

searcher.run(NUM_GENERATIONS)

pandas_logger.to_dataframe().mean_eval.plot()
