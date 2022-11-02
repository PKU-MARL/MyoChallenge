import numpy as np
from mjrl.utils.fc_network import FCNetwork
from mjrl.policies.gaussian_mlp import MLP
import torch
from torch.autograd import Variable



class MyMLP:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 anchor_dims=[0],
                 anchors=[[0]],
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        self.num_anchors = len(anchors)
        self.num_anchor_dims = len(anchor_dims)
        self.anchors = np.array(anchors)
        self.anchor_dims = np.array(anchor_dims)

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # create sub networks
        self.models = [MLP(env_spec, hidden_sizes, min_log_std, init_log_std, seed=seed) for i in range(self.num_anchors)]
        self.trainable_params = []
        for model in self.models :
            self.trainable_params += model.trainable_params
        self.old_models = [MLP(env_spec, hidden_sizes, min_log_std, init_log_std, seed=seed) for i in range(self.num_anchors)]
        self.old_params = []
        for old_model in self.old_models :
            self.old_params += model.trainable_params
        
    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data

    def get_model(self, observation) :
        o = np.float32(observation.reshape(-1))
        feature = o[self.anchor_dims]
        dist = np.linalg.norm(feature.reshape(1, self.num_anchor_dims) - self.anchors, axis=-1)
        dim = np.argmin(dist)
        return self.models[dim]

    # Main functions
    # ============================================
    def get_action(self, observation):
        model = self.get_model(observation)
        return model.get_action(observation)

    def mean_LL(self, observations, actions, model=None, log_std=None):

        model = self.get_model(observations)
        return model.mean_LL(observations)

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)
