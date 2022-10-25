import torch
from torch import nn


class Policy(nn.Module):
    def __init__(
        self,
        obs_length: int,
        act_length: int,
        hidden_dim: int = 64,
        obs_space: dict = {},
    ):
        """General policy network is a 2 layer MLP with a recurrent connection from the second layer to the first.
        The recurrent connection passes through another 1-layer MLP.
        The activation is LeakyReLU, and layer normalization is used after every hidden layer.
        Args:
            obs_length (int): The length of the observation vector
            act_length (int): The length of the expected action layer
            hidden_dim (int): The hidden dimension of the policy, used for all hidden layers.
            obs_space (dict): The observation space of the environment -- ignored for this example
        """
        super().__init__()

        # First layer of MLP takes the observation vector and the recurrent hidden vector
        self.l1 = nn.Linear(obs_length + hidden_dim, hidden_dim)
        # Second layer of MLP takes the output of the first layer
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        # Recurrent connection takes the output of the second layer and maps back to the first
        self.rec = nn.Linear(hidden_dim, hidden_dim)
        # Final linear layer maps the output of the second layer to the action space
        self.l3 = nn.Linear(hidden_dim, act_length)

        # Layer normalization is applied after every hidden layer e.g. self.l1, self.l2, self.rec
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # LeakyReLU activation is applied after every hidden layer e.g. self.l1, self.l2, self.rec
        self.act = nn.LeakyReLU()

        # The hidden state is initialized as zeros
        self._hidden_state = torch.zeros(hidden_dim)

    def reset(self):
        """Reset is called every time the environment is also reset, so that hidden states don't persist
        between episodes and individual solution evaluations.
        """
        # Reset simply by zeroing the internal hidden state
        self._hidden_state.zero_()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward step of policy produces actions from observations"""
        # First, the observations are concatenated with the internal hidden state to pass to the first layer
        x = torch.cat([obs, self._hidden_state], dim=-1)
        # The first layer is applied with layer normalization and activation
        x = self.ln(self.act(self.l1(x)))
        # The second layer is applied with layer normalization and activation
        x = self.ln(self.act(self.l2(x)))
        # The internal hidden state is updated using self.rec
        self._hidden_state = self.ln(self.act(self.rec(x)))
        # The actions are computed from the output of the second layer
        act = self.l3(x)
        return act
