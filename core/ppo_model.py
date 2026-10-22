import copy
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Three-layer fully connected policy network."""

    def __init__(self, state_dim=28, action_dim=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)


class ValueNetwork(nn.Module):
    """Critic network estimating the value of a traffic state."""

    def __init__(self, state_dim=28):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.network(state)


class ActorCritic(nn.Module):
    def __init__(self, state_dim=28, action_dim=8):
        super().__init__()

        self.actor = PolicyNetwork(state_dim, action_dim)
        self.old_actor = copy.deepcopy(self.actor)
        self.critic = ValueNetwork(state_dim)

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.reshape(1, -1)

        with torch.no_grad():
            probabilities = self.old_actor(state)
            value = self.critic(state)

            distribution = Categorical(probabilities)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)

        return (
            action.item(),
            log_probability.item(),
            value.item(),
        )

    def synchronize_old_actor(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
