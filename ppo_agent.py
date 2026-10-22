import torch
import torch.nn as nn
from torch.distributions import Categorical

from ppo_model import ActorCritic


class PPOAgent:
    def __init__(
        self,
        state_dim=28,
        action_dim=8,
        gamma=0.9,
        clip_epsilon=0.2,
        actor_lr=0.0001,
        critic_lr=0.001,
        batch_size=20,
        grad_clip=0.5,
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.model = ActorCritic(state_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(),
            lr=actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=critic_lr,
        )

    def select_action(self, state):
        return self.model.select_action(state)

    def update(self, states, actions, old_log_probs, rewards,
               next_states, dones):
        states = torch.as_tensor(__import__("numpy").asarray(states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        old_log_probs = torch.as_tensor(
            old_log_probs, dtype=torch.float32
        )
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(__import__("numpy").asarray(next_states), dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        # One-step TD target and advantage, as described in the paper.
        with torch.no_grad():
            values = self.model.critic(states).squeeze(-1)
            next_values = self.model.critic(next_states).squeeze(-1)

            td_targets = rewards + self.gamma * next_values * (1.0 - dones)
            advantages = td_targets - values

        # Current policy probabilities for the recorded actions.
        probabilities = self.model.actor(states)
        distribution = Categorical(probabilities)
        new_log_probs = distribution.log_prob(actions)

        ratios = torch.exp(new_log_probs - old_log_probs)

        unclipped = ratios * advantages.detach()
        clipped_ratios = torch.clamp(
            ratios,
            1.0 - self.clip_epsilon,
            1.0 + self.clip_epsilon,
        )
        clipped = clipped_ratios * advantages.detach()

        # Minimize the negative clipped surrogate objective.
        actor_loss = -torch.minimum(unclipped, clipped).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.actor.parameters(), self.grad_clip
        )
        self.actor_optimizer.step()

        # Critic minimizes the squared one-step TD error.
        predicted_values = self.model.critic(states).squeeze(-1)
        critic_loss = torch.mean(
            (td_targets.detach() - predicted_values) ** 2
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.critic.parameters(), self.grad_clip
        )
        self.critic_optimizer.step()

        # Prepare the old policy for the next data-collection round.
        self.model.synchronize_old_actor()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_advantage": advantages.mean().item(),
        }

