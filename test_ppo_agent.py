import numpy as np
import torch
from ppo_agent import PPOAgent

np.random.seed(42)
torch.manual_seed(42)

agent = PPOAgent()

states = np.random.rand(20, 28).astype(np.float32)
next_states = np.random.rand(20, 28).astype(np.float32)
actions = []
old_log_probs = []

# Generate actions and log probabilities using the old policy.
for state in states:
    action, log_prob, _ = agent.select_action(state)
    actions.append(action)
    old_log_probs.append(log_prob)

rewards = np.random.randn(20).astype(np.float32)
dones = np.zeros(20, dtype=np.float32)
dones[-1] = 1.0

metrics = agent.update(
    states,
    actions,
    old_log_probs,
    rewards,
    next_states,
    dones,
)

print("PPO synthetic update successful!")
print("Batch size:", len(states))
print("Actor loss:", round(metrics["actor_loss"], 5))
print("Critic loss:", round(metrics["critic_loss"], 5))
print("Mean advantage:", round(metrics["mean_advantage"], 5))
print("Action range:", min(actions), "to", max(actions))
