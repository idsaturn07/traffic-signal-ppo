import numpy as np
import torch

from traffic_env import TrafficSignalEnv
from ppo_agent import PPOAgent


def train_short_run(max_decisions=40):
    torch.manual_seed(42)
    np.random.seed(42)

    env = TrafficSignalEnv()
    agent = PPOAgent()

    states = []
    actions = []
    log_probs = []
    rewards = []
    next_states = []
    dones = []

    state = env.reset()
    episode_reward = 0.0
    decision = 0

    try:
        while decision < max_decisions:
            action, log_prob, _ = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))

            state = next_state
            episode_reward += reward
            decision += 1

            if decision % 10 == 0 or done:
                print(
                    f"decision={decision}, "
                    f"time={info['time']:.0f}s, "
                    f"action={action}, "
                    f"reward={reward:.3f}, "
                    f"queue={info['avg_queue']:.2f}, "
                    f"avg_wait={info['avg_wait']:.2f}"
                )

            # Update after collecting one batch.
            if len(states) == agent.batch_size or done:
                metrics = agent.update(
                    states,
                    actions,
                    log_probs,
                    rewards,
                    next_states,
                    dones,
                )

                print(
                    f"  PPO update | actor_loss={metrics['actor_loss']:.4f} "
                    f"| critic_loss={metrics['critic_loss']:.4f}"
                )

                states.clear()
                actions.clear()
                log_probs.clear()
                rewards.clear()
                next_states.clear()
                dones.clear()

            if done:
                break

        # Train on any remaining transitions.
        if states:
            metrics = agent.update(
                states,
                actions,
                log_probs,
                rewards,
                next_states,
                dones,
            )
            print(
                f"Final partial batch | actor_loss={metrics['actor_loss']:.4f} "
                f"| critic_loss={metrics['critic_loss']:.4f}"
            )

        print("\n--- SHORT TRAINING TEST COMPLETE ---")
        print("Decisions:", decision)
        print("Total reward:", round(episode_reward, 3))

    finally:
        env.close()


if __name__ == "__main__":
    train_short_run()
