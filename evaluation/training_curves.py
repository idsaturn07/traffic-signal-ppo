import pandas as pd
import matplotlib.pyplot as plt

def smooth(series, window=10):
    return series.rolling(window=window, min_periods=1).mean()

indep = pd.read_csv("results/training_log.csv")
neigh = pd.read_csv("results/training_log_neighbor.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(indep["episode"], indep["total_reward"], alpha=0.25, color="tab:blue")
axes[0].plot(indep["episode"], smooth(indep["total_reward"]), color="tab:blue", label="Independent PPO (smoothed)")
axes[0].plot(neigh["episode"], neigh["total_reward"], alpha=0.25, color="tab:orange")
axes[0].plot(neigh["episode"], smooth(neigh["total_reward"]), color="tab:orange", label="Neighbor-aware PPO (smoothed)")
axes[0].set_xlabel("Training episode")
axes[0].set_ylabel("Total reward (sum over 3 agents)")
axes[0].set_title("Training reward over time")
axes[0].legend()

axes[1].plot(indep["episode"], indep["arrived"], alpha=0.25, color="tab:blue")
axes[1].plot(indep["episode"], smooth(indep["arrived"]), color="tab:blue", label="Independent PPO (smoothed)")
axes[1].plot(neigh["episode"], neigh["arrived"], alpha=0.25, color="tab:orange")
axes[1].plot(neigh["episode"], smooth(neigh["arrived"]), color="tab:orange", label="Neighbor-aware PPO (smoothed)")
axes[1].set_xlabel("Training episode")
axes[1].set_ylabel("Vehicles arrived per episode")
axes[1].set_title("Throughput over training")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
print("Saved results/training_curves.png")