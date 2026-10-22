import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/demand_shift_ppo.csv")

summary = df.groupby(["method", "demand_scale"]).agg(
    arrived_mean=("arrived", "mean"),
    arrived_std=("arrived", "std"),
    travel_time_mean=("mean_travel_time", "mean"),
    travel_time_std=("mean_travel_time", "std"),
    queue_mean=("final_queue", "mean"),
    queue_std=("final_queue", "std"),
).reset_index()

print("\n=== PPO demand-shift summary (mean +/- std across 5 seeds) ===")
print(summary.to_string(index=False))

summary.to_csv("results/demand_shift_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for method in df["method"].unique():
    sub = summary[summary["method"] == method].sort_values("demand_scale")
    axes[0].errorbar(sub["demand_scale"], sub["arrived_mean"], yerr=sub["arrived_std"],
                      marker="o", capsize=4, label=method)
    axes[1].errorbar(sub["demand_scale"], sub["queue_mean"], yerr=sub["queue_std"],
                      marker="o", capsize=4, label=method)

axes[0].set_xlabel("Demand scale")
axes[0].set_ylabel("Vehicles arrived")
axes[0].set_title("Throughput vs demand scale")
axes[0].legend()

axes[1].set_xlabel("Demand scale")
axes[1].set_ylabel("Final queue length")
axes[1].set_title("Leftover queue vs demand scale")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/demand_shift_comparison.png", dpi=150)
print("\nPlot saved to results/demand_shift_comparison.png")
print("Table saved to results/demand_shift_summary.csv")