import glob
import pandas as pd
import matplotlib.pyplot as plt

baseline_files = (
    glob.glob("results/results_fixed_time_seed*.csv")
    + glob.glob("results/results_max_pressure_seed*.csv")
    + glob.glob("results/results_fixed_time_scale*_seed*.csv")
    + glob.glob("results/results_max_pressure_scale*_seed*.csv")
)

baseline_df = pd.concat([pd.read_csv(f) for f in baseline_files], ignore_index=True)

baseline_final = (
    baseline_df.sort_values("time")
    .groupby(["method", "seed", "demand_scale"])
    .tail(1)
    .rename(columns={
        "mean_completed_travel_time": "travel_time",
        "total_queue": "queue",
    })[["method", "demand_scale", "seed", "arrived", "travel_time", "queue"]]
)

ppo_df = pd.read_csv("results/demand_shift_ppo.csv").rename(columns={
    "mean_travel_time": "travel_time",
    "final_queue": "queue",
})[["method", "demand_scale", "seed", "arrived", "travel_time", "queue"]]

combined = pd.concat([baseline_final, ppo_df], ignore_index=True)

summary = combined.groupby(["method", "demand_scale"]).agg(
    arrived_mean=("arrived", "mean"),
    arrived_std=("arrived", "std"),
    travel_time_mean=("travel_time", "mean"),
    travel_time_std=("travel_time", "std"),
    queue_mean=("queue", "mean"),
    queue_std=("queue", "std"),
).reset_index().sort_values(["demand_scale", "method"])

print("\n=== FINAL robustness summary: all methods x all demand scales ===")
print(summary.to_string(index=False))

summary.to_csv("results/final_robustness_summary.csv", index=False)
combined.to_csv("results/final_robustness_raw.csv", index=False)

method_order = ["fixed_time", "max_pressure", "independent_ppo", "neighbor_ppo"]
markers = {"fixed_time": "s", "max_pressure": "^", "independent_ppo": "o", "neighbor_ppo": "D"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for method in method_order:
    sub = summary[summary["method"] == method].sort_values("demand_scale")
    if sub.empty:
        continue
    axes[0].errorbar(sub["demand_scale"], sub["arrived_mean"], yerr=sub["arrived_std"],
                      marker=markers[method], capsize=4, label=method)
    axes[1].errorbar(sub["demand_scale"], sub["queue_mean"], yerr=sub["queue_std"],
                      marker=markers[method], capsize=4, label=method)

axes[0].set_xlabel("Demand scale")
axes[0].set_ylabel("Vehicles arrived")
axes[0].set_title("Throughput vs demand scale (all methods)")
axes[0].legend()

axes[1].set_xlabel("Demand scale")
axes[1].set_ylabel("Final queue length")
axes[1].set_title("Leftover queue vs demand scale (all methods)")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/final_robustness_comparison.png", dpi=150)
print("\nPlot saved to results/final_robustness_comparison.png")
print("Full table saved to results/final_robustness_summary.csv")