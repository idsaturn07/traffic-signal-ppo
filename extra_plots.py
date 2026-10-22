import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

summary = pd.read_csv("results/final_robustness_summary.csv")
methods = ["fixed_time", "max_pressure", "independent_ppo", "neighbor_ppo"]
labels = ["Fixed-Time", "Max-Pressure", "IPPO-local", "IPPO-neighbor"]

# --- Plot 1: grouped bar comparison at normal (1.0x) demand, 3 metrics side by side ---
sub = summary[summary["demand_scale"] == 1.0].set_index("method").loc[methods]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = [
    ("arrived_mean", "arrived_std", "Vehicles arrived"),
    ("travel_time_mean", "travel_time_std", "Mean travel time (s)"),
    ("queue_mean", "queue_std", "Final queue length"),
]
for ax, (mcol, scol, title) in zip(axes, metrics):
    ax.bar(labels, sub[mcol], yerr=sub[scol], capsize=4, color=["#999999", "#4C72B0", "#55A868", "#C44E52"])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)

plt.suptitle("All four methods at normal (1.0x) demand")
plt.tight_layout()
plt.savefig("results/intro_comparison_1.0x.png", dpi=150)
print("Saved results/intro_comparison_1.0x.png")

# --- Plot 2: travel time vs demand scale, all 4 methods (completes the robustness story) ---
fig, ax = plt.subplots(figsize=(8, 5.5))
markers = {"fixed_time": "s", "max_pressure": "^", "independent_ppo": "o", "neighbor_ppo": "D"}
for method, label in zip(methods, labels):
    sub2 = summary[summary["method"] == method].sort_values("demand_scale")
    ax.errorbar(sub2["demand_scale"], sub2["travel_time_mean"], yerr=sub2["travel_time_std"],
                marker=markers[method], capsize=4, label=label)
ax.set_xlabel("Demand scale")
ax.set_ylabel("Mean travel time (s)")
ax.set_title("Travel time vs demand scale (all methods)")
ax.legend()
plt.tight_layout()
plt.savefig("results/travel_time_vs_demand.png", dpi=150)
print("Saved results/travel_time_vs_demand.png")