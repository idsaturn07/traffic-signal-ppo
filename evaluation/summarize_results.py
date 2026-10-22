import glob
import pandas as pd
import matplotlib.pyplot as plt

rows = []
for f in glob.glob("results/results_*_seed*.csv"):
    df = pd.read_csv(f)
    rows.append(df)
all_data = pd.concat(rows, ignore_index=True)

final_rows = (
    all_data.sort_values("time")
    .groupby(["method", "seed"])
    .tail(1)
)

summary = final_rows.groupby("method").agg(
    arrived_mean=("arrived", "mean"),
    arrived_std=("arrived", "std"),
    travel_time_mean=("mean_completed_travel_time", "mean"),
    travel_time_std=("mean_completed_travel_time", "std"),
    queue_mean=("total_queue", "mean"),
).round(2)

print("\n=== Final-state summary (mean +/- std across 5 seeds) ===")
print(summary)
summary.to_csv("results/summary_table.csv")

avg_over_time = all_data.groupby(["method", "time"]).agg(
    total_queue=("total_queue", "mean"),
    arrived=("arrived", "mean"),
).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

for method, grp in avg_over_time.groupby("method"):
    axes[0].plot(grp["time"], grp["total_queue"], label=method)
    axes[1].plot(grp["time"], grp["arrived"], label=method)

axes[0].set_ylabel("Total queue length")
axes[0].set_title("Queue Length Over Time (mean of 5 seeds)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel("Cumulative vehicles arrived")
axes[1].set_xlabel("Simulation time (s)")
axes[1].set_title("Cumulative Completed Vehicles Over Time (mean of 5 seeds)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/baseline_comparison.png", dpi=150)
print("\nPlot saved to results/baseline_comparison.png")
print("Table saved to results/summary_table.csv")