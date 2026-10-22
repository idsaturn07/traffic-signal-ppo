import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results/per_intersection_final_queue.csv")
melted = df.melt(id_vars=["method", "demand_scale", "seed"], value_vars=["I1", "I2", "I3"],
                  var_name="junction", value_name="final_queue")

summary = melted.groupby(["method", "demand_scale", "junction"]).agg(
    mean_queue=("final_queue", "mean"),
    std_queue=("final_queue", "std"),
).reset_index()

print(summary.to_string(index=False))
summary.to_csv("results/per_intersection_summary.csv", index=False)

sub = summary[summary["demand_scale"] == 1.4]
methods = ["fixed_time", "max_pressure", "independent_ppo", "neighbor_ppo"]
junctions = ["I1", "I2", "I3"]

x = np.arange(len(junctions))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 5.5))
for i, method in enumerate(methods):
    vals, errs = [], []
    for j in junctions:
        row = sub[(sub["method"] == method) & (sub["junction"] == j)]
        vals.append(row["mean_queue"].values[0] if not row.empty else 0)
        errs.append(row["std_queue"].values[0] if not row.empty else 0)
    ax.bar(x + i * width, vals, width, yerr=errs, capsize=3, label=method)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(junctions)
ax.set_ylabel("Final queue length")
ax.set_title("Per-intersection final queue at 1.4x demand (surge)")
ax.legend()
plt.tight_layout()
plt.savefig("results/per_intersection_1.4x.png", dpi=150)
print("\nSaved results/per_intersection_1.4x.png")
print("Saved results/per_intersection_summary.csv")