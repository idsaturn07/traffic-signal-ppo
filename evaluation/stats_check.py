import pandas as pd
from scipy import stats

df = pd.read_csv("results/final_robustness_raw.csv")

rows = []
for scale in sorted(df["demand_scale"].unique()):
    a = df[(df["method"] == "independent_ppo") & (df["demand_scale"] == scale)]
    b = df[(df["method"] == "neighbor_ppo") & (df["demand_scale"] == scale)]
    for metric in ["arrived", "travel_time", "queue"]:
        t, p = stats.ttest_ind(a[metric], b[metric], equal_var=False)
        rows.append({
            "demand_scale": scale,
            "metric": metric,
            "independent_mean": round(a[metric].mean(), 2),
            "neighbor_mean": round(b[metric].mean(), 2),
            "t_stat": round(t, 3),
            "p_value": round(p, 4),
            "significant_at_0.05": p < 0.05,
        })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
result.to_csv("results/ppo_variant_significance.csv", index=False)
print("\nSaved results/ppo_variant_significance.csv")
print("Note: n=5 seeds per group is a small sample -- a non-significant p-value means")
print("'not enough evidence to confidently distinguish', not 'confirmed equal'.")