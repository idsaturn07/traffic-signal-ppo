# Neighbor-Aware Reward Sharing for Independent PPO Traffic Signal Control

Deep learning course project — adaptive traffic signal control for a 3-intersection
corridor (I1–I2–I3) in SUMO, comparing Fixed-Time and Max-Pressure baselines against
two Independent PPO (IPPO) variants that differ only in their reward signal:

- **IPPO-local (α = 1.0)** — each agent is rewarded purely for its own intersection.
- **IPPO-neighbor (α = 0.5)** — each agent's reward is 50% local, 50% the average of
  its neighbors' rewards, with zero change to the network architecture.

**Research question:** does this zero-architecture-change, reward-only coordination
improve network-wide performance over local-reward IPPO, and does any improvement
survive a change in traffic demand after training (agents are trained once at 1.0×
demand, frozen, then evaluated at 0.6×, 1.0× and 1.4×)?

## Requirements

- Python 3.10 (a virtual environment is strongly recommended — see Setup)
- [SUMO](https://sumo.dlr.de/docs/Installation.html) with `traci` on your `PYTHONPATH`
  (SUMO's own Python bindings — installing SUMO and setting the `SUMO_HOME` environment
  variable is normally enough)
- Python packages: `torch`, `numpy`, `pandas`, `matplotlib`, `scipy` (for the
  significance tests) — see `requirements.txt`

## Setup (Windows / PowerShell)

```powershell
cd D:\Desktop\traffic_signal_ppo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Confirm SUMO/TraCI is reachable before doing anything else:

```powershell
python -c "import traci; print('traci OK')"
```

## Project structure

```
traffic_signal_ppo/
├── sumo/
│   └── research/
│       ├── research.sumocfg        # SUMO config for the 3-intersection corridor
│       └── ...                     # network, route and additional files
├── train_independent_ppo.py        # trains IPPO-local (alpha = 1.0)
├── train_neighbor_ppo.py           # trains IPPO-neighbor (alpha = 0.5)
├── research_experiment.py          # Fixed-Time / Max-Pressure baseline runner (single demand scale, 1.0x)
├── research_experiment_demand_shift.py  # same baselines, run across 0.6x / 1.0x / 1.4x demand
├── evaluate_per_intersection.py    # evaluates all 4 methods, logs true end-of-episode state, per-intersection queues
├── evaluate_demand_shift.py        # trains/evaluates IPPO-local and IPPO-neighbor across demand scales
├── final_summary.py                # aggregates every results/*.csv into the final robustness table + plot
├── stats_check.py                  # Welch's t-tests, IPPO-local vs IPPO-neighbor
├── results/                        # all CSV outputs, trained model weights (.pt) and generated PNGs land here
├── PPO_Traffic_Signal_Control_Report.pdf  # final project report
└── README.md
```

## Running the baseline evaluations

**1. Non-learning baselines (Fixed-Time, Max-Pressure) at training-level demand (1.0×):**

```powershell
foreach ($method in "fixed_time","max_pressure") {
  foreach ($seed in 1..5) {
    python research_experiment.py --method $method --demand-scale 1.0 --seed $seed
  }
}
```

**2. Non-learning baselines across all three demand scales (for the robustness table):**

```powershell
foreach ($method in "fixed_time","max_pressure") {
  foreach ($scale in 0.6,1.0,1.4) {
    foreach ($seed in 1..5) {
      python research_experiment_demand_shift.py --method $method --demand-scale $scale --seed $seed
    }
  }
}
```

This produces `results/results_{method}_seed{seed}.csv` (1.0× only) and
`results/results_{method}_scale{scale}_seed{seed}.csv` (all three scales) — 30 runs
total (2 methods × 3 scales × 5 seeds).

**3. Train the PPO agents:**

```powershell
python train_independent_ppo.py
python train_neighbor_ppo.py
```

**4. Evaluate the trained PPO agents (per-intersection breakdown, all 4 methods):**

```powershell
python evaluate_per_intersection.py
```

**5. Evaluate the trained PPO agents across demand scales (IPPO-local / IPPO-neighbor):**

```powershell
python evaluate_demand_shift.py
```

Both evaluators measure the simulation's **true final-step state** when the 600-second
episode loop exits — see the note on `research_experiment.py`'s logging interval below.

## Aggregating results

Once all of the above have run, build the final robustness table and comparison plot:

```powershell
python final_summary.py
```

This reads every `results/results_fixed_time*.csv`, `results/results_max_pressure*.csv`
and `results/demand_shift_ppo.csv`, takes the last logged row per `(method, seed,
demand_scale)` group, and writes:

- `results/final_robustness_summary.csv` — mean ± std of arrived vehicles, mean
  completed travel time and final queue, per method per demand scale (this is the
  table reported in the project report, Section 6.3)
- `results/final_robustness_raw.csv` — the same data before aggregation
- `results/final_robustness_comparison.png` — the comparison plot

For the statistical-significance check between IPPO-local and IPPO-neighbor:

```powershell
python stats_check.py
```

## Important configuration notes

- **SUMO teleportation is disabled.** `sumo/research/research.sumocfg` sets
  `<time-to-teleport value="-1"/>`, so SUMO never teleports gridlocked vehicles out of
  the simulation. A teleport count of 0 in any evaluation run is therefore a property
  of this config, **not** evidence that no vehicle was ever gridlocked — treat it as a
  sanity-check footnote, not a robustness claim.
- **Fixed-interval logging vs. true final state (already patched).** `research_experiment.py`
  and `research_experiment_demand_shift.py` originally only wrote a metrics row every
  ~10 simulated seconds, so the "final" row `final_summary.py` picked up could be stale
  by up to 9 seconds (missing late-episode congestion). Both scripts have since been
  patched to also write an unconditional final-state row right before `traci.close()`.
  If you ever see baseline queue numbers that look suspiciously low compared with
  `evaluate_per_intersection.py`'s per-run numbers for the same seed/scale/method,
  this is the first thing to check — confirm the patched final-row block is still
  present in both scripts before re-running.
- **PPO evaluation uses stochastic action sampling**, matching the training regime —
  not greedy/deterministic action selection. Greedy evaluation of this policy was found
  to collapse to a single constant action for an entire episode (one phase locked in),
  causing severe one-directional starvation. All reported PPO results use stochastic
  sampling.
- **Network identity.** The 3-intersection network (`sumo/research/research.net.xml`) is
  a custom synthetic corridor built for this project — junctions evenly spaced at
  500-unit intervals, no real-world coordinate projection. It is not an import of
  RESCO's Cologne benchmark network; RESCO is cited in the report only as a
  benchmark-design reference.
