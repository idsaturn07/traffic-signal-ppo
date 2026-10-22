import argparse
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import traci

CONFIG = "sumo/research/research.sumocfg"
INTERSECTIONS = ["I1", "I2", "I3"]
GREEN_PHASES = [0, 2]
GREEN_SECONDS = 10
YELLOW_SECONDS = 3
QUEUE_NORM = 30.0
WAIT_NORM = 500.0


def lanes_for(junction):
    edges = traci.junction.getIncomingEdges(junction)
    return [
        f"{edge}_{i}"
        for edge in edges
        for i in range(traci.edge.getLaneNumber(edge))
    ]


def queue_for(junction):
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes_for(junction))


def waiting_for(junction):
    vehicles = {v for lane in lanes_for(junction) for v in traci.lane.getLastStepVehicleIDs(lane)}
    total = 0.0
    for v in vehicles:
        try:
            total += traci.vehicle.getWaitingTime(v)
        except Exception:
            continue
    return total


def get_state(junction):
    q = queue_for(junction)
    w = waiting_for(junction)
    obs = np.array([
        min(q / QUEUE_NORM, 1.0),
        min(w / WAIT_NORM, 1.0),
    ], dtype=np.float32)
    return obs, q, w


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=2, hidden=32, n_actions=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def act(self, obs):
        with torch.no_grad():
            logits, value = self.forward(torch.as_tensor(obs, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.item(), logp.item(), value.item()


def load_nets(prefix):
    nets = {}
    for j in INTERSECTIONS:
        net = ActorCritic()
        net.load_state_dict(torch.load(f"results/{prefix}_{j}.pt"))
        net.eval()
        nets[j] = net
    return nets


def run_eval_episode(nets, duration, scale, seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    traci.start([
        "sumo", "-c", CONFIG,
        "--end", str(duration),
        "--scale", str(scale),
        "--seed", str(seed),
        "--no-warnings", "true",
    ])

    switch_at = {j: 0 for j in INTERSECTIONS}
    current_action = {j: 0 for j in INTERSECTIONS}
    target_action = {j: 0 for j in INTERSECTIONS}
    yellow_until = {j: -1 for j in INTERSECTIONS}
    total_teleports = 0

    phase_states = {}
    for j in INTERSECTIONS:
        logic = traci.trafficlight.getAllProgramLogics(j)[0]
        phase_states[j] = [p.state for p in logic.phases]
        traci.trafficlight.setProgram(j, "off")
        traci.trafficlight.setRedYellowGreenState(j, phase_states[j][0])

    departed_at, trip_times = {}, []

    while traci.simulation.getTime() < duration:
        now = traci.simulation.getTime()

        for j in INTERSECTIONS:
            if yellow_until[j] >= 0:
                if now >= yellow_until[j]:
                    traci.trafficlight.setRedYellowGreenState(
                        j, phase_states[j][GREEN_PHASES[target_action[j]]])
                    current_action[j] = target_action[j]
                    switch_at[j] = now
                    yellow_until[j] = -1
                continue

            if now - switch_at[j] >= GREEN_SECONDS:
                obs, q, w = get_state(j)
                action, logp, value = nets[j].act(obs)

                if action != current_action[j]:
                    yellow_phase = GREEN_PHASES[current_action[j]] + 1
                    traci.trafficlight.setRedYellowGreenState(j, phase_states[j][yellow_phase])
                    target_action[j] = action
                    yellow_until[j] = now + YELLOW_SECONDS
                else:
                    switch_at[j] = now

        traci.simulationStep()
        total_teleports += traci.simulation.getStartingTeleportNumber()
        current_time = traci.simulation.getTime()

        for v in traci.simulation.getDepartedIDList():
            departed_at[v] = current_time
        for v in traci.simulation.getArrivedIDList():
            if v in departed_at:
                trip_times.append(current_time - departed_at.pop(v))

    final_queue = sum(queue_for(j) for j in INTERSECTIONS)
    traci.close()

    arrived = len(trip_times)
    mean_tt = sum(trip_times) / len(trip_times) if trip_times else 0.0
    return arrived, mean_tt, final_queue, total_teleports


def main():
    variants = {
        "independent_ppo": "ppo",
        "neighbor_ppo": "ppo_neighbor",
    }
    scales = [0.6, 1.0, 1.4]
    seeds = [1, 2, 3, 4, 5]

    rows = []
    for variant_name, prefix in variants.items():
        nets = load_nets(prefix)
        for scale in scales:
            for seed in seeds:
                arrived, mean_tt, final_q, teleports = run_eval_episode(nets, 600, scale, seed)
                print(f"{variant_name} scale={scale} seed={seed}: "
                      f"arrived={arrived} mean_tt={mean_tt:.2f} final_queue={final_q} teleports={teleports}")
                rows.append([variant_name, scale, seed, arrived, mean_tt, final_q, teleports])

    os.makedirs("results", exist_ok=True)
    with open("results/demand_shift_ppo.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "demand_scale", "seed", "arrived", "mean_travel_time", "final_queue", "teleports"])
        writer.writerows(rows)

    print("\nSaved results/demand_shift_ppo.csv")


if __name__ == "__main__":
    main()

