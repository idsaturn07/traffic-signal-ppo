import csv
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
    obs = np.array([min(q / QUEUE_NORM, 1.0), min(w / WAIT_NORM, 1.0)], dtype=np.float32)
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

    def act_greedy(self, obs):
        with torch.no_grad():
            logits, _ = self.forward(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.argmax(logits).item())


def load_nets():
    nets = {}
    for j in INTERSECTIONS:
        net = ActorCritic()
        net.load_state_dict(torch.load(f"results/ppo_{j}.pt"))
        net.eval()
        nets[j] = net
    return nets


def run_ppo_episode(nets, duration, scale, seed):
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

    phase_states = {}
    for j in INTERSECTIONS:
        logic = traci.trafficlight.getAllProgramLogics(j)[0]
        phase_states[j] = [p.state for p in logic.phases]
        traci.trafficlight.setProgram(j, "off")
        traci.trafficlight.setRedYellowGreenState(j, phase_states[j][0])

    departed_at, trip_times = {}, []
    rows = []
    last_metrics_time = -10

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
                action = nets[j].act_greedy(obs) if False else int(torch.distributions.Categorical(logits=nets[j].forward(torch.as_tensor(obs, dtype=torch.float32))[0]).sample().item())

                if action != current_action[j]:
                    yellow_phase = GREEN_PHASES[current_action[j]] + 1
                    traci.trafficlight.setRedYellowGreenState(j, phase_states[j][yellow_phase])
                    target_action[j] = action
                    yellow_until[j] = now + YELLOW_SECONDS
                else:
                    switch_at[j] = now

        traci.simulationStep()
        current_time = traci.simulation.getTime()

        for v in traci.simulation.getDepartedIDList():
            departed_at[v] = current_time
        for v in traci.simulation.getArrivedIDList():
            if v in departed_at:
                trip_times.append(current_time - departed_at.pop(v))

        if current_time - last_metrics_time >= 10:
            total_queue = sum(queue_for(j) for j in INTERSECTIONS)
            rows.append({
                "time": current_time,
                "method": "independent_ppo",
                "seed": seed,
                "total_queue": total_queue,
                "arrived": len(trip_times),
                "mean_completed_travel_time": sum(trip_times) / len(trip_times) if trip_times else 0,
            })
            last_metrics_time = current_time

    traci.close()
    return rows, trip_times


def main():
    nets = load_nets()
    all_rows = []
    for seed in range(1, 6):
        rows, trip_times = run_ppo_episode(nets, duration=600, scale=1.0, seed=seed)
        all_rows.extend(rows)
        arrived = len(trip_times)
        mean_tt = sum(trip_times) / len(trip_times) if trip_times else 0
        print(f"seed {seed}: arrived={arrived}  mean_travel_time={mean_tt:.2f}")

    fields = ["time", "method", "seed", "total_queue", "arrived", "mean_completed_travel_time"]
    for seed in range(1, 6):
        seed_rows = [r for r in all_rows if r["seed"] == seed]
        with open(f"results/results_independent_ppo_seed{seed}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(seed_rows)

    print("\nSaved results/results_independent_ppo_seed1.csv ... seed5.csv")


if __name__ == "__main__":
    main()
