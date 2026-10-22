import os
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
    return [f"{edge}_{i}" for edge in edges for i in range(traci.edge.getLaneNumber(edge))]


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
    return np.array([min(q / QUEUE_NORM, 1.0), min(w / WAIT_NORM, 1.0)], dtype=np.float32)


def phase_pressure(junction, phase_index, phase_states):
    state = phase_states[junction][phase_index]
    links = traci.trafficlight.getControlledLinks(junction)
    score = 0
    for i, link_group in enumerate(links):
        if i >= len(state) or state[i] not in ("G", "g"):
            continue
        for incoming, outgoing, _ in link_group:
            upstream = traci.lane.getLastStepVehicleNumber(incoming)
            downstream = traci.lane.getLastStepVehicleNumber(outgoing)
            score += upstream - downstream
    return score


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=2, hidden=32, n_actions=2):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
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
        return action.item()


def load_nets(prefix):
    nets = {}
    for j in INTERSECTIONS:
        net = ActorCritic()
        net.load_state_dict(torch.load(f"results/{prefix}_{j}.pt"))
        net.eval()
        nets[j] = net
    return nets


def run_episode(method, duration, scale, seed, nets=None):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    traci.start([
        "sumo", "-c", CONFIG,
        "--end", str(duration),
        "--scale", str(scale),
        "--seed", str(seed),
        "--no-warnings", "true",
    ])

    phase_states = {}
    for j in INTERSECTIONS:
        logic = traci.trafficlight.getAllProgramLogics(j)[0]
        phase_states[j] = [p.state for p in logic.phases]

    uses_control = method in ("max_pressure", "independent_ppo", "neighbor_ppo")
    if uses_control:
        for j in INTERSECTIONS:
            traci.trafficlight.setProgram(j, "off")
            traci.trafficlight.setRedYellowGreenState(j, phase_states[j][0])

    switch_at = {j: 0 for j in INTERSECTIONS}
    current_phase = {j: 0 for j in INTERSECTIONS}
    target_phase = {j: 0 for j in INTERSECTIONS}
    current_action = {j: 0 for j in INTERSECTIONS}
    target_action = {j: 0 for j in INTERSECTIONS}
    yellow_until = {j: -1 for j in INTERSECTIONS}
    total_teleports = 0

    while traci.simulation.getTime() < duration:
        now = traci.simulation.getTime()

        if method == "max_pressure":
            for j in INTERSECTIONS:
                if yellow_until[j] >= 0:
                    if now >= yellow_until[j]:
                        traci.trafficlight.setRedYellowGreenState(j, phase_states[j][target_phase[j]])
                        current_phase[j] = target_phase[j]
                        switch_at[j] = now
                        yellow_until[j] = -1
                    continue
                if now - switch_at[j] >= GREEN_SECONDS:
                    candidates = [(phase_pressure(j, p, phase_states), p) for p in GREEN_PHASES]
                    _, chosen = max(candidates)
                    if chosen != current_phase[j]:
                        yellow_phase = current_phase[j] + 1
                        traci.trafficlight.setRedYellowGreenState(j, phase_states[j][yellow_phase])
                        target_phase[j] = chosen
                        yellow_until[j] = now + YELLOW_SECONDS
                    else:
                        switch_at[j] = now

        elif method in ("independent_ppo", "neighbor_ppo"):
            for j in INTERSECTIONS:
                if yellow_until[j] >= 0:
                    if now >= yellow_until[j]:
                        traci.trafficlight.setRedYellowGreenState(j, phase_states[j][GREEN_PHASES[target_action[j]]])
                        current_action[j] = target_action[j]
                        switch_at[j] = now
                        yellow_until[j] = -1
                    continue
                if now - switch_at[j] >= GREEN_SECONDS:
                    obs = get_state(j)
                    action = nets[j].act(obs)
                    if action != current_action[j]:
                        yellow_phase = GREEN_PHASES[current_action[j]] + 1
                        traci.trafficlight.setRedYellowGreenState(j, phase_states[j][yellow_phase])
                        target_action[j] = action
                        yellow_until[j] = now + YELLOW_SECONDS
                    else:
                        switch_at[j] = now

        traci.simulationStep()
        total_teleports += traci.simulation.getStartingTeleportNumber()

    final_queues = {j: queue_for(j) for j in INTERSECTIONS}
    traci.close()
    return final_queues, total_teleports


def main():
    scales = [0.6, 1.0, 1.4]
    seeds = [1, 2, 3, 4, 5]
    methods = ["fixed_time", "max_pressure", "independent_ppo", "neighbor_ppo"]

    nets_cache = {
        "independent_ppo": load_nets("ppo"),
        "neighbor_ppo": load_nets("ppo_neighbor"),
    }

    rows = []
    for method in methods:
        nets = nets_cache.get(method)
        for scale in scales:
            for seed in seeds:
                fq, teleports = run_episode(method, 600, scale, seed, nets=nets)
                row = {"method": method, "demand_scale": scale, "seed": seed}
                row.update(fq)
                row["teleports"] = teleports
                rows.append(row)
                print(f"{method} scale={scale} seed={seed}: I1={fq['I1']} I2={fq['I2']} I3={fq['I3']} teleports={teleports}")

    os.makedirs("results", exist_ok=True)
    with open("results/per_intersection_final_queue.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "demand_scale", "seed", "I1", "I2", "I3", "teleports"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved results/per_intersection_final_queue.csv")


if __name__ == "__main__":
    main()

