import numpy as np
import torch
import torch.nn as nn
import traci
from collections import Counter

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
    return np.array([min(q / QUEUE_NORM, 1.0), min(w / WAIT_NORM, 1.0)], dtype=np.float32), q, w


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=2, hidden=32, n_actions=2):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def act_greedy(self, obs):
        with torch.no_grad():
            logits, _ = self.forward(torch.as_tensor(obs, dtype=torch.float32))
            probs = torch.softmax(logits, dim=-1)
            return int(torch.argmax(logits).item()), probs.numpy()


nets = {}
for j in INTERSECTIONS:
    net = ActorCritic()
    net.load_state_dict(torch.load(f"results/ppo_{j}.pt"))
    net.eval()
    nets[j] = net

traci.start(["sumo", "-c", CONFIG, "--end", "600", "--scale", "1.0", "--seed", "1", "--no-warnings", "true"])

switch_at = {j: 0 for j in INTERSECTIONS}
current_action = {j: 0 for j in INTERSECTIONS}
target_action = {j: 0 for j in INTERSECTIONS}
yellow_until = {j: -1 for j in INTERSECTIONS}
action_counts = {j: Counter() for j in INTERSECTIONS}

phase_states = {}
for j in INTERSECTIONS:
    logic = traci.trafficlight.getAllProgramLogics(j)[0]
    phase_states[j] = [p.state for p in logic.phases]
    traci.trafficlight.setProgram(j, "off")
    traci.trafficlight.setRedYellowGreenState(j, phase_states[j][0])

first_print = {j: True for j in INTERSECTIONS}

while traci.simulation.getTime() < 600:
    now = traci.simulation.getTime()
    for j in INTERSECTIONS:
        if yellow_until[j] >= 0:
            if now >= yellow_until[j]:
                traci.trafficlight.setRedYellowGreenState(j, phase_states[j][GREEN_PHASES[target_action[j]]])
                current_action[j] = target_action[j]
                switch_at[j] = now
                yellow_until[j] = -1
            continue
        if now - switch_at[j] >= GREEN_SECONDS:
            obs, q, w = get_state(j)
            action, probs = nets[j].act_greedy(obs)
            action_counts[j][action] += 1
            if first_print[j] or action_counts[j].total() % 10 == 0:
                print(f"t={now:.0f} {j}: q={q} w={w:.0f} probs={probs} -> action={action}")
                first_print[j] = False
            if action != current_action[j]:
                yellow_phase = GREEN_PHASES[current_action[j]] + 1
                traci.trafficlight.setRedYellowGreenState(j, phase_states[j][yellow_phase])
                target_action[j] = action
                yellow_until[j] = now + YELLOW_SECONDS
            else:
                switch_at[j] = now
    traci.simulationStep()

traci.close()

print("\n=== Action distribution per intersection ===")
for j in INTERSECTIONS:
    print(j, dict(action_counts[j]))