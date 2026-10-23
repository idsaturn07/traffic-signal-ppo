"""
Live SUMO-GUI demo runner for the panel presentation.

Drop this file anywhere in your traffic-signal-ppo repo (repo root is
simplest) and run it with the SUMO-GUI window popping up and the chosen
controller actually driving the three-intersection corridor in real time.

It is the exact same control logic your evaluate_per_intersection.py uses
for fixed_time / max_pressure / independent_ppo / neighbor_ppo -- just
pointed at "sumo-gui" instead of headless "sumo", with a visible per-step
delay so a panel can actually watch it, and no file-writing side effects.

USAGE (from the repo root, with your sumo/research/research.sumocfg and
results/*.pt files in place):

    python demo_live.py --method fixed_time
    python demo_live.py --method max_pressure
    python demo_live.py --method independent_ppo
    python demo_live.py --method neighbor_ppo

Useful flags:
    --scale 1.0        traffic demand multiplier (0.6 light / 1.0 normal / 1.4 heavy)
    --duration 600      how many simulated seconds to run (default 600 = 10 min)
    --seed 1            random seed, so you can rerun the identical scenario
    --delay 200          milliseconds the GUI pauses between steps (bigger = slower/easier to watch)
    --no-quit            keep the SUMO-GUI window open after the run ends (default) so the
                          panel can look at the final state; pass --quit to auto-close instead

To show a side-by-side comparison for the panel, just run the same command
four times back to back, once per method, pointing out the queue lengths /
waiting times printed to the console at the end of each run.
"""
import argparse
import os
import sys

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
        net.load_state_dict(torch.load(f"results/{prefix}_{j}.pt", map_location="cpu"))
        net.eval()
        nets[j] = net
    return nets


def run_demo(method, duration, scale, seed, delay_ms, quit_on_end):
    torch.manual_seed(seed)

    nets = None
    if method == "independent_ppo":
        nets = load_nets("ppo")
    elif method == "neighbor_ppo":
        nets = load_nets("ppo_neighbor")

    sumo_cmd = [
        "sumo-gui", "-c", CONFIG,
        "--end", str(duration),
        "--scale", str(scale),
        "--seed", str(seed),
        "--no-warnings", "true",
        "--delay", str(delay_ms),
        "--start",  # auto-press play, no need to click the button in front of the panel
    ]
    if quit_on_end:
        sumo_cmd.append("--quit-on-end")

    print(f"Launching sumo-gui for method={method!r}  scale={scale}  seed={seed}  duration={duration}s")
    traci.start(sumo_cmd)

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

    try:
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
        final_waits = {j: waiting_for(j) for j in INTERSECTIONS}
        print("\n--- Run finished ---")
        print(f"Method: {method}")
        for j in INTERSECTIONS:
            print(f"  {j}: final queue={final_queues[j]}  final waiting={final_waits[j]:.1f}s")
        print(f"  teleports (gridlock events): {total_teleports}")

    finally:
        if traci.isLoaded():
            traci.close()


def main():
    parser = argparse.ArgumentParser(description="Live SUMO-GUI demo of one traffic-signal controller.")
    parser.add_argument(
        "--method", required=True,
        choices=["fixed_time", "max_pressure", "independent_ppo", "neighbor_ppo"],
        help="which controller to run live",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="demand multiplier (0.6 / 1.0 / 1.4 in the paper)")
    parser.add_argument("--duration", type=int, default=600, help="simulated seconds to run")
    parser.add_argument("--seed", type=int, default=1, help="random seed, for a repeatable run")
    parser.add_argument("--delay", type=int, default=200, help="ms delay per simulation step in the GUI (bigger = slower)")
    parser.add_argument("--quit", dest="quit_on_end", action="store_true", help="auto-close the GUI window when the run ends")
    args = parser.parse_args()

    if not os.path.exists(CONFIG):
        print(f"ERROR: can't find {CONFIG} -- run this script from your repo root "
              f"(the folder containing the sumo/ directory).", file=sys.stderr)
        sys.exit(1)

    run_demo(args.method, args.duration, args.scale, args.seed, args.delay, args.quit_on_end)


if __name__ == "__main__":
    main()