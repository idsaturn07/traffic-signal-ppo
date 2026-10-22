import argparse
import os
import csv
import random
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

    def evaluate(self, obs_batch, act_batch):
        logits, values = self.forward(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_batch)
        entropy = dist.entropy()
        return logp, values.squeeze(-1), entropy


class RolloutBuffer:
    def __init__(self):
        self.obs, self.acts, self.logps, self.rews, self.vals, self.dones = [], [], [], [], [], []

    def add(self, o, a, lp, r, v, d):
        self.obs.append(o); self.acts.append(a); self.logps.append(lp)
        self.rews.append(r); self.vals.append(v); self.dones.append(d)

    def compute_returns(self, last_value, gamma=0.99, lam=0.95):
        adv = np.zeros(len(self.rews), dtype=np.float32)
        last_gae = 0.0
        vals = self.vals + [last_value]
        for t in reversed(range(len(self.rews))):
            mask = 1.0 - self.dones[t]
            delta = self.rews[t] + gamma * vals[t + 1] * mask - vals[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        returns = adv + np.array(self.vals, dtype=np.float32)
        return adv, returns

    def clear(self):
        self.__init__()


def ppo_update(net, opt, buf, last_value, clip=0.2, epochs=4):
    adv, ret = buf.compute_returns(last_value)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs = torch.as_tensor(np.array(buf.obs), dtype=torch.float32)
    acts = torch.as_tensor(np.array(buf.acts), dtype=torch.int64)
    old_logp = torch.as_tensor(np.array(buf.logps), dtype=torch.float32)
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)

    for _ in range(epochs):
        logp, values, entropy = net.evaluate(obs, acts)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((values - ret_t) ** 2).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()


def run_episode(nets, duration, scale, seed, train=True, buffers=None):
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

    last_obs = {j: None for j in INTERSECTIONS}
    ep_reward = {j: 0.0 for j in INTERSECTIONS}
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
                reward = -(q + w / 50.0)

                if last_obs[j] is not None and train:
                    prev_obs, prev_action, prev_logp, prev_value = last_obs[j]
                    buffers[j].add(prev_obs, prev_action, prev_logp, reward, prev_value, 0.0)

                ep_reward[j] += reward
                action, logp, value = nets[j].act(obs)

                if action != current_action[j]:
                    yellow_phase = GREEN_PHASES[current_action[j]] + 1
                    traci.trafficlight.setRedYellowGreenState(j, phase_states[j][yellow_phase])
                    target_action[j] = action
                    yellow_until[j] = now + YELLOW_SECONDS
                else:
                    switch_at[j] = now

                last_obs[j] = (obs, action, logp, value)

        traci.simulationStep()
        current_time = traci.simulation.getTime()

        for v in traci.simulation.getDepartedIDList():
            departed_at[v] = current_time
        for v in traci.simulation.getArrivedIDList():
            if v in departed_at:
                trip_times.append(current_time - departed_at.pop(v))

    last_values = {}
    for j in INTERSECTIONS:
        if last_obs[j] is not None and train:
            _, _, _, value = last_obs[j]
            last_values[j] = value
        else:
            last_values[j] = 0.0

    traci.close()
    return ep_reward, trip_times, last_values


def train(episodes, duration, scale, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nets = {j: ActorCritic() for j in INTERSECTIONS}
    opts = {j: torch.optim.Adam(nets[j].parameters(), lr=3e-4) for j in INTERSECTIONS}
    buffers = {j: RolloutBuffer() for j in INTERSECTIONS}

    log = []
    for ep in range(episodes):
        for j in INTERSECTIONS:
            buffers[j].clear()

        ep_reward, trip_times, last_values = run_episode(
            nets, duration, scale, seed=ep + 1, train=True, buffers=buffers)

        for j in INTERSECTIONS:
            if len(buffers[j].obs) > 1:
                ppo_update(nets[j], opts[j], buffers[j], last_values[j])

        arrived = len(trip_times)
        mean_tt = sum(trip_times) / len(trip_times) if trip_times else 0
        total_r = sum(ep_reward.values())
        log.append((ep, total_r, arrived, mean_tt))
        print(f"episode {ep+1}/{episodes}  reward={total_r:.1f}  arrived={arrived}  mean_tt={mean_tt:.1f}")

    os.makedirs("results", exist_ok=True)
    for j in INTERSECTIONS:
        torch.save(nets[j].state_dict(), f"results/ppo_{j}.pt")

    with open("results/training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "arrived", "mean_travel_time"])
        writer.writerows(log)

    print("\nTraining done. Models saved to results/ppo_I1.pt, ppo_I2.pt, ppo_I3.pt")
    print("Training log saved to results/training_log.csv")
    return nets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.episodes, args.duration, args.demand_scale, args.seed)