"""
Live demo control panel for the panel presentation.

A small desktop window (Tkinter, already built into Python -- nothing extra
to install) that sits next to the SUMO-GUI window. Pick the controller,
demand level, seed, run length and playback speed, hit Launch, and it opens
sumo-gui and drives the simulation for you -- while this panel shows live
per-intersection queue length and waiting time updating as the run plays,
and a log of what happened.

NOTE on "changing parameters live": SUMO loads the road network and traffic
demand once, at startup, from research.sumocfg -- that's a SUMO limitation,
not something this script can work around. So demand scale / seed / duration
are set *before* each Launch, not edited mid-run. What IS live is the
playback speed and pause/resume -- both this panel's delay field and
SUMO-GUI's own built-in speed slider / step button adjust that while it's
running. Showing the panel values changing between one run and the next,
side by side, is exactly how you'd demo "the parameters you can change".

USAGE: place this file in your repo root (same folder as demo_live.py and
the sumo/ directory) and run:

    python demo_dashboard.py

Requires the same things demo_live.py does: sumo-gui on PATH, torch/numpy,
and (for the two PPO methods) results/ppo_*.pt / results/ppo_neighbor_*.pt.
"""
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

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

METHOD_LABELS = {
    "Fixed-Time": "fixed_time",
    "Max-Pressure": "max_pressure",
    "Independent PPO": "independent_ppo",
    "Neighbor-Aware PPO": "neighbor_ppo",
}


# ---------------------------------------------------------------- SUMO side

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


def run_simulation(method, duration, scale, seed, delay_ms, msg_queue, stop_event):
    """Runs entirely in a background thread. Only talks to the UI via msg_queue."""
    try:
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
            "--start",
        ]
        msg_queue.put(("log", f"Launching sumo-gui: method={method} scale={scale} seed={seed} duration={duration}s"))
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
        step_count = 0

        while traci.simulation.getTime() < duration:
            if stop_event.is_set():
                msg_queue.put(("log", "Stopped by user."))
                break

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
            step_count += 1

            if step_count % 5 == 0:
                metrics = {
                    "time": now,
                    "queues": {j: queue_for(j) for j in INTERSECTIONS},
                    "waits": {j: waiting_for(j) for j in INTERSECTIONS},
                    "teleports": total_teleports,
                }
                msg_queue.put(("metrics", metrics))

        final_queues = {j: queue_for(j) for j in INTERSECTIONS}
        final_waits = {j: waiting_for(j) for j in INTERSECTIONS}
        msg_queue.put(("log", "--- Run finished ---"))
        for j in INTERSECTIONS:
            msg_queue.put(("log", f"  {j}: final queue={final_queues[j]}  final waiting={final_waits[j]:.1f}s"))
        msg_queue.put(("log", f"  teleports (gridlock events): {total_teleports}"))

    except Exception as exc:
        msg_queue.put(("log", f"ERROR: {exc}"))
    finally:
        if traci.isLoaded():
            traci.close()
        msg_queue.put(("done", None))


# ----------------------------------------------------------------- UI side

class DemoDashboard:
    def __init__(self, root):
        self.root = root
        root.title("Traffic Signal RL — Live Demo Control Panel")
        root.geometry("560x520")

        self.msg_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        pad = {"padx": 8, "pady": 4}

        form = ttk.Frame(root)
        form.pack(fill="x", **pad)

        ttk.Label(form, text="Controller:").grid(row=0, column=0, sticky="w")
        self.method_var = tk.StringVar(value="Fixed-Time")
        ttk.Combobox(
            form, textvariable=self.method_var, values=list(METHOD_LABELS.keys()),
            state="readonly", width=20,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(form, text="Demand scale:").grid(row=1, column=0, sticky="w")
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            form, from_=0.4, to=1.8, variable=self.scale_var, orient="horizontal", length=160,
            command=lambda _=None: self.scale_label.config(text=f"{self.scale_var.get():.2f}"),
        ).grid(row=1, column=1, sticky="w")
        self.scale_label = ttk.Label(form, text="1.00")
        self.scale_label.grid(row=1, column=2, sticky="w")

        ttk.Label(form, text="Seed:").grid(row=2, column=0, sticky="w")
        self.seed_var = tk.IntVar(value=1)
        ttk.Spinbox(form, from_=1, to=999, textvariable=self.seed_var, width=8).grid(row=2, column=1, sticky="w")

        ttk.Label(form, text="Duration (sim sec):").grid(row=3, column=0, sticky="w")
        self.duration_var = tk.IntVar(value=600)
        ttk.Spinbox(form, from_=100, to=1800, increment=100, textvariable=self.duration_var, width=8).grid(row=3, column=1, sticky="w")

        ttk.Label(form, text="GUI delay (ms/step):").grid(row=4, column=0, sticky="w")
        self.delay_var = tk.IntVar(value=200)
        ttk.Spinbox(form, from_=0, to=1000, increment=50, textvariable=self.delay_var, width=8).grid(row=4, column=1, sticky="w")

        btns = ttk.Frame(root)
        btns.pack(fill="x", **pad)
        self.launch_btn = ttk.Button(btns, text="Launch", command=self.launch)
        self.launch_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(root, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", **pad)

        metrics_frame = ttk.LabelFrame(root, text="Live metrics")
        metrics_frame.pack(fill="x", **pad)
        self.metric_labels = {}
        for i, j in enumerate(INTERSECTIONS):
            ttk.Label(metrics_frame, text=f"{j}:").grid(row=i, column=0, sticky="w", padx=4)
            lbl = ttk.Label(metrics_frame, text="queue=-  waiting=-")
            lbl.grid(row=i, column=1, sticky="w", padx=4)
            self.metric_labels[j] = lbl

        ttk.Label(root, text="Log:").pack(anchor="w", padx=8)
        self.log_box = scrolledtext.ScrolledText(root, height=14, state="disabled")
        self.log_box.pack(fill="both", expand=True, **pad)

        self.root.after(150, self.poll_queue)

    def log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def launch(self):
        if not os.path.exists(CONFIG):
            self.log(f"ERROR: can't find {CONFIG} — run this from your repo root.")
            return
        method = METHOD_LABELS[self.method_var.get()]
        scale = round(self.scale_var.get(), 2)
        seed = self.seed_var.get()
        duration = self.duration_var.get()
        delay = self.delay_var.get()

        self.stop_event.clear()
        self.status_var.set(f"Running: {self.method_var.get()}  (scale={scale}, seed={seed})")
        self.launch_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.worker = threading.Thread(
            target=run_simulation,
            args=(method, duration, scale, seed, delay, self.msg_queue, self.stop_event),
            daemon=True,
        )
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("Stopping…")

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "metrics":
                    for j in INTERSECTIONS:
                        q = payload["queues"][j]
                        w = payload["waits"][j]
                        self.metric_labels[j].config(text=f"queue={q}  waiting={w:.1f}s")
                    self.status_var.set(
                        f"Running — sim time {payload['time']:.0f}s  "
                        f"(teleports so far: {payload['teleports']})"
                    )
                elif kind == "done":
                    self.status_var.set("Idle")
                    self.launch_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(150, self.poll_queue)


def main():
    root = tk.Tk()
    DemoDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()