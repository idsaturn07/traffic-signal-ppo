import argparse
import csv
import os
import shutil
import xml.etree.ElementTree as ET
import traci

CONFIG = "sumo/research/research.sumocfg"
INTERSECTIONS = ["I1", "I2", "I3"]

GREEN_PHASES = [0, 2]
GREEN_SECONDS = 10
YELLOW_SECONDS = 3


def lanes_for(junction):
    edges = traci.junction.getIncomingEdges(junction)
    return [
        f"{edge}_{i}"
        for edge in edges
        for i in range(traci.edge.getLaneNumber(edge))
    ]


def queue_for(junction):
    return sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in lanes_for(junction)
    )


def waiting_for(junction):
    vehicles = {
        vehicle
        for lane in lanes_for(junction)
        for vehicle in traci.lane.getLastStepVehicleIDs(lane)
    }
    return sum(
        traci.vehicle.getWaitingTime(vehicle)
        for vehicle in vehicles
    )


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


def parse_tripinfo(path):
    if not os.path.exists(path):
        return [], []
    tree = ET.parse(path)
    waits, losses = [], []
    for trip in tree.getroot().findall("tripinfo"):
        waits.append(float(trip.attrib["waitingTime"]))
        losses.append(float(trip.attrib["timeLoss"]))
    return waits, losses


def run(method, duration, scale, seed):
    os.makedirs("results", exist_ok=True)
    output = f"results/results_{method}_scale{scale}_seed{seed}.csv"
    tripinfo_tmp = "results/tripinfo_tmp_demand.xml"
    tripinfo_final = f"results/tripinfo_{method}_scale{scale}_seed{seed}.xml"

    traci.start([
        "sumo", "-c", CONFIG,
        "--end", str(duration),
        "--scale", str(scale),
        "--seed", str(seed),
        "--tripinfo-output", tripinfo_tmp,
        "--no-warnings", "true",
    ])

    rows = []
    departed_at = {}
    trip_times = []
    trip_waiting = {}
    completed_wait_times = []

    switch_at = {j: 0 for j in INTERSECTIONS}
    current_green = {j: 0 for j in INTERSECTIONS}
    target_green = {j: 0 for j in INTERSECTIONS}
    yellow_until = {j: -1 for j in INTERSECTIONS}

    last_metrics_time = -10

    try:
        phase_states = {}
        for junction in INTERSECTIONS:
            logic = traci.trafficlight.getAllProgramLogics(junction)[0]
            phase_states[junction] = [
                phase.state for phase in logic.phases
            ]

        if method == "max_pressure":
            for junction in INTERSECTIONS:
                traci.trafficlight.setProgram(junction, "off")
                traci.trafficlight.setRedYellowGreenState(
                    junction, phase_states[junction][0]
                )

        while traci.simulation.getTime() < duration:
            now = traci.simulation.getTime()

            if method == "max_pressure":
                for junction in INTERSECTIONS:

                    if yellow_until[junction] >= 0:
                        if now >= yellow_until[junction]:
                            traci.trafficlight.setRedYellowGreenState(
                                junction,
                                phase_states[junction][
                                    target_green[junction]
                                ]
                            )
                            current_green[junction] = target_green[junction]
                            switch_at[junction] = now
                            yellow_until[junction] = -1
                        continue

                    if now - switch_at[junction] >= GREEN_SECONDS:
                        candidates = [
                            (
                                phase_pressure(
                                    junction, phase, phase_states
                                ),
                                phase,
                            )
                            for phase in GREEN_PHASES
                        ]
                        _, chosen = max(candidates)

                        if chosen != current_green[junction]:
                            yellow_phase = current_green[junction] + 1
                            traci.trafficlight.setRedYellowGreenState(
                                junction,
                                phase_states[junction][yellow_phase]
                            )
                            target_green[junction] = chosen
                            yellow_until[junction] = (
                                now + YELLOW_SECONDS
                            )
                        else:
                            switch_at[junction] = now

            traci.simulationStep()
            current_time = traci.simulation.getTime()

            for vehicle in traci.simulation.getDepartedIDList():
                departed_at[vehicle] = current_time
                trip_waiting[vehicle] = 0.0

            for vehicle in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(vehicle) < 0.1:
                    trip_waiting[vehicle] = (
                        trip_waiting.get(vehicle, 0.0) + 1.0
                    )

            for vehicle in traci.simulation.getArrivedIDList():
                if vehicle in departed_at:
                    trip_times.append(
                        current_time - departed_at.pop(vehicle)
                    )
                    completed_wait_times.append(
                        trip_waiting.pop(vehicle, 0.0)
                    )

            if current_time - last_metrics_time >= 10:
                queues = [
                    queue_for(junction)
                    for junction in INTERSECTIONS
                ]
                waits = [
                    waiting_for(junction)
                    for junction in INTERSECTIONS
                ]

                rows.append({
                    "time": current_time,
                    "method": method,
                    "seed": seed,
                    "demand_scale": scale,
                    "total_queue": sum(queues),
                    "mean_queue_per_intersection": sum(queues) / 3,
                    "mean_waiting_per_intersection": sum(waits) / 3,
                    "arrived": len(trip_times),
                    "mean_completed_travel_time": (
                        sum(trip_times) / len(trip_times)
                        if trip_times else 0
                    ),
                    "mean_completed_waiting_time_approx": (
                        sum(completed_wait_times)
                        / len(completed_wait_times)
                        if completed_wait_times else 0
                    ),
                })
                last_metrics_time = current_time

        final_time = traci.simulation.getTime()
        queues = [queue_for(junction) for junction in INTERSECTIONS]
        waits = [waiting_for(junction) for junction in INTERSECTIONS]
        rows.append({
            "time": final_time,
            "method": method,
            "seed": seed,
            "demand_scale": scale,
            "total_queue": sum(queues),
            "mean_queue_per_intersection": sum(queues) / 3,
            "mean_waiting_per_intersection": sum(waits) / 3,
            "arrived": len(trip_times),
            "mean_completed_travel_time": (
                sum(trip_times) / len(trip_times) if trip_times else 0
            ),
            "mean_completed_waiting_time_approx": (
                sum(completed_wait_times) / len(completed_wait_times)
                if completed_wait_times else 0
            ),
        })

    finally:
        traci.close()

    if os.path.exists(tripinfo_tmp):
        shutil.move(tripinfo_tmp, tripinfo_final)

    real_waits, real_losses = parse_tripinfo(tripinfo_final)

    fields = list(rows[0].keys()) if rows else [
        "time", "method", "seed", "demand_scale",
        "total_queue", "mean_queue_per_intersection",
        "mean_waiting_per_intersection", "arrived",
        "mean_completed_travel_time",
        "mean_completed_waiting_time_approx",
    ]

    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Method: {method} | scale: {scale} | seed: {seed} | arrived: {len(trip_times)}")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fixed_time", "max_pressure"], default="fixed_time")
    parser.add_argument("--duration", type=int, default=600)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run(args.method, args.duration, args.demand_scale, args.seed)