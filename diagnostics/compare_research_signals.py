
import traci

CONFIG = "sumo/research/research.sumocfg"
INTERSECTIONS = ["I1", "I2", "I3"]

GREEN_PHASES = [0, 2]
GREEN_SECONDS = 42
YELLOW_SECONDS = 3


def phase_pressure(junction, phase_index):
    logic = traci.trafficlight.getAllProgramLogics(junction)[0]
    state = logic.phases[phase_index].state
    links = traci.trafficlight.getControlledLinks(junction)

    score = 0
    for i, link_group in enumerate(links):
        if state[i] not in ("G", "g"):
            continue

        for incoming, outgoing, _ in link_group:
            upstream = traci.lane.getLastStepVehicleNumber(incoming)
            downstream = traci.lane.getLastStepVehicleNumber(outgoing)
            score += upstream - downstream

    return score


traci.start([
    "sumo", "-c", CONFIG,
    "--end", "180",
    "--scale", "1.0",
    "--no-warnings", "true",
])

switch_at = {j: 0 for j in INTERSECTIONS}
current_green = {j: 0 for j in INTERSECTIONS}
target_green = {j: 0 for j in INTERSECTIONS}
yellow_until = {j: -1 for j in INTERSECTIONS}

try:
    # Disable the default program and explicitly initialize green states.
    for junction in INTERSECTIONS:
        traci.trafficlight.setProgram(junction, "off")
        logic = traci.trafficlight.getAllProgramLogics(junction)[0]
        traci.trafficlight.setRedYellowGreenState(
            junction, logic.phases[0].state
        )

    print("Time | Intersection | Phase | Signal state")

    while traci.simulation.getTime() < 180:
        now = traci.simulation.getTime()

        for junction in INTERSECTIONS:
            if yellow_until[junction] >= 0:
                if now >= yellow_until[junction]:
                    logic = traci.trafficlight.getAllProgramLogics(
                        junction
                    )[0]
                    green_state = logic.phases[
                        target_green[junction]
                    ].state

                    traci.trafficlight.setRedYellowGreenState(
                        junction, green_state
                    )

                    current_green[junction] = target_green[junction]
                    switch_at[junction] = now
                    yellow_until[junction] = -1

                continue

            if now - switch_at[junction] >= GREEN_SECONDS:
                candidates = [
                    (phase_pressure(junction, p), p)
                    for p in GREEN_PHASES
                ]
                _, chosen = max(candidates)

                print(
                    f"Decision t={now:.0f}, {junction}: "
                    f"scores={candidates}, chosen={chosen}"
                )

                if chosen != current_green[junction]:
                    logic = traci.trafficlight.getAllProgramLogics(
                        junction
                    )[0]
                    yellow_phase = current_green[junction] + 1

                    traci.trafficlight.setRedYellowGreenState(
                        junction,
                        logic.phases[yellow_phase].state
                    )

                    target_green[junction] = chosen
                    yellow_until[junction] = now + YELLOW_SECONDS
                else:
                    switch_at[junction] = now

        traci.simulationStep()
        t = int(traci.simulation.getTime())

        if t in (1, 42, 43, 45, 46, 87, 88, 90, 91, 132, 135, 180):
            for junction in INTERSECTIONS:
                print(
                    f"{t:4d} | {junction:12} | "
                    f"{traci.trafficlight.getPhase(junction):5} | "
                    f"{traci.trafficlight.getRedYellowGreenState(junction)}"
                )

finally:
    traci.close()