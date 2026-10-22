import traci
from actions import ACTIONS

CONFIG = "sumo/config/simulation.sumocfg"
TLS_ID = "center"
GREEN_SECONDS = 10
YELLOW_SECONDS = 3

def make_state(green_links, count):
    state = ["r"] * count
    for index in green_links:
        state[index] = "G"
    return "".join(state)

def make_yellow(old_green, count):
    state = ["r"] * count
    for index in old_green:
        state[index] = "y"
    return "".join(state)

try:
    traci.start(["sumo", "-c", CONFIG])
    count = len(traci.trafficlight.getRedYellowGreenState(TLS_ID))

    previous_green = []

    for action_id, action in ACTIONS.items():
        green_links = action["green_links"]

        if previous_green:
            yellow_state = make_yellow(previous_green, count)
            traci.trafficlight.setRedYellowGreenState(
                TLS_ID, yellow_state
            )
            for _ in range(YELLOW_SECONDS):
                traci.simulationStep()

        green_state = make_state(green_links, count)
        traci.trafficlight.setRedYellowGreenState(
            TLS_ID, green_state
        )

        print(
            f"Testing action {action_id + 1}: "
            f"{action['name']} at t={traci.simulation.getTime():.0f}s"
        )

        for _ in range(GREEN_SECONDS):
            traci.simulationStep()

        previous_green = green_links

    print("\nController smoke test completed.")
    print("Simulated time:", traci.simulation.getTime())

finally:
    if traci.isLoaded():
        traci.close()