import traci
from actions import ACTIONS

CONFIG = "sumo/config/simulation.sumocfg"
TLS_ID = "center"
GREEN_SECONDS = 10
YELLOW_SECONDS = 3

INCOMING_EDGES = [
    "north_in", "south_in", "east_in", "west_in"
]

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

def collect_metrics():
    vehicle_ids = traci.vehicle.getIDList()
    waiting = [
        traci.vehicle.getWaitingTime(v)
        for v in vehicle_ids
    ]
    queues = [
        traci.edge.getLastStepHaltingNumber(edge)
        for edge in INCOMING_EDGES
    ]

    return {
        "vehicles": len(vehicle_ids),
        "avg_wait": sum(waiting) / len(waiting) if waiting else 0.0,
        "queue": sum(queues),
    }

try:
    traci.start(["sumo", "-c", CONFIG])
    count = len(traci.trafficlight.getRedYellowGreenState(TLS_ID))
    previous_green = []

    for action_id, action in ACTIONS.items():
        if previous_green:
            traci.trafficlight.setRedYellowGreenState(
                TLS_ID, make_yellow(previous_green, count)
            )
            for _ in range(YELLOW_SECONDS):
                traci.simulationStep()

        traci.trafficlight.setRedYellowGreenState(
            TLS_ID, make_state(action["green_links"], count)
        )

        for _ in range(GREEN_SECONDS):
            traci.simulationStep()

        metrics = collect_metrics()
        print(
            f"Action {action_id + 1} ({action['name']}): "
            f"vehicles={metrics['vehicles']}, "
            f"avg_wait={metrics['avg_wait']:.2f}s, "
            f"queue={metrics['queue']}"
        )

        previous_green = action["green_links"]

    print("\nEvaluation smoke test completed.")
    print("Simulated time:", traci.simulation.getTime())

finally:
    if traci.isLoaded():
        traci.close()