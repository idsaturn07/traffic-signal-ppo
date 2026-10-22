import traci

CONFIG = "sumo/config/simulation.sumocfg"

try:
    traci.start(["sumo", "-c", CONFIG])

    print("\n--- OUTGOING EDGE LENGTHS ---")
    for edge_id in ["north_out", "south_out", "east_out", "west_out"]:
        lanes = traci.edge.getLaneNumber(edge_id)
        lengths = [
            traci.lane.getLength(f"{edge_id}_{i}")
            for i in range(lanes)
        ]
        print(f"{edge_id}: lanes={lanes}, lengths={lengths}")

    for _ in range(600):
        if traci.simulation.getMinExpectedNumber() == 0:
            break
        traci.simulationStep()

    print("\n--- AFTER 600 SECONDS ---")
    print("Simulation time:", traci.simulation.getTime())
    print("Arrived:", traci.simulation.getArrivedNumber())
    print("Vehicles in network:", traci.vehicle.getIDCount())
    print("Vehicles waiting to depart:", traci.simulation.getPendingVehicles())

finally:
    if traci.isLoaded():
        traci.close()
