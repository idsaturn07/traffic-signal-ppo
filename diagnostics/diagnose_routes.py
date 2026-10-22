import traci

CONFIG = "sumo/config/simulation.sumocfg"

try:
    traci.start(["sumo", "-c", CONFIG])

    # Run a short simulation using SUMO's default signal program.
    for _ in range(300):
        if traci.simulation.getMinExpectedNumber() == 0:
            break
        traci.simulationStep()

    print("\n--- ROUTE AND MOVEMENT CHECK ---")
    print("Simulation time:", traci.simulation.getTime())
    print("Vehicles in network:", traci.vehicle.getIDCount())
    print("Vehicles arrived:", traci.simulation.getArrivedNumber())
    print("Vehicles waiting to depart:", traci.simulation.getPendingVehicles())

    print("\n--- SAMPLE VEHICLES ---")
    for vehicle_id in traci.vehicle.getIDList()[:10]:
        print(
            vehicle_id,
            "| route:", traci.vehicle.getRoute(vehicle_id),
            "| road:", traci.vehicle.getRoadID(vehicle_id),
            "| lane:", traci.vehicle.getLaneID(vehicle_id),
            "| speed:", round(traci.vehicle.getSpeed(vehicle_id), 2)
        )

finally:
    if traci.isLoaded():
        traci.close()
