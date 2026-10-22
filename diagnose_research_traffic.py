
import traci

CONFIG = "sumo/research/research.sumocfg"
DURATION = 600

try:
    traci.start([
        "sumo", "-c", CONFIG,
        "--end", str(DURATION),
        "--scale", "1.0",
        "--no-warnings", "true",
    ])

    departed_at = {}
    total_departed = 0
    total_arrived = 0
    completed_trip_times = []

    print("Time | Total departed | Total arrived | In network | Completed trips")

    while traci.simulation.getTime() < DURATION:
        traci.simulationStep()
        now = traci.simulation.getTime()

        for vehicle in traci.simulation.getDepartedIDList():
            departed_at[vehicle] = now
            total_departed += 1

        for vehicle in traci.simulation.getArrivedIDList():
            total_arrived += 1
            if vehicle in departed_at:
                completed_trip_times.append(now - departed_at.pop(vehicle))

        if now % 60 == 0:
            print(
                f"{now:4.0f} | "
                f"{total_departed:14d} | "
                f"{total_arrived:13d} | "
                f"{traci.vehicle.getIDCount():10d} | "
                f"{len(completed_trip_times):15d}"
            )

    print("\n--- Final summary ---")
    print("Simulated seconds:", traci.simulation.getTime())
    print("Total departed:", total_departed)
    print("Total arrived:", total_arrived)
    print("Vehicles still in network:", traci.vehicle.getIDCount())

    if completed_trip_times:
        print(
            "Mean completed travel time:",
            round(sum(completed_trip_times) / len(completed_trip_times), 2),
            "seconds"
        )
    else:
        print("Mean completed travel time: N/A (no completed trips)")

finally:
    if traci.isLoaded():
        traci.close()