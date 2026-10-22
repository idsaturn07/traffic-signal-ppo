import xml.etree.ElementTree as ET
import traci

CONFIG = "sumo/config/simulation.sumocfg"
NET_FILE = "sumo/network/network.net.xml"

try:
    tree = ET.parse(NET_FILE)
    root = tree.getroot()

    incoming_edges = {"north_in", "south_in", "east_in", "west_in"}
    incoming_lanes = []

    for edge in root.findall("edge"):
        if edge.get("id") in incoming_edges:
            for lane in edge.findall("lane"):
                incoming_lanes.append(lane.get("id"))

    traci.start(["sumo", "-c", CONFIG])

    tls_id = "center"
    signal_count = len(
        traci.trafficlight.getRedYellowGreenState(tls_id)
    )

    # Diagnostic only: hold all signal links green.
    traci.trafficlight.setRedYellowGreenState(
        tls_id, "G" * signal_count
    )

    for _ in range(600):
        if traci.simulation.getMinExpectedNumber() == 0:
            break
        traci.simulationStep()

    waiting = sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in incoming_lanes
    )

    print("\n--- TRAFFIC DIAGNOSTIC ---")
    print("Simulation time:", traci.simulation.getTime())
    print("Vehicles currently in network:", traci.vehicle.getIDCount())
    print("Vehicles arrived:", traci.simulation.getArrivedNumber())
    print("Vehicles waiting on incoming lanes:", waiting)
    print("Incoming lanes:", len(incoming_lanes))
    print("Signal links:", signal_count)

finally:
    if traci.isLoaded():
        traci.close()
