"""Shared controller utilities for the three-intersection experiments."""

import traci

INTERSECTIONS = ["I1", "I2", "I3"]
GREEN_SECONDS = 10
YELLOW_SECONDS = 3

# Each intersection's controlled signal links are discovered from SUMO.
# The generated signal programs are inspected at runtime rather than
# assuming link indices are identical across intersections.

def get_signal_ids():
    return {
        junction: traci.trafficlight.getControlledLinks(junction)
        for junction in INTERSECTIONS
    }



def get_incoming_lanes(junction):
    """Return all incoming lane IDs for one intersection."""
    incoming_edges = traci.junction.getIncomingEdges(junction)

    incoming = {
        lane_id
        for edge_id in incoming_edges
        for lane_id in traci.edge.getLaneIDs(edge_id)
    }

    return sorted(incoming)


def get_queue(junction):
    """Count halted vehicles on the intersection's incoming lanes."""
    incoming_edges = traci.junction.getIncomingEdges(junction)
    lanes = [
        lane_id
        for edge_id in incoming_edges
        for lane_id in traci.edge.getLaneIDs(edge_id)
    ]
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)


def get_waiting_time(junction):
    """Sum accumulated waiting time for vehicles on incoming lanes."""
    incoming_edges = traci.junction.getIncomingEdges(junction)
    lanes = [
        lane_id
        for edge_id in incoming_edges
        for lane_id in traci.edge.getLaneIDs(edge_id)
    ]
    vehicle_ids = {
        vehicle_id
        for lane_id in lanes
        for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id)
    }
    return sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)


def get_metrics():
    """Return shared evaluation metrics for all controller methods."""
    vehicle_ids = traci.vehicle.getIDList()
    return {
        "time": traci.simulation.getTime(),
        "vehicles_in_network": len(vehicle_ids),
        "arrived": traci.simulation.getArrivedNumber(),
        "total_queue": sum(get_queue(j) for j in INTERSECTIONS),
        "total_waiting": sum(get_waiting_time(j) for j in INTERSECTIONS),
    }
