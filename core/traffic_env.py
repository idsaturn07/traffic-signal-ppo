import xml.etree.ElementTree as ET

import numpy as np
import traci

CONFIG = "sumo/config/simulation.sumocfg"
NET_FILE = "sumo/network/network.net.xml"
TLS_ID = "center"

# Paper settings
ACTION_SECONDS = 10
YELLOW_SECONDS = 3
EPISODE_SECONDS = 7200

# Temporary fixed weights; the paper's exact dynamic-weight rule
# has not yet been established.
W_WAIT = 1.0
W_QUEUE = 1.0

ACTIONS = {
    0: [1, 2, 3, 4, 12, 13, 14, 15],
    1: [7, 8, 9, 18, 19, 20],
    2: [5, 16],
    3: [10, 21],
    4: [1, 2, 3, 4, 5],
    5: [7, 8, 9, 10],
    6: [12, 13, 14, 15, 16],
    7: [18, 19, 20, 21],
}


class TrafficSignalEnv:
    def __init__(
        self,
        config=CONFIG,
        action_seconds=ACTION_SECONDS,
        yellow_seconds=YELLOW_SECONDS,
        episode_seconds=EPISODE_SECONDS,
        w_wait=W_WAIT,
        w_queue=W_QUEUE,
    ):
        self.config = config
        self.action_seconds = action_seconds
        self.yellow_seconds = yellow_seconds
        self.episode_seconds = episode_seconds
        self.w_wait = w_wait
        self.w_queue = w_queue

        net_root = ET.parse(NET_FILE).getroot()
        self.lanes_by_edge = {
            edge.get("id"): [
                lane.get("id") for lane in edge.findall("lane")
            ]
            for edge in net_root.findall("edge")
        }

        self.incoming_lanes = [
            lane
            for edge in ("north_in", "south_in", "east_in", "west_in")
            for lane in self.lanes_by_edge[edge]
        ]

        self.signal_count = None
        self.previous_action = None
        self.previous_avg_wait = 0.0
        self.previous_avg_queue = 0.0
        self.steps = 0

        # Track each vehicle's cumulative stopped time while it is
        # present in this simulation episode.
        self.vehicle_waiting_totals = {}

    def reset(self):
        if traci.isLoaded():
            traci.close()

        traci.start(["sumo", "-c", self.config])

        self.signal_count = len(
            traci.trafficlight.getRedYellowGreenState(TLS_ID)
        )
        self.previous_action = None
        self.steps = 0
        self.vehicle_waiting_totals = {}

        self.previous_avg_wait, self.previous_avg_queue = (
            self._traffic_metrics()
        )

        return self._observation()

    def _advance_simulation(self):
        """Advance SUMO one second and accumulate vehicle stopping time."""
        traci.simulationStep()
        delta_t = traci.simulation.getDeltaT()

        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id not in self.vehicle_waiting_totals:
                self.vehicle_waiting_totals[vehicle_id] = 0.0

            if traci.vehicle.getSpeed(vehicle_id) < 0.1:
                self.vehicle_waiting_totals[vehicle_id] += delta_t

    def _observation(self):
        # Two features per incoming lane:
        # 1. Number of halted vehicles.
        # 2. Cumulative stopped time of the stopped vehicle nearest
        #    the junction.
        values = []

        for lane_id in self.incoming_lanes:
            values.append(
                float(traci.lane.getLastStepHaltingNumber(lane_id))
            )
            values.append(float(self._head_waiting_time(lane_id)))

        return np.asarray(values, dtype=np.float32)

    def _head_waiting_time(self, lane_id):
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

        stopped = [
            vehicle_id
            for vehicle_id in vehicle_ids
            if traci.vehicle.getSpeed(vehicle_id) < 0.1
        ]

        if not stopped:
            return 0.0

        head_vehicle = max(
            stopped,
            key=lambda vehicle_id:
                traci.vehicle.getLanePosition(vehicle_id),
        )

        return self.vehicle_waiting_totals.get(head_vehicle, 0.0)

    def _traffic_metrics(self):
        # Paper Eq. (9): average cumulative stopping time of vehicles
        # currently in the road network.
        vehicle_ids = traci.vehicle.getIDList()
        avg_wait = (
            sum(
                self.vehicle_waiting_totals.get(vehicle_id, 0.0)
                for vehicle_id in vehicle_ids
            ) / len(vehicle_ids)
            if vehicle_ids else 0.0
        )

        # Paper Eq. (10): average halting vehicles per incoming lane.
        queue_counts = [
            traci.lane.getLastStepHaltingNumber(lane_id)
            for lane_id in self.incoming_lanes
        ]
        avg_queue = (
            sum(queue_counts) / len(queue_counts)
            if queue_counts else 0.0
        )

        return avg_wait, avg_queue

    def _signal_state(self, green_links, yellow=False):
        state = ["r"] * self.signal_count

        for index in green_links:
            state[index] = "y" if yellow else "G"

        return "".join(state)

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if traci.simulation.getTime() >= self.episode_seconds:
            return self._observation(), 0.0, True, {
                "time": traci.simulation.getTime(),
                "avg_wait": self.previous_avg_wait,
                "avg_queue": self.previous_avg_queue,
                "steps": self.steps,
            }

        remaining = self.episode_seconds - traci.simulation.getTime()

        # When switching phases, yellow time is part of the 10-second
        # decision interval, not an extra 3 seconds.
        if (
            self.previous_action is not None
            and action != self.previous_action
        ):
            yellow_time = min(self.yellow_seconds, int(remaining))
            old_links = ACTIONS[self.previous_action]

            traci.trafficlight.setRedYellowGreenState(
                TLS_ID,
                self._signal_state(old_links, yellow=True),
            )

            for _ in range(yellow_time):
                if traci.simulation.getTime() >= self.episode_seconds:
                    break
                self._advance_simulation()

            green_time = max(0, self.action_seconds - yellow_time)
        else:
            green_time = self.action_seconds

        traci.trafficlight.setRedYellowGreenState(
            TLS_ID, self._signal_state(ACTIONS[action])
        )

        for _ in range(green_time):
            if traci.simulation.getTime() >= self.episode_seconds:
                break
            self._advance_simulation()

        observation = self._observation()
        avg_wait, avg_queue = self._traffic_metrics()

        reward = (
            self.w_wait * (self.previous_avg_wait - avg_wait)
            + self.w_queue * (self.previous_avg_queue - avg_queue)
        )

        self.previous_avg_wait = avg_wait
        self.previous_avg_queue = avg_queue
        self.previous_action = action
        self.steps += 1

        done = (
            traci.simulation.getTime() >= self.episode_seconds
            or traci.simulation.getMinExpectedNumber() == 0
        )

        info = {
            "time": traci.simulation.getTime(),
            "avg_wait": avg_wait,
            "avg_queue": avg_queue,
            "steps": self.steps,
        }

        return observation, float(reward), done, info

    def close(self):
        if traci.isLoaded():
            traci.close()
