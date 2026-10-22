from traffic_env import TrafficSignalEnv, ACTIONS
import traci

env = TrafficSignalEnv(episode_seconds=120)

try:
    env.reset()
    print("Signal link count:", env.signal_count)

    for action_id, green_links in ACTIONS.items():
        state = env._signal_state(green_links)

        assert len(state) == env.signal_count, (
            f"Action {action_id + 1}: invalid signal-state length"
        )

        traci.trafficlight.setRedYellowGreenState("center", state)
        traci.simulationStep()

        actual_state = traci.trafficlight.getRedYellowGreenState("center")

        print(
            f"Action {action_id + 1}: "
            f"state length={len(state)}, "
            f"SUMO accepted={'YES' if len(actual_state) == len(state) else 'NO'}"
        )

    print("\nAll eight signal states were applied.")

finally:
    env.close()
