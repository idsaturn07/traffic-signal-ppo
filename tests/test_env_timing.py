from traffic_env import TrafficSignalEnv
import traci

env = TrafficSignalEnv(episode_seconds=100)

try:
    state = env.reset()
    print("Initial state shape:", state.shape)
    print("Initial simulation time:", traci.simulation.getTime())

    # First action: 10 seconds of green
    _, reward, done, info = env.step(0)
    print("\nAfter first action:")
    print("Time:", info["time"])
    print("Queue:", info["avg_queue"])
    print("Average waiting time:", info["avg_wait"])

    # Different action: 3 seconds yellow + 7 seconds green
    _, reward, done, info = env.step(1)
    print("\nAfter switching action:")
    print("Time:", info["time"])
    print("Queue:", info["avg_queue"])
    print("Average waiting time:", info["avg_wait"])

    assert info["time"] == 20, "Action timing is not 10 seconds per decision!"
    assert state.shape == (28,), "Observation should contain 28 features."

    print("\nPASS: 28-feature state and 10-second action intervals verified.")

finally:
    env.close()
