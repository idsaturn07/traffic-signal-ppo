from traffic_env import TrafficSignalEnv

env = TrafficSignalEnv()

try:
    state = env.reset()
    print("Initial state shape:", state.shape)

    for action in [0, 1, 2, 3, 4, 5, 6, 7]:
        state, reward, done, info = env.step(action)

        print(
            f"Action {action + 1}: "
            f"time={info['time']:.0f}s, "
            f"reward={reward:.2f}, "
            f"queue={info['avg_queue']}, "
            f"total_waiting={info['avg_wait']:.2f}, "
            f"done={done}"
        )

        if done:
            print("Simulation ended.")
            break

finally:
    env.close()
    print("Environment closed.")