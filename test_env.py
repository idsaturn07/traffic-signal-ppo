from traffic_env import TrafficSignalEnv

env = TrafficSignalEnv()

try:
    state = env.reset()

    print("Environment started successfully!")
    print("Initial observation shape:", state.shape)
    print("Initial observation length:", len(state))
    print("Initial observation:", state)

    next_state, reward, done, info = env.step(0)

    print("\nAfter Action 1:")
    print("Next observation shape:", next_state.shape)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

finally:
    env.close()
    print("\nSUMO closed successfully.")