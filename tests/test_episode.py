from traffic_env import TrafficSignalEnv

env = TrafficSignalEnv()

try:
    state = env.reset()
    done = False

    while not done:
        state, reward, done, info = env.step(0)

        if info["steps"] % 50 == 0 or done:
            print(
                f"step={info['steps']}, "
                f"time={info['time']:.0f}s, "
                f"reward={reward:.2f}, "
                f"queue={info['avg_queue']:.2f}, "
                f"avg_wait={info['avg_wait']:.2f}, "
                f"done={done}"
            )

    print("Episode terminated.")
finally:
    env.close()
