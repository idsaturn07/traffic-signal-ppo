from traffic_env import TrafficSignalEnv

env = TrafficSignalEnv()

try:
    state = env.reset()
    done = False
    action = 0

    while not done:
        state, reward, done, info = env.step(action)

        if info["steps"] % 10 == 0 or done:
            print(
                f"step={info['steps']}, "
                f"time={info['time']:.0f}s, "
                f"action={action}, "
                f"reward={reward:.2f}, "
                f"queue={info['avg_queue']:.2f}, "
                f"avg_wait={info['avg_wait']:.2f}, "
                f"done={done}"
            )

        action = (action + 1) % 8

finally:
    env.close()
