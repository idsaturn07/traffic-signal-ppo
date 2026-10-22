def patch(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    anchor = "\n\n    finally:\n        traci.close()"
    if anchor not in content:
        raise RuntimeError(f"Anchor not found in {path}")
    guaranteed_snapshot = """

        final_time = traci.simulation.getTime()
        queues = [queue_for(junction) for junction in INTERSECTIONS]
        waits = [waiting_for(junction) for junction in INTERSECTIONS]
        rows.append({
            "time": final_time,
            "method": method,
            "seed": seed,
            "demand_scale": scale,
            "total_queue": sum(queues),
            "mean_queue_per_intersection": sum(queues) / 3,
            "mean_waiting_per_intersection": sum(waits) / 3,
            "arrived": len(trip_times),
            "mean_completed_travel_time": (
                sum(trip_times) / len(trip_times) if trip_times else 0
            ),
            "mean_completed_waiting_time_approx": (
                sum(completed_wait_times) / len(completed_wait_times)
                if completed_wait_times else 0
            ),
        })
"""
    content = content.replace(anchor, guaranteed_snapshot + "\n    finally:\n        traci.close()", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Patched {path}")

patch("research_experiment.py")
patch("research_experiment_demand_shift.py")