def patch(path, replacements):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    for old, new in replacements:
        if old not in content:
            raise RuntimeError(f"Anchor not found in {path}:\n{old!r}")
        content = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Patched {path}")


patch("evaluate_per_intersection.py", [
    (
        "yellow_until = {j: -1 for j in INTERSECTIONS}\n\n    while traci.simulation.getTime() < duration:",
        "yellow_until = {j: -1 for j in INTERSECTIONS}\n    total_teleports = 0\n\n    while traci.simulation.getTime() < duration:",
    ),
    (
        "traci.simulationStep()\n\n    final_queues = {j: queue_for(j) for j in INTERSECTIONS}",
        "traci.simulationStep()\n        total_teleports += traci.simulation.getStartingTeleportNumber()\n\n    final_queues = {j: queue_for(j) for j in INTERSECTIONS}",
    ),
    (
        "return final_queues\n",
        "return final_queues, total_teleports\n",
    ),
    (
        "fq = run_episode(method, 600, scale, seed, nets=nets)",
        "fq, teleports = run_episode(method, 600, scale, seed, nets=nets)",
    ),
    (
        "row.update(fq)",
        "row.update(fq)\n                row[\"teleports\"] = teleports",
    ),
    (
        "I2={fq['I2']} I3={fq['I3']}",
        "I2={fq['I2']} I3={fq['I3']} teleports={teleports}",
    ),
    (
        '["method", "demand_scale", "seed", "I1", "I2", "I3"]',
        '["method", "demand_scale", "seed", "I1", "I2", "I3", "teleports"]',
    ),
])

patch("evaluate_demand_shift.py", [
    (
        "yellow_until = {j: -1 for j in INTERSECTIONS}\n\n    phase_states = {}",
        "yellow_until = {j: -1 for j in INTERSECTIONS}\n    total_teleports = 0\n\n    phase_states = {}",
    ),
    (
        "traci.simulationStep()\n        current_time = traci.simulation.getTime()",
        "traci.simulationStep()\n        total_teleports += traci.simulation.getStartingTeleportNumber()\n        current_time = traci.simulation.getTime()",
    ),
    (
        "return arrived, mean_tt, final_queue\n",
        "return arrived, mean_tt, final_queue, total_teleports\n",
    ),
    (
        "arrived, mean_tt, final_q = run_eval_episode(nets, 600, scale, seed)",
        "arrived, mean_tt, final_q, teleports = run_eval_episode(nets, 600, scale, seed)",
    ),
    (
        "final_queue={final_q}",
        "final_queue={final_q} teleports={teleports}",
    ),
    (
        "rows.append([variant_name, scale, seed, arrived, mean_tt, final_q])",
        "rows.append([variant_name, scale, seed, arrived, mean_tt, final_q, teleports])",
    ),
    (
        '["method", "demand_scale", "seed", "arrived", "mean_travel_time", "final_queue"]',
        '["method", "demand_scale", "seed", "arrived", "mean_travel_time", "final_queue", "teleports"]',
    ),
])