import traci

CONFIG = "sumo/research/research.sumocfg"

try:
    traci.start(["sumo", "-c", CONFIG])

    print("\n=== RESEARCH NETWORK SIGNAL PROGRAMS ===")
    signal_ids = traci.trafficlight.getIDList()
    print("Traffic-light IDs:", signal_ids)

    for tls_id in signal_ids:
        print(f"\nIntersection: {tls_id}")

        controlled_links = traci.trafficlight.getControlledLinks(tls_id)
        print("Controlled links:", len(controlled_links))

        for index, link_group in enumerate(controlled_links):
            for link in link_group:
                incoming_lane, outgoing_lane, via_lane = link
                print(
                    f"  Link {index}: {incoming_lane} -> "
                    f"{outgoing_lane} (via {via_lane})"
                )

        logic_list = traci.trafficlight.getAllProgramLogics(tls_id)

        for logic in logic_list:
            print(
                f"\n  Program: {logic.programID}, "
                f"phases: {len(logic.phases)}"
            )

            for index, phase in enumerate(logic.phases):
                print(
                    f"    Phase {index}: duration={phase.duration}s, "
                    f"state={phase.state}"
                )

finally:
    if traci.isLoaded():
        traci.close()
