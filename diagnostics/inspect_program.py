import traci

CONFIG = "sumo/config/simulation.sumocfg"

try:
    traci.start(["sumo", "-c", CONFIG])

    tls_id = "center"

    print("\n--- TRAFFIC LIGHT PROGRAM ---")
    print("Signal ID:", tls_id)

    for program in traci.trafficlight.getAllProgramLogics(tls_id):
        print("\nProgram ID:", program.programID)
        print("Number of phases:", len(program.phases))

        for i, phase in enumerate(program.phases):
            print(
                f"Phase {i}: duration={phase.duration}s, "
                f"state={phase.state}"
            )

    print("\n--- SIGNAL-LINK CONNECTIONS ---")
    for connection in traci.trafficlight.getControlledLinks(tls_id):
        print(connection)

finally:
    if traci.isLoaded():
        traci.close()
