import xml.etree.ElementTree as ET

edges_root = ET.parse("sumo/network/edges.edg.xml").getroot()
connections_root = ET.parse("sumo/network/connections.con.xml").getroot()

edges = {
    edge.get("id"): int(edge.get("numLanes"))
    for edge in edges_root.findall("edge")
}

errors = []

for c in connections_root.findall("connection"):
    source = c.get("from")
    target = c.get("to")
    from_lane = int(c.get("fromLane"))
    to_lane = int(c.get("toLane"))

    if source not in edges or target not in edges:
        errors.append(f"Unknown edge: {source} -> {target}")
        continue

    if from_lane >= edges[source]:
        errors.append(f"Invalid source lane: {source}[{from_lane}]")

    if to_lane >= edges[target]:
        errors.append(f"Invalid target lane: {target}[{to_lane}]")

if errors:
    print("VALIDATION FAILED:")
    for error in errors:
        print("-", error)
else:
    print(f"Basic validation passed: {len(connections_root.findall('connection'))} connections.")
    print("Next we must verify the turn directions and signal compatibility.")
