import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()

for connection in root.findall("connection"):
    if connection.get("tl") is not None:
        print(
            f"signal_index={connection.get('linkIndex')}, "
            f"from={connection.get('from')}[{connection.get('fromLane')}], "
            f"to={connection.get('to')}[{connection.get('toLane')}], "
            f"direction={connection.get('dir')}"
        )

print("\n--- PHASES ---")
for tls in root.findall("tlLogic"):
    print("Traffic light:", tls.get("id"))
    for i, phase in enumerate(tls.findall("phase")):
        print(i, "duration =", phase.get("duration"),
              "state =", phase.get("state"))
