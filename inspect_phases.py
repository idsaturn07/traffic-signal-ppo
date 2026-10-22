import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()

for tls in root.findall("tlLogic"):
    print(f"\nTraffic light: {tls.get('id')}")
    print(f"Program ID: {tls.get('programID')}")
    print(f"Type: {tls.get('type')}")

    for i, phase in enumerate(tls.findall("phase")):
        print(f"\nPhase {i}")
        print("  Duration:", phase.get("duration"))
        print("  State:   ", phase.get("state"))
        print("  State length:", len(phase.get("state", "")))
