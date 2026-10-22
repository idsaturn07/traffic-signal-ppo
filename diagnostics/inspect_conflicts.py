import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()

print("JUNCTIONS AT THE INTERSECTION")
for junction in root.findall("junction"):
    if junction.get("id") == "center":
        print("ID:", junction.get("id"))
        print("Type:", junction.get("type"))
        print("Incoming lanes:", junction.get("incLanes"))
        print("Internal lanes:", junction.get("intLanes"))

print("\nCONTROLLED CONNECTIONS")
for c in sorted(
    [c for c in root.findall("connection") if c.get("tl") == "center"],
    key=lambda x: int(x.get("linkIndex"))
):
    print(
        f"Index {c.get('linkIndex'):>2}: "
        f"{c.get('from')}[{c.get('fromLane')}] -> "
        f"{c.get('to')}[{c.get('toLane')}] "
        f"turn={c.get('dir')} "
        f"via={c.get('via', 'N/A')}"
    )
