import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()

connections = [
    c for c in root.findall("connection")
    if c.get("tl") == "center"
]

connections.sort(key=lambda c: int(c.get("linkIndex")))

for c in connections:
    print(
        f"{int(c.get('linkIndex')):2}: "
        f"{c.get('from')}[{c.get('fromLane')}] -> "
        f"{c.get('to')}[{c.get('toLane')}] "
        f"({c.get('dir')})"
    )
