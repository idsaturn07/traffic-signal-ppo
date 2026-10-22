import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()
junction = root.find("junction[@id='center']")
requests = junction.findall("request")

connections = sorted(
    [c for c in root.findall("connection") if c.get("tl") == "center"],
    key=lambda c: int(c.get("linkIndex"))
)

print("Request index | Signal link | Movement")
for request in sorted(requests, key=lambda r: int(r.get("index"))):
    index = int(request.get("index"))
    matching = [
        c for c in connections
        if int(c.get("linkIndex")) == index
    ]

    if matching:
        c = matching[0]
        movement = (
            f"{c.get('from')}[{c.get('fromLane')}] -> "
            f"{c.get('to')}[{c.get('toLane')}]"
        )
    else:
        movement = "NO MATCH"

    print(f"{index:13} | {index:11} | {movement}")