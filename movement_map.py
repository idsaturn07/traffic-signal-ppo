import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()

connections = sorted(
    [c for c in root.findall("connection") if c.get("tl") is not None],
    key=lambda c: int(c.get("linkIndex"))
)

groups = {
    "NS_STRAIGHT": [],
    "NS_LEFT": [],
    "EW_STRAIGHT": [],
    "EW_LEFT": [],
    "RIGHT_TURNS": [],
    "TURNAROUNDS": [],
    "OTHER": []
}

for c in connections:
    source = c.get("from")
    direction = c.get("dir")
    index = int(c.get("linkIndex"))

    if direction == "t":
        group = "TURNAROUNDS"
    elif direction == "r":
        group = "RIGHT_TURNS"
    elif source in ("north_in", "south_in") and direction == "s":
        group = "NS_STRAIGHT"
    elif source in ("north_in", "south_in") and direction == "l":
        group = "NS_LEFT"
    elif source in ("east_in", "west_in") and direction == "s":
        group = "EW_STRAIGHT"
    elif source in ("east_in", "west_in") and direction == "l":
        group = "EW_LEFT"
    else:
        group = "OTHER"

    groups[group].append(index)

for name, indices in groups.items():
    print(f"{name}: {indices}")