import xml.etree.ElementTree as ET

net_file = "sumo/network/network.net.xml"
root = ET.parse(net_file).getroot()

junction = root.find(".//junction[@id='center']")
if junction is None:
    raise RuntimeError("Junction 'center' not found.")

requests = junction.findall("request")

print("JUNCTION:", junction.get("id"))
print("REQUEST ENTRIES:", len(requests))
print("\nRIGHT-TURN LINKS AND THEIR CONFLICT DATA")

right_turns = {0, 6, 11, 17}

for i in sorted(right_turns):
    req = next(
        (r for r in requests if int(r.get("index", "-1")) == i),
        None
    )

    if req is None:
        print(f"\nLink {i}: No matching request entry found.")
        continue

    foes = req.get("foes", "")
    response = req.get("response", "")

    print(f"\nLink {i}")
    print("  Foes bit string:    ", foes)
    print("  Response bit string:", response)

    conflicting = []
    for j, bit in enumerate(reversed(foes)):
        if bit == "1":
            conflicting.append(j)

    print("  Foe request indices:", conflicting)
