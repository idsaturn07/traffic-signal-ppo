import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))

import xml.etree.ElementTree as ET
from actions import ACTIONS

root = ET.parse("sumo/network/network.net.xml").getroot()
junction = root.find("junction[@id='center']")

requests = {
    int(r.get("index")): r.get("foes", "")
    for r in junction.findall("request")
}

connections = {
    int(c.get("linkIndex")): c
    for c in root.findall("connection")
    if c.get("tl") == "center"
}

def movement_name(i):
    c = connections[i]
    return (
        f"{c.get('from')}[{c.get('fromLane')}]"
        f" -> {c.get('to')}[{c.get('toLane')}]"
    )

for action_id, action in ACTIONS.items():
    selected = sorted(set(action["green_links"]))
    conflicts = []

    for pos, i in enumerate(selected):
        foes = requests.get(i, "")

        for j in selected[pos + 1:]:
            if j < len(foes) and foes[-1 - j] == "1":
                conflicts.append((i, j))

    print(f"\nAction {action_id + 1}: {action['name']}")

    if conflicts:
        print("  CONFLICTS FOUND:")
        for i, j in conflicts:
            print(f"  {i} vs {j}: {movement_name(i)} | {movement_name(j)}")
    else:
        print("  No pairwise conflicts indicated by the foes data.")

print("\nThis is a diagnostic check; simulation validation is still required.")