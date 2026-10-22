import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core'))

import xml.etree.ElementTree as ET
from actions import ACTIONS

net = ET.parse("sumo/network/network.net.xml").getroot()

tls = net.find("tlLogic[@id='center']")
if tls is None:
    raise RuntimeError("Traffic light 'center' was not found.")

phases = tls.findall("phase")
if not phases:
    raise RuntimeError("No traffic-light phases found.")

signal_count = len(phases[0].get("state", ""))

controlled_indices = {
    int(c.get("linkIndex"))
    for c in net.findall("connection")
    if c.get("tl") == "center"
}

print("Traffic light:", tls.get("id"))
print("Signal-state length:", signal_count)
print("Controlled connections:", len(controlled_indices))
print()

for action_id, action in ACTIONS.items():
    links = set(action["green_links"])
    invalid = links - controlled_indices

    print(f"Action {action_id + 1}: {action['name']}")
    print("  Requested green links:", sorted(links))

    if invalid:
        print("  ERROR: invalid signal indices:", sorted(invalid))
    else:
        print("  Index check: PASS")

    if any(index < 0 or index >= signal_count for index in links):
        print("  ERROR: an index is outside the signal-state string")
    else:
        print("  Signal-string bounds: PASS")

    print()

print("Note: index checks do not prove that simultaneous greens are conflict-free.")