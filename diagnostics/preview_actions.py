from actions import ACTIONS
import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()
tls = root.find("tlLogic[@id='center']")
signal_count = len(tls.findall("phase")[0].get("state"))

for action_id, action in ACTIONS.items():
    state = ["r"] * signal_count

    for index in action["green_links"]:
        state[index] = "G"

    print(f"\nAction {action_id + 1}: {action['name']}")
    print("".join(state))
