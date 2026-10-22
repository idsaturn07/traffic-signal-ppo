from pathlib import Path

path = Path("traffic_env.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "import numpy as np",
    "import numpy as np\nimport xml.etree.ElementTree as ET"
)

text = text.replace(
    "        self.signal_count = None",
    '''        self.signal_count = None
        net_root = ET.parse("sumo/network/network.net.xml").getroot()
        self.lanes_by_edge = {
            edge.get("id"): [
                lane.get("id") for lane in edge.findall("lane")
            ]
            for edge in net_root.findall("edge")
        }'''
)

text = text.replace(
    "lane_ids = traci.edge.getLaneIDs(edge)",
    "lane_ids = self.lanes_by_edge[edge]"
)

text = text.replace(
    "for lane_id in traci.edge.getLaneIDs(edge)",
    "for lane_id in self.lanes_by_edge[edge]"
)

path.write_text(text, encoding="utf-8")
print("Updated traffic_env.py")