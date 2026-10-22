import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()
tls = root.find("tlLogic[@id='center']")

print("TRAFFIC LIGHT:", tls.get("id"))
print("SIGNAL LINKS:", len(tls.findall("phase")[0].get("state", "")))

print("\nLINK INDEX -> MOVEMENT")
for c in sorted(
    [c for c in root.findall("connection") if c.get("tl") == "center"],
    key=lambda c: int(c.get("linkIndex"))
):
    print(
        f"linkIndex={int(c.get('linkIndex')):2} | "
        f"{c.get('from')}[{c.get('fromLane')}] -> "
        f"{c.get('to')}[{c.get('toLane')}] | "
        f"turn={c.get('dir')}"
    )

print("\nJUNCTION REQUEST COUNT")
junction = root.find("junction[@id='center']")
print("Request entries:", len(junction.findall("request")))