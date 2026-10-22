import xml.etree.ElementTree as ET

root = ET.parse("sumo/network/network.net.xml").getroot()

print("JUNCTION FOE INFORMATION")
for junction in root.findall("junction"):
    if junction.get("id") == "center":
        print("Junction:", junction.get("id"))
        print("Request entries:")
        for request in junction.findall("request"):
            print(
                f"  index={request.get('index')}, "
                f"foes={request.get('foes')}, "
                f"response={request.get('response')}, "
                f"cont={request.get('cont')}"
            )

print("\nINTERNAL CONNECTION DETAILS")
for c in root.findall("connection"):
    if c.get("via") and c.get("tl") == "center":
        print(
            f"link={c.get('linkIndex')}, "
            f"via={c.get('via')}, "
            f"foes={c.get('foes', 'not provided')}"
        )
