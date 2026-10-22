# Eight candidate actions from the Wang et al. (2025) baseline.
# Signal indices refer to sumo/network/network.net.xml.

ACTIONS = {
    0: {
        "name": "NS_STRAIGHT",
        "green_links": [1, 2, 3, 4, 12, 13, 14, 15],
    },
    1: {
        "name": "EW_STRAIGHT",
        "green_links": [7, 8, 9, 18, 19, 20],
    },
    2: {
        "name": "NS_LEFT",
        "green_links": [5, 16],
    },
    3: {
        "name": "EW_LEFT",
        "green_links": [10, 21],
    },
    4: {
        "name": "NORTH_STRAIGHT_LEFT",
        "green_links": [1, 2, 3, 4, 5],
    },
    5: {
        "name": "EAST_STRAIGHT_LEFT",
        "green_links": [7, 8, 9, 10],
    },
    6: {
        "name": "SOUTH_STRAIGHT_LEFT",
        "green_links": [12, 13, 14, 15, 16],
    },
    7: {
        "name": "WEST_STRAIGHT_LEFT",
        "green_links": [18, 19, 20, 21],
    },
}

if __name__ == "__main__":
    for action_id, action in ACTIONS.items():
        print(
            f"Action {action_id + 1}: {action['name']} | "
            f"green links = {action['green_links']}"
        )