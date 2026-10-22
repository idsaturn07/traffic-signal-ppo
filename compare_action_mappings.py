from actions import ACTIONS
from traffic_env import ACTIONS as ENV_ACTIONS

print("Checking all 8 action mappings...\n")

all_match = True

for action_id in range(8):
    checked = set(ACTIONS[action_id]["green_links"])
    environment = set(ENV_ACTIONS[action_id])

    match = checked == environment
    print(
        f"Action {action_id + 1}: "
        f"{'MATCH' if match else 'MISMATCH'}"
    )

    if not match:
        print("  Checked:", sorted(checked))
        print("  Environment:", sorted(environment))
        all_match = False

print("\nRESULT:", "PASS" if all_match else "FAIL")
