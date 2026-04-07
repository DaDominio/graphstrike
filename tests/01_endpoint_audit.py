import requests
import json

BASE_URL = "https://pandago-graphstrike.hf.space"
TASKS = ["easy", "medium", "hard"]
SEED = 42

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def short(obj, n=900):
    try:
        return json.dumps(obj, indent=2)[:n]
    except Exception:
        return str(obj)[:n]

def reset_task(task, seed=SEED):
    return requests.post(f"{BASE_URL}/reset", json={"task": task, "seed": seed}, timeout=20)

def step_action(action_type, account_id=None):
    payload = {"action_type": action_type}
    if account_id is not None:
        payload["account_id"] = account_id
    return requests.post(f"{BASE_URL}/step", json=payload, timeout=20)

def get_visible_ids(reset_json):
    return reset_json.get("observation", {}).get("visible_account_ids", [])

print_section("1) ROOT / HEALTH")
try:
    r = requests.get(BASE_URL, timeout=15)
    print("Status:", r.status_code)
    print("Body preview:", r.text[:300])
except Exception as e:
    print("Root failed:", e)

print_section("2) /tasks")
tasks_json = None
try:
    r = requests.get(f"{BASE_URL}/tasks", timeout=15)
    print("Status:", r.status_code)
    if r.status_code == 200:
        tasks_json = r.json()
        print(short(tasks_json, 1200))
except Exception as e:
    print("/tasks failed:", e)

print_section("3) TASK SEPARATION VIA /reset")
reset_snapshots = {}
for task in TASKS:
    try:
        r = reset_task(task)
        print(f"\nTask={task} | Status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            reset_snapshots[task] = data
            print(short(data, 900))
        else:
            print(r.text[:400])
    except Exception as e:
        print(f"Task={task} reset failed:", e)

print("\nTask comparison:")
pairs = [("easy", "medium"), ("easy", "hard"), ("medium", "hard")]
for a, b in pairs:
    if a in reset_snapshots and b in reset_snapshots:
        print(f"{a} vs {b}:", "DIFFERENT" if reset_snapshots[a] != reset_snapshots[b] else "IDENTICAL")

print_section("4) /state")
try:
    reset_task("easy")
    r = requests.get(f"{BASE_URL}/state", timeout=15)
    print("Status:", r.status_code)
    if r.status_code == 200:
        print(short(r.json(), 800))
    else:
        print(r.text[:400])
except Exception as e:
    print("/state failed:", e)

print_section("5) VALID ACTION FLOW")
for task in TASKS:
    print(f"\n--- Task: {task} ---")
    try:
        r = reset_task(task)
        if r.status_code != 200:
            print("Reset failed:", r.status_code, r.text[:300])
            continue

        data = r.json()
        visible_ids = get_visible_ids(data)
        print("Visible IDs count:", len(visible_ids))

        if not visible_ids:
            print("No visible account ids found.")
            continue

        acc = visible_ids[0]
        print("Using account_id:", acc)

        for action_type in ["inspect", "investigate_network", "flag", "unflag"]:
            rs = step_action(action_type, acc)
            print(f"{action_type:20s} -> {rs.status_code}")
            if rs.status_code == 200:
                print(short(rs.json(), 700))
            else:
                print(rs.text[:300])
    except Exception as e:
        print("Action flow failed:", e)

print_section("6) GRADER CHECK AFTER SUBMIT")
for task in TASKS:
    print(f"\n--- Task: {task} ---")
    try:
        r = reset_task(task)
        if r.status_code != 200:
            print("Reset failed:", r.status_code, r.text[:300])
            continue

        visible_ids = get_visible_ids(r.json())
        if not visible_ids:
            print("No visible IDs.")
            continue

        acc = visible_ids[0]

        r1 = step_action("inspect", acc)
        r2 = step_action("flag", acc)
        r3 = step_action("submit")

        print("inspect ->", r1.status_code)
        print("flag    ->", r2.status_code)
        print("submit  ->", r3.status_code)

        rg = requests.get(f"{BASE_URL}/grader", timeout=20)
        print("grader  ->", rg.status_code)

        if rg.status_code == 200:
            g = rg.json()
            print(short(g, 600))
            score = g.get("score", g.get("final_score", g.get("reward")))
            print("Extracted final score:", score)
            try:
                score = float(score)
                print("In [0,1]:", 0.0 <= score <= 1.0)
            except Exception:
                print("Could not parse score as float.")
        else:
            print(rg.text[:300])
    except Exception as e:
        print("Grader check failed:", e)

print_section("7) /baseline")
try:
    r = requests.get(f"{BASE_URL}/baseline", timeout=60)
    print("Status:", r.status_code)
    if r.status_code == 200:
        print(short(r.json(), 1000))
    else:
        print(r.text[:400])
except Exception as e:
    print("/baseline failed:", e)

print_section("8) SUMMARY HINTS")
print("- If /tasks lists easy, medium, hard: task design is exposed correctly.")
print("- If /reset changes state by task: single environment with task-specific behavior works.")
print("- If /grader returns score in [0,1]: final scoring is compliant.")
print("- If /baseline is 404: likely missing a required endpoint.")