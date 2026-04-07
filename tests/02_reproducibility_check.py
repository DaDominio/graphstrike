import requests
import json
import time

BASE_URL = "https://pandago-graphstrike.hf.space"
TASKS = ["easy", "medium", "hard"]
SEED = 42

def reset_task(task, seed=SEED):
    return requests.post(f"{BASE_URL}/reset", json={"task": task, "seed": seed}, timeout=20)

def step_action(action_type, account_id=None):
    payload = {"action_type": action_type}
    if account_id is not None:
        payload["account_id"] = account_id
    return requests.post(f"{BASE_URL}/step", json=payload, timeout=20)

def get_visible_ids(reset_json):
    return reset_json.get("observation", {}).get("visible_account_ids", [])

def run_episode(task, seed=SEED):
    rr = reset_task(task, seed)
    if rr.status_code != 200:
        return {"error": f"reset failed: {rr.status_code} {rr.text[:200]}"}

    data = rr.json()
    visible_ids = get_visible_ids(data)
    if not visible_ids:
        return {"error": "no visible_account_ids in reset output"}

    acc = visible_ids[0]
    history = []

    for action_type, account_id in [
        ("inspect", acc),
        ("flag", acc),
        ("submit", None),
    ]:
        rs = step_action(action_type, account_id)
        if rs.status_code != 200:
            return {"error": f"step failed: {action_type} {rs.status_code} {rs.text[:200]}"}

        out = rs.json()
        history.append({
            "action_type": action_type,
            "account_id": account_id,
            "reward": out.get("reward"),
            "done": out.get("done"),
            "message": out.get("message") or out.get("observation", {}).get("message")
        })

    rg = requests.get(f"{BASE_URL}/grader", timeout=20)
    if rg.status_code != 200:
        return {"error": f"grader failed: {rg.status_code} {rg.text[:200]}"}

    grader = rg.json()
    score = grader.get("score", grader.get("final_score", grader.get("reward")))

    return {
        "task": task,
        "seed": seed,
        "target_account": acc,
        "history": history,
        "final_score": score,
    }

print("=" * 70)
print("REPRODUCIBILITY CHECK")
print("=" * 70)

for task in TASKS:
    print(f"\nTask: {task}")
    run1 = run_episode(task, SEED)
    time.sleep(1)
    run2 = run_episode(task, SEED)

    print("Run 1:")
    print(json.dumps(run1, indent=2)[:1200])

    print("Run 2:")
    print(json.dumps(run2, indent=2)[:1200])

    if run1 == run2:
        print("RESULT: REPRODUCIBLE")
    else:
        print("RESULT: NOT REPRODUCIBLE")