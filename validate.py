"""Pre-submission validator for the Fake Gang Detection OpenEnv environment.

Checks all submission requirements and prints pass/fail for each.
Exits 0 if all checks pass, 1 if any fail.

Usage:
    python validate.py                        # server must be running on :8000
    python validate.py --url http://host:8001
    python validate.py --local                # skip HTTP checks, test locally
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "[PASS]"
_FAIL = "[FAIL]"
_results: List[Tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = _PASS if ok else _FAIL
    line = f"  {tag} {name}"
    if detail and not ok:
        line += f"\n         → {detail}"
    print(line)
    _results.append((ok, name))
    return ok


def _get(url: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read()), None
    except Exception as exc:
        return None, str(exc)


def _post(url: str, body: Any = None) -> Tuple[Optional[dict], Optional[str]]:
    try:
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as exc:
        try:
            body_bytes = exc.read()
            return None, f"HTTP {exc.code}: {body_bytes.decode()}"
        except Exception:
            return None, f"HTTP {exc.code}"
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# HTTP checks
# ---------------------------------------------------------------------------

def run_http_checks(base_url: str) -> None:
    print(f"\nHTTP checks against {base_url}\n")

    # /health
    data, err = _get(f"{base_url}/health")
    check("/health reachable", data is not None and data.get("status") == "healthy", err or "")

    # /tasks — must have action_schema and 3 tasks
    data, err = _get(f"{base_url}/tasks")
    if check("/tasks reachable", data is not None, err or ""):
        has_schema = isinstance(data.get("action_schema"), dict)
        has_3_tasks = isinstance(data.get("tasks"), list) and len(data["tasks"]) == 3
        has_score_range = "score_range" in data
        check("/tasks has action_schema", has_schema, str(data))
        check("/tasks has 3 tasks", has_3_tasks, str(data))
        check("/tasks has score_range", has_score_range, str(data))

    # /reset for each task
    for task in ["easy", "medium", "hard"]:
        data, err = _post(f"{base_url}/reset", {"task": task, "seed": 0})
        check(f"/reset task={task}", data is not None and "observation" in data, err or "")

    # /step — INSPECT, FLAG, SUBMIT cycle
    _post(f"{base_url}/reset", {"task": "easy", "seed": 0})
    obs_resp, err = _post(f"{base_url}/step", {"action_type": "inspect",
                                                "account_id": "acc_0000"})
    check("/step INSPECT", obs_resp is not None, err or "")

    # Get a visible account ID from the observation to flag
    acc_to_flag = None
    if obs_resp:
        vis_ids = obs_resp.get("observation", {}).get("visible_account_ids", [])
        if vis_ids:
            acc_to_flag = vis_ids[0]

    if acc_to_flag:
        flag_resp, err = _post(f"{base_url}/step", {"action_type": "flag",
                                                     "account_id": acc_to_flag})
        check("/step FLAG", flag_resp is not None, err or "")

    sub_resp, err = _post(f"{base_url}/step", {"action_type": "submit"})
    check("/step SUBMIT", sub_resp is not None and sub_resp.get("done") is True, err or "")

    # /grader — must return float in [0, 1]
    data, err = _get(f"{base_url}/grader")
    if check("/grader reachable", data is not None, err or ""):
        score = data.get("score")
        check("/grader returns [0,1] float",
              isinstance(score, (int, float)) and 0.0 <= score <= 1.0,
              f"score={score}")

    # /baseline — must return 3 task scores in [0, 1]
    data, err = _post(f"{base_url}/baseline")
    if check("/baseline reachable", data is not None, err or ""):
        scores = data.get("scores", {})
        all_valid = (
            set(scores.keys()) == {"easy", "medium", "hard"}
            and all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0
                    for v in scores.values())
        )
        check("/baseline returns 3 valid scores", all_valid,
              f"got: {scores}")


# ---------------------------------------------------------------------------
# Local checks (no server needed)
# ---------------------------------------------------------------------------

def run_local_checks() -> None:
    print("\nLocal checks\n")

    # scoring.py importable and correct
    try:
        from scoring import (  # type: ignore[import]
            compute_fake_risk, compute_hub_legitimacy, grader_score
        )
        gang_risk = compute_fake_risk(0.75, 0.65, 0.85, 0.10)
        hub = compute_hub_legitimacy(2_000_000, 200, 2000, 0.05)
        celeb_risk = compute_fake_risk(0.02, 0.02, 0.10, hub)
        # Perfect score: 10 TP, 0 FP, 0 FN, 0 steps used → efficiency=1.0 → score=1.0
        perfect = grader_score(10, 0, 0, 0, 30)
        ok = (gang_risk >= 0.60 and celeb_risk < 0.20 and perfect == 1.0)
        check("scoring.py math correct", ok,
              f"gang_risk={gang_risk} celeb_risk={celeb_risk} perfect={perfect}")
    except Exception as exc:
        check("scoring.py importable", False, str(exc))

    # models.py has AccountStatus + new fields
    try:
        from models import AccountStatus, AccountProfile, FakeGangObservation  # type: ignore[import]
        p = AccountProfile(
            account_id="acc_0001", follower_count=100, following_count=50,
            post_count=10, avg_post_hour=14.0, photo_reuse_score=0.8,
            bio_template_score=0.7, account_age_days=60,
        )
        check("models.py AccountProfile has fake_risk_score",
              hasattr(p, "fake_risk_score"), "")
        check("models.py FakeGangObservation has suspect_ids",
              hasattr(FakeGangObservation(), "suspect_ids"), "")
        check("models.py AccountStatus enum exists",
              AccountStatus.SUSPECT == "suspect", "")
    except Exception as exc:
        check("models.py new fields", False, str(exc))

    # environment.py runs episode + status cascade
    try:
        from environment import FakeGangEnvironment  # type: ignore[import]
        from models import FakeGangAction, ActionType  # type: ignore[import]
        env = FakeGangEnvironment()
        obs = env.reset(task="easy", seed=0)
        ep_path = _ROOT / "episodes" / "easy_000.json"
        if ep_path.exists():
            gang_id = json.loads(ep_path.read_text())["gang_member_ids"][0]
            obs = env.step(FakeGangAction(action_type=ActionType.INSPECT, account_id=gang_id))
            obs = env.step(FakeGangAction(action_type=ActionType.FLAG, account_id=gang_id))
            cascade_ok = len(obs.suspect_ids) > 0
            check("environment.py SUSPECT cascade works", cascade_ok,
                  f"suspect_ids={obs.suspect_ids[:3]}")
            p_flagged = next((p for p in obs.visible_accounts if p.account_id == gang_id), None)
            check("environment.py fake_risk_score computed",
                  p_flagged is not None and p_flagged.fake_risk_score > 0, "")
        else:
            check("episode file exists for cascade test", False,
                  f"run python server/generator.py first")
    except Exception as exc:
        check("environment.py status cascade", False, str(exc))

    # inference.py importable + runs one episode locally
    try:
        from inference import run_rule_based_episode  # type: ignore[import]
        from environment import FakeGangEnvironment  # type: ignore[import]
        env2 = FakeGangEnvironment()
        score = run_rule_based_episode(env2, task="easy", seed=1)
        check("inference.py runs locally",
              isinstance(score, float) and 0.0 <= score <= 1.0,
              f"score={score}")
    except Exception as exc:
        check("inference.py importable", False, str(exc))

    # Episodes have new features
    ep_path = _ROOT / "episodes" / "easy_000.json"
    if ep_path.exists():
        ep = json.loads(ep_path.read_text())
        accounts = ep["network"]["accounts"]
        first = accounts[0]["features"]
        has_features = "comment_repeat_score" in first and "shared_ip_count" in first
        check("episodes have new features (comment_repeat_score, shared_ip_count)",
              has_features, f"keys: {list(first.keys())}")
        has_celebs = "celeb_ids" in ep
        check("episodes have celeb_ids field", has_celebs, "")
    else:
        check("episodes directory has files", False, "run python server/generator.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--local", action="store_true",
                        help="Run local checks only (no server needed)")
    args = parser.parse_args()

    run_local_checks()

    if not args.local:
        run_http_checks(args.url)

    total = len(_results)
    passed = sum(1 for ok, _ in _results if ok)
    failed = total - passed

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        failed_names = [name for ok, name in _results if not ok]
        for name in failed_names:
            print(f"  - {name}")
        print()
        sys.exit(1)
    else:
        print("  — all OK")
        print()
        sys.exit(0)
