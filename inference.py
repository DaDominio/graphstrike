"""Inference script for the Fake Gang Detection environment.

Two modes:
  1. Direct (library):  call run_rule_based_episode(env, task, seed) → float
  2. HTTP client:       python inference.py [--url URL] → prints JSON with baseline scores

The rule-based strategy uses fake_risk_score thresholds — no LLM required.
It is the same strategy called by the /baseline server endpoint.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# Allow running from project root
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))

from models import ActionType, FakeGangAction, FakeGangObservation

# ---------------------------------------------------------------------------
# Per-task thresholds (tune down for harder tasks since signals are noisier)
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, float] = {
    "easy": 0.60,
    "medium": 0.50,
    "hard": 0.45,
}


# ---------------------------------------------------------------------------
# Rule-based episode runner (library API)
# ---------------------------------------------------------------------------

def run_rule_based_episode(env, task: str, seed: int = 0) -> float:
    """Run one complete episode using the rule-based policy.

    Returns the grader_score in [0.0, 1.0].
    Called directly by the /baseline endpoint (no HTTP overhead).
    """
    obs: FakeGangObservation = env.reset(task=task, seed=seed)
    threshold = THRESHOLDS[task]

    while not obs.done:
        # Priority 1: Inspect SUSPECT accounts (auto-elevated by FLAG cascade)
        uninspected_suspects = [s for s in obs.suspect_ids if s not in obs.inspected_ids]
        if uninspected_suspects:
            obs = env.step(FakeGangAction(action_type=ActionType.INSPECT,
                                          account_id=uninspected_suspects[0]))
            continue

        # Priority 2: Flag any inspected account above risk threshold
        flagged_this_turn = False
        for p in obs.visible_accounts:
            if p.fake_risk_score >= threshold and p.account_id not in obs.flagged_ids:
                obs = env.step(FakeGangAction(action_type=ActionType.FLAG,
                                              account_id=p.account_id))
                flagged_this_turn = True
                break

        if flagged_this_turn:
            continue

        # Priority 3: Inspect the highest-risk uninspected account
        uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]
        if uninspected and obs.steps_remaining > 3:
            obs = env.step(FakeGangAction(action_type=ActionType.INSPECT,
                                          account_id=uninspected[0]))
        else:
            obs = env.step(FakeGangAction(action_type=ActionType.SUBMIT))
            break

        if obs.steps_remaining <= 1 and not obs.done:
            env.step(FakeGangAction(action_type=ActionType.SUBMIT))
            break

    return env._last_grader_score


# ---------------------------------------------------------------------------
# HTTP client mode (command-line entrypoint)
# ---------------------------------------------------------------------------

def _http_post(url: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def run_via_http(base_url: str) -> Dict[str, float]:
    """Call the /baseline endpoint and return {task: score} dict."""
    result = _http_post(f"{base_url}/baseline")
    return result["scores"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rule-based baseline inference")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Base URL of the running environment server")
    parser.add_argument("--local", action="store_true",
                        help="Run directly (no server needed) — import environment locally")
    args = parser.parse_args()

    if args.local:
        from environment import FakeGangEnvironment  # type: ignore[import]
        env = FakeGangEnvironment()
        scores: Dict[str, float] = {}
        for task in ["easy", "medium", "hard"]:
            scores[task] = run_rule_based_episode(env, task=task, seed=0)
    else:
        scores = run_via_http(args.url)

    print(json.dumps({"scores": scores, "agent": "rule_based"}, indent=2))
