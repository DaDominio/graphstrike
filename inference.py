"""
GraphStrike — OpenEnv Inference Script
=======================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM (default: HF router)
    MODEL_NAME     The model identifier for inference
    HF_TOKEN       Your Hugging Face / API key
    IMAGE_NAME     Docker image name (for from_docker_image mode)

STDOUT FORMAT:
    [START] task=<task_name> env=graphstrike model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

TWO MODES:
    1. LLM inference (default):  Uses OpenAI client to call an LLM that decides actions
    2. Library mode:             run_rule_based_episode(env, task, seed) -> float
       (used internally by /baseline endpoint — no LLM, deterministic)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from project root
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))

from models import ActionType, FakeGangAction, FakeGangObservation

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME", "graphstrike")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "graphstrike"
MAX_STEPS_OVERRIDE = None  # Use environment's max_steps
TEMPERATURE = 0.4
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Thresholds (for rule-based baseline)
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, float] = {
    "easy": 0.60,
    "medium": 0.50,
    "hard": 0.45,
}

_BOOTSTRAP_RAW_THRESHOLD = 0.40
_SHARED_IP_GANG_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM decision-making via OpenAI client
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI detective investigating a social network to find 10 coordinated
    fake accounts (a "gang"). You interact with the environment via actions:

    ACTIONS:
    - INSPECT <account_id>: Reveal full profile + follow list (costs 1 step)
    - INVESTIGATE_NETWORK <account_id>: Reveal 2-hop neighbor IDs (costs 2 steps)
    - FLAG <account_id>: Mark account as fake (free, triggers SUSPECT cascade)
    - UNFLAG <account_id>: Remove flag (free)
    - SUBMIT: End episode and get scored

    STRATEGY:
    1. INSPECT accounts to reveal their risk profiles
    2. FLAG accounts with high fake_risk_score (>= 0.60)
    3. After flagging, check suspect_ids — these are auto-elevated neighbors
    4. Always INSPECT suspects first (highest priority)
    5. SUBMIT when you've flagged 10 accounts or are running low on steps

    KEY SIGNALS:
    - fake_risk_score: Composite risk (0-1), higher = more likely fake
    - suspect_ids: Accounts auto-elevated by FLAG cascade — inspect these next
    - flagged_neighbor_count: How many flagged accounts follow this one
    - hub_legitimacy_score > 0.70: Likely a celebrity, do NOT flag

    Reply with EXACTLY one action line:
    INSPECT acc_XXXX
    FLAG acc_XXXX
    SUBMIT
""").strip()


def _format_obs_for_llm(obs_data: dict) -> str:
    """Format observation as text prompt for the LLM."""
    lines = []
    lines.append(f"TASK: {obs_data.get('task', '?').upper()} | Steps remaining: {obs_data.get('steps_remaining', '?')}")
    flagged = obs_data.get("flagged_ids", [])
    lines.append(f"Currently flagged ({len(flagged)}/10): {', '.join(flagged) if flagged else 'none'}")
    suspects = obs_data.get("suspect_ids", [])
    inspected = obs_data.get("inspected_ids", [])
    uninspected_suspects = [s for s in suspects if s not in inspected]
    if uninspected_suspects:
        lines.append(f"SUSPECTS not yet inspected ({len(uninspected_suspects)}): {', '.join(uninspected_suspects)}")
    lines.append("")

    accounts = obs_data.get("visible_accounts", [])
    if accounts:
        lines.append("PROFILED ACCOUNTS (sorted by risk):")
        sorted_accs = sorted(accounts, key=lambda a: a.get("fake_risk_score", 0), reverse=True)
        for a in sorted_accs[:15]:
            status = a.get("status", "normal").upper()
            aid = a.get("account_id", "?")
            risk = a.get("fake_risk_score", 0)
            hub = a.get("hub_legitimacy_score", 0)
            fnbr = a.get("flagged_neighbor_count", 0)
            flagged_marker = " FLAGGED" if aid in flagged else ""
            hub_marker = " [HUB]" if hub > 0.70 else ""
            fnbr_marker = f" fnbr={fnbr}(!)" if fnbr > 0 else ""
            lines.append(f"  {status:15s} {aid}{flagged_marker}: risk={risk:.3f} hub={hub:.2f}{hub_marker}{fnbr_marker}")

    visible_ids = obs_data.get("visible_account_ids", [])
    uninspected_ids = [i for i in visible_ids if i not in inspected]
    if uninspected_ids:
        lines.append(f"\nUninspected visible IDs ({len(uninspected_ids)}): {', '.join(uninspected_ids[:10])}{'...' if len(uninspected_ids) > 10 else ''}")

    lines.append(f"\nMessage: {obs_data.get('message', '')}")
    return "\n".join(lines)


def _parse_llm_action(text: str, obs_data: dict) -> str:
    """Parse LLM response into an action string like 'INSPECT acc_0042'."""
    text = text.strip()
    for line in text.split("\n"):
        line = line.strip().upper()
        if line.startswith("INSPECT ") or line.startswith("FLAG ") or line.startswith("UNFLAG "):
            return line
        if line.startswith("INVESTIGATE_NETWORK "):
            return line
        if line == "SUBMIT":
            return line
    # Fallback: inspect first uninspected suspect or visible account
    suspects = obs_data.get("suspect_ids", [])
    inspected = obs_data.get("inspected_ids", [])
    for s in suspects:
        if s not in inspected:
            return f"INSPECT {s}"
    visible = obs_data.get("visible_account_ids", [])
    for v in visible:
        if v not in inspected:
            return f"INSPECT {v}"
    return "SUBMIT"


def _action_str_to_dict(action_str: str) -> dict:
    """Convert 'INSPECT acc_0042' to {action_type: 'inspect', account_id: 'acc_0042'}."""
    parts = action_str.strip().split(maxsplit=1)
    action_type = parts[0].lower()
    account_id = parts[1] if len(parts) > 1 else None
    d = {"action_type": action_type}
    if account_id:
        d["account_id"] = account_id.lower()
    return d


# ---------------------------------------------------------------------------
# Rule-based episode runner (library API — used by /baseline)
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

        # Priority 2: Flag any inspected account exceeding thresholds
        flagged_this_turn = False
        for p in sorted(obs.visible_accounts, key=lambda x: x.fake_risk_score, reverse=True):
            if p.account_id in obs.flagged_ids:
                continue
            if p.hub_legitimacy_score > 0.75:
                continue

            bootstrap_raw = (
                0.30 * p.photo_reuse_score
                + 0.20 * p.bio_template_score
                + 0.50 * p.comment_repeat_score
            )
            should_flag = (
                p.fake_risk_score >= threshold
                or bootstrap_raw >= _BOOTSTRAP_RAW_THRESHOLD
                or p.shared_ip_count >= _SHARED_IP_GANG_THRESHOLD
            )
            if should_flag:
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
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_post(url: str, body: Optional[dict] = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _http_get(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# LLM inference loop (main entrypoint)
# ---------------------------------------------------------------------------

def run_llm_episode(base_url: str, task: str, seed: int = 0) -> float:
    """Run one episode using an LLM agent via OpenAI client.

    Connects to the environment server at base_url, uses the OpenAI-compatible
    API (HF router or any endpoint) for decision-making.
    Returns the grader score in [0.0, 1.0].
    """
    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_resp = _http_post(f"{base_url}/reset", {"task": task, "seed": seed})
        obs_data = reset_resp.get("observation", reset_resp)
        done = reset_resp.get("done", False)
        max_steps = obs_data.get("steps_remaining", 80)

        for step in range(1, max_steps + 1):
            if done:
                break

            # Build prompt from observation
            obs_text = _format_obs_for_llm(obs_data)

            # Call LLM
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_text},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                llm_text = (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                print(f"[DEBUG] LLM call failed: {exc}", flush=True)
                llm_text = ""

            # Parse action
            action_str = _parse_llm_action(llm_text, obs_data)
            action_dict = _action_str_to_dict(action_str)

            # Step environment
            step_resp = _http_post(f"{base_url}/step", action_dict)
            obs_data = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward") or 0.0
            done = step_resp.get("done", False)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Get grader score
        grader_resp = _http_get(f"{base_url}/grader")
        score = grader_resp.get("score", 0.0)
        success = score >= 0.815  # win threshold

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GraphStrike inference script")
    parser.add_argument("--url", default="http://localhost:7860",
                        help="Base URL of the running environment server")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"],
                        help="Task difficulty")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument("--local", action="store_true",
                        help="Run rule-based baseline locally (no server, no LLM)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run rule-based baseline via /baseline endpoint")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run LLM inference on all 3 tasks")
    args = parser.parse_args()

    if args.local:
        # Direct library mode — no server, no LLM
        from environment import FakeGangEnvironment  # type: ignore[import]
        env = FakeGangEnvironment()
        scores: Dict[str, float] = {}
        for t in ["easy", "medium", "hard"]:
            scores[t] = run_rule_based_episode(env, task=t, seed=0)
        print(json.dumps({"scores": scores, "agent": "rule_based"}, indent=2))

    elif args.baseline:
        # Call /baseline endpoint
        result = _http_post(f"{args.url}/baseline")
        print(json.dumps(result, indent=2))

    elif args.all_tasks:
        # LLM inference on all tasks
        scores = {}
        for t in ["easy", "medium", "hard"]:
            scores[t] = run_llm_episode(args.url, task=t, seed=args.seed)
        print(json.dumps({"scores": scores, "agent": MODEL_NAME}, indent=2))

    else:
        # Single-task LLM inference
        score = run_llm_episode(args.url, task=args.task, seed=args.seed)
        print(json.dumps({"score": score, "task": args.task, "agent": MODEL_NAME}, indent=2))
