"""
GraphStrike — OpenEnv Inference Script
=======================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM (default: HF router)
    MODEL_NAME     The model identifier for inference
    HF_TOKEN       Your Hugging Face / API key
    LOCAL_IMAGE_NAME  Docker image name (optional, for from_docker_image mode)

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

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional — from_docker_image mode

# Resolved API key: HF_TOKEN is primary, API_KEY is fallback
API_KEY = HF_TOKEN or os.getenv("API_KEY")

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
    You are an AI detective finding 10 coordinated fake accounts in a social network.

    ━━━ ROUND 2: PLATFORM-ADAPTIVE DETECTION ━━━
    Episodes run on specific platforms (Instagram/Snapchat) with different thresholds and enforcement priorities.

    ACTIONS (reply with exactly ONE line):
    - GET_POLICY              — get platform policy (FREE, call first!)
    - INSPECT acc_XXXX        — reveal profile (costs 1 step)
    - REVERSE_IMAGE_SEARCH acc_XXXX  — reveal photo_reuse_score (costs 1 step)
    - ANALYZE_BIO acc_XXXX    — reveal bio_template_score (costs 1 step)
    - CHECK_IP acc_XXXX       — reveal ip_cluster_id (costs 2 steps, expensive!)
    - FLAG acc_XXXX           — mark as fake (FREE, no step cost, triggers suspect cascade)
    - SUBMIT                  — end episode, get scored

    DECISION RULES (Round 2, apply top-to-bottom):
    1. First action of episode → GET_POLICY (learn platform threshold and primary signal)
    2. If suspect_ids lists accounts you haven't inspected → INSPECT the first one
    3. If ANY profiled account has shared_ip_count >= 5 → CHECK_IP to confirm cluster, then FLAG
    4. If photo_reuse_score or bio_template_score is 0.0 on suspicious accounts → use REVERSE_IMAGE_SEARCH or ANALYZE_BIO
    5. If ANY profiled account has photo_reuse >= 0.50 AND bio_template >= 0.40 and hub < 0.70 → FLAG
    6. If fake_risk_score >= platform_threshold and hub < 0.70 → FLAG
    7. If uninspected visible accounts and steps > 3 → INSPECT the next one
    8. If you have 10 flags OR steps <= 3 → SUBMIT

    PLATFORM-SPECIFIC STRATEGIES:
    - Instagram (threshold ~0.08, high FP penalty): Be precise, use REVERSE_IMAGE_SEARCH on borderline cases
    - Snapchat (threshold ~0.74, low FP penalty): Be aggressive, flag when fake_risk >= 0.74

    IMPORTANT:
    - Hidden signals (photo_reuse, bio_template, ip_cluster) start as 0.0/None — use tools to reveal!
    - GET_POLICY is FREE and shows platform threshold — always call first
    - FLAG is FREE (costs 0 steps) — flag aggressively when you see suspicious signals
    - CHECK_IP costs 2 steps (expensive) — only use when shared_ip_count >= 5
    - hub_legitimacy_score > 0.70 means celebrity — do NOT flag

    Reply with EXACTLY one line, nothing else:
    GET_POLICY
    REVERSE_IMAGE_SEARCH acc_XXXX
    ANALYZE_BIO acc_XXXX
    CHECK_IP acc_XXXX
    FLAG acc_XXXX
    INSPECT acc_XXXX
    SUBMIT
""").strip()


def _format_obs_for_llm(obs_data: dict) -> str:
    """Format observation as text prompt for the LLM — shows raw signals prominently."""
    lines = []

    # Round 2: Show platform context
    platform = obs_data.get("platform", "Unknown")
    platform_info = f" | PLATFORM: {platform}" if platform != "Unknown" else ""

    lines.append(f"TASK: {obs_data.get('task', '?').upper()}{platform_info} | Steps remaining: {obs_data.get('steps_remaining', '?')}")
    flagged = obs_data.get("flagged_ids", [])
    lines.append(f"Flagged ({len(flagged)}/10): {', '.join(flagged) if flagged else 'none'}")

    suspects = obs_data.get("suspect_ids", [])
    inspected = obs_data.get("inspected_ids", [])
    uninspected_suspects = [s for s in suspects if s not in inspected]
    if uninspected_suspects:
        lines.append(f"*** SUSPECTS (uninspected) → INSPECT THESE FIRST: {', '.join(uninspected_suspects)} ***")

    accounts = obs_data.get("visible_accounts", [])
    if accounts:
        unflagged_suspicious = []
        flagged_accs = []
        clean_accs = []
        for a in sorted(accounts, key=lambda x: x.get("fake_risk_score", 0), reverse=True):
            aid = a.get("account_id", "?")
            if aid in flagged:
                flagged_accs.append(a)
            elif (a.get("shared_ip_count", 0) >= 5 or
                  (a.get("photo_reuse_score", 0) >= 0.50 and a.get("bio_template_score", 0) >= 0.40)):
                unflagged_suspicious.append(a)
            else:
                clean_accs.append(a)

        if unflagged_suspicious:
            lines.append(f"\n!!! ACTION NEEDED — FLAG THESE ({len(unflagged_suspicious)} suspicious):")
            for a in unflagged_suspicious:
                aid = a.get("account_id", "?")
                lines.append(f"  → FLAG {aid}: risk={a.get('fake_risk_score',0):.3f} photo={a.get('photo_reuse_score',0):.2f} bio={a.get('bio_template_score',0):.2f} ip_shared={a.get('shared_ip_count',0)} hub={a.get('hub_legitimacy_score',0):.2f}")

        if flagged_accs:
            lines.append(f"\nALREADY FLAGGED ({len(flagged_accs)}):")
            for a in flagged_accs[:5]:
                lines.append(f"  ✓ {a.get('account_id','?')}")

        if clean_accs:
            lines.append(f"\nCLEAN ({len(clean_accs)}):")
            for a in clean_accs[:8]:
                aid = a.get("account_id", "?")
                hub = a.get("hub_legitimacy_score", 0)
                hub_mark = " [CELEBRITY]" if hub > 0.70 else ""
                lines.append(f"  {aid}: risk={a.get('fake_risk_score',0):.3f} photo={a.get('photo_reuse_score',0):.2f} bio={a.get('bio_template_score',0):.2f} hub={hub:.2f}{hub_mark}")

    visible_ids = obs_data.get("visible_account_ids", [])
    uninspected_ids = [i for i in visible_ids if i not in inspected]
    if uninspected_ids:
        lines.append(f"\nUninspected IDs ({len(uninspected_ids)}): {', '.join(uninspected_ids[:10])}{'...' if len(uninspected_ids) > 10 else ''}")

    lines.append(f"\nMessage: {obs_data.get('message', '')}")
    return "\n".join(lines)


def _parse_llm_action(text: str, obs_data: dict) -> str:
    """Parse LLM response into an action string like 'INSPECT acc_0042'."""
    text = text.strip()
    for line in text.split("\n"):
        line = line.strip()
        parts = line.split(maxsplit=1)
        verb = parts[0].upper()
        acc = parts[1].lower() if len(parts) > 1 else None

        # Round 2: New tool actions
        if verb == "GET_POLICY":
            return "GET_POLICY"
        if verb == "REVERSE_IMAGE_SEARCH" and acc:
            return f"REVERSE_IMAGE_SEARCH {acc}"
        if verb == "ANALYZE_BIO" and acc:
            return f"ANALYZE_BIO {acc}"
        if verb == "CHECK_IP" and acc:
            return f"CHECK_IP {acc}"

        # Round 1 actions
        if verb in ("INSPECT", "FLAG", "UNFLAG", "INVESTIGATE_NETWORK"):
            return f"{verb} {acc}" if acc else verb
        if verb == "SUBMIT":
            return "SUBMIT"

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


def _rule_prefilter(obs_data: dict) -> Optional[str]:
    """Return an obvious rule-based action string without calling the LLM.

    Only fires when the correct action is completely unambiguous — this avoids
    wasting LLM calls (and wall-clock time) on decisions that don't need reasoning.
    Returns None when the LLM should decide.

    Round 2: Handles GET_POLICY as first action and hidden signals (0.0 values).
    """
    flagged = set(obs_data.get("flagged_ids", []))
    inspected = set(obs_data.get("inspected_ids", []))
    steps_remaining = obs_data.get("steps_remaining", 999)

    # Round 2: GET_POLICY should be first action if platform unknown
    # (This is handled by LLM, but we don't auto-fire it here to allow flexibility)

    # Forced submit when out of steps
    if steps_remaining <= 0:
        return "SUBMIT"

    # Forced submit when all 10 slots filled
    if len(flagged) >= 10:
        return "SUBMIT"

    # Uninspected suspects are always highest priority (cascade just fired)
    suspects = obs_data.get("suspect_ids", [])
    for s in suspects:
        if s not in inspected and s not in flagged:
            return f"INSPECT {s}"

    # Any already-inspected account with unambiguous gang signals → FLAG immediately
    accounts = obs_data.get("visible_accounts", [])
    for a in sorted(accounts, key=lambda x: x.get("fake_risk_score", 0), reverse=True):
        aid = a.get("account_id", "")
        if aid in flagged:
            continue
        if a.get("hub_legitimacy_score", 0) > 0.75:
            continue  # protect celebrities

        # Shared IP is the strongest signal: all gang members share ip_gang_{seed}
        if a.get("shared_ip_count", 0) >= 5:
            return f"FLAG {aid}"

        # Round 2: Only flag on content signals if they're revealed (non-zero)
        # If signals are 0.0, they may be hidden → let LLM decide whether to investigate
        photo = a.get("photo_reuse_score", 0)
        bio = a.get("bio_template_score", 0)

        # Both content signals very high AND revealed → clear fake, flag without LLM
        if photo >= 0.65 and bio >= 0.55:
            return f"FLAG {aid}"

    # No obvious action — let the LLM decide (exploration, borderline flags, tool usage)
    return None


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

    Architecture:
    - Rule pre-filter handles unambiguous decisions (suspects, shared-IP flags, forced submit)
      WITHOUT calling the LLM. This saves ~50% of LLM calls and keeps runtime well
      under the 20-minute budget on 2 vCPU / 8 GB RAM.
    - The LLM (OpenAI client → HF router) decides only when genuine reasoning is needed:
      exploration choices, borderline risk accounts, evasion adaptation.
    - Loop terminates on `done=True` from env, not a fixed iteration counter.
      FLAG actions cost 0 steps, so the loop can take more actions than max_steps.
    """
    from openai import OpenAI

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    action_count = 0  # total actions taken (including free FLAGs)
    llm_calls = 0

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    score = 0.0
    success = False

    try:
        reset_resp = _http_post(f"{base_url}/reset", {"task": task, "seed": seed})
        obs_data = reset_resp.get("observation", reset_resp)
        done = reset_resp.get("done", False)

        # Safety cap: max actions = 4 × max_steps to handle free FLAG chains
        task_max_steps = {"easy": 30, "medium": 50, "hard": 80}
        max_actions = task_max_steps.get(task, 80) * 4

        while not done and action_count < max_actions:
            action_count += 1

            # --- Rule pre-filter: skip LLM for unambiguous decisions ---
            action_str = _rule_prefilter(obs_data)

            if action_str is None:
                # LLM decides
                obs_text = _format_obs_for_llm(obs_data)
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
                    llm_calls += 1
                except Exception as exc:
                    # print(f"[DEBUG] LLM call failed: {exc}", flush=True)
                    llm_text = ""

                action_str = _parse_llm_action(llm_text, obs_data)

            action_dict = _action_str_to_dict(action_str)

            step_resp = _http_post(f"{base_url}/step", action_dict)
            obs_data = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward") or 0.0
            done = step_resp.get("done", False)

            rewards.append(reward)

            log_step(step=action_count, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # print(f"[DEBUG] LLM calls: {llm_calls}/{action_count} actions", flush=True)

        grader_resp = _http_get(f"{base_url}/grader")
        score = grader_resp.get("score", 0.0)
        success = score >= 0.815  # win threshold (recall≥0.8, precision≥0.7)

    except Exception:
        pass

    log_end(success=success, steps=action_count, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# Environment variables (used by judge / automated evaluators):
#   ENV_URL    — base URL of the running environment server
#   TASK_NAME  — "easy" | "medium" | "hard" | "all"  (default: "all")
#   SEED       — integer seed (default: 0)
#
# CLI flags are optional overrides for local development:
#   --url, --task, --seed, --baseline, --local, --all-tasks

if __name__ == "__main__":
    import argparse

    # ── Defaults from env vars (judge interface) ─────────────────────────────
    _default_url  = os.getenv("ENV_URL", "https://pandago-graphstrike.hf.space")
    _default_task = os.getenv("TASK_NAME", "all")
    _default_seed = int(os.getenv("SEED", "0"))

    parser = argparse.ArgumentParser(description="GraphStrike inference script")
    parser.add_argument("--url",  default=_default_url,
                        help="Base URL of the running environment server (env: ENV_URL)")
    parser.add_argument("--task", default=_default_task,
                        choices=["easy", "medium", "hard", "all"],
                        help="Task difficulty or 'all' (env: TASK_NAME)")
    parser.add_argument("--seed", type=int, default=_default_seed,
                        help="Episode seed (env: SEED)")
    parser.add_argument("--local", action="store_true",
                        help="Rule-based baseline locally (no server, no LLM)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run rule-based baseline via /baseline endpoint")
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

    elif args.task == "all":
        # LLM inference on all 3 tasks (default when run bare: python3 inference.py)
        scores = {}
        for t in ["easy", "medium", "hard"]:
            scores[t] = run_llm_episode(args.url, task=t, seed=args.seed)
        print(json.dumps({"scores": scores, "agent": MODEL_NAME}, indent=2))

    else:
        # Single-task LLM inference
        score = run_llm_episode(args.url, task=args.task, seed=args.seed)
        print(json.dumps({"score": score, "task": args.task, "agent": MODEL_NAME}, indent=2))
