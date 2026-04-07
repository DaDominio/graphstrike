"""
GraphStrike — Custom Inference Script (AWS Bedrock backend)
===========================================================

Same as inference.py but uses AWS Bedrock instead of HF router.

MANDATORY ENVIRONMENT VARIABLES:
    AWS_ACCESS_KEY_ID      AWS credentials
    AWS_SECRET_ACCESS_KEY  AWS credentials
    AWS_DEFAULT_REGION     AWS region (default: us-east-1)
    BEDROCK_MODEL_ID       Bedrock model ID (default: qwen.qwen3-next-80b-a3b)

OPTIONAL:
    ENV_URL    Environment server URL (default: https://pandago-graphstrike.hf.space)
    TASK_NAME  "easy" | "medium" | "hard" | "all"  (default: "all")
    SEED       Integer seed (default: 0)

STDOUT FORMAT:
    [START] task=<task_name> env=graphstrike model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
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

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "qwen.qwen3-next-80b-a3b")
AWS_REGION       = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_ACCESS_KEY   = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY   = os.getenv("AWS_SECRET_ACCESS_KEY")

BENCHMARK    = "graphstrike"
TEMPERATURE  = 0.3
MAX_TOKENS   = 256

# ---------------------------------------------------------------------------
# Thresholds (for rule-based baseline)
# ---------------------------------------------------------------------------

THRESHOLDS: Dict[str, float] = {
    "easy":   0.60,
    "medium": 0.50,
    "hard":   0.45,
}

_BOOTSTRAP_RAW_THRESHOLD  = 0.40
_SHARED_IP_GANG_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
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
# System prompt (shared with inference.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI detective finding 10 coordinated fake accounts in a social network.

    ACTIONS (reply with exactly ONE line):
    - INSPECT acc_XXXX  — reveal profile (costs 1 step)
    - FLAG acc_XXXX     — mark as fake (FREE, no step cost, triggers suspect cascade)
    - SUBMIT            — end episode, get scored

    DECISION RULES (apply top-to-bottom, first match wins):
    1. If suspect_ids lists accounts you haven't inspected → INSPECT the first one
    2. If ANY profiled account has shared_ip_count >= 5 and is NOT flagged → FLAG it immediately
    3. If ANY profiled account has photo_reuse >= 0.50 AND bio_template >= 0.40 and hub < 0.70 and NOT flagged → FLAG it
    4. If ANY profiled account has fake_risk_score >= 0.30 and hub < 0.70 and NOT flagged → FLAG it
    5. If there are uninspected visible accounts and steps > 3 → INSPECT the next one
    6. If you have 10 flags OR steps <= 3 → SUBMIT

    IMPORTANT:
    - FLAG is FREE (costs 0 steps) — flag aggressively when you see suspicious signals
    - After each FLAG, new suspects appear — always inspect suspects before other accounts
    - hub_legitimacy_score > 0.70 means celebrity — do NOT flag
    - shared_ip_count >= 5 is the strongest gang signal (all 10 share one IP)
    - Do NOT re-inspect already inspected accounts

    Reply with EXACTLY one line, nothing else:
    FLAG acc_XXXX
    INSPECT acc_XXXX
    SUBMIT
""").strip()


# ---------------------------------------------------------------------------
# AWS Bedrock LLM call
# ---------------------------------------------------------------------------

def _call_bedrock(prompt: str) -> str:
    """Call LLM via AWS Bedrock. Tries converse() first, falls back to invoke_model()."""
    import boto3
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    # converse API (boto3 >= 1.34.x) — preferred
    if hasattr(client, "converse"):
        resp = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": TEMPERATURE},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()

    # Fallback: invoke_model (works with all boto3 versions)
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
    })
    resp = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    if "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    if "content" in result:
        content = result["content"]
        if isinstance(content, list):
            return content[0].get("text", "").strip()
        return str(content).strip()
    if "output" in result:
        return result["output"].get("text", "").strip()
    return str(result).strip()


def _call_llm_with_retry(prompt: str) -> str:
    """Call Bedrock with up to 3 retries, strip Qwen3 <think> blocks."""
    for attempt in range(3):
        try:
            raw = _call_bedrock(prompt)
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return cleaned if cleaned else raw
        except Exception:
            if attempt == 2:
                return ""
            wait = 3 * (attempt + 1)
            time.sleep(wait)
    return ""


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def _format_obs_for_llm(obs_data: dict) -> str:
    lines = []
    lines.append(f"TASK: {obs_data.get('task', '?').upper()} | Steps remaining: {obs_data.get('steps_remaining', '?')}")
    flagged  = obs_data.get("flagged_ids", [])
    lines.append(f"Flagged ({len(flagged)}/10): {', '.join(flagged) if flagged else 'none'}")

    suspects = obs_data.get("suspect_ids", [])
    inspected = obs_data.get("inspected_ids", [])
    uninspected_suspects = [s for s in suspects if s not in inspected]
    if uninspected_suspects:
        lines.append(f"*** SUSPECTS (uninspected) → INSPECT THESE FIRST: {', '.join(uninspected_suspects)} ***")

    accounts = obs_data.get("visible_accounts", [])
    if accounts:
        unflagged_suspicious, flagged_accs, clean_accs = [], [], []
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
                lines.append(
                    f"  → FLAG {aid}: risk={a.get('fake_risk_score',0):.3f} "
                    f"photo={a.get('photo_reuse_score',0):.2f} bio={a.get('bio_template_score',0):.2f} "
                    f"ip_shared={a.get('shared_ip_count',0)} hub={a.get('hub_legitimacy_score',0):.2f}"
                )

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
                lines.append(
                    f"  {aid}: risk={a.get('fake_risk_score',0):.3f} "
                    f"photo={a.get('photo_reuse_score',0):.2f} bio={a.get('bio_template_score',0):.2f} "
                    f"hub={hub:.2f}{hub_mark}"
                )

    visible_ids   = obs_data.get("visible_account_ids", [])
    uninspected_ids = [i for i in visible_ids if i not in inspected]
    if uninspected_ids:
        lines.append(
            f"\nUninspected IDs ({len(uninspected_ids)}): "
            f"{', '.join(uninspected_ids[:10])}{'...' if len(uninspected_ids) > 10 else ''}"
        )

    lines.append(f"\nMessage: {obs_data.get('message', '')}")
    return "\n".join(lines)


def _parse_llm_action(text: str, obs_data: dict) -> str:
    """Parse LLM response into an action string like 'INSPECT acc_0042'."""
    text = text.strip()
    for line in text.split("\n"):
        line  = line.strip()
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        verb = parts[0].upper()
        acc  = parts[1].lower() if len(parts) > 1 else None
        if verb in ("INSPECT", "FLAG", "UNFLAG", "INVESTIGATE_NETWORK"):
            return f"{verb} {acc}" if acc else verb
        if verb == "SUBMIT":
            return "SUBMIT"
    # Fallback: inspect first uninspected suspect or visible account
    suspects  = obs_data.get("suspect_ids", [])
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
    parts       = action_str.strip().split(maxsplit=1)
    action_type = parts[0].lower()
    account_id  = parts[1] if len(parts) > 1 else None
    d = {"action_type": action_type}
    if account_id:
        d["account_id"] = account_id.lower()
    return d


def _rule_prefilter(obs_data: dict) -> Optional[str]:
    """Return an obvious rule-based action string without calling the LLM."""
    flagged   = set(obs_data.get("flagged_ids", []))
    inspected = set(obs_data.get("inspected_ids", []))
    steps_remaining = obs_data.get("steps_remaining", 999)

    if steps_remaining <= 0:
        return "SUBMIT"
    if len(flagged) >= 10:
        return "SUBMIT"

    suspects = obs_data.get("suspect_ids", [])
    for s in suspects:
        if s not in inspected and s not in flagged:
            return f"INSPECT {s}"

    accounts = obs_data.get("visible_accounts", [])
    for a in sorted(accounts, key=lambda x: x.get("fake_risk_score", 0), reverse=True):
        aid = a.get("account_id", "")
        if aid in flagged:
            continue
        if a.get("hub_legitimacy_score", 0) > 0.75:
            continue
        if a.get("shared_ip_count", 0) >= 5:
            return f"FLAG {aid}"
        if a.get("photo_reuse_score", 0) >= 0.65 and a.get("bio_template_score", 0) >= 0.55:
            return f"FLAG {aid}"

    return None


# ---------------------------------------------------------------------------
# Rule-based episode runner (library API — used by /baseline)
# ---------------------------------------------------------------------------

def run_rule_based_episode(env, task: str, seed: int = 0) -> float:
    """Run one complete episode using the rule-based policy. Returns grader_score."""
    obs: FakeGangObservation = env.reset(task=task, seed=seed)
    threshold = THRESHOLDS[task]

    while not obs.done:
        uninspected_suspects = [s for s in obs.suspect_ids if s not in obs.inspected_ids]
        if uninspected_suspects:
            obs = env.step(FakeGangAction(action_type=ActionType.INSPECT,
                                          account_id=uninspected_suspects[0]))
            continue

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
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _http_get(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Bedrock inference loop
# ---------------------------------------------------------------------------

def run_llm_episode(base_url: str, task: str, seed: int = 0) -> float:
    """Run one episode using an LLM agent via AWS Bedrock."""
    rewards: List[float] = []
    action_count = 0
    llm_calls    = 0

    log_start(task=task, env=BENCHMARK, model=BEDROCK_MODEL_ID)

    score   = 0.0
    success = False

    try:
        reset_resp = _http_post(f"{base_url}/reset", {"task": task, "seed": seed})
        obs_data   = reset_resp.get("observation", reset_resp)
        done       = reset_resp.get("done", False)

        task_max_steps = {"easy": 30, "medium": 50, "hard": 80}
        max_actions    = task_max_steps.get(task, 80) * 4

        while not done and action_count < max_actions:
            action_count += 1

            # Rule pre-filter: skip LLM for unambiguous decisions
            action_str = _rule_prefilter(obs_data)

            if action_str is None:
                obs_text   = _format_obs_for_llm(obs_data)
                llm_text   = _call_llm_with_retry(obs_text)
                llm_calls += 1
                action_str = _parse_llm_action(llm_text, obs_data)

            action_dict = _action_str_to_dict(action_str)

            step_resp = _http_post(f"{base_url}/step", action_dict)
            obs_data  = step_resp.get("observation", step_resp)
            reward    = step_resp.get("reward") or 0.0
            done      = step_resp.get("done", False)

            rewards.append(reward)
            log_step(step=action_count, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # print(f"[DEBUG] LLM calls: {llm_calls}/{action_count} actions", flush=True)

        grader_resp = _http_get(f"{base_url}/grader")
        score       = grader_resp.get("score", 0.0)
        success     = score >= 0.815

    except Exception:
        pass

    log_end(success=success, steps=action_count, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# Judge interface — only AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY required.
# Everything else has sensible defaults.
#
# Environment variables:
#   ENV_URL    — environment server URL  (default: https://pandago-graphstrike.hf.space)
#   TASK_NAME  — "easy" | "medium" | "hard" | "all"  (default: "all")
#   SEED       — integer seed  (default: 0)
#
# CLI flags override env vars (for local dev/testing only):
#   --url, --task, --seed, --baseline, --local

if __name__ == "__main__":
    import argparse

    _default_url  = os.getenv("ENV_URL",   "http://localhost:7862")
    _default_task = os.getenv("TASK_NAME", "all")
    _default_seed = int(os.getenv("SEED",  "0"))

    parser = argparse.ArgumentParser(description="GraphStrike custom inference (AWS Bedrock)")
    parser.add_argument("--url",  default=_default_url,
                        help="Environment server URL (env: ENV_URL)")
    parser.add_argument("--task", default=_default_task,
                        choices=["easy", "medium", "hard", "all"],
                        help="Task difficulty or 'all' (env: TASK_NAME)")
    parser.add_argument("--seed", type=int, default=_default_seed,
                        help="Episode seed (env: SEED)")
    parser.add_argument("--local",    action="store_true",
                        help="Rule-based baseline locally (no server, no LLM)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run rule-based baseline via /baseline endpoint")
    args = parser.parse_args()

    if args.local:
        from environment import FakeGangEnvironment  # type: ignore[import]
        env = FakeGangEnvironment()
        scores: Dict[str, float] = {}
        for t in ["easy", "medium", "hard"]:
            scores[t] = run_rule_based_episode(env, task=t, seed=0)
        print(json.dumps({"scores": scores, "agent": "rule_based"}, indent=2))

    elif args.baseline:
        result = _http_post(f"{args.url}/baseline")
        print(json.dumps(result, indent=2))

    elif args.task == "all":
        scores = {}
        for t in ["easy", "medium", "hard"]:
            scores[t] = run_llm_episode(args.url, task=t, seed=args.seed)
        print(json.dumps({"scores": scores, "agent": BEDROCK_MODEL_ID}, indent=2))

    else:
        score = run_llm_episode(args.url, task=args.task, seed=args.seed)
        print(json.dumps({"score": score, "task": args.task, "agent": BEDROCK_MODEL_ID}, indent=2))
