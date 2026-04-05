#!/usr/bin/env python3
"""
Judge Evaluation Simulator
==========================

Simulates EXACTLY how hackathon judges will evaluate your environment:

1. Baseline re-run: POST /baseline → verify scores are stable
2. Standard Open LLM agent: Run an LLM (via HF router) against all 3 tasks
3. Score variance check: Run same task multiple seeds, check variance

USAGE:
    # Against live HF Space (requires HF_TOKEN):
    export HF_TOKEN="hf_..."
    python test_judge_eval.py --url https://pandago-graphstrike.hf.space

    # Against local server:
    export HF_TOKEN="hf_..."
    python test_judge_eval.py --url http://localhost:7860

    # Choose model (default: Qwen/Qwen2.5-72B-Instruct):
    export MODEL_NAME="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    python test_judge_eval.py --url https://pandago-graphstrike.hf.space

    # Just test endpoints (no LLM needed):
    python test_judge_eval.py --url https://pandago-graphstrike.hf.space --endpoints-only
"""

import json
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Config from env vars (same as judges will set)
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta.llama4-scout-17b-instruct-v1:0")

# AWS Bedrock config (use --bedrock flag)
LLM_BACKEND = "hf"  # "hf" or "bedrock"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "meta.llama4-scout-17b-instruct-v1:0")

SYSTEM_PROMPT = """You are an AI detective finding 10 coordinated fake accounts in a social network.

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
SUBMIT"""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _retry(fn, retries=3, backoff=3):
    """Retry a function on network errors."""
    for attempt in range(retries):
        try:
            return fn()
        except OSError as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (attempt + 1)
            print(f"    [RETRY] Network error: {e} — retrying in {wait}s ({attempt+1}/{retries})")
            time.sleep(wait)


def http_post(url: str, body: Optional[dict] = None) -> dict:
    def _do():
        data = json.dumps(body or {}).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    return _retry(_do)


def http_get(url: str, expect_json: bool = True) -> dict:
    def _do():
        with urllib.request.urlopen(url, timeout=120) as resp:
            body = resp.read()
            if not expect_json:
                return {"_status": resp.status, "_body_len": len(body)}
            return json.loads(body)
    return _retry(_do)


# ---------------------------------------------------------------------------
# LLM call via OpenAI-compatible API
# ---------------------------------------------------------------------------

def _call_hf(prompt: str) -> str:
    """Call LLM via HF router (OpenAI-compatible)."""
    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_bedrock(prompt: str) -> str:
    """Call LLM via AWS Bedrock. Tries converse() first, falls back to invoke_model()."""
    import boto3
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    # Try converse API first (boto3 >= 1.34.x)
    if hasattr(client, "converse"):
        resp = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": SYSTEM_PROMPT}],
            inferenceConfig={"maxTokens": 256, "temperature": 0.3},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()
    # Fallback: invoke_model (works with all boto3 versions)
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.3,
    })
    resp = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(resp["body"].read())
    # Handle both OpenAI-style and Bedrock-native response formats
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


def call_llm(prompt: str) -> str:
    """Call LLM with retries. Uses HF router or Bedrock based on LLM_BACKEND."""
    fn = _call_bedrock if LLM_BACKEND == "bedrock" else _call_hf
    for attempt in range(3):
        try:
            raw = fn(prompt)
            if os.getenv("DEBUG_LLM"):
                print(f"    [LLM RAW] {raw[:200]}")
            # Strip Qwen3 <think>...</think> reasoning blocks
            import re
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            return cleaned if cleaned else raw
        except Exception as e:
            if attempt == 2:
                print(f"    [LLM ERROR] {e} (gave up after 3 attempts)")
                return ""
            wait = 3 * (attempt + 1)
            print(f"    [LLM RETRY] {e} — retrying in {wait}s")
            time.sleep(wait)
    return ""


def format_obs(obs: dict) -> str:
    """Format observation as text for LLM — shows raw signals prominently."""
    lines = []
    lines.append(f"TASK: {obs.get('task','?').upper()} | Steps remaining: {obs.get('steps_remaining','?')}")

    flagged = obs.get("flagged_ids", [])
    lines.append(f"Flagged ({len(flagged)}/10): {', '.join(flagged) if flagged else 'none'}")

    suspects = obs.get("suspect_ids", [])
    inspected = obs.get("inspected_ids", [])
    uninspected_suspects = [s for s in suspects if s not in inspected]
    if uninspected_suspects:
        lines.append(f"*** SUSPECTS (uninspected) → INSPECT THESE FIRST: {', '.join(uninspected_suspects)} ***")

    accounts = obs.get("visible_accounts", [])
    if accounts:
        # Split: unflagged accounts that should be flagged vs rest
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
            lines.append(f"\n!!! ACTION NEEDED — FLAG THESE ({len(unflagged_suspicious)} accounts with strong fake signals):")
            for a in unflagged_suspicious:
                aid = a.get("account_id", "?")
                lines.append(f"  → FLAG {aid}: risk={a.get('fake_risk_score',0):.3f} photo={a.get('photo_reuse_score',0):.2f} bio={a.get('bio_template_score',0):.2f} ip_shared={a.get('shared_ip_count',0)} hub={a.get('hub_legitimacy_score',0):.2f}")

        if flagged_accs:
            lines.append(f"\nALREADY FLAGGED ({len(flagged_accs)}):")
            for a in flagged_accs[:5]:
                lines.append(f"  ✓ {a.get('account_id','?')}")

        if clean_accs:
            lines.append(f"\nCLEAN ACCOUNTS ({len(clean_accs)}):")
            for a in clean_accs[:5]:
                aid = a.get("account_id", "?")
                hub = a.get("hub_legitimacy_score", 0)
                hub_mark = " [CELEBRITY]" if hub > 0.70 else ""
                lines.append(f"  {aid}: risk={a.get('fake_risk_score',0):.3f} photo={a.get('photo_reuse_score',0):.2f} bio={a.get('bio_template_score',0):.2f} hub={hub:.2f}{hub_mark}")

    visible = obs.get("visible_account_ids", [])
    uninspected = [i for i in visible if i not in inspected]
    if uninspected:
        lines.append(f"\nUninspected IDs ({len(uninspected)}): {', '.join(uninspected[:8])}{'...' if len(uninspected) > 8 else ''}")

    lines.append(f"\nMessage: {obs.get('message', '')}")
    return "\n".join(lines)


def parse_action(llm_text: str, obs: dict) -> dict:
    """Parse LLM output to action dict."""
    for line in llm_text.split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("INSPECT ") or upper.startswith("FLAG ") or upper.startswith("INVESTIGATE_NETWORK ") or upper.startswith("UNFLAG "):
            parts = line.split(maxsplit=1)
            return {"action_type": parts[0].lower(), "account_id": parts[1].lower() if len(parts) > 1 else None}
        if upper == "SUBMIT":
            return {"action_type": "submit"}

    # Fallback: inspect first uninspected suspect
    suspects = obs.get("suspect_ids", [])
    inspected = obs.get("inspected_ids", [])
    for s in suspects:
        if s not in inspected:
            return {"action_type": "inspect", "account_id": s}
    visible = obs.get("visible_account_ids", [])
    for v in visible:
        if v not in inspected:
            return {"action_type": "inspect", "account_id": v}
    return {"action_type": "submit"}


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------

def test_endpoints(base_url: str) -> bool:
    """Phase 0: Verify all required endpoints respond correctly."""
    print("\n" + "="*60)
    print("PHASE 0: Endpoint Verification")
    print("="*60)

    checks = [
        ("GET",  "/health",   None,  True),
        ("GET",  "/tasks",    None,  True),
        ("GET",  "/metadata", None,  True),
        ("GET",  "/schema",   None,  True),
        ("GET",  "/web",      None,  False),   # returns HTML, not JSON
        ("POST", "/reset",    {"task": "easy", "seed": 0}, True),
        ("GET",  "/state",    None,  True),
        ("POST", "/step",     {"action_type": "inspect", "account_id": "acc_0000"}, True),
        ("POST", "/step",     {"action_type": "submit"}, True),
        ("GET",  "/grader",   None,  True),
        ("POST", "/mcp",      {"jsonrpc": "2.0", "method": "tools/list", "id": 1}, True),
        ("POST", "/baseline", None,  True),
    ]

    all_ok = True
    for method, path, body, expect_json in checks:
        try:
            if method == "GET":
                http_get(f"{base_url}{path}", expect_json=expect_json)
            else:
                http_post(f"{base_url}{path}", body)
            print(f"  ✓ {method} {path}")
        except Exception as e:
            print(f"  ✗ {method} {path} — {e}")
            all_ok = False

    return all_ok


def test_baseline_stability(base_url: str) -> bool:
    """Phase 1: Baseline re-run (must produce identical scores)."""
    print("\n" + "="*60)
    print("PHASE 1: Baseline Stability (3 runs)")
    print("="*60)

    scores_list = []
    for i in range(3):
        r = http_post(f"{base_url}/baseline")
        scores = r["scores"]
        scores_list.append(scores)
        print(f"  Run {i+1}: easy={scores['easy']:.4f}  medium={scores['medium']:.4f}  hard={scores['hard']:.4f}")

    # Check all identical
    stable = all(s == scores_list[0] for s in scores_list)
    if stable:
        print("  ✓ All 3 runs identical — baseline is deterministic")
    else:
        print("  ✗ SCORES DIFFER — baseline is non-deterministic!")
    return stable


def test_llm_agent(base_url: str, task: str, seed: int = 0) -> float:
    """Phase 2: Run an LLM agent against one task (simulates judge's Nemotron run)."""
    _model = f"Bedrock/{BEDROCK_MODEL_ID}" if LLM_BACKEND == "bedrock" else MODEL_NAME
    print(f"\n  --- LLM Agent: task={task}, seed={seed}, model={_model} ---")

    # Reset
    reset_resp = http_post(f"{base_url}/reset", {"task": task, "seed": seed})
    obs = reset_resp.get("observation", reset_resp)
    done = reset_resp.get("done", False)

    step_num = 0
    while not done:
        step_num += 1
        prompt = format_obs(obs)
        llm_text = call_llm(prompt)
        action = parse_action(llm_text, obs)

        action_str = f"{action['action_type'].upper()} {action.get('account_id', '')}".strip()

        step_resp = http_post(f"{base_url}/step", action)
        obs = step_resp.get("observation", step_resp)
        done = step_resp.get("done", False)
        reward = step_resp.get("reward")

        flagged_n = len(obs.get("flagged_ids", []))
        suspects_n = len(obs.get("suspect_ids", []))
        steps_left = obs.get("steps_remaining", "?")

        print(f"    Step {step_num:2d}: {action_str:35s} flagged={flagged_n}/10  suspects={suspects_n}  steps_left={steps_left}")

        if done and reward is not None:
            msg = step_resp.get("message", obs.get("message", ""))
            print(f"    → Episode ended: {msg[:100]}")

    # Get grader score
    grader = http_get(f"{base_url}/grader")
    score = grader["score"]
    print(f"    ★ GRADER SCORE: {score:.4f}")
    return score


def test_llm_all_tasks(base_url: str) -> Dict[str, float]:
    """Phase 2: Run LLM agent on all 3 tasks."""
    print("\n" + "="*60)
    _model = f"Bedrock/{BEDROCK_MODEL_ID}" if LLM_BACKEND == "bedrock" else MODEL_NAME
    print(f"PHASE 2: LLM Agent Evaluation (model={_model})")
    print("="*60)

    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = test_llm_agent(base_url, task=task, seed=0)

    print(f"\n  Summary: easy={scores['easy']:.4f}  medium={scores['medium']:.4f}  hard={scores['hard']:.4f}")
    return scores


def test_variance(base_url: str, seeds: List[int] = [0, 1, 2, 3, 4]) -> None:
    """Phase 3: Score variance check (multiple seeds per task)."""
    print("\n" + "="*60)
    print(f"PHASE 3: Score Variance (seeds={seeds})")
    print("="*60)

    for task in ["easy", "medium", "hard"]:
        task_scores = []
        for seed in seeds:
            score = test_llm_agent(base_url, task=task, seed=seed)
            task_scores.append(score)

        mean = sum(task_scores) / len(task_scores)
        variance = sum((s - mean) ** 2 for s in task_scores) / len(task_scores)
        print(f"\n  {task}: scores={[f'{s:.3f}' for s in task_scores]}  mean={mean:.4f}  var={variance:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Judge Evaluation Simulator for GraphStrike")
    parser.add_argument("--url", required=True, help="Environment server URL")
    parser.add_argument("--bedrock", action="store_true", help="Use AWS Bedrock instead of HF router")
    parser.add_argument("--endpoints-only", action="store_true", help="Only test endpoints (no LLM)")
    parser.add_argument("--skip-variance", action="store_true", help="Skip variance check (faster)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds for variance check")
    args = parser.parse_args()

    if args.bedrock:
        LLM_BACKEND = "bedrock"

    base = args.url.rstrip("/")
    model_display = f"Bedrock/{BEDROCK_MODEL_ID}" if LLM_BACKEND == "bedrock" else MODEL_NAME
    print(f"GraphStrike Judge Evaluation Simulator")
    print(f"Target:  {base}")
    print(f"Backend: {LLM_BACKEND}")
    print(f"Model:   {model_display}")
    print(f"Token:   {'set' if (HF_TOKEN or os.getenv('AWS_ACCESS_KEY_ID')) else 'NOT SET'}")

    # Phase 0: Endpoints
    if not test_endpoints(base):
        print("\n✗ Endpoint check failed. Fix before proceeding.")
        sys.exit(1)

    # Phase 1: Baseline stability
    test_baseline_stability(base)

    if args.endpoints_only:
        print("\n✓ Endpoint-only mode — skipping LLM tests.")
        sys.exit(0)

    if LLM_BACKEND == "bedrock":
        if not os.getenv("AWS_ACCESS_KEY_ID"):
            print("\n✗ AWS_ACCESS_KEY_ID not set. Cannot run Bedrock LLM tests.")
            sys.exit(1)
    elif not HF_TOKEN:
        print("\n✗ HF_TOKEN not set. Cannot run LLM agent tests.")
        print("  export HF_TOKEN='hf_...'  OR  use --bedrock with AWS creds")
        sys.exit(1)

    # Phase 2: LLM on all tasks
    scores = test_llm_all_tasks(base)

    # Phase 3: Variance
    if not args.skip_variance:
        test_variance(base, seeds=list(range(args.seeds)))

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
