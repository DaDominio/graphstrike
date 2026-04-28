#!/usr/bin/env python3
"""
Training script for Fake Gang Detection agent.

Supports:
- AWS Bedrock models (Qwen, Claude, Llama)
- HuggingFace router models
- Local rule-based baseline
- Platform-specific training (Instagram/Snapchat)
- Metrics tracking and visualization
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from server.environment import FakeGangEnvironment
from models import FakeGangAction, FakeGangObservation, ActionType


# ============================================================================
# LLM Backends
# ============================================================================

def call_bedrock(prompt: str, system_prompt: str, model_id: str) -> str:
    """Call AWS Bedrock model."""
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
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.3, "maxTokens": 512},
        )
        return resp["output"]["message"]["content"][0]["text"]
    else:
        # Fallback to invoke_model
        body = json.dumps({
            "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:",
            "max_tokens": 512,
            "temperature": 0.3,
        })
        resp = client.invoke_model(modelId=model_id, body=body)
        result = json.loads(resp["body"].read())
        return result.get("completion", result.get("content", [{"text": "SUBMIT"}])[0]["text"])


def call_hf(prompt: str, system_prompt: str, model_name: str, api_key: str) -> str:
    """Call HuggingFace router model."""
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=api_key,
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=512,
    )

    return (resp.choices[0].message.content or "").strip()


# ============================================================================
# System Prompt (Round 2)
# ============================================================================

SYSTEM_PROMPT = """You are an AI detective finding 10 coordinated fake accounts in a social network.

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
SUBMIT"""


# ============================================================================
# Agent Policy
# ============================================================================

def format_observation(obs: FakeGangObservation) -> str:
    """Format observation as text prompt for LLM."""
    lines = []

    # Platform context
    platform_info = f" | PLATFORM: {obs.platform}" if obs.platform else ""
    lines.append(f"TASK: {obs.task.upper()}{platform_info} | Steps remaining: {obs.steps_remaining}")

    flagged = obs.flagged_ids
    lines.append(f"Flagged ({len(flagged)}/10): {', '.join(flagged) if flagged else 'none'}")

    # Suspects (high priority)
    suspects = obs.suspect_ids
    inspected = obs.inspected_ids
    uninspected_suspects = [s for s in suspects if s not in inspected]
    if uninspected_suspects:
        lines.append(f"*** SUSPECTS (uninspected) → INSPECT THESE FIRST: {', '.join(uninspected_suspects)} ***")

    # Accounts
    if obs.visible_accounts:
        unflagged_suspicious = []
        flagged_accs = []
        clean_accs = []

        for a in sorted(obs.visible_accounts, key=lambda x: x.fake_risk_score, reverse=True):
            aid = a.account_id
            if aid in flagged:
                flagged_accs.append(a)
            elif (a.shared_ip_count >= 5 or
                  (a.photo_reuse_score >= 0.50 and a.bio_template_score >= 0.40)):
                unflagged_suspicious.append(a)
            else:
                clean_accs.append(a)

        if unflagged_suspicious:
            lines.append(f"\n!!! ACTION NEEDED — FLAG THESE ({len(unflagged_suspicious)} suspicious):")
            for a in unflagged_suspicious:
                lines.append(f"  → FLAG {a.account_id}: risk={a.fake_risk_score:.3f} photo={a.photo_reuse_score:.2f} bio={a.bio_template_score:.2f} ip_shared={a.shared_ip_count} hub={a.hub_legitimacy_score:.2f}")

        if flagged_accs:
            lines.append(f"\nALREADY FLAGGED ({len(flagged_accs)}):")
            for a in flagged_accs[:5]:
                lines.append(f"  ✓ {a.account_id}")

        if clean_accs:
            lines.append(f"\nCLEAN ({len(clean_accs)}):")
            for a in clean_accs[:8]:
                hub_mark = " [CELEBRITY]" if a.hub_legitimacy_score > 0.70 else ""
                lines.append(f"  {a.account_id}: risk={a.fake_risk_score:.3f} photo={a.photo_reuse_score:.2f} bio={a.bio_template_score:.2f} hub={a.hub_legitimacy_score:.2f}{hub_mark}")

    visible_ids = obs.visible_account_ids
    uninspected_ids = [i for i in visible_ids if i not in inspected]
    if uninspected_ids:
        lines.append(f"\nUninspected IDs ({len(uninspected_ids)}): {', '.join(uninspected_ids[:10])}{'...' if len(uninspected_ids) > 10 else ''}")

    lines.append(f"\nMessage: {obs.message}")
    return "\n".join(lines)


def parse_action(text: str, obs: FakeGangObservation) -> FakeGangAction:
    """Parse LLM response into action."""
    text = text.strip().upper()

    for line in text.split("\n"):
        line = line.strip()
        parts = line.split(maxsplit=1)
        verb = parts[0]
        acc = parts[1].lower() if len(parts) > 1 else None

        # Round 2 actions
        if verb == "GET_POLICY":
            return FakeGangAction(action_type=ActionType.GET_POLICY)
        if verb == "REVERSE_IMAGE_SEARCH" and acc:
            return FakeGangAction(action_type=ActionType.REVERSE_IMAGE_SEARCH, account_id=acc)
        if verb == "ANALYZE_BIO" and acc:
            return FakeGangAction(action_type=ActionType.ANALYZE_BIO, account_id=acc)
        if verb == "CHECK_IP" and acc:
            return FakeGangAction(action_type=ActionType.CHECK_IP, account_id=acc)

        # Round 1 actions
        if verb in ("INSPECT", "FLAG", "UNFLAG", "INVESTIGATE_NETWORK"):
            if acc:
                return FakeGangAction(action_type=ActionType[verb], account_id=acc)
        if verb == "SUBMIT":
            return FakeGangAction(action_type=ActionType.SUBMIT)

    # Fallback: inspect first uninspected
    for s in obs.suspect_ids:
        if s not in obs.inspected_ids:
            return FakeGangAction(action_type=ActionType.INSPECT, account_id=s)

    for v in obs.visible_account_ids:
        if v not in obs.inspected_ids:
            return FakeGangAction(action_type=ActionType.INSPECT, account_id=v)

    return FakeGangAction(action_type=ActionType.SUBMIT)


# ============================================================================
# Episode Runner
# ============================================================================

def run_episode(
    env: FakeGangEnvironment,
    task: str,
    seed: int,
    backend: str,
    model_id: str,
    verbose: bool = False,
) -> Dict:
    """Run one episode and return metrics."""

    obs = env.reset(task=task, seed=seed)
    platform = obs.platform

    actions_taken = []
    tool_counts = {
        "GET_POLICY": 0,
        "REVERSE_IMAGE_SEARCH": 0,
        "ANALYZE_BIO": 0,
        "CHECK_IP": 0,
        "INSPECT": 0,
        "FLAG": 0,
    }

    start_time = time.time()

    while not obs.done:
        # Format observation
        prompt = format_observation(obs)

        # Get action from LLM or rule-based
        if backend == "bedrock":
            response = call_bedrock(prompt, SYSTEM_PROMPT, model_id)
            action = parse_action(response, obs)
        elif backend == "hf":
            api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
            response = call_hf(prompt, SYSTEM_PROMPT, model_id, api_key)
            action = parse_action(response, obs)
        else:  # rule-based
            action = get_rule_based_action(obs)

        # Track action
        actions_taken.append(action.action_type.value)
        if action.action_type.value.upper() in tool_counts:
            tool_counts[action.action_type.value.upper()] += 1

        # Step environment
        obs = env.step(action)

        if verbose:
            print(f"  [{obs.steps_remaining:2d}] {action.action_type.value:20s} {action.account_id or ''}")

    elapsed = time.time() - start_time

    # Get final metrics
    grader = env._last_grader_score

    # Count TP/FP/FN
    flagged = set(env._flagged)
    gang_members = set(env._ep["gang_member_ids"])
    tp = len(flagged & gang_members)
    fp = len(flagged - gang_members)
    fn = len(gang_members - flagged)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "episode": seed,
        "platform": platform,
        "task": task,
        "reward": obs.reward or 0.0,
        "grader_score": grader,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "steps_used": env._step_count,
        "max_steps": env._max_steps,
        "tool_counts": tool_counts,
        "total_tools": sum(tool_counts.values()) - tool_counts["INSPECT"] - tool_counts["FLAG"],  # Only investigation tools
        "actions": actions_taken,
        "elapsed_seconds": elapsed,
    }


def get_rule_based_action(obs: FakeGangObservation) -> FakeGangAction:
    """Simple rule-based policy for baseline."""
    # Priority 1: Inspect suspects
    for s in obs.suspect_ids:
        if s not in obs.inspected_ids:
            return FakeGangAction(action_type=ActionType.INSPECT, account_id=s)

    # Priority 2: Flag high-risk accounts
    for p in obs.visible_accounts:
        if p.account_id in obs.flagged_ids:
            continue
        if p.hub_legitimacy_score > 0.75:
            continue
        if p.shared_ip_count >= 5 or p.fake_risk_score >= 0.60:
            return FakeGangAction(action_type=ActionType.FLAG, account_id=p.account_id)

    # Priority 3: Inspect uninspected
    uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]
    if uninspected and obs.steps_remaining > 3:
        return FakeGangAction(action_type=ActionType.INSPECT, account_id=uninspected[0])

    return FakeGangAction(action_type=ActionType.SUBMIT)


# ============================================================================
# Training Loop
# ============================================================================

def run_training(
    episodes: int,
    task: str,
    backend: str,
    model_id: str,
    output_file: Optional[str],
    verbose: bool,
) -> List[Dict]:
    """Run training loop and collect metrics."""

    env = FakeGangEnvironment()
    results = []

    print(f"Starting training: {episodes} episodes, task={task}, backend={backend}, model={model_id}")
    print("=" * 80)

    for ep in range(episodes):
        try:
            result = run_episode(env, task, seed=ep, backend=backend, model_id=model_id, verbose=verbose)
            results.append(result)

            # Print summary
            print(f"Episode {ep:3d} ({result['platform']:9s}): "
                  f"reward={result['reward']:+.3f} | "
                  f"grader={result['grader_score']:.3f} | "
                  f"TP={result['tp']:2d} FP={result['fp']:2d} FN={result['fn']:2d} | "
                  f"P={result['precision']:.2f} R={result['recall']:.2f} | "
                  f"tools={result['total_tools']} | "
                  f"{result['elapsed_seconds']:.1f}s")

        except Exception as e:
            print(f"Episode {ep:3d} FAILED: {e}")
            continue

    print("=" * 80)

    # Aggregate metrics
    instagram_results = [r for r in results if r["platform"] == "Instagram"]
    snapchat_results = [r for r in results if r["platform"] == "Snapchat"]

    def avg(lst, key):
        vals = [r[key] for r in lst]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"\n=== Training Summary ===")
    print(f"Total Episodes: {len(results)}/{episodes}")
    print(f"\nInstagram ({len(instagram_results)} episodes):")
    print(f"  Avg Reward:      {avg(instagram_results, 'reward'):+.3f}")
    print(f"  Avg Grader:      {avg(instagram_results, 'grader_score'):.3f}")
    print(f"  Avg Precision:   {avg(instagram_results, 'precision'):.3f}")
    print(f"  Avg Recall:      {avg(instagram_results, 'recall'):.3f}")
    print(f"  Win Rate:        {sum(1 for r in instagram_results if r['grader_score'] >= 0.815) / len(instagram_results) * 100:.1f}%")
    print(f"  Avg Tools:       {avg(instagram_results, 'total_tools'):.1f}")

    print(f"\nSnapchat ({len(snapchat_results)} episodes):")
    print(f"  Avg Reward:      {avg(snapchat_results, 'reward'):+.3f}")
    print(f"  Avg Grader:      {avg(snapchat_results, 'grader_score'):.3f}")
    print(f"  Avg Precision:   {avg(snapchat_results, 'precision'):.3f}")
    print(f"  Avg Recall:      {avg(snapchat_results, 'recall'):.3f}")
    print(f"  Win Rate:        {sum(1 for r in snapchat_results if r['grader_score'] >= 0.815) / len(snapchat_results) * 100:.1f}%")
    print(f"  Avg Tools:       {avg(snapchat_results, 'total_tools'):.1f}")

    # Tool usage breakdown
    all_tools = {}
    for r in results:
        for tool, count in r["tool_counts"].items():
            all_tools[tool] = all_tools.get(tool, 0) + count

    print(f"\nTool Usage (total):")
    for tool, count in sorted(all_tools.items(), key=lambda x: -x[1]):
        print(f"  {tool:25s}: {count:4d} calls")

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": {
                "episodes": episodes,
                "task": task,
                "backend": backend,
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
            },
            "results": results,
            "summary": {
                "instagram": {
                    "count": len(instagram_results),
                    "avg_reward": avg(instagram_results, "reward"),
                    "avg_grader": avg(instagram_results, "grader_score"),
                    "avg_precision": avg(instagram_results, "precision"),
                    "avg_recall": avg(instagram_results, "recall"),
                    "win_rate": sum(1 for r in instagram_results if r["grader_score"] >= 0.815) / len(instagram_results) if instagram_results else 0,
                    "avg_tools": avg(instagram_results, "total_tools"),
                },
                "snapchat": {
                    "count": len(snapchat_results),
                    "avg_reward": avg(snapchat_results, "reward"),
                    "avg_grader": avg(snapchat_results, "grader_score"),
                    "avg_precision": avg(snapchat_results, "precision"),
                    "avg_recall": avg(snapchat_results, "recall"),
                    "win_rate": sum(1 for r in snapchat_results if r["grader_score"] >= 0.815) / len(snapchat_results) if snapchat_results else 0,
                    "avg_tools": avg(snapchat_results, "total_tools"),
                },
                "tool_usage": all_tools,
            },
        }

        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\n✓ Results saved to {output_file}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Fake Gang Detection agent")

    # Training params
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy", help="Task difficulty")

    # Model selection
    parser.add_argument("--backend", choices=["bedrock", "hf", "rule"], default="rule",
                        help="LLM backend (bedrock=AWS, hf=HuggingFace, rule=baseline)")
    parser.add_argument("--model-id", default="qwen.qwen3-next-80b-a3b",
                        help="Model ID (Bedrock: qwen.qwen3-next-80b-a3b, HF: Qwen/Qwen2.5-72B-Instruct)")

    # Output
    parser.add_argument("--output", "-o", default="results/training_results.json",
                        help="Output JSON file for metrics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed action log")

    args = parser.parse_args()

    # Run training
    run_training(
        episodes=args.episodes,
        task=args.task,
        backend=args.backend,
        model_id=args.model_id,
        output_file=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
