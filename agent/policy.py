"""LLM policy for the Fake Gang Detection agent.

Uses Qwen3 via AWS Bedrock (configured in bedrock_model.py) to:
  1. Format the current observation into a readable detective briefing
  2. Inject past reflections and a best-trajectory few-shot example
  3. Call Qwen to reason and output the next action
  4. Parse the LLM text output into a typed FakeGangAction

Learning mechanism:
  The agent does NOT fine-tune model weights (Bedrock is a black-box API).
  Instead it learns via **Reflexion** — accumulated reflections and successful
  episode demonstrations are injected into every prompt, causing measurably
  better decisions over time.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Resolve imports regardless of working directory
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))

from bedrock_model import invoke_qwen
from models import FakeGangAction, FakeGangObservation, ActionType

# ---------------------------------------------------------------------------
# System prompt — static, sent as the "system" role to Qwen
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an elite digital forensics detective. Your mission: identify ALL 10 members \
of a coordinated fake account gang hidden inside a large social network.

━━━ AVAILABLE ACTIONS ━━━
  GET_POLICY                              → retrieve platform policy + threshold    (FREE — call first!)
  INSPECT <account_id>                    → reveal full profile + follow list      (costs 1 step)
  REVERSE_IMAGE_SEARCH <account_id>       → reveal photo_reuse_score               (costs 1 step)
  ANALYZE_BIO <account_id>                → reveal bio_template_score              (costs 1 step)
  CHECK_IP <account_id>                   → reveal ip_cluster_id                   (costs 2 steps)
  INVESTIGATE_NETWORK <account_id>        → expand graph 2 hops, find new IDs      (costs 2 steps)
  FLAG <account_id>                       → mark account as gang member            (free)
  UNFLAG <account_id>                     → remove a flag                          (free)
  SUBMIT                                  → finalise your answer, end episode

━━━ ROUND 2: PLATFORM-ADAPTIVE DETECTION ━━━
  ⚠ CRITICAL: Call GET_POLICY first! Each episode runs on a specific platform (Instagram/Snapchat)
  with different detection thresholds and cost models:

  • Instagram: θ = 0.08 (STRICT) — FP penalty = 0.1 (HIGH precision required)
    Primary signal: photo_reuse (content-focused enforcement)
    Strategy: Only flag when fake_risk >= 0.08 AND primary signals confirmed

  • Snapchat: θ = 0.74 (LENIENT) — FP penalty = 0.01 (HIGH recall required)
    Primary signal: bio_template (behavior-focused enforcement)
    Strategy: Flag aggressively when fake_risk >= 0.74, maximize recall

━━━ HIDDEN SIGNALS (Round 2) ━━━
  ⚠ photo_reuse_score, bio_template_score, ip_cluster_id start as 0.0/None!
  You MUST use tool actions to reveal them:

  • REVERSE_IMAGE_SEARCH → reveals photo_reuse_score (costs 1 step)
    Use when: Instagram platform, or profile looks suspicious but fake_risk borderline

  • ANALYZE_BIO → reveals bio_template_score (costs 1 step)
    Use when: Snapchat platform, or comment_repeat_score already elevated

  • CHECK_IP → reveals ip_cluster_id (costs 2 steps — expensive!)
    Use when: Multiple accounts with similar patterns (shared_ip_count > 5)

  Strategy: Don't blindly call tools on every account. Use tools strategically on:
    1. Accounts with status=SUSPECT (flagged neighbor cascade)
    2. Borderline accounts (fake_risk near platform threshold)
    3. Primary signal for the platform (photo_reuse for Instagram, bio_template for Snapchat)

━━━ RISK SCORE GUIDE (platform-adaptive) ━━━
  fake_risk_score computation changes based on platform's primary signal:
  • If photo_reuse is primary → node_risk weighted 0.45 (vs default 0.30)
  • If ip_cluster is primary → behavior_risk weighted 0.40 (vs default 0.25)

  Thresholds (use platform threshold from GET_POLICY):
  • fake_risk >= θ + 0.25  → CONFIRMED_FAKE status, safe to FLAG
  • θ <= fake_risk < θ + 0.25 → SUSPECT status, investigate with tools first
  • fake_risk < θ → NORMAL status, skip unless connected to gang cluster
  • hub_legitimacy_score > 0.70 → likely celebrity, do NOT flag

━━━ RAW SIGNAL THRESHOLDS ━━━
  • photo_reuse_score > 0.5   → stealing celebrity photos (strong on Instagram)
  • bio_template_score > 0.4  → copy-paste bios (strong on Snapchat)
  • comment_repeat_score > 0.6 → copy-paste spam comments
  • shared_ip_count > 5       → sharing IP subnet (hint: use CHECK_IP to confirm cluster)
  • mutual_follow_rate > 0.6  → gang members mutually inflate each other
  • avg_post_hour clustered   → all posting in same narrow time window

━━━ CORE STRATEGY (Round 2) ━━━
  1. Call GET_POLICY first → learn platform (Instagram/Snapchat) + threshold + primary signal
  2. INSPECT starting accounts → identify high fake_risk_score accounts
  3. Use tool actions strategically:
     - Instagram: REVERSE_IMAGE_SEARCH on borderline suspects
     - Snapchat: ANALYZE_BIO on borderline suspects
     - CHECK_IP only when shared_ip_count > 5 (confirm gang cluster)
  4. INSPECT any status=SUSPECT accounts (cascade from flagged neighbors)
  5. FLAG when fake_risk >= platform threshold + confirmed signals
  6. SUBMIT when you have ~10 flagged OR steps_remaining < 5
  7. Platform bonuses:
     - Instagram: +2.0 reward if precision >= 0.95 (avoid FPs!)
     - Snapchat: +2.0 reward if recall >= 0.95 (catch all fakes!)

━━━ RESPONSE FORMAT (always use this exactly) ━━━
<thinking>
[your reasoning — platform context, which signals to investigate, which account is most suspicious and why]
</thinking>
<action>
[exactly one action, e.g.: GET_POLICY   or   INSPECT acc_0042   or   REVERSE_IMAGE_SEARCH acc_0007   or   SUBMIT]
</action>\
"""

# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

_STATUS_BADGE = {
    "confirmed_fake": "CONFIRMED_FAKE",
    "suspect":        "SUSPECT      ",
    "normal":         "NORMAL       ",
}


def _format_observation(obs: FakeGangObservation) -> str:
    flagged_str = ", ".join(obs.flagged_ids) if obs.flagged_ids else "none yet"
    uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]
    suspect_uninspected = [s for s in obs.suspect_ids if s not in obs.inspected_ids]

    # Round 2: Show platform context if available
    platform_info = ""
    if hasattr(obs, "platform") and obs.platform:
        platform_info = f" | PLATFORM: {obs.platform}"

    lines = [
        f"TASK: {obs.task.upper()}{platform_info} | Steps remaining: {obs.steps_remaining}",
        f"Evasion triggered: {obs.evasion_triggered} (events so far: {obs.evasion_count})",
        f"Currently flagged ({len(obs.flagged_ids)}/10): {flagged_str}",
        f"Accounts inspected: {len(obs.inspected_ids)} | Not yet inspected: {len(uninspected)}",
    ]
    if suspect_uninspected:
        lines.append(
            f"SUSPECTS not yet inspected ({len(suspect_uninspected)}): "
            + ", ".join(suspect_uninspected[:8])
            + (f" +{len(suspect_uninspected)-8} more" if len(suspect_uninspected) > 8 else "")
        )
    lines.append("")

    if obs.visible_accounts:
        lines.append("PROFILED ACCOUNTS (sorted by fake_risk_score — highest first):")
        lines.append(
            "  [status | risk | node beh graph hub | photo bio mutual | comment ip_count]"
        )
        scored = sorted(obs.visible_accounts, key=lambda p: p.fake_risk_score, reverse=True)
        for p in scored[:18]:
            status_str = getattr(p.status, "value", str(p.status)) if hasattr(p, "status") else "normal"
            badge = _STATUS_BADGE.get(status_str, "NORMAL       ")
            flagged_marker = " ◀ FLAGGED" if p.account_id in obs.flagged_ids else ""
            nc = f" name_chg={p.name_change_count}" if p.name_change_count else ""
            fnbr = f" fnbr={p.flagged_neighbor_count}(!)" if p.flagged_neighbor_count else ""
            hub_warn = " [HUB?]" if p.hub_legitimacy_score > 0.70 else ""
            lines.append(
                f"  {badge} {p.account_id}{flagged_marker}: "
                f"risk={p.fake_risk_score:.3f} | "
                f"node={p.node_risk:.2f} beh={p.behavior_risk:.2f} "
                f"graph={p.graph_risk:.2f} hub={p.hub_legitimacy_score:.2f}{hub_warn} | "
                f"photo={p.photo_reuse_score:.3f} bio={p.bio_template_score:.3f} "
                f"mutual={p.mutual_follow_rate:.2f}"
                f"{fnbr}{nc}"
            )
        if len(obs.visible_accounts) > 18:
            lines.append(f"  … +{len(obs.visible_accounts) - 18} more inspected accounts")
    else:
        lines.append("No accounts profiled yet — pick one from the known IDs below and INSPECT it.")

    if uninspected:
        sample = uninspected[:12]
        more = f" +{len(uninspected)-12} more" if len(uninspected) > 12 else ""
        lines.append(f"\nKNOWN UNINSPECTED IDs: {', '.join(sample)}{more}")

    lines.append(f"\nEnvironment message: {obs.message}")
    return "\n".join(lines)


def _format_reflections(reflections: List[str]) -> str:
    if not reflections:
        return ""
    parts = ["\n━━━ LESSONS FROM YOUR PAST CASES ━━━"]
    for i, r in enumerate(reflections, 1):
        parts.append(f"{i}. {r.strip()}")
    return "\n".join(parts)


def _format_few_shot(example: Optional[Dict]) -> str:
    if not example:
        return ""
    log = example.get("action_log", [])
    preview = log[:14]
    tail = f"\n  … [{len(log)-14} more steps] …" if len(log) > 14 else ""
    steps_str = "\n".join(f"  {s}" for s in preview) + tail
    return (
        f"\n━━━ EXAMPLE SUCCESSFUL CASE (task={example['task']}, reward={example['reward']:+.2f}) ━━━\n"
        f"{steps_str}\n"
        f"  → {example['final_message']}"
    )


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def _parse_action(raw: str, obs: FakeGangObservation) -> FakeGangAction:
    """Extract a FakeGangAction from the LLM's raw text output."""
    # Prefer content inside <action>…</action>
    m = re.search(r"<action>\s*(.*?)\s*</action>", raw, re.DOTALL | re.IGNORECASE)
    text = m.group(1).strip() if m else raw.strip()
    upper = text.upper()

    # Round 2: New tool actions (parse before INSPECT to avoid prefix collision)
    if "GET_POLICY" in upper:
        return FakeGangAction(action_type=ActionType.GET_POLICY)

    if hit := re.search(r"REVERSE_IMAGE_SEARCH\s+(acc_\w+)", upper):
        acc_id = hit.group(1).lower()
        return FakeGangAction(action_type=ActionType.REVERSE_IMAGE_SEARCH, account_id=acc_id)

    if hit := re.search(r"ANALYZE_BIO\s+(acc_\w+)", upper):
        acc_id = hit.group(1).lower()
        return FakeGangAction(action_type=ActionType.ANALYZE_BIO, account_id=acc_id)

    if hit := re.search(r"CHECK_IP\s+(acc_\w+)", upper):
        acc_id = hit.group(1).lower()
        return FakeGangAction(action_type=ActionType.CHECK_IP, account_id=acc_id)

    # INVESTIGATE_NETWORK must come before INSPECT (it's a longer prefix)
    if hit := re.search(r"INVESTIGATE_NETWORK\s+(acc_\w+)", upper):
        acc_id = hit.group(1).lower()
        if acc_id in obs.visible_account_ids:
            return FakeGangAction(action_type=ActionType.INVESTIGATE_NETWORK, account_id=acc_id)

    if hit := re.search(r"INSPECT\s+(acc_\w+)", upper):
        acc_id = hit.group(1).lower()
        if acc_id in obs.visible_account_ids:
            return FakeGangAction(action_type=ActionType.INSPECT, account_id=acc_id)

    if hit := re.search(r"UNFLAG\s+(acc_\w+)", upper):
        return FakeGangAction(action_type=ActionType.UNFLAG, account_id=hit.group(1).lower())

    if hit := re.search(r"FLAG\s+(acc_\w+)", upper):
        return FakeGangAction(action_type=ActionType.FLAG, account_id=hit.group(1).lower())

    if "SUBMIT" in upper:
        return FakeGangAction(action_type=ActionType.SUBMIT)

    # Fallback: inspect the highest-scored uninspected account
    uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]
    if uninspected:
        return FakeGangAction(action_type=ActionType.INSPECT, account_id=uninspected[0])
    return FakeGangAction(action_type=ActionType.SUBMIT)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_action(
    obs: FakeGangObservation,
    reflections: List[str],
    few_shot_example: Optional[Dict] = None,
    temperature: float = 0.4,
    max_retries: int = 3,
) -> Tuple[FakeGangAction, str]:
    """
    Query Qwen3 (via AWS Bedrock) for the next detective action.

    Returns:
        (action, raw_llm_output)

    Learning signal flows in through `reflections` and `few_shot_example`:
      - reflections: list of past post-episode lessons (from agent/reflection.py)
      - few_shot_example: best saved trajectory (from agent/memory.py)
    Both grow richer over training, causing measurably better decisions.
    """
    reflections_text = _format_reflections(reflections)
    few_shot_text = _format_few_shot(few_shot_example)
    obs_text = _format_observation(obs)

    prompt = (
        f"{reflections_text}"
        f"{few_shot_text}"
        f"\n\n━━━ CURRENT CASE ━━━\n"
        f"{obs_text}"
        f"\n\nWhat is your next action?"
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            raw = invoke_qwen(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_tokens=512,
                temperature=temperature,
            )
            action = _parse_action(raw, obs)
            return action, raw
        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            print(f"  [policy] Bedrock call failed (attempt {attempt+1}): {exc} — retrying in {wait}s")
            time.sleep(wait)

    # All retries exhausted — fall back to heuristic
    print(f"  [policy] All retries exhausted: {last_error}. Using heuristic fallback.")
    return _heuristic_fallback(obs), "[FALLBACK — Bedrock unavailable]"


def _heuristic_fallback(obs: FakeGangObservation) -> FakeGangAction:
    """Simple heuristic used when Bedrock is unavailable."""
    uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]

    if uninspected and obs.steps_remaining > 3:
        return FakeGangAction(action_type=ActionType.INSPECT, account_id=uninspected[0])

    # Flag high-signal accounts
    for p in obs.visible_accounts:
        if (p.photo_reuse_score > 0.5 and p.bio_template_score > 0.4
                and p.account_id not in obs.flagged_ids):
            return FakeGangAction(action_type=ActionType.FLAG, account_id=p.account_id)

    return FakeGangAction(action_type=ActionType.SUBMIT)
