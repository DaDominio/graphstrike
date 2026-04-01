"""Post-episode reflection generator.

After each episode the agent calls `generate_reflection()`.
Qwen3 analyses the action log and outcome, then writes a concrete lesson
that gets stored in AgentMemory and injected into future prompts.

This is the core **learning mechanism**:
  Episode N fails  →  reflection generated  →  stored in memory
  Episode N+1      →  reflection in prompt  →  agent avoids past mistake
  ...
  Win rate rises measurably over episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from bedrock_model import invoke_qwen

# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM = """\
You are a senior detective reviewing a FAKE INSTAGRAM ACCOUNT detection case debrief.
This environment detects coordinated fake social media accounts — NOT financial fraud.

Signals available in this environment (use ONLY these):
  • comment_repeat_score > 0.6  → copy-paste spam comments (gang: 0.6-0.9, real: 0.0-0.08)
  • shared_ip_count > 5         → shares IP subnet (all 10 gang members have count=9)
  • photo_reuse_score > 0.5     → stolen profile photos
  • bio_template_score > 0.4    → copy-paste bio text
  • fake_risk_score > 0.75      → high-confidence gang member (composite score)
  • hub_legitimacy_score > 0.70 → celebrity account, do NOT flag
  • After FLAG: visible neighbors auto-become SUSPECT (priority targets)

Available actions: INSPECT (1 step, reveals profile), INVESTIGATE_NETWORK (2 steps, 2-hop expand),
FLAG, UNFLAG, SUBMIT.

CRITICAL: Write lessons about fake social media signals and INSPECT/INVESTIGATE_NETWORK strategy
ONLY. Do NOT mention transactions, financial transfers, banking, or any concepts not listed above.
Output only the lesson text — no headers, no bullet points, just 2-3 plain sentences.\
"""


def generate_reflection(
    task: str,
    action_log: List[str],
    final_message: str,
    won: bool,
    steps_used: int,
    max_steps: int,
    episode_num: int,
) -> str:
    """
    Ask Qwen3 to generate a concrete lesson from one completed episode.
    Returns a short string (2-3 sentences) suitable for memory storage.
    """
    outcome = "SUCCESS" if won else "FAILURE"
    log_preview = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(action_log[:20]))
    if len(action_log) > 20:
        log_preview += f"\n  … [{len(action_log) - 20} more steps]"

    prompt = f"""\
FAKE INSTAGRAM ACCOUNT DETECTION — Episode {episode_num}
Task difficulty: {task.upper()}
Outcome: {outcome}
Steps used: {steps_used}/{max_steps}
Result: {final_message}

AVAILABLE SIGNALS (reference for your lesson):
  comment_repeat_score > 0.6 | shared_ip_count > 5 | photo_reuse_score > 0.5
  fake_risk_score > 0.75 | hub_legitimacy_score > 0.70 (celebrity, skip)
  After FLAG → neighbors become SUSPECT (inspect them immediately)
  INVESTIGATE_NETWORK on a flagged account reveals their 2-hop gang cluster

INVESTIGATION LOG:
{log_preview}

Write a 2-3 sentence lesson for your future self based on this case.
Focus on: which of the above signals were most diagnostic, whether using
INVESTIGATE_NETWORK after the first FLAG would have helped, and how to
better allocate the step budget. Be concrete and actionable.\
"""

    try:
        reflection = invoke_qwen(
            prompt=prompt,
            system=_REFLECTION_SYSTEM,
            max_tokens=180,
            temperature=0.6,
        )
        return reflection.strip()
    except Exception as exc:
        # If Bedrock fails, generate a minimal rule-based reflection
        return _rule_based_reflection(won, steps_used, max_steps, final_message)


def _rule_based_reflection(
    won: bool, steps_used: int, max_steps: int, final_message: str
) -> str:
    """Minimal fallback reflection when Bedrock is unavailable."""
    if won and steps_used < max_steps * 0.6:
        return (
            "Early INVESTIGATE_NETWORK calls efficiently expanded the graph to all gang members. "
            "Flagging accounts with both high photo_reuse AND bio_template scores maintained precision. "
            "Submitting with budget remaining earned an efficiency bonus."
        )
    if won:
        return (
            "Found all gang members but used most of the step budget. "
            "Look for intra-gang follow density earlier — once you find one member, INVESTIGATE_NETWORK immediately. "
            "Flag faster to leave budget for verification."
        )
    if "Recall=0.00" in final_message or "TP=0" in final_message:
        return (
            "Zero gang members found — the starting accounts were all real. "
            "After inspecting 3-4 low-signal accounts, use INVESTIGATE_NETWORK to jump to a different part of the graph. "
            "Gang members have photo_reuse > 0.5 and bio_template > 0.4 simultaneously."
        )
    return (
        "Partial recall — found some gang members but missed others. "
        "After flagging the first gang member, immediately use INVESTIGATE_NETWORK: gang members follow each other heavily. "
        "Don't waste steps inspecting low-signal accounts one-by-one."
    )


# ---------------------------------------------------------------------------
# Post-win reflection (reinforces what worked)
# ---------------------------------------------------------------------------

def generate_success_reflection(
    task: str,
    action_log: List[str],
    final_message: str,
    steps_used: int,
    max_steps: int,
    episode_num: int,
) -> str:
    """Generate a reinforcement reflection after a WIN to capture what worked."""
    log_preview = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(action_log[:15]))

    prompt = f"""\
SUCCESSFUL CASE — Episode {episode_num}
Task: {task.upper()} | Steps used: {steps_used}/{max_steps}
Result: {final_message}

INVESTIGATION LOG (first 15 steps):
{log_preview}

In 2-3 sentences, describe the specific strategy that led to success.
What did you do right? What should you repeat in future cases?\
"""

    try:
        return invoke_qwen(
            prompt=prompt,
            system=_REFLECTION_SYSTEM,
            max_tokens=150,
            temperature=0.5,
        ).strip()
    except Exception:
        return _rule_based_reflection(True, steps_used, max_steps, final_message)
