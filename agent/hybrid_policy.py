"""Hybrid confidence-weighted policy for the Fake Gang Detection agent.

Blends a deterministic rule-based agent with the LLM (Qwen3) agent
using a dynamic trust weight α ∈ [0.20, 1.00]:

    α = 0.20 + 0.80 * recent_win_rate * reflection_factor

Action selection when LLM and rules DISAGREE:
    • rules win  if rule_confidence >= α   (α low → rules trusted more)
    • LLM  wins  if rule_confidence <  α   (α high → LLM trusted more)
When they AGREE the action is used as-is (mode="agree").

α dynamics:
  α starts at 0.20 (rules dominate — LLM has no history yet).
  As the LLM accumulates wins and reflections, α climbs toward 1.0.
  Episode 1, 0 wins → α ≈ 0.20 (almost always rule-guided)
  Episode 20, 70% wins, 4 reflections → α ≈ 0.76 (LLM leads, rules as safety net)
  Episode 40, 90% wins, 8 reflections → α ≈ 0.92 (LLM trusted, rules intervene rarely)

Rule confidence levels:
  1.00 — forced SUBMIT (out of steps)
  0.95 — INSPECT a SUSPECT account (cascade-elevated neighbor)
  0.90 — FLAG an account with fake_risk_score ≥ 0.85
  0.80 — SUBMIT with 10 flags in place
  0.70 — FLAG an account with fake_risk_score in [threshold, 0.85)
  0.30 — exploratory INSPECT (no strong signal, just scanning)

This means:
  • At α=0.20 → rules win all disagreements (confidence always ≥ 0.20)
  • At α=0.50 → rules win when confidence ≥ 0.50 (suspect/high-risk actions)
  • At α=0.80 → rules win only when confidence ≥ 0.80 (critical overrides only)
  • At α=1.00 → rules never win (pure LLM mode)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))

from models import ActionType, FakeGangAction, FakeGangObservation
from agent.policy import get_action

# Per-task thresholds mirror inference.py
_THRESHOLDS: Dict[str, float] = {
    "easy": 0.60,
    "medium": 0.50,
    "hard": 0.45,
}

# Bootstrap raw-feature score — same calibration as inference.py
# 0.30*photo + 0.20*bio + 0.50*comment_repeat >= 0.40
# Gang member (any task): ~0.57–0.78; decoy: ~0.25; real: ~0.07
_BOOTSTRAP_RAW_THRESHOLD = 0.40
_SHARED_IP_GANG_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Rule-based single-step decision
# ---------------------------------------------------------------------------

def get_rule_action(obs: FakeGangObservation) -> Tuple[FakeGangAction, float]:
    """Return the rule-based action for the current observation and its confidence.

    Returns:
        (action, confidence)  where confidence ∈ [0.0, 1.0]
    """
    threshold = _THRESHOLDS.get(obs.task, 0.50)

    # Priority 1 — forced end (out of steps)
    if obs.steps_remaining <= 0:
        return FakeGangAction(action_type=ActionType.SUBMIT), 1.00

    # Priority 2 — INSPECT SUSPECT accounts (auto-cascaded from FLAG)
    uninspected_suspects = [s for s in obs.suspect_ids if s not in obs.inspected_ids]
    if uninspected_suspects:
        return (
            FakeGangAction(action_type=ActionType.INSPECT, account_id=uninspected_suspects[0]),
            0.95,
        )

    # Priority 3 — FLAG high-risk inspected accounts
    # Two signal paths: composite fake_risk (active post-cascade) OR bootstrap raw
    # node score (catches first gang members before graph signals are established).
    for p in sorted(obs.visible_accounts, key=lambda x: x.fake_risk_score, reverse=True):
        if p.account_id in obs.flagged_ids:
            continue
        if p.hub_legitimacy_score > 0.75:
            continue  # protect celebrities

        bootstrap_raw = (
            0.30 * p.photo_reuse_score
            + 0.20 * p.bio_template_score
            + 0.50 * p.comment_repeat_score
        )

        if p.shared_ip_count >= _SHARED_IP_GANG_THRESHOLD:
            return FakeGangAction(action_type=ActionType.FLAG, account_id=p.account_id), 0.97

        if p.fake_risk_score >= threshold:
            confidence = min(0.95, 0.70 + (p.fake_risk_score - threshold) * 0.60)
            return FakeGangAction(action_type=ActionType.FLAG, account_id=p.account_id), confidence

        if bootstrap_raw >= _BOOTSTRAP_RAW_THRESHOLD:
            # Bootstrap confidence: how far above threshold the raw score is
            confidence = min(0.88, 0.60 + (bootstrap_raw - _BOOTSTRAP_RAW_THRESHOLD) * 0.80)
            return FakeGangAction(action_type=ActionType.FLAG, account_id=p.account_id), confidence

    # Priority 4 — SUBMIT if fully confident (10 flagged or almost out of steps)
    if len(obs.flagged_ids) >= 10:
        return FakeGangAction(action_type=ActionType.SUBMIT), 0.85

    if obs.steps_remaining <= 3:
        return FakeGangAction(action_type=ActionType.SUBMIT), 0.90

    # Priority 5 — INSPECT the highest-risk uninspected account (exploratory)
    uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]
    if uninspected:
        # Sort uninspected: SUSPECT status first, then by whatever we know
        suspects_set = set(obs.suspect_ids)
        uninspected.sort(key=lambda i: (i not in suspects_set, i))
        return FakeGangAction(action_type=ActionType.INSPECT, account_id=uninspected[0]), 0.30

    # Fallback
    return FakeGangAction(action_type=ActionType.SUBMIT), 0.75


# ---------------------------------------------------------------------------
# Alpha computation
# ---------------------------------------------------------------------------

def compute_alpha(recent_win_rate: float, n_reflections: int) -> float:
    """Compute α (LLM trust weight) from recent performance.

    α = 0.20 + 0.80 * win_rate * reflection_factor

    reflection_factor ramps from 0 → 1 as reflections accumulate (0–4).
    The LLM needs at least ~2 reflections before getting meaningful trust above 0.20.

    Examples:
        0 wins,    0 reflections → α = 0.20
        50% wins,  2 reflections → α = 0.50
        80% wins,  4 reflections → α = 0.71
        100% wins, 4 reflections → α = 1.00
    """
    reflection_factor = min(1.0, n_reflections / 4.0)
    raw = 0.20 + 0.80 * recent_win_rate * reflection_factor
    return round(max(0.20, min(1.00, raw)), 3)


# ---------------------------------------------------------------------------
# Hybrid decision
# ---------------------------------------------------------------------------

def get_hybrid_action(
    obs: FakeGangObservation,
    reflections: List[str],
    few_shot_example: Optional[dict] = None,
    alpha: float = 0.30,
    temperature: float = 0.40,
) -> Tuple[FakeGangAction, str, str]:
    """Return the blended action, the raw LLM output, and the decision mode.

    Decision logic:
        1. Get rule action + confidence
        2. Get LLM action (with Reflexion context)
        3. If they agree → "agree" (unanimous)
        4. If disagree:
           - rule_confidence >= alpha → rule wins  ("rule_override")
           - rule_confidence <  alpha → LLM wins   ("llm")

    Returns:
        (action, raw_llm_output, mode_str)

    mode_str is one of:
        "agree"                     — both said the same thing
        "rule_override(c=X,α=Y)"    — rule overrode LLM
        "llm(c=X,α=Y)"              — LLM won over rule
    """
    rule_action, rule_conf = get_rule_action(obs)
    llm_action, raw_llm = get_action(
        obs=obs,
        reflections=reflections,
        few_shot_example=few_shot_example,
        temperature=temperature,
    )

    agree = (
        rule_action.action_type == llm_action.action_type
        and rule_action.account_id == llm_action.account_id
    )

    if agree:
        return llm_action, raw_llm, "agree"

    if rule_conf >= alpha:
        mode = f"rule_override(c={rule_conf:.2f},α={alpha:.2f})"
        return rule_action, raw_llm, mode
    else:
        mode = f"llm(c={rule_conf:.2f}<α={alpha:.2f})"
        return llm_action, raw_llm, mode
