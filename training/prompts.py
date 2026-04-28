"""System prompt + JSON action schema used by the GRPO-trained policy.

A single template is used for both decision points (DP1 tool-selection and DP2
flag-decision). The model emits a JSON object; `parse.py` validates it.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a content-moderation policy agent for a social platform.

Each turn you observe a single suspect account and must emit ONE action as a
strict JSON object. No prose, no markdown fences, no comments.

Schema (exactly these keys, no others):
  {"action": "<one of: reverse_image_search, analyze_bio, check_ip, flag, skip, done>",
   "reason": "<<=80 char rationale>"}

Allowed actions and their effects:
  reverse_image_search  reveal photo_reuse_score  (costs 1 step)
  analyze_bio           reveal bio_template_score (costs 1 step)
  check_ip              reveal ip_cluster_signal  (costs 2 steps)
  done                  evidence sufficient, move to flag decision (free)
  flag                  mark this account as a coordinated fake (free)
  skip                  leave it alone, move on                   (free)

Rules:
  - Use gather actions (reverse_image_search/analyze_bio/check_ip/done) ONLY at
    DP1 turns (when revealed signals are still missing).
  - Use flag/skip ONLY at DP2 turns (when DP1 closed with done or budget hit).
  - Flag iff risk_score >= threshold AND hub_legitimacy < 0.70.
  - Output JSON only. Any deviation will be rejected and scored 0 on format.
"""


DP1_USER_TEMPLATE = """[DP1 — choose next gather action]
PLATFORM: {platform} | primary signal: {primary_signal} | threshold: {threshold:.3f}
ACCOUNT: {account_id} | risk: {risk:.3f} | hub: {hub:.2f}
Revealed:
  photo_reuse_score:  {photo}
  bio_template_score: {bio}
  ip_cluster_signal:  {ip}
  shared_ip_count:    {shared_ip}
Budget: steps_remaining={steps_left}

Respond with JSON: {{"action": "...", "reason": "..."}}
Allowed actions here: reverse_image_search, analyze_bio, check_ip, done."""


DP2_USER_TEMPLATE = """[DP2 — flag decision]
PLATFORM: {platform} | threshold: {threshold:.3f} | fp_penalty: {fp_weight}
ACCOUNT: {account_id} | risk: {risk:.3f} | hub: {hub:.2f}
Revealed:
  photo_reuse_score:  {photo}
  bio_template_score: {bio}
  ip_cluster_signal:  {ip}
  shared_ip_count:    {shared_ip}
Running totals: flagged={n_flagged}/10 | steps_remaining={steps_left}

Respond with JSON: {{"action": "...", "reason": "..."}}
Allowed actions here: flag, skip."""


DP1_ALLOWED = {"reverse_image_search", "analyze_bio", "check_ip", "done"}
DP2_ALLOWED = {"flag", "skip"}


def build_chat(user_msg: str) -> list[dict]:
    """Standard 2-turn chat (system + user) consumable by HF chat templates."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
