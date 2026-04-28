"""Three-component reward used during GRPO.

  R_grader  = episode-level grader_score in [0, 1]      (weight 1.0)
  R_step    = clip(step_reward * 0.1, -0.5, 0.5)        (weight 0.3)
  R_format  = 1.0 if completion parses as valid JSON action else 0.0  (weight 0.2)

Per-turn scalar reward returned to GRPO is:
  r = w_grader*R_grader + w_step*R_step + w_format*R_format

`R_grader` is broadcast to every turn in the episode (TRL GRPO needs a scalar
per (prompt, completion) pair).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

W_GRADER = 1.0
W_STEP = 0.3
W_FORMAT = 0.2

STEP_SCALE = 0.1
STEP_CLIP = 0.5


@dataclass
class RewardBreakdown:
    grader: float
    step: float
    format: float
    total: float

    def as_dict(self) -> dict:
        return {
            "r_grader": self.grader,
            "r_step": self.step,
            "r_format": self.format,
            "r_total": self.total,
        }


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_reward(
    grader_score: Optional[float],
    step_reward: Optional[float],
    format_ok: bool,
) -> RewardBreakdown:
    """Compose the per-turn scalar from the three signals."""
    g = float(grader_score) if grader_score is not None else 0.0
    s = float(step_reward) if step_reward is not None else 0.0
    s = _clip(s * STEP_SCALE, -STEP_CLIP, STEP_CLIP)
    f = 1.0 if format_ok else 0.0
    total = W_GRADER * g + W_STEP * s + W_FORMAT * f
    return RewardBreakdown(grader=g, step=s, format=f, total=total)
