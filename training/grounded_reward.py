"""Env-grounded reward computation for GRPO.

Given a generated completion and the (task, seed, decision_index) it was sampled
for, we replay the full episode using a heuristic teacher and inject the
completion at the matching turn. The episode's grader_score becomes the
per-turn grader signal; step_reward is the env delta at the injected turn.

Each call to `score_completion` runs ONE episode against the env. For GRPO with
group size K and N prompts, that's N*K episodes per training step — keep N*K
small (phase 1: 4*4 = 16, phase 2: 8*4 = 32).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_PARENT / "eval-models"))

from _round2_runner import _run_episode  # noqa: E402
from client import FakeGangEnvClient  # noqa: E402

from training.parse import parse_completion
from training.rewards import compute_reward, RewardBreakdown
from training.build_dataset import heuristic_teacher


def _injecting_llm(
    target_idx: int,
    completion: str,
    baseline: Callable[[str], str],
):
    """Wrap a baseline call_llm so it returns `completion` exactly once at
    decision index `target_idx`. All other turns fall back to the baseline."""
    state = {"i": 0, "captured_step_reward": None}

    def call(prompt: str) -> str:
        idx = state["i"]
        state["i"] += 1
        if idx == target_idx:
            return completion
        return baseline(prompt)

    return call, state


def score_completion(
    base_url: str,
    platform: str,
    task: str,
    seed: int,
    decision_index: int,
    decision_type: str,
    completion: str,
    client: Optional[FakeGangEnvClient] = None,
) -> Tuple[RewardBreakdown, dict]:
    """Replay one episode with `completion` injected at turn `decision_index`.

    Returns (RewardBreakdown, debug_info). The reward composes:
      - grader (episode-level)
      - step  (env delta at the injected turn — looked up from collected tuples)
      - format (1.0 iff `completion` parses for `decision_type`)
    """
    client = client or FakeGangEnvClient(base_url=base_url)
    baseline = heuristic_teacher()
    inject, state = _injecting_llm(decision_index, completion, baseline)

    log, tuples = _run_episode(
        client, "grpo:rollout", platform, task, seed, inject,
        collect_tuples=True,
    )

    # Find the tuple at the target index (if the episode reached that turn).
    step_reward = None
    matched = None
    if 0 <= decision_index < len(tuples):
        matched = tuples[decision_index]
        step_reward = matched.get("step_reward")
    _, _, format_ok = parse_completion(completion, decision_type)
    rb = compute_reward(
        grader_score=log.grader_score,
        step_reward=step_reward,
        format_ok=format_ok,
    )
    debug = {
        "episode_id": log.episode_id,
        "grader": log.grader_score,
        "n_decisions_executed": len(tuples),
        "matched_turn": matched is not None,
        "format_ok": format_ok,
    }
    return rb, debug
