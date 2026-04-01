"""Baseline agents for all three tasks of the Fake Gang Detection environment.

Run directly:
    python baseline_agent.py --task easy --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from client import FakeGangEnvClient, StepResult
from models import AccountProfile, FakeGangObservation, ActionType


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _gang_score(profile: AccountProfile, task: str) -> float:
    """Simple composite fake-likelihood score (0–1)."""
    # Normalise account age: newer = more suspicious (fakes created recently)
    age_score = max(0.0, 1.0 - profile.account_age_days / 500.0)
    # Posting hour clustering: peak hour 14 is suspicious
    hour_score = max(0.0, 1.0 - abs(profile.avg_post_hour - 14.0) / 12.0)
    return (
        profile.photo_reuse_score * 0.40
        + profile.bio_template_score * 0.30
        + age_score * 0.20
        + hour_score * 0.10
    )


# ---------------------------------------------------------------------------
# Easy baseline: signal scanner
# ---------------------------------------------------------------------------

def easy_agent(env: FakeGangEnvClient, seed: int = 0) -> StepResult:
    result = env.reset(task="easy", seed=seed)
    obs = result.observation

    PHOTO_THRESH = 0.50
    BIO_THRESH = 0.40

    while not obs.done:
        # Pick an uninspected account to inspect
        uninspected = [
            i for i in obs.visible_account_ids if i not in obs.inspected_ids
        ]

        if uninspected:
            target = uninspected[0]
            result = env.inspect(target)
            obs = result.observation

            # Flag if it matches both signal thresholds
            profile = next((p for p in obs.visible_accounts if p.account_id == target), None)
            if profile and (
                profile.photo_reuse_score > PHOTO_THRESH
                and profile.bio_template_score > BIO_THRESH
            ):
                result = env.flag(target)
                obs = result.observation

                # Explore its network if we don't have enough flags yet
                if len(obs.flagged_ids) < 10:
                    result = env.investigate_network(target)
                    obs = result.observation
        else:
            # Lower thresholds if we're running low on steps
            if obs.steps_remaining < 10:
                PHOTO_THRESH = max(0.20, PHOTO_THRESH - 0.10)
                BIO_THRESH = max(0.20, BIO_THRESH - 0.10)
                # Re-evaluate all inspected accounts with looser thresholds
                for p in obs.visible_accounts:
                    if p.account_id not in obs.flagged_ids:
                        if (p.photo_reuse_score > PHOTO_THRESH
                                and p.bio_template_score > BIO_THRESH):
                            result = env.flag(p.account_id)
                            obs = result.observation

            result = env.submit()
            obs = result.observation

        if obs.steps_remaining <= 0 or obs.done:
            if not obs.done:
                result = env.submit()
                obs = result.observation
            break

    return result


# ---------------------------------------------------------------------------
# Medium baseline: time-aware scanner
# ---------------------------------------------------------------------------

def medium_agent(env: FakeGangEnvClient, seed: int = 0) -> StepResult:
    result = env.reset(task="medium", seed=seed)
    obs = result.observation

    max_steps = 50
    evasion_step = 20

    while not obs.done:
        steps_used = max_steps - obs.steps_remaining

        uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]

        # Phase 1: race against evasion — use graph traversal
        if steps_used < evasion_step - 5 and uninspected:
            target = uninspected[0]
            result = env.inspect(target)
            obs = result.observation
            profile = next((p for p in obs.visible_accounts if p.account_id == target), None)
            if profile and profile.photo_reuse_score > 0.35:
                result = env.flag(target)
                obs = result.observation
                result = env.investigate_network(target)
                obs = result.observation

        # Phase 2: after evasion — rely on features only
        elif uninspected:
            target = uninspected[0]
            result = env.inspect(target)
            obs = result.observation
            profile = next((p for p in obs.visible_accounts if p.account_id == target), None)
            if profile:
                score = _gang_score(profile, "medium")
                if score > 0.45:
                    result = env.flag(target)
                    obs = result.observation

        else:
            # No more to inspect — submit
            result = env.submit()
            obs = result.observation
            break

        if obs.steps_remaining <= 2 or obs.done:
            if not obs.done:
                result = env.submit()
                obs = result.observation
            break

    return result


# ---------------------------------------------------------------------------
# Hard baseline: feature-only detective
# ---------------------------------------------------------------------------

def hard_agent(env: FakeGangEnvClient, seed: int = 0) -> StepResult:
    result = env.reset(task="hard", seed=seed)
    obs = result.observation

    max_steps = 80
    submit_by_step = 60   # submit before too many evasion events

    while not obs.done:
        steps_used = max_steps - obs.steps_remaining
        uninspected = [i for i in obs.visible_account_ids if i not in obs.inspected_ids]

        # Inspect until we have a dataset or time's up
        if uninspected and steps_used < submit_by_step:
            target = uninspected[0]
            result = env.inspect(target)
            obs = result.observation

            # Discover more IDs via network expansion (spend the extra step)
            if len(obs.visible_account_ids) < 100 and obs.steps_remaining > 15:
                result = env.investigate_network(target)
                obs = result.observation

        else:
            # Score all inspected accounts and flag top-12
            scored = sorted(
                obs.visible_accounts,
                key=lambda p: _gang_score(p, "hard") + 0.3 * p.name_change_count,
                reverse=True,
            )
            # Unflag everything first
            for fid in list(obs.flagged_ids):
                result = env.unflag(fid)
                obs = result.observation

            # Flag top 12 (a bit generous to improve recall)
            for profile in scored[:12]:
                result = env.flag(profile.account_id)
                obs = result.observation

            result = env.submit()
            obs = result.observation
            break

        if obs.steps_remaining <= 2 or obs.done:
            if not obs.done:
                result = env.submit()
                obs = result.observation
            break

    return result


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

AGENTS = {
    "easy": easy_agent,
    "medium": medium_agent,
    "hard": hard_agent,
}


def evaluate(task: str, episodes: int = 10, base_url: str = "http://localhost:8000") -> None:
    agent_fn = AGENTS[task]
    wins = 0
    total_reward = 0.0

    with FakeGangEnvClient(base_url=base_url) as env:
        for seed in range(episodes):
            result = agent_fn(env, seed=seed)
            msg = result.message
            reward = result.reward or 0.0
            total_reward += reward
            won = "[WIN]" in msg
            if won:
                wins += 1
            print(f"Episode {seed:3d} | {'WIN ' if won else 'LOSS'} | reward={reward:+.2f} | {msg}")

    print(f"\n{task.upper()} — wins: {wins}/{episodes} ({100*wins/episodes:.0f}%) | avg reward: {total_reward/episodes:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline agent against the Fake Gang env.")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    evaluate(args.task, args.episodes, args.url)
