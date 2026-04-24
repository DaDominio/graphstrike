"""Synthetic episode generator for the Fake Gang Detection environment."""

from __future__ import annotations

import json
import math
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "easy": {
        "network_size": 50,
        "gang_size": 10,
        "decoy_count": 0,
        "max_steps": 30,
        "starting_visible": 5,
        "post_hour_mean": 14.0,
        "post_hour_std": 0.5,
        "photo_reuse_range": (0.70, 0.95),
        "bio_template_range": (0.60, 0.90),
        "intra_gang_density": 0.80,
        "evasion_schedule": [],
        "win_recall": 0.8,
        "win_precision": 0.7,
    },
    "medium": {
        "network_size": 200,
        "gang_size": 10,
        "decoy_count": 20,
        "max_steps": 50,
        "starting_visible": 10,
        "post_hour_mean": 14.0,
        "post_hour_std": 1.5,
        "photo_reuse_range": (0.40, 0.70),
        "bio_template_range": (0.30, 0.60),
        "intra_gang_density": 0.70,
        "evasion_schedule": [
            {"step": 20, "event": "unfollow_intragang", "drop_rate": 0.5}
        ],
        "win_recall": 0.8,
        "win_precision": 0.7,
    },
    "hard": {
        "network_size": 1000,
        "gang_size": 10,
        "decoy_count": 50,
        "max_steps": 80,
        "starting_visible": 20,
        "post_hour_mean": 14.0,
        "post_hour_std": 2.5,
        "photo_reuse_range": (0.30, 0.55),
        "bio_template_range": (0.20, 0.50),
        "intra_gang_density": 0.60,
        "evasion_schedule": [
            {"step": 15, "event": "unfollow_intragang", "drop_rate": 0.3, "rename_count": 3},
            {"step": 30, "event": "unfollow_intragang", "drop_rate": 0.3, "rename_count": 3},
            {"step": 45, "event": "unfollow_intragang", "drop_rate": 0.3, "rename_count": 3},
            {"step": 60, "event": "unfollow_intragang", "drop_rate": 0.3, "rename_count": 3},
        ],
        "win_recall": 0.9,
        "win_precision": 0.8,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lognormal_int(rng: random.Random, mu: float, sigma: float, lo: int, hi: int) -> int:
    val = math.exp(rng.gauss(mu, sigma))
    return max(lo, min(hi, int(val)))


def _beta(rng: random.Random, a: float, b: float) -> float:
    """Simple beta distribution via gamma approximation."""
    x = rng.gammavariate(a, 1)
    y = rng.gammavariate(b, 1)
    return x / (x + y)


# ---------------------------------------------------------------------------
# Account generators
# ---------------------------------------------------------------------------

def _gen_real_account(rng: random.Random, acc_id: str) -> Dict[str, Any]:
    # Round 2: photo_reuse, bio_template, ip_cluster moved to hidden_signals
    return {
        "id": acc_id,
        "is_fake": False,
        "gang_id": None,
        "features": {
            "follower_count": _lognormal_int(rng, 6.0, 1.5, 10, 100_000),
            "following_count": _lognormal_int(rng, 5.5, 1.2, 10, 5_000),
            "post_count": _lognormal_int(rng, 4.0, 1.0, 1, 5_000),
            "avg_post_hour": rng.uniform(6, 23),
            "account_age_days": rng.randint(30, 1825),
            "comment_repeat_score": round(_beta(rng, 1, 20), 4),  # mostly 0.0-0.08
            "shared_ip_count": 0,  # Still visible (hint signal)
            "name_change_count": 0,
        },
        # Hidden signals (returned separately, revealed by tool calls)
        "hidden_signals": {
            "photo_reuse_score": round(_beta(rng, 1, 10), 4),
            "bio_template_score": round(_beta(rng, 1, 8), 4),
            "ip_cluster_id": f"ip_real_{acc_id}",
        },
        "true_edges": {"follows": [], "followed_by": []},
    }


def _gen_decoy_account(rng: random.Random, acc_id: str) -> Dict[str, Any]:
    # Round 2: Override hidden signals for decoys (elevated fake signals but not gang)
    acc = _gen_real_account(rng, acc_id)
    acc["hidden_signals"]["photo_reuse_score"] = round(rng.uniform(0.2, 0.4), 4)
    acc["hidden_signals"]["bio_template_score"] = round(rng.uniform(0.2, 0.4), 4)
    acc["features"]["comment_repeat_score"] = round(rng.uniform(0.10, 0.30), 4)
    return acc


def _gen_celebrity_account(rng: random.Random, acc_id: str) -> Dict[str, Any]:
    # Round 2: Celebrities have low fake signals (hidden)
    return {
        "id": acc_id,
        "is_fake": False,
        "gang_id": None,
        "features": {
            "follower_count": rng.randint(100_000, 5_000_000),
            "following_count": rng.randint(50, 500),
            "post_count": rng.randint(500, 5000),
            "avg_post_hour": rng.uniform(6, 23),
            "account_age_days": rng.randint(1000, 3000),
            "comment_repeat_score": round(rng.uniform(0.0, 0.03), 4),
            "shared_ip_count": 0,
            "name_change_count": 0,
        },
        "hidden_signals": {
            "photo_reuse_score": round(rng.uniform(0.0, 0.05), 4),
            "bio_template_score": round(rng.uniform(0.0, 0.05), 4),
            "ip_cluster_id": f"ip_celeb_{acc_id}",
        },
        "true_edges": {"follows": [], "followed_by": []},
    }


def _gen_gang_accounts(
    rng: random.Random,
    ids: List[str],
    cfg: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    # Round 2: Gang accounts have high fake signals (hidden until agent calls tools)
    gang_size = len(ids)
    base_age = rng.randint(30, 180)
    base_hour = cfg["post_hour_mean"]
    ph_lo, ph_hi = cfg["photo_reuse_range"]
    bt_lo, bt_hi = cfg["bio_template_range"]
    std = cfg["post_hour_std"]
    accounts = []
    for acc_id in ids:
        accounts.append({
            "id": acc_id,
            "is_fake": True,
            "gang_id": "gang_A",
            "features": {
                "follower_count": rng.randint(1_000, 8_000),
                "following_count": rng.randint(200, 2_000),
                "post_count": rng.randint(50, 500),
                "avg_post_hour": round(max(0, min(23, rng.gauss(base_hour, std))), 2),
                "account_age_days": base_age + rng.randint(0, 7),
                "comment_repeat_score": round(rng.uniform(0.60, 0.90), 4),
                "shared_ip_count": gang_size - 1,  # Hint: all gang members share IP
                "name_change_count": 0,
            },
            "hidden_signals": {
                "photo_reuse_score": round(rng.uniform(ph_lo, ph_hi), 4),
                "bio_template_score": round(rng.uniform(bt_lo, bt_hi), 4),
                "ip_cluster_id": f"ip_gang_{seed}",
            },
            "true_edges": {"follows": [], "followed_by": []},
        })
    return accounts


# ---------------------------------------------------------------------------
# Edge generation
# ---------------------------------------------------------------------------

def _build_edges(
    rng: random.Random,
    accounts: List[Dict[str, Any]],
    gang_ids: List[str],
    density: float,
) -> None:
    """Add follower edges in-place."""
    acc_map = {a["id"]: a for a in accounts}
    all_ids = [a["id"] for a in accounts]

    # Intra-gang edges (high density)
    for g in gang_ids:
        for h in gang_ids:
            if g != h and rng.random() < density:
                acc_map[g]["true_edges"]["follows"].append(h)
                acc_map[h]["true_edges"]["followed_by"].append(g)

    # Real/decoy accounts: sparse preferential attachment
    non_gang = [a["id"] for a in accounts if a["id"] not in gang_ids]
    for acc_id in non_gang:
        n_follows = rng.randint(5, min(50, len(all_ids) - 1))
        targets = rng.sample([x for x in all_ids if x != acc_id], min(n_follows, len(all_ids) - 1))
        for t in targets:
            acc_map[acc_id]["true_edges"]["follows"].append(t)
            acc_map[t]["true_edges"]["followed_by"].append(acc_id)


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------

def generate_episode(task: str, seed: int, platform: str | None = None) -> Dict[str, Any]:
    """
    Generate synthetic episode with platform-specific configuration.

    Round 2: Adds platform parameter and hidden_signals section.
    Platform assignment: even seeds → Instagram, odd seeds → Snapchat
    """
    cfg = TASK_CONFIG[task]
    rng = random.Random(seed)

    # Round 2: Platform assignment (deterministic by seed)
    if platform is None:
        platform = "Instagram" if seed % 2 == 0 else "Snapchat"

    network_size = cfg["network_size"]
    gang_size = cfg["gang_size"]
    decoy_count = cfg["decoy_count"]
    real_count = network_size - gang_size - decoy_count

    # Generate IDs
    all_ids = [f"acc_{i:04d}" for i in range(network_size)]
    rng.shuffle(all_ids)
    gang_ids = all_ids[:gang_size]
    decoy_ids = all_ids[gang_size: gang_size + decoy_count]
    real_ids = all_ids[gang_size + decoy_count:]

    # Partition real IDs: last 2 → celebrities, 2 before that → zero-edge isolates
    # Need at least 4 real accounts for this split; fall back gracefully if not.
    if len(real_ids) >= 4:
        celeb_ids = real_ids[-2:]
        zero_edge_ids = real_ids[-4:-2]
        standard_real_ids = real_ids[:-4]
    elif len(real_ids) >= 2:
        celeb_ids = real_ids[-2:]
        zero_edge_ids = []
        standard_real_ids = real_ids[:-2]
    else:
        celeb_ids = []
        zero_edge_ids = []
        standard_real_ids = real_ids

    accounts: List[Dict[str, Any]] = []
    accounts.extend(_gen_gang_accounts(rng, gang_ids, cfg, seed))
    accounts.extend(_gen_decoy_account(rng, did) for did in decoy_ids)
    accounts.extend(_gen_real_account(rng, rid) for rid in standard_real_ids)

    # Zero-edge isolates: real accounts with no connections (edge cases for the agent)
    for rid in zero_edge_ids:
        acc = _gen_real_account(rng, rid)
        acc["features"]["follower_count"] = 0
        acc["features"]["following_count"] = 0
        accounts.append(acc)

    # Celebrity accounts: large legit hubs (high hub_legitimacy → low fake_risk)
    for cid in celeb_ids:
        accounts.append(_gen_celebrity_account(rng, cid))

    _build_edges(rng, accounts, gang_ids, cfg["intra_gang_density"])

    # Choose starting visible accounts.
    # Guarantee exactly 1 gang member is included so the cascade CAN start
    # regardless of seed. The agent still has to identify WHICH account is fake
    # (requires inspecting profiles) — so difficulty is preserved.
    # Without this, ~31% of easy episodes and ~82% of hard episodes start with
    # zero gang members visible, making score variance seed-luck rather than
    # agent skill.
    starting_count = cfg["starting_visible"]
    forced_gang = rng.sample(gang_ids, 1)          # exactly 1 gang member
    rest_pool = [i for i in all_ids if i not in forced_gang]
    additional = rng.sample(rest_pool, starting_count - 1)
    starting_visible = forced_gang + additional
    rng.shuffle(starting_visible)                   # don't reveal which is fake

    # Round 2: Extract hidden signals from accounts into centralized store
    hidden_signals: Dict[str, Dict[str, Any]] = {
        "photo_reuse": {},
        "bio_template": {},
        "ip_cluster": {},
    }

    # Move hidden_signals from account-level to episode-level
    for acc in accounts:
        acc_id = acc["id"]
        if "hidden_signals" in acc:
            hs = acc.pop("hidden_signals")  # Remove from account
            hidden_signals["photo_reuse"][acc_id] = hs.get("photo_reuse_score", 0.0)
            hidden_signals["bio_template"][acc_id] = hs.get("bio_template_score", 0.0)
            hidden_signals["ip_cluster"][acc_id] = hs.get("ip_cluster_id", "")

    return {
        "episode_id": f"{task}_{seed:03d}_{platform}",  # Include platform in ID
        "task": task,
        "seed": seed,
        "platform": platform,  # Round 2: Platform field
        "max_steps": cfg["max_steps"],
        "win_recall": cfg["win_recall"],
        "win_precision": cfg["win_precision"],
        "starting_visible": starting_visible,
        "gang_member_ids": gang_ids,
        "decoy_ids": decoy_ids,
        "celeb_ids": celeb_ids,
        "zero_edge_ids": zero_edge_ids,
        "network": {"accounts": accounts},
        "hidden_signals": hidden_signals,  # Round 2: Centralized hidden signals
        "evasion_schedule": cfg["evasion_schedule"],
    }


# ---------------------------------------------------------------------------
# Bulk generation
# ---------------------------------------------------------------------------

def generate_all_episodes(
    output_dir: Path,
    counts: Dict[str, int] | None = None,
) -> None:
    if counts is None:
        counts = {"easy": 50, "medium": 50, "hard": 50}

    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for task, n in counts.items():
        for i in range(n):
            seed = i  # deterministic
            ep = generate_episode(task, seed)
            fname = output_dir / f"{task}_{i:03d}.json"
            fname.write_text(json.dumps(ep, indent=2))
            total += 1
    print(f"Generated {total} episodes in {output_dir}")


def main() -> None:
    """Entry point: called by `uv run generate-episodes` or directly."""
    here = Path(__file__).parent.parent / "episodes"
    generate_all_episodes(here)


if __name__ == "__main__":
    main()
