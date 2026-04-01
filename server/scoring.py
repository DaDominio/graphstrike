"""Pure-math risk scoring engine for the Fake Gang Detection environment.

All functions are stateless — no imports from other project modules.
Implements the formulas from formulas.md exactly.
"""

from __future__ import annotations

import math


def compute_node_risk(photo_reuse: float, bio_template: float) -> float:
    """Content-based risk: stolen photos + copy-paste bios."""
    return round(0.60 * photo_reuse + 0.40 * bio_template, 4)


def compute_behavior_risk(account_age_days: int, post_hour_cluster_score: float) -> float:
    """Temporal risk: recently created + posting in the gang's time window."""
    age_norm = min(1.0, account_age_days / 365.0)
    return round(0.55 * (1.0 - age_norm) + 0.45 * post_hour_cluster_score, 4)


def compute_graph_risk(
    flagged_neighbor_ratio: float,
    mutual_follow_rate: float,
    avg_neighbor_photo_reuse: float,
) -> float:
    """Structural risk: embedded in a flagged cluster + inflated mutual follows."""
    return round(
        0.45 * flagged_neighbor_ratio
        + 0.35 * mutual_follow_rate
        + 0.20 * avg_neighbor_photo_reuse,
        4,
    )


def compute_hub_legitimacy(
    follower_count: int,
    following_count: int,
    account_age_days: int,
    suspicious_mutual_ratio: float,
) -> float:
    """Celebrity/hub discount: large established accounts are unlikely to be fakes.

    High value → high legitimacy → subtract from fake_risk.
    """
    F_MAX = 1_000_000
    followers_norm = min(1.0, math.log1p(follower_count) / math.log1p(F_MAX))
    follow_ratio_norm = min(1.0, (following_count / max(follower_count, 1)) / 5.0)
    age_norm = min(1.0, account_age_days / 365.0)
    return round(
        0.45 * followers_norm
        + 0.25 * (1.0 - follow_ratio_norm)
        + 0.20 * age_norm
        + 0.10 * (1.0 - suspicious_mutual_ratio),
        4,
    )


def compute_fake_risk(
    node_risk: float,
    behavior_risk: float,
    graph_risk: float,
    hub_legitimacy: float,
) -> float:
    """Composite fake risk score in [0.0, 1.0].

    Graph risk carries the most weight (0.45) because structural signals
    are hardest to fake at scale. Hub legitimacy discounts celebrities.
    """
    raw = (
        0.30 * node_risk
        + 0.25 * behavior_risk
        + 0.45 * graph_risk
        - 0.25 * hub_legitimacy
    )
    return round(max(0.0, min(1.0, raw)), 4)


def classify_risk(fake_risk: float) -> str:
    """Map a fake_risk score to an account status string."""
    if fake_risk < 0.35:
        return "normal"
    if fake_risk < 0.60:
        return "suspect"
    return "confirmed_fake"


def grader_score(tp: int, fp: int, fn: int, steps_used: int, max_steps: int) -> float:
    """Normalised [0.0, 1.0] submission score used by /grader endpoint.

    Win condition (recall >= 0.8 AND precision >= 0.7):
        score = 0.55 + 0.20*recall + 0.15*precision + 0.10*efficiency
    Otherwise (partial credit):
        score = 0.30*recall + 0.10*precision
    """
    recall = tp / 10.0
    precision = tp / max(tp + fp, 1)
    efficiency = max(0.0, (max_steps - steps_used) / max_steps)

    if recall >= 0.8 and precision >= 0.7:
        score = 0.55 + 0.20 * recall + 0.15 * precision + 0.10 * efficiency
    else:
        score = 0.30 * recall + 0.10 * precision

    return round(max(0.0, min(1.0, score)), 4)
