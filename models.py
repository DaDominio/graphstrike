from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# OpenEnv base types
# Use real SDK when available; fall back to stubs for local dev without SDK.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server import Action, Observation, State  # type: ignore
except ImportError:
    class Action(BaseModel):  # type: ignore[no-redef]
        pass

    class Observation(BaseModel):  # type: ignore[no-redef]
        done: bool = False
        reward: Optional[float] = None

    class State(BaseModel):  # type: ignore[no-redef]
        episode_id: str = ""
        step_count: int = 0

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    INSPECT = "inspect"                       # reveal full profile + edges, costs 1 step
    INVESTIGATE_NETWORK = "investigate_network"  # expand graph 1 hop, costs 2 steps
    FLAG = "flag"                             # mark account fake (free)
    UNFLAG = "unflag"                         # unmark account (free)
    SUBMIT = "submit"                         # end episode, trigger scoring


class AccountStatus(str, Enum):
    NORMAL = "normal"
    SUSPECT = "suspect"           # auto-elevated when a neighbor is flagged
    CONFIRMED_FAKE = "confirmed_fake"  # agent explicitly flagged this account


class FakeGangAction(Action):
    action_type: ActionType
    account_id: Optional[str] = None  # required for all actions except SUBMIT


class AccountProfile(BaseModel):
    account_id: str
    follower_count: int
    following_count: int
    post_count: int
    avg_post_hour: float        # 0–23
    photo_reuse_score: float    # 0–1 — pre-computed: fraction of posts using stolen celebrity photos
    bio_template_score: float   # 0–1 — pre-computed: cosine similarity to known fake bio templates
    account_age_days: int
    name_change_count: int = 0  # incremented by hard-mode evasion events

    # ── Derived graph features (computed at INSPECT time from live graph state) ──
    flagged_neighbor_count: int = 0    # how many of this account's follows are currently flagged
                                       # high value = deep inside a cluster you're already tracking
    mutual_follow_rate: float = 0.0    # fraction of follows that also follow back (0–1)
                                       # real fans: low; fake gangs: high (they mutually inflate each other)
    avg_neighbor_photo_reuse: float = 0.0  # mean photo_reuse_score of inspected follows
                                           # gang members cluster: if neighbors are fake, this is high

    visible_follows: List[str] = []    # IDs of accounts this account follows (revealed by INSPECT)

    # ── Account status ──
    status: AccountStatus = AccountStatus.NORMAL

    # ── Full risk breakdown (computed via scoring.py at INSPECT time) ──
    fake_risk_score: float = 0.0
    node_risk: float = 0.0
    behavior_risk: float = 0.0
    graph_risk: float = 0.0
    hub_legitimacy_score: float = 0.0

    # ── New raw features (from generator) ──
    comment_repeat_score: float = 0.0   # fakes: 0.6-0.9 | decoys: 0.1-0.3 | reals: 0.0-0.08
    shared_ip_count: int = 0            # fakes: 9 (gang shares 1 IP) | reals: 0-1

    # ── Extended runtime graph features ──
    inspected_neighbor_count: int = 0   # denominator for flagged_neighbor_ratio
    post_hour_cluster_score: float = 0.0  # hour alignment to flagged cluster mean
    suspicious_mutual_ratio: float = 0.0  # used in hub legitimacy computation


class FakeGangObservation(Observation):
    visible_accounts: List[AccountProfile] = []
    visible_account_ids: List[str] = []   # all account IDs the agent knows exist
    flagged_ids: List[str] = []
    inspected_ids: List[str] = []
    graph_edges: Dict[str, List[str]] = {}  # account_id -> list of accounts it follows
    steps_remaining: int = 0
    evasion_triggered: bool = False
    evasion_count: int = 0
    task: str = "easy"
    message: str = ""
    suspect_ids: List[str] = []  # auto-elevated neighbors of flagged accounts


class FakeGangState(State):
    task: str = "easy"
    score_so_far: float = 0.0
    evasion_count: int = 0
    network_size: int = 0
    gang_size: int = 10
    episode_seed: int = 0
