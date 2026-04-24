"""Core environment logic for the Fake Gang Detection RL environment."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    AccountProfile,
    AccountStatus,
    FakeGangAction,
    FakeGangObservation,
    FakeGangState,
    ActionType,
)
from server.generator import generate_episode, TASK_CONFIG
from server.scoring import (
    compute_node_risk,
    compute_behavior_risk,
    compute_graph_risk,
    compute_hub_legitimacy,
    compute_fake_risk,
    classify_risk,
    grader_score as _compute_grader_score,
)

# Use the real OpenEnv Environment base class when the SDK is installed;
# fall back to a plain object so the env works without it.
try:
    from openenv.core.env_server import Environment as _OpenEnvBase  # type: ignore
except ImportError:
    class _OpenEnvBase:  # type: ignore[no-redef]
        pass

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

EPISODES_DIR = Path(__file__).parent.parent / "episodes"


class FakeGangEnvironment(_OpenEnvBase):
    """OpenEnv-compatible environment for fake Instagram gang detection."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._ep: Dict[str, Any] = {}
        self._accounts: Dict[str, Dict[str, Any]] = {}   # id -> account dict
        self._live_edges: Dict[str, List[str]] = {}       # id -> follows (mutable, affected by evasion)
        self._reverse_edges: Dict[str, List[str]] = {}    # id -> who follows this id (kept in sync)
        self._gang_ids: List[str] = []
        self._inspected: List[str] = []
        self._flagged: List[str] = []
        self._visible_ids: List[str] = []                 # known to exist
        self._profiled: Dict[str, AccountProfile] = {}    # fully revealed profiles
        self._account_statuses: Dict[str, str] = {}       # id -> "normal"|"suspect"|"confirmed_fake"
        self._last_grader_score: float = 0.0
        self._step_count: int = 0
        self._max_steps: int = 30
        self._task: str = "easy"
        self._evasion_count: int = 0
        self._evasion_triggered: bool = False
        self._episode_id: str = ""
        self._done: bool = False
        self._score: float = 0.0
        self._seed: int = 0

        # Round 2: Platform-specific state
        self._platform: str = ""
        self._policy: Optional[Any] = None  # PlatformPolicy
        self._revealed_signals: Dict[str, Dict[str, Any]] = {
            "photo_reuse": {},
            "bio_template": {},
            "ip_cluster": {},
        }
        # Round 2: tool-use shaping bookkeeping (per-episode)
        self._tool_calls: Dict[str, set] = {  # action_type -> set(account_id)
            "reverse_image_search": set(),
            "analyze_bio": set(),
            "check_ip": set(),
        }
        self._policy_called_early: bool = False
        self._action_count: int = 0  # total step() calls including 0-cost ones
        self._decision_package: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task: str = "easy",
        episode_id: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> FakeGangObservation:
        self._task = task
        self._step_count = 0
        self._evasion_count = 0
        self._evasion_triggered = False
        # Clear evasion-fired flags from previous episodes
        for attr in [a for a in vars(self) if a.startswith('_fired_')]:
            delattr(self, attr)
        self._inspected = []
        self._flagged = []
        self._profiled = {}
        self._account_statuses = {}
        self._last_grader_score = 0.0
        self._done = False
        self._score = 0.0

        # Load or generate episode
        if seed is None:
            seed = random.randint(0, 9999)
        self._seed = seed

        ep = self._load_episode(task, seed)
        self._ep = ep
        self._episode_id = ep["episode_id"]
        self._max_steps = ep["max_steps"]
        self._gang_ids = ep["gang_member_ids"]

        # Round 2: Load platform policy and initialize revealed signals
        self._platform = ep.get("platform", "Instagram")
        self._policy = self._load_policy(self._platform)
        self._revealed_signals = {
            "photo_reuse": {},
            "bio_template": {},
            "ip_cluster": {},
        }
        self._tool_calls = {
            "reverse_image_search": set(),
            "analyze_bio": set(),
            "check_ip": set(),
        }
        self._policy_called_early = False
        self._action_count = 0
        self._decision_package = {}

        # Build account map and live edges
        self._accounts = {a["id"]: a for a in ep["network"]["accounts"]}
        self._live_edges = {
            a["id"]: list(a["true_edges"]["follows"])
            for a in ep["network"]["accounts"]
        }

        # Build reverse index: who follows each account (kept in sync with _live_edges)
        self._reverse_edges = {}
        for follower, targets in self._live_edges.items():
            for target in targets:
                self._reverse_edges.setdefault(target, []).append(follower)

        # Initial visible IDs (not yet profiled)
        self._visible_ids = list(ep["starting_visible"])

        return self._make_observation(message="Episode started. Investigate accounts to find the fake gang.")

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: FakeGangAction, **kwargs: Any) -> FakeGangObservation:
        if self._done:
            return self._make_observation(message="Episode is already over.")

        atype = action.action_type
        acc_id = action.account_id

        # Round 2: shaping — bonus when GET_POLICY is the very first action.
        # Recorded once per episode; surfaced inside _do_get_policy.
        self._action_count += 1

        # Trigger evasion if due BEFORE processing the action
        self._maybe_trigger_evasion()

        if atype == ActionType.SUBMIT:
            return self._do_submit()

        if atype == ActionType.FLAG:
            return self._do_flag(acc_id)

        if atype == ActionType.UNFLAG:
            return self._do_unflag(acc_id)

        if atype == ActionType.INSPECT:
            return self._do_inspect(acc_id)

        if atype == ActionType.INVESTIGATE_NETWORK:
            return self._do_investigate(acc_id)

        # Round 2: New tool-call actions
        if atype == ActionType.REVERSE_IMAGE_SEARCH:
            return self._do_reverse_image_search(acc_id)

        if atype == ActionType.ANALYZE_BIO:
            return self._do_analyze_bio(acc_id)

        if atype == ActionType.CHECK_IP:
            return self._do_check_ip(acc_id)

        if atype == ActionType.GET_POLICY:
            return self._do_get_policy()

        return self._make_observation(message=f"Unknown action: {atype}")

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> FakeGangState:
        return FakeGangState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task=self._task,
            score_so_far=self._score,
            evasion_count=self._evasion_count,
            network_size=len(self._accounts),
            gang_size=len(self._gang_ids),
            episode_seed=self._seed,
            platform=self._platform,  # Round 2
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_inspect(self, acc_id: Optional[str]) -> FakeGangObservation:
        if acc_id is None or acc_id not in self._accounts:
            return self._make_observation(message=f"Cannot INSPECT: account '{acc_id}' not found.")

        self._step_count += 1
        self._score -= 0.01  # time cost

        if acc_id not in self._inspected:
            self._inspected.append(acc_id)

        if acc_id not in self._visible_ids:
            self._visible_ids.append(acc_id)

        # Reveal profile
        self._profiled[acc_id] = self._build_profile(acc_id)

        # Reveal the accounts this one follows
        neighbors = self._live_edges.get(acc_id, [])
        for n in neighbors:
            if n not in self._visible_ids:
                self._visible_ids.append(n)

        # Check step limit
        if self._step_count >= self._max_steps:
            return self._do_submit(forced=True)

        return self._make_observation(
            message=f"Inspected {acc_id}. Found {len(neighbors)} outgoing connections."
        )

    def _do_investigate(self, acc_id: Optional[str]) -> FakeGangObservation:
        if acc_id is None or acc_id not in self._accounts:
            return self._make_observation(message=f"Cannot INVESTIGATE_NETWORK: account '{acc_id}' not found.")

        self._step_count += 2  # costs 2 steps
        self._score -= 0.02

        if acc_id not in self._inspected:
            self._inspected.append(acc_id)

        if acc_id not in self._visible_ids:
            self._visible_ids.append(acc_id)

        # Reveal neighbors AND their neighbors (2-hop), traversing BOTH follow directions.
        # Unidirectional (outgoing-only) expansion misses gang members who follow the target
        # but aren't followed back — with density=0.70 this leaves ~30% unreachable per hop.
        new_ids = set()

        def _add_visible(nid: str) -> None:
            if nid not in self._visible_ids:
                self._visible_ids.append(nid)
                new_ids.add(nid)

        # Outgoing: accounts that acc_id follows
        for n in self._live_edges.get(acc_id, []):
            _add_visible(n)
            for n2 in self._live_edges.get(n, []):
                _add_visible(n2)
            for n2 in self._reverse_edges.get(n, []):
                _add_visible(n2)

        # Incoming: accounts that follow acc_id (reverse edges)
        for n in self._reverse_edges.get(acc_id, []):
            _add_visible(n)
            for n2 in self._live_edges.get(n, []):
                _add_visible(n2)
            for n2 in self._reverse_edges.get(n, []):
                _add_visible(n2)

        # Re-cascade SUSPECT to newly visible accounts using two complementary signals:
        #
        # Signal 1 — follow-graph: newly visible accounts that a flagged account follows.
        # Survives post-evasion because it re-checks live_edges (already updated by evasion).
        for flagged_id in self._flagged:
            for neighbor in self._live_edges.get(flagged_id, []):
                if (neighbor in self._visible_ids
                        and self._account_statuses.get(neighbor, "normal") == "normal"):
                    self._account_statuses[neighbor] = "suspect"
        #
        # Signal 2 — IP cluster: newly revealed accounts sharing the same IP subnet as any
        # flagged account. This catches gang members connected via incoming follow edges that
        # evasion may have removed from live_edges. Zero false positives (gang: shared IP;
        # real/decoy: unique IP per account).
        flagged_ips = {
            self._accounts[fid]["features"].get("ip_cluster_id")
            for fid in self._flagged
            if fid in self._accounts
        }
        flagged_ips.discard(None)
        for new_id in new_ids:
            if new_id not in self._flagged and self._account_statuses.get(new_id, "normal") == "normal":
                vid_ip = self._accounts.get(new_id, {}).get("features", {}).get("ip_cluster_id")
                if vid_ip in flagged_ips:
                    self._account_statuses[new_id] = "suspect"

        # Refresh profiles for already-inspected accounts whose status changed so that
        # Priority 3 in the rule engine sees updated fake_risk (not stale pre-cascade values).
        for inspected_id in list(self._inspected):
            new_status = self._account_statuses.get(inspected_id, "normal")
            if new_status != "normal" and inspected_id in self._profiled:
                cached_status = self._profiled[inspected_id].status.value
                if cached_status != new_status:
                    self._profiled[inspected_id] = self._build_profile(inspected_id)

        if self._step_count >= self._max_steps:
            return self._do_submit(forced=True)

        return self._make_observation(
            message=f"Investigated network around {acc_id}. Discovered {len(new_ids)} new account IDs."
        )

    def _do_flag(self, acc_id: Optional[str]) -> FakeGangObservation:
        if acc_id is None or acc_id not in self._accounts:
            return self._make_observation(message=f"Cannot FLAG: account '{acc_id}' not found.")
        if acc_id not in self._flagged:
            self._flagged.append(acc_id)
            self._account_statuses[acc_id] = "confirmed_fake"
            # Cascade 1 — follow-graph: mark accounts that acc_id follows as SUSPECT.
            # Gang members follow each other (density 0.70+), so this is high-precision.
            for neighbor in self._live_edges.get(acc_id, []):
                if (neighbor in self._visible_ids
                        and self._account_statuses.get(neighbor, "normal") == "normal"):
                    self._account_statuses[neighbor] = "suspect"
            # Cascade 2 — IP cluster: any visible account sharing the same IP subnet is
            # a gang cohort. Gang: shared_ip_count=9, ip_cluster_id="ip_gang_<seed>".
            # Real/decoy: unique ip_cluster_id. Zero false positives.
            flagged_ip = self._accounts[acc_id]["features"].get("ip_cluster_id")
            if flagged_ip:
                for vid in self._visible_ids:
                    if (vid not in self._flagged
                            and self._account_statuses.get(vid, "normal") == "normal"):
                        vid_ip = self._accounts.get(vid, {}).get("features", {}).get("ip_cluster_id")
                        if vid_ip == flagged_ip:
                            self._account_statuses[vid] = "suspect"
            # Refresh profiles for already-inspected accounts that FOLLOW acc_id,
            # because their flagged_neighbor_count just increased (risk score changes).
            for inspected_id in self._inspected:
                if acc_id in self._live_edges.get(inspected_id, []):
                    self._profiled[inspected_id] = self._build_profile(inspected_id)
        return self._make_observation(message=f"Flagged {acc_id} as suspected fake.")

    def _do_unflag(self, acc_id: Optional[str]) -> FakeGangObservation:
        if acc_id is None:
            return self._make_observation(message="Cannot UNFLAG: no account_id provided.")
        if acc_id in self._flagged:
            self._flagged.remove(acc_id)
            self._account_statuses.pop(acc_id, None)
        return self._make_observation(message=f"Removed flag from {acc_id}.")

    # ------------------------------------------------------------------
    # Round 2: New tool-call action handlers
    # ------------------------------------------------------------------

    def _do_reverse_image_search(self, acc_id: Optional[str]) -> FakeGangObservation:
        """Reveal photo_reuse_score for account (costs 1 step)."""
        if acc_id is None or acc_id not in self._accounts:
            return self._make_observation(message=f"Cannot REVERSE_IMAGE_SEARCH: account '{acc_id}' not found.")

        # Shaping: penalize redundant tool reuse on same account
        if acc_id in self._tool_calls["reverse_image_search"]:
            self._score -= 0.05
        self._tool_calls["reverse_image_search"].add(acc_id)

        self._step_count += 1
        self._score -= 0.01  # time cost

        # Reveal from hidden store (or from features if old episode format)
        if "hidden_signals" in self._ep and "photo_reuse" in self._ep["hidden_signals"]:
            score = self._ep["hidden_signals"]["photo_reuse"].get(acc_id, 0.0)
        else:
            # Fallback: old episode format
            score = self._accounts[acc_id]["features"].get("photo_reuse_score", 0.0)

        self._revealed_signals["photo_reuse"][acc_id] = score

        # Update account features
        self._accounts[acc_id]["features"]["photo_reuse_score"] = score

        # Refresh profile if already inspected
        if acc_id in self._profiled:
            self._profiled[acc_id] = self._build_profile(acc_id)

        # Check step limit
        if self._step_count >= self._max_steps:
            return self._do_submit(forced=True)

        count = int(score * 100)  # Simulate "found N matches"
        return self._make_observation(
            message=f"Reverse image search: {acc_id} has photo_reuse_score={score:.2f} (found {count} matching images)"
        )

    def _do_analyze_bio(self, acc_id: Optional[str]) -> FakeGangObservation:
        """Reveal bio_template_score for account (costs 1 step)."""
        if acc_id is None or acc_id not in self._accounts:
            return self._make_observation(message=f"Cannot ANALYZE_BIO: account '{acc_id}' not found.")

        if acc_id in self._tool_calls["analyze_bio"]:
            self._score -= 0.05
        self._tool_calls["analyze_bio"].add(acc_id)

        self._step_count += 1
        self._score -= 0.01  # time cost

        # Reveal from hidden store (or from features if old episode format)
        if "hidden_signals" in self._ep and "bio_template" in self._ep["hidden_signals"]:
            score = self._ep["hidden_signals"]["bio_template"].get(acc_id, 0.0)
        else:
            # Fallback: old episode format
            score = self._accounts[acc_id]["features"].get("bio_template_score", 0.0)

        self._revealed_signals["bio_template"][acc_id] = score

        # Update account features
        self._accounts[acc_id]["features"]["bio_template_score"] = score

        # Refresh profile if already inspected
        if acc_id in self._profiled:
            self._profiled[acc_id] = self._build_profile(acc_id)

        # Check step limit
        if self._step_count >= self._max_steps:
            return self._do_submit(forced=True)

        count = int(score * 10)  # Simulate "matches N templates"
        return self._make_observation(
            message=f"Bio analysis: {acc_id} has bio_template_score={score:.2f} (matches {count} known templates)"
        )

    def _do_check_ip(self, acc_id: Optional[str]) -> FakeGangObservation:
        """Reveal ip_cluster_signal for account (costs 2 steps - expensive!)."""
        if acc_id is None or acc_id not in self._accounts:
            return self._make_observation(message=f"Cannot CHECK_IP: account '{acc_id}' not found.")

        if acc_id in self._tool_calls["check_ip"]:
            self._score -= 0.10  # CHECK_IP is expensive — heavier redundancy penalty
        self._tool_calls["check_ip"].add(acc_id)

        self._step_count += 2  # Higher cost (requires warrant/legal approval)
        self._score -= 0.02

        # Reveal from hidden store (or from features if old episode format)
        if "hidden_signals" in self._ep and "ip_cluster" in self._ep["hidden_signals"]:
            cluster_id = self._ep["hidden_signals"]["ip_cluster"].get(acc_id, "")
        else:
            # Fallback: old episode format
            cluster_id = self._accounts[acc_id]["features"].get("ip_cluster_id", "")

        self._revealed_signals["ip_cluster"][acc_id] = cluster_id

        # Update account features
        self._accounts[acc_id]["features"]["ip_cluster_id"] = cluster_id

        # Count how many visible accounts share this cluster
        if "hidden_signals" in self._ep and "ip_cluster" in self._ep["hidden_signals"]:
            cluster_count = sum(
                1 for vid in self._visible_ids
                if self._ep["hidden_signals"]["ip_cluster"].get(vid) == cluster_id
            )
        else:
            # Fallback: old episode format
            cluster_count = sum(
                1 for vid in self._visible_ids
                if self._accounts.get(vid, {}).get("features", {}).get("ip_cluster_id") == cluster_id
            )

        # Refresh profile if already inspected
        if acc_id in self._profiled:
            self._profiled[acc_id] = self._build_profile(acc_id)

        # Check step limit
        if self._step_count >= self._max_steps:
            return self._do_submit(forced=True)

        return self._make_observation(
            message=f"IP check: {acc_id} shares cluster '{cluster_id}' with {cluster_count} visible accounts"
        )

    def _do_get_policy(self) -> FakeGangObservation:
        """Return platform policy information (costs 0 steps)."""
        # Shaping: small bonus for calling GET_POLICY before any other action
        if not self._policy_called_early and self._action_count == 1:
            self._score += 0.20
            self._policy_called_early = True

        if self._policy is None:
            return self._make_observation(message="Policy not available")

        policy_info = (
            f"Platform: {self._policy.platform} | "
            f"Threshold: {self._policy.threshold:.3f} | "
            f"Primary Signal: {self._policy.primary_enforcement_signal} | "
            f"FP Penalty: {self._policy.fp_penalty_weight}x | "
            f"({self._policy.fn_cost_signal} fn_cost, {self._policy.fp_cost_signal} fp_cost)"
        )

        return self._make_observation(message=f"Policy compiled: {policy_info}")

    # ------------------------------------------------------------------

    def _do_submit(self, forced: bool = False) -> FakeGangObservation:
        self._done = True
        gang_set = set(self._gang_ids)
        flagged_set = set(self._flagged)

        tp = len(gang_set & flagged_set)
        fp = len(flagged_set - gang_set)
        fn = len(gang_set - flagged_set)

        # Round 2: Platform-specific FP penalty
        fp_penalty = self._policy.fp_penalty_weight if self._policy else 0.5
        reward = tp * 1.0 - fp * fp_penalty - fn * 0.3

        recall = tp / len(gang_set) if gang_set else 0.0
        precision = tp / len(flagged_set) if flagged_set else 0.0

        win_recall = self._ep.get("win_recall", 0.8)
        win_precision = self._ep.get("win_precision", 0.7)

        if recall >= win_recall and precision >= win_precision:
            reward += 5.0  # full win bonus
            if tp == len(gang_set):
                reward += 3.0  # perfect recall bonus
        elif recall >= win_recall:
            reward += 2.0  # partial win

        # Efficiency bonus
        steps_left = self._max_steps - self._step_count
        if not forced and steps_left >= self._max_steps * 0.5:
            reward += 1.0

        # Round 2: Platform-specific bonus
        if self._policy:
            if self._policy.platform == "Instagram" and precision >= 0.95:
                reward += 2.0  # Instagram rewards high precision
            elif self._policy.platform == "Snapchat" and recall >= 0.95:
                reward += 2.0  # Snapchat rewards high recall

        # Evasion penalty (hard mode)
        if self._task == "hard":
            reward -= self._evasion_count * 1.0

        if forced:
            reward -= 2.0  # ran out of steps

        # Round 2: insufficient-evidence penalty.
        # Flagging an account whose hidden signals were never revealed is
        # speculative; charge a small per-flag penalty.
        revealed_any = (
            set(self._revealed_signals["photo_reuse"])
            | set(self._revealed_signals["bio_template"])
            | set(self._revealed_signals["ip_cluster"])
        )
        unsupported_flags = [f for f in self._flagged if f not in revealed_any]
        if unsupported_flags:
            reward -= 0.15 * len(unsupported_flags)

        self._score += reward

        # Round 2: Platform-aware grader score
        threshold = self._policy.threshold if self._policy else 0.35
        self._last_grader_score = _compute_grader_score(
            tp, fp, fn, self._step_count, self._max_steps, threshold, fp_penalty
        )

        won = recall >= win_recall and precision >= win_precision

        # Round 2: build moderation decision package.
        # The recommended action is policy-aware: precision-leaning platforms
        # (low FP penalty already paid for) escalate; recall-leaning platforms
        # batch-takedown when recall is high.
        if precision >= 0.95 and recall >= 0.9:
            recommended_action = "batch_takedown"
        elif precision >= 0.8 and recall >= 0.7:
            recommended_action = "scheduled_ban"
        elif precision >= 0.6:
            recommended_action = "temporary_hold"
        else:
            recommended_action = "queue_for_review"

        evidence_summary = {
            "flagged": list(self._flagged),
            "revealed_photo_reuse": len(self._revealed_signals["photo_reuse"]),
            "revealed_bio_template": len(self._revealed_signals["bio_template"]),
            "revealed_ip_cluster": len(self._revealed_signals["ip_cluster"]),
            "unsupported_flags": unsupported_flags,
        }
        policy_rationale = (
            f"{self._platform} policy θ={threshold:.3f} "
            f"(primary={self._policy.primary_enforcement_signal if self._policy else '?'}, "
            f"FP penalty={fp_penalty}); "
            f"observed precision={precision:.2f}, recall={recall:.2f}."
        )
        self._decision_package = {
            "platform": self._platform,
            "flagged_accounts": list(self._flagged),
            "recommended_action": recommended_action,
            "evidence_summary": evidence_summary,
            "policy_rationale": policy_rationale,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "reward": self._score,
            "grader_score": self._last_grader_score,
        }

        msg = (
            f"{'[WIN] ' if won else '[LOSS] '}"
            f"TP={tp} FP={fp} FN={fn} "
            f"Recall={recall:.2f} Precision={precision:.2f} "
            f"Episode reward={self._score:.2f} | "
            f"Decision: {recommended_action} | {policy_rationale}"
        )
        return self._make_observation(message=msg, terminal_reward=self._score)

    # ------------------------------------------------------------------
    # Evasion
    # ------------------------------------------------------------------

    def _maybe_trigger_evasion(self) -> None:
        for event in self._ep.get("evasion_schedule", []):
            if self._step_count >= event["step"] and not self._event_fired(event):
                self._fire_evasion(event)

    def _event_fired(self, event: Dict[str, Any]) -> bool:
        # Track which events have fired by step threshold
        key = f"_fired_{event['step']}"
        return getattr(self, key, False)

    def _fire_evasion(self, event: Dict[str, Any]) -> None:
        step_key = f"_fired_{event['step']}"
        setattr(self, step_key, True)
        self._evasion_count += 1
        self._evasion_triggered = True

        if event["event"] == "unfollow_intragang":
            drop_rate = event.get("drop_rate", 0.5)
            rng = random.Random(self._seed + self._evasion_count)
            gang_set = set(self._gang_ids)
            for g in self._gang_ids:
                follows = self._live_edges.get(g, [])
                kept = [f for f in follows if f not in gang_set or rng.random() > drop_rate]
                dropped = set(follows) - set(kept)
                self._live_edges[g] = kept
                # Keep reverse_edges in sync: remove dropped edges
                for target in dropped:
                    rev = self._reverse_edges.get(target, [])
                    if g in rev:
                        rev.remove(g)

        rename_count = event.get("rename_count", 0)
        if rename_count > 0:
            rng = random.Random(self._seed + self._evasion_count + 1000)
            targets = rng.sample(self._gang_ids, min(rename_count, len(self._gang_ids)))
            for t in targets:
                self._accounts[t]["features"]["name_change_count"] += 1
                # Update profiled cache if already inspected
                if t in self._profiled:
                    self._profiled[t] = self._build_profile(t)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_post_hour_cluster_score(self, acc_hour: float) -> float:
        """How closely does this account's posting hour match the flagged accounts' mean hour?"""
        if not self._flagged:
            return 0.0
        hours = [
            self._accounts[fid]["features"]["avg_post_hour"]
            for fid in self._flagged
            if fid in self._accounts
        ]
        if not hours:
            return 0.0
        mean_h = sum(hours) / len(hours)
        diff = abs(acc_hour - mean_h)
        diff = min(diff, 24.0 - diff)  # wrap-around distance on 24-hour clock
        return round(max(0.0, 1.0 - diff / 6.0), 4)

    def _compute_suspicious_mutual_ratio(self, acc_id: str, follows: List[str]) -> float:
        """Fraction of suspicious follows that also mutually follow this account."""
        suspicious = [
            fid for fid in follows
            if self._account_statuses.get(fid, "normal") in {"suspect", "confirmed_fake"}
        ]
        if not suspicious:
            return 0.0
        mutual = [fid for fid in suspicious if acc_id in self._live_edges.get(fid, [])]
        return round(len(mutual) / len(suspicious), 4)

    def _build_profile(self, acc_id: str) -> AccountProfile:
        a = self._accounts[acc_id]
        f = a["features"]
        follows = list(self._live_edges.get(acc_id, []))

        # ── Derived graph features (computed from live graph state at inspect time) ──

        # How many of this account's follows are already flagged?
        flagged_neighbor_count = sum(1 for fid in follows if fid in self._flagged)

        # Mutual follow rate: fraction of follows that also follow this account back.
        if follows:
            mutual_follow_rate = round(
                sum(1 for fid in follows if acc_id in self._live_edges.get(fid, [])) / len(follows),
                4,
            )
        else:
            mutual_follow_rate = 0.0

        # Average photo_reuse_score among already-inspected neighbors.
        inspected_neighbors = [fid for fid in follows if fid in self._profiled]
        inspected_neighbor_count = len(inspected_neighbors)
        if inspected_neighbors:
            avg_neighbor_photo_reuse = round(
                sum(self._profiled[fid].photo_reuse_score for fid in inspected_neighbors)
                / inspected_neighbor_count,
                4,
            )
        else:
            avg_neighbor_photo_reuse = 0.0

        # ── Full risk score computation ──
        post_hour_cluster_score = self._compute_post_hour_cluster_score(f["avg_post_hour"])
        suspicious_mutual_ratio = self._compute_suspicious_mutual_ratio(acc_id, follows)
        flagged_neighbor_ratio = flagged_neighbor_count / max(inspected_neighbor_count, 1)

        # Round 2: Use 0.0 for signals that haven't been revealed yet
        photo_reuse = f.get("photo_reuse_score", 0.0)
        bio_template = f.get("bio_template_score", 0.0)

        node_risk = compute_node_risk(photo_reuse, bio_template)
        behavior_risk = compute_behavior_risk(f["account_age_days"], post_hour_cluster_score)
        graph_risk = compute_graph_risk(flagged_neighbor_ratio, mutual_follow_rate, avg_neighbor_photo_reuse)
        hub_legitimacy = compute_hub_legitimacy(
            f["follower_count"], f["following_count"],
            f["account_age_days"], suspicious_mutual_ratio,
        )
        fake_risk = compute_fake_risk(node_risk, behavior_risk, graph_risk, hub_legitimacy)

        # Status: explicit (flagged/suspected) takes precedence over formula-derived
        formula_status = classify_risk(fake_risk)
        explicit_status = self._account_statuses.get(acc_id, "normal")
        final_status_str = explicit_status if explicit_status != "normal" else formula_status
        final_status = AccountStatus(final_status_str)

        return AccountProfile(
            account_id=acc_id,
            follower_count=f["follower_count"],
            following_count=f["following_count"],
            post_count=f["post_count"],
            avg_post_hour=f["avg_post_hour"],
            photo_reuse_score=photo_reuse,  # Round 2: Use variable (0.0 if not revealed)
            bio_template_score=bio_template,  # Round 2: Use variable (0.0 if not revealed)
            account_age_days=f["account_age_days"],
            name_change_count=f.get("name_change_count", 0),
            flagged_neighbor_count=flagged_neighbor_count,
            mutual_follow_rate=mutual_follow_rate,
            avg_neighbor_photo_reuse=avg_neighbor_photo_reuse,
            visible_follows=follows,
            status=final_status,
            fake_risk_score=fake_risk,
            node_risk=node_risk,
            behavior_risk=behavior_risk,
            graph_risk=graph_risk,
            hub_legitimacy_score=hub_legitimacy,
            comment_repeat_score=f.get("comment_repeat_score", 0.0),
            shared_ip_count=f.get("shared_ip_count", 0),
            inspected_neighbor_count=inspected_neighbor_count,
            post_hour_cluster_score=post_hour_cluster_score,
            suspicious_mutual_ratio=suspicious_mutual_ratio,
        )

    def _build_hint(self) -> str:
        """Generate actionable hints for the agent based on current state."""
        hints = []

        # Hint 1: Uninspected suspects (highest priority)
        suspect_ids = [
            sid for sid in self._visible_ids
            if sid not in self._flagged
            and self._account_statuses.get(sid, "normal") == "suspect"
        ]
        uninspected_suspects = [s for s in suspect_ids if s not in self._inspected]
        if uninspected_suspects:
            hints.append(f"HINT: {len(uninspected_suspects)} SUSPECT accounts need inspection — INSPECT {uninspected_suspects[0]} next (auto-elevated by cascade, likely gang member).")

        # Hint 2: Unflagged accounts with strong fake signals
        unflagged_fakes = []
        for acc_id in self._inspected:
            if acc_id in self._flagged:
                continue
            p = self._profiled.get(acc_id)
            if not p:
                continue
            if (p.shared_ip_count >= 5
                or (p.photo_reuse_score >= 0.50 and p.bio_template_score >= 0.40
                    and p.hub_legitimacy_score < 0.70)):
                unflagged_fakes.append(acc_id)
        if unflagged_fakes and not uninspected_suspects:
            hints.append(f"HINT: FLAG {unflagged_fakes[0]} — strong fake signals detected (photo_reuse/bio_template/shared_ip). FLAG is FREE (costs 0 steps).")

        # Hint 3: Submit reminder
        steps_left = max(0, self._max_steps - self._step_count)
        if len(self._flagged) >= 10:
            hints.append("HINT: You have 10 flags — SUBMIT now to end the episode and get scored.")
        elif steps_left <= 3 and not self._done:
            hints.append(f"HINT: Only {steps_left} steps left — consider SUBMIT to lock in your score.")

        return " ".join(hints)

    def _make_observation(
        self,
        message: str = "",
        terminal_reward: Optional[float] = None,
    ) -> FakeGangObservation:
        # Append hints to message for agent guidance
        hint = self._build_hint() if not self._done else ""
        full_message = f"{message} {hint}".strip() if hint else message

        return FakeGangObservation(
            done=self._done,
            reward=terminal_reward,
            visible_accounts=[
                self._profiled[i] for i in self._inspected if i in self._profiled
            ],
            visible_account_ids=list(self._visible_ids),
            flagged_ids=list(self._flagged),
            inspected_ids=list(self._inspected),
            graph_edges={
                acc_id: list(self._live_edges.get(acc_id, []))
                for acc_id in self._inspected
            },
            steps_remaining=max(0, self._max_steps - self._step_count),
            evasion_triggered=self._evasion_triggered,
            evasion_count=self._evasion_count,
            task=self._task,
            message=full_message,
            suspect_ids=[
                sid for sid in self._visible_ids
                if sid not in self._flagged
                and self._account_statuses.get(sid, "normal") == "suspect"
            ],
            platform=self._platform,  # Round 2: Pass platform to observation
        )

    def _load_episode(self, task: str, seed: int) -> Dict[str, Any]:
        """Load pre-generated episode JSON or generate on the fly."""
        fname = EPISODES_DIR / f"{task}_{seed:03d}.json"
        if fname.exists():
            return json.loads(fname.read_text())
        # Generate on the fly and cache
        ep = generate_episode(task, seed)
        EPISODES_DIR.mkdir(parents=True, exist_ok=True)
        fname.write_text(json.dumps(ep, indent=2))
        return ep

    def _load_policy(self, platform: str) -> Optional[Any]:
        """Load platform policy from compiler."""
        try:
            from server.policy_compiler import get_policy
            return get_policy(platform)
        except Exception as e:
            print(f"Warning: Failed to load policy for {platform}: {e}")
            # Return a minimal fallback policy
            from models import PlatformPolicy
            return PlatformPolicy(
                platform=platform,
                threshold=0.35,  # Default threshold
                base_rate=0.005,
                fn_cost_signal="high",
                fp_cost_signal="medium",
                harm_weight=1.0,
                primary_enforcement_signal="photo_reuse",
                fp_penalty_weight=0.5,
                sources=[],
                confidence=0.0,
                compiled_at=""
            )
