"""
Round 2 shared evaluation runner.

Deterministic loop + two LLM decision points per suspicious account.

Framework logic (deterministic, no LLM):
  - GET_POLICY as step 0
  - Loop over visible accounts sorted by risk_score desc
  - INSPECT if not yet inspected
  - INVESTIGATE_NETWORK on very high-risk suspects (risk >= 0.80, once)
  - SUBMIT at end

Per-account LLM decision points:
  - DP1 (tool selection): "reverse_image_search" | "analyze_bio" | "check_ip" | "done"
  - DP2 (flag decision):  "flag" | "skip"

Entry point:
    run_evaluation(model_name, call_llm, platform, base_url, tasks, seeds, out_path)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Make parent (fake_gang_env/) importable so the local client + models resolve
# AHEAD of any same-named modules in site-packages. Always insert at index 0
# (and drop any pre-cached aliases) — a no-op `if not in sys.path` check is
# unsafe because the path may already be present at lower priority.
_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PARENT))
for _stale in ("models", "client"):
    sys.modules.pop(_stale, None)

from client import FakeGangEnvClient, StepResult  # noqa: E402
from models import ActionType, FakeGangAction  # noqa: E402


# ---------------------------------------------------------------------------
# Prompts for the two decision points
# ---------------------------------------------------------------------------

DP1_PROMPT = """You are evaluating a suspicious account. Pick the next tool to call, or 'done' if evidence is sufficient.

PLATFORM: {platform} | primary signal: {primary_signal} | threshold θ*: {threshold:.3f}
ACCOUNT: {account_id} | risk_score: {risk:.3f} | hub_legitimacy: {hub:.2f}

Revealed signals (None = not yet revealed):
  photo_reuse_score:   {photo}
  bio_template_score:  {bio}
  ip_cluster_signal:   {ip}
  shared_ip_count:     {shared_ip}

Budget: steps_remaining={steps_left}  |  tool costs: reverse_image_search=1, analyze_bio=1, check_ip=2

Reply with EXACTLY one token (no punctuation, no explanation):
  reverse_image_search   (reveal photo_reuse_score)
  analyze_bio            (reveal bio_template_score)
  check_ip               (reveal ip_cluster_signal, costs 2 steps)
  done                   (evidence is sufficient, move to flag decision)
"""

DP2_PROMPT = """You must decide whether to FLAG this account as a coordinated fake.

PLATFORM: {platform} | threshold θ*: {threshold:.3f} | fp_penalty: {fp_weight}
ACCOUNT: {account_id} | risk_score: {risk:.3f} | hub_legitimacy: {hub:.2f}

Revealed signals:
  photo_reuse_score:   {photo}
  bio_template_score:  {bio}
  ip_cluster_signal:   {ip}
  shared_ip_count:     {shared_ip}

Running totals: flagged={n_flagged}/10  |  steps_remaining={steps_left}

Guidance:
  - Flag ONLY if risk >= threshold and hub_legitimacy < 0.70 (celebrities are hubs).
  - High photo_reuse + bio_template or shared_ip_count >= 5 is strong evidence.
  - Skip on ambiguous cases when fp_penalty is high.

Reply with EXACTLY one token:
  flag   (mark this account as a coordinated fake)
  skip   (leave it alone, move on)
"""


# ---------------------------------------------------------------------------
# Episode log record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLog:
    model: str
    platform: str
    task: str
    seed: int
    episode_id: str = ""
    threshold: float = 0.0
    primary_signal: str = ""
    steps_taken: int = 0
    inspected: int = 0
    tool_calls: Dict[str, int] = field(default_factory=lambda: {
        "reverse_image_search": 0, "analyze_bio": 0, "check_ip": 0,
        "get_policy": 0, "investigate_network": 0,
    })
    flagged: int = 0
    dp1_calls: int = 0
    dp2_calls: int = 0
    dp1_invalid: int = 0
    dp2_invalid: int = 0
    reward: Optional[float] = None
    grader_score: Optional[float] = None
    final_message: str = ""
    wall_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_account(obs, account_id: str):
    for a in obs.visible_accounts:
        if a.account_id == account_id:
            return a
    return None


def _render_signal(val) -> str:
    if val is None:
        return "None"
    if isinstance(val, float):
        return f"{val:.3f}" if val > 0 else "0.000"
    return str(val)


def _parse_dp1(text: str) -> Optional[str]:
    t = text.strip().lower().split("\n")[0].strip().strip("'\"`.,")
    valid = {"reverse_image_search", "analyze_bio", "check_ip", "done"}
    for tok in valid:
        if tok in t:
            return tok
    return None


def _parse_dp2(text: str) -> Optional[str]:
    t = text.strip().lower().split("\n")[0].strip().strip("'\"`.,")
    if "flag" in t and "unflag" not in t:
        return "flag"
    if "skip" in t or "no" == t or "keep" in t:
        return "skip"
    return None


def _seeds_for_platform(seeds: List[int], platform: Optional[str]) -> List[int]:
    """The env assigns platform by seed%2 (even=Instagram, odd=Snapchat).
    Offset seeds so the requested platform is actually used."""
    if platform is None:
        return seeds
    p = platform.lower()
    if p == "instagram":
        return [s if s % 2 == 0 else s + 1 for s in seeds]
    if p == "snapchat":
        return [s if s % 2 == 1 else s + 1 for s in seeds]
    return seeds  # unknown platform → env will fall back to its own mapping


def _policy_from_message(msg: str) -> Dict[str, float]:
    """Parse the message returned by GET_POLICY into a dict.
    Message format: 'Policy compiled: Platform: X | Threshold: 0.081 | Primary Signal: photo_reuse | FP Penalty: 0.5x | ...'
    """
    out = {"threshold": 0.0, "primary_signal": "", "platform": "", "fp_weight": "?"}
    try:
        body = msg.split("Policy compiled:", 1)[-1]
        for part in body.split("|"):
            k, _, v = part.strip().partition(":")
            k = k.strip().lower()
            v = v.strip()
            if k == "platform":
                out["platform"] = v
            elif k == "threshold":
                out["threshold"] = float(v)
            elif k == "primary signal":
                out["primary_signal"] = v
            elif k == "fp penalty":
                out["fp_weight"] = v
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Per-account loop (DP1 + DP2)
# ---------------------------------------------------------------------------

def _gather_and_flag(
    client: FakeGangEnvClient,
    obs,
    account_id: str,
    policy: Dict,
    call_llm: Callable[[str], str],
    log: EpisodeLog,
    max_dp1_iters: int = 4,
    tuples: Optional[List[Dict]] = None,
) -> StepResult:
    """Run DP1 tool-gathering loop then DP2 flag decision for one account.
    Returns the latest StepResult (observation reflects any actions taken).

    If `tuples` is provided, each LLM decision appends a dict with keys:
      prompt, completion, step_reward, decision_type, platform, threshold,
      fp_penalty, step_index
    """
    last = StepResult(observation=obs, done=False, reward=None, message="")
    last.observation = obs
    plat = policy.get("platform") or log.platform
    fp_w = policy.get("fp_weight", "?")
    thr = float(policy.get("threshold", 0.0) or 0.0)

    # DP1 loop
    for _ in range(max_dp1_iters):
        acc = _find_account(last.observation, account_id)
        if acc is None:
            return last

        photo = acc.photo_reuse_score if acc.photo_reuse_score > 0 else None
        bio = acc.bio_template_score if acc.bio_template_score > 0 else None
        ip_signal = None  # ip_cluster_signal is returned only in the step message; treat as unknown until check_ip called

        prompt = DP1_PROMPT.format(
            platform=policy.get("platform") or log.platform,
            primary_signal=policy.get("primary_signal", "?"),
            threshold=policy.get("threshold", 0.0),
            account_id=account_id,
            risk=acc.fake_risk_score,
            hub=acc.hub_legitimacy_score,
            photo=_render_signal(photo),
            bio=_render_signal(bio),
            ip=_render_signal(ip_signal),
            shared_ip=acc.shared_ip_count,
            steps_left=last.observation.steps_remaining,
        )
        log.dp1_calls += 1
        resp = call_llm(prompt)
        choice = _parse_dp1(resp)
        if choice is None:
            log.dp1_invalid += 1
            if tuples is not None:
                tuples.append({
                    "prompt": prompt, "completion": resp, "step_reward": 0.0,
                    "decision_type": "dp1", "platform": plat, "threshold": thr,
                    "fp_penalty": fp_w, "step_index": log.steps_taken,
                })
            break
        if choice == "done":
            if tuples is not None:
                tuples.append({
                    "prompt": prompt, "completion": resp, "step_reward": 0.0,
                    "decision_type": "dp1", "platform": plat, "threshold": thr,
                    "fp_penalty": fp_w, "step_index": log.steps_taken,
                })
            break

        atype = {
            "reverse_image_search": ActionType.REVERSE_IMAGE_SEARCH,
            "analyze_bio": ActionType.ANALYZE_BIO,
            "check_ip": ActionType.CHECK_IP,
        }[choice]
        last = client.step(FakeGangAction(action_type=atype, account_id=account_id))
        log.tool_calls[choice] = log.tool_calls.get(choice, 0) + 1
        log.steps_taken += 1
        if tuples is not None:
            tuples.append({
                "prompt": prompt, "completion": resp,
                "step_reward": float(last.reward) if last.reward is not None else 0.0,
                "decision_type": "dp1", "platform": plat, "threshold": thr,
                "fp_penalty": fp_w, "step_index": log.steps_taken,
            })
        if last.done or last.observation.steps_remaining <= 1:
            return last

        # Stop early if all cheap signals revealed
        acc2 = _find_account(last.observation, account_id)
        if acc2 and acc2.photo_reuse_score > 0 and acc2.bio_template_score > 0:
            break

    # DP2
    acc = _find_account(last.observation, account_id)
    if acc is None:
        return last
    prompt = DP2_PROMPT.format(
        platform=policy.get("platform") or log.platform,
        threshold=policy.get("threshold", 0.0),
        fp_weight=policy.get("fp_weight", "?"),
        account_id=account_id,
        risk=acc.fake_risk_score,
        hub=acc.hub_legitimacy_score,
        photo=_render_signal(acc.photo_reuse_score if acc.photo_reuse_score > 0 else None),
        bio=_render_signal(acc.bio_template_score if acc.bio_template_score > 0 else None),
        ip=_render_signal(None),
        shared_ip=acc.shared_ip_count,
        n_flagged=len(last.observation.flagged_ids),
        steps_left=last.observation.steps_remaining,
    )
    log.dp2_calls += 1
    resp = call_llm(prompt)
    choice = _parse_dp2(resp)
    step_reward = 0.0
    if choice is None:
        log.dp2_invalid += 1
    elif choice == "flag":
        last = client.step(FakeGangAction(action_type=ActionType.FLAG, account_id=account_id))
        log.flagged = len(last.observation.flagged_ids)
        step_reward = float(last.reward) if last.reward is not None else 0.0
    if tuples is not None:
        tuples.append({
            "prompt": prompt, "completion": resp, "step_reward": step_reward,
            "decision_type": "dp2", "platform": plat, "threshold": thr,
            "fp_penalty": fp_w, "step_index": log.steps_taken,
        })
    return last


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def _run_episode(
    client: FakeGangEnvClient,
    model: str,
    platform: str,
    task: str,
    seed: int,
    call_llm: Callable[[str], str],
    max_accounts_per_episode: int = 15,
    collect_tuples: bool = False,
):
    """Run one episode. Returns EpisodeLog by default, or
    (EpisodeLog, list[tuple_dict]) when collect_tuples=True."""
    log = EpisodeLog(model=model, platform=platform, task=task, seed=seed)
    tuples: List[Dict] = [] if collect_tuples else None  # type: ignore[assignment]
    t0 = time.time()

    res = client.reset(task=task, seed=seed, platform=platform)
    obs = res.observation

    # Deterministic step 0: GET_POLICY
    res = client.step(FakeGangAction(action_type=ActionType.GET_POLICY))
    log.tool_calls["get_policy"] += 1
    policy = _policy_from_message(res.message or res.observation.message)
    log.threshold = float(policy.get("threshold", 0.0) or 0.0)
    log.primary_signal = str(policy.get("primary_signal", "") or "")
    obs = res.observation

    investigated: set[str] = set()
    processed: set[str] = set()

    while not res.done and obs.steps_remaining > 1 and len(processed) < max_accounts_per_episode:
        # Candidate pool: suspects first, then visible not-yet-processed, ranked by risk.
        candidates = [a for a in obs.visible_accounts if a.account_id not in processed]
        suspects = set(obs.suspect_ids)
        candidates.sort(
            key=lambda a: (a.account_id in suspects, a.fake_risk_score),
            reverse=True,
        )
        if not candidates:
            break
        acc = candidates[0]
        aid = acc.account_id

        # Ensure inspected
        if aid not in obs.inspected_ids:
            res = client.step(FakeGangAction(action_type=ActionType.INSPECT, account_id=aid))
            log.steps_taken += 1
            log.inspected += 1
            obs = res.observation
            if res.done or obs.steps_remaining <= 1:
                break

        # Expand network once for very risky hubs
        acc_now = _find_account(obs, aid)
        if (
            acc_now
            and acc_now.fake_risk_score >= 0.80
            and aid not in investigated
            and obs.steps_remaining >= 5
        ):
            res = client.step(FakeGangAction(action_type=ActionType.INVESTIGATE_NETWORK, account_id=aid))
            log.tool_calls["investigate_network"] += 1
            log.steps_taken += 2
            investigated.add(aid)
            obs = res.observation
            if res.done or obs.steps_remaining <= 1:
                break

        # DP1 + DP2 per account
        res = _gather_and_flag(client, obs, aid, policy, call_llm, log, tuples=tuples)
        obs = res.observation
        processed.add(aid)
        if res.done:
            break

    # Final submit (if not already done)
    if not res.done:
        res = client.step(FakeGangAction(action_type=ActionType.SUBMIT))

    log.flagged = len(res.observation.flagged_ids)
    log.reward = res.reward
    log.final_message = (res.message or "")[:400]
    log.episode_id = getattr(res.observation, "episode_id", "") or f"{task}_{seed:03d}_{platform}"
    log.wall_seconds = round(time.time() - t0, 2)

    # Grader endpoint (optional)
    try:
        import requests
        g = requests.get(f"{client.base_url}/grader", timeout=30).json()
        log.grader_score = g.get("score")
    except Exception:
        pass

    if collect_tuples:
        # Attach episode-level identifiers to each decision tuple.
        for t in tuples:
            t["episode_id"] = log.episode_id
            t["grader_score"] = log.grader_score
        return log, tuples
    return log


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    model_name: str,
    call_llm: Callable[[str], str],
    platform: str,
    base_url: str = "http://localhost:8000",
    tasks: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    out_path: Optional[str] = None,
) -> List[EpisodeLog]:
    tasks = tasks or ["easy", "medium", "hard"]
    seeds = seeds or [0, 1, 2]
    seeds = _seeds_for_platform(seeds, platform)

    out_path = out_path or str(
        _PARENT / "eval-models" / "results" /
        f"{model_name.replace('/', '_')}_{platform.lower()}_results.jsonl"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Round 2 evaluation | model={model_name} | platform={platform}")
    print(f"Target: {base_url} | tasks={tasks} | seeds={seeds}")
    print(f"Log:    {out_path}")
    print(f"{'='*70}")

    logs: List[EpisodeLog] = []
    client = FakeGangEnvClient(base_url=base_url)

    with open(out_path, "w") as f:
        for task in tasks:
            for seed in seeds:
                print(f"\n--- episode: task={task} seed={seed} ---")
                try:
                    log = _run_episode(client, model_name, platform, task, seed, call_llm)
                except Exception as e:
                    print(f"  ✗ episode failed: {e}")
                    log = EpisodeLog(
                        model=model_name, platform=platform, task=task, seed=seed,
                        final_message=f"EXCEPTION: {e}",
                    )
                logs.append(log)
                f.write(json.dumps(asdict(log)) + "\n")
                f.flush()
                print(
                    f"  → steps={log.steps_taken} inspected={log.inspected} "
                    f"flagged={log.flagged} dp1={log.dp1_calls}(bad={log.dp1_invalid}) "
                    f"dp2={log.dp2_calls}(bad={log.dp2_invalid}) "
                    f"reward={log.reward} grader={log.grader_score} ({log.wall_seconds}s)"
                )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    rewards = [l.reward for l in logs if l.reward is not None]
    graders = [l.grader_score for l in logs if l.grader_score is not None]
    if rewards:
        print(f"  mean reward:  {sum(rewards)/len(rewards):.4f}  (n={len(rewards)})")
    if graders:
        print(f"  mean grader:  {sum(graders)/len(graders):.4f}  (n={len(graders)})")
    total_dp1 = sum(l.dp1_calls for l in logs)
    total_dp2 = sum(l.dp2_calls for l in logs)
    bad_dp1 = sum(l.dp1_invalid for l in logs)
    bad_dp2 = sum(l.dp2_invalid for l in logs)
    print(f"  DP1 calls:    {total_dp1}  (invalid: {bad_dp1})")
    print(f"  DP2 calls:    {total_dp2}  (invalid: {bad_dp2})")
    print(f"  Logged to:    {out_path}")
    return logs


# ---------------------------------------------------------------------------
# Standard CLI argument parser (reused by every model shim)
# ---------------------------------------------------------------------------

def build_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Round 2 eval runner")
    parser.add_argument("--url", default=os.getenv("API_BASE_URL_ENV", "http://localhost:8000"),
                        help="Environment server URL")
    parser.add_argument("--platform", default="Instagram", help="Platform name (Instagram/Snapchat/...)")
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default=None, help="Output JSONL path")
    return parser
