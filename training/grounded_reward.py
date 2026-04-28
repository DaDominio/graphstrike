"""Env-grounded reward computation for GRPO."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import requests as _requests

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_PARENT / "eval-models"))

from _round2_runner import _run_episode  # noqa: E402
from client import FakeGangEnvClient  # noqa: E402

from training.parse import parse_completion
from training.rewards import compute_reward, RewardBreakdown
from training.build_dataset import heuristic_teacher

EPISODE_TIMEOUT_S = 45   # hard-kill per episode so 429 loops never stall the trainer
_MAX_ATTEMPTS     = 8    # per individual HTTP request


class SmartSession:
    """requests.Session wrapper that respects Retry-After on 429 and backs off
    on 5xx. Never raises on 429 — callers handle non-2xx status codes normally
    after all retries are exhausted."""

    def __init__(self):
        self._s = _requests.Session()

    def _call(self, method: str, url: str, **kwargs):
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = getattr(self._s, method)(url, **kwargs)
            except _requests.RequestException as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
                continue

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", min(5 * (2 ** attempt), 60)))
                print(f"  [env 429] waiting {wait}s (attempt {attempt + 1}/{_MAX_ATTEMPTS})")
                # Ping /health every 5s during the backoff window.
                # Keeps the env Space warm so the retry hits a hot instance,
                # not a cold-start that adds another 30s stall.
                health_url = url.split("/reset")[0].split("/step")[0] + "/health"
                elapsed = 0
                while elapsed < wait:
                    chunk = min(5, wait - elapsed)
                    time.sleep(chunk)
                    elapsed += chunk
                    if elapsed < wait:
                        try:
                            self._s.get(health_url, timeout=3)
                        except Exception:
                            pass
                continue

            return resp  # 2xx, 4xx-other, 5xx — let caller decide

        return resp  # last response after all retries

    def post(self, url, **kwargs):
        return self._call("post", url, **kwargs)

    def get(self, url, **kwargs):
        return self._call("get", url, **kwargs)

    def close(self):
        self._s.close()


_SESSION = SmartSession()


def _make_client(base_url: str) -> FakeGangEnvClient:
    client = FakeGangEnvClient(base_url=base_url)
    if hasattr(client, "_session"):
        client._session = _SESSION
    return client


def _injecting_llm(target_idx: int, completion: str, baseline: Callable[[str], str]):
    state = {"i": 0}

    def call(prompt: str) -> str:
        idx = state["i"]
        state["i"] += 1
        return completion if idx == target_idx else baseline(prompt)

    return call


def _timeout_handler(signum, frame):
    raise TimeoutError("episode timeout")


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
    """Replay one episode injecting `completion` at turn `decision_index`.

    Returns reward=0.0 on timeout or env error — never raises.
    """
    _, _, format_ok = parse_completion(completion, decision_type)
    zero_rb = compute_reward(grader_score=0.0, step_reward=None, format_ok=format_ok)

    client = client or _make_client(base_url)
    inject = _injecting_llm(decision_index, completion, heuristic_teacher())

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(EPISODE_TIMEOUT_S)
    try:
        log, tuples = _run_episode(
            client, "grpo:rollout", platform, task, seed, inject,
            collect_tuples=True,
        )
        signal.alarm(0)
    except (TimeoutError, Exception) as exc:
        signal.alarm(0)
        print(f"  [grounded_reward] aborted ({type(exc).__name__}): {exc}")
        return zero_rb, {"timeout": True, "grader": 0.0}

    step_reward = None
    matched = None
    if 0 <= decision_index < len(tuples):
        matched = tuples[decision_index]
        step_reward = matched.get("step_reward")

    rb = compute_reward(
        grader_score=log.grader_score,
        step_reward=step_reward,
        format_ok=format_ok,
    )
    return rb, {
        "episode_id": log.episode_id,
        "grader": log.grader_score,
        "n_decisions": len(tuples),
        "matched_turn": matched is not None,
        "format_ok": format_ok,
        "timeout": False,
    }
