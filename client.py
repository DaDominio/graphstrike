"""Python client for the Fake Gang Detection OpenEnv environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from models import (
    AccountProfile,
    FakeGangAction,
    FakeGangObservation,
    FakeGangState,
    ActionType,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    observation: FakeGangObservation
    done: bool
    reward: Optional[float]
    message: str


# ---------------------------------------------------------------------------
# Sync HTTP client
# ---------------------------------------------------------------------------

class FakeGangEnvClient:
    """Synchronous HTTP client for the Fake Gang Detection environment."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        if requests is None:
            raise ImportError("Install 'requests' to use FakeGangEnvClient.")
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        task: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> StepResult:
        payload = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        resp = self._post("/reset", payload)
        return self._parse_result(resp)

    def step(self, action: FakeGangAction) -> StepResult:
        resp = self._post("/step", action.model_dump())
        return self._parse_result(resp)

    def state(self) -> FakeGangState:
        resp = self._session.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return FakeGangState(**resp.json())

    def health(self) -> Dict[str, str]:
        resp = self._session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def inspect(self, account_id: str) -> StepResult:
        return self.step(FakeGangAction(action_type=ActionType.INSPECT, account_id=account_id))

    def investigate_network(self, account_id: str) -> StepResult:
        return self.step(FakeGangAction(action_type=ActionType.INVESTIGATE_NETWORK, account_id=account_id))

    def flag(self, account_id: str) -> StepResult:
        return self.step(FakeGangAction(action_type=ActionType.FLAG, account_id=account_id))

    def unflag(self, account_id: str) -> StepResult:
        return self.step(FakeGangAction(action_type=ActionType.UNFLAG, account_id=account_id))

    def submit(self) -> StepResult:
        return self.step(FakeGangAction(action_type=ActionType.SUBMIT))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(f"{self.base_url}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs_data = payload["observation"]
        profiles = [AccountProfile(**p) for p in obs_data.get("visible_accounts", [])]
        obs = FakeGangObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            visible_accounts=profiles,
            visible_account_ids=obs_data.get("visible_account_ids", []),
            flagged_ids=obs_data.get("flagged_ids", []),
            inspected_ids=obs_data.get("inspected_ids", []),
            graph_edges=obs_data.get("graph_edges", {}),
            steps_remaining=obs_data.get("steps_remaining", 0),
            evasion_triggered=obs_data.get("evasion_triggered", False),
            evasion_count=obs_data.get("evasion_count", 0),
            task=obs_data.get("task", "easy"),
            message=obs_data.get("message", ""),
        )
        return StepResult(
            observation=obs,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            message=payload.get("message", ""),
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "FakeGangEnvClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._session.close()
