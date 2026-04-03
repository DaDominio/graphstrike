"""FastAPI server for the Fake Gang Detection OpenEnv environment."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing sibling modules when running from server/ dir
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from models import FakeGangAction, FakeGangObservation, FakeGangState
from environment import FakeGangEnvironment

# ---------------------------------------------------------------------------
# App setup — use real OpenEnv SDK wiring when available
# ---------------------------------------------------------------------------

try:
    from openenv.core.env_server import create_fastapi_app  # type: ignore
    app = create_fastapi_app(FakeGangEnvironment)
    _using_sdk = True
except ImportError:
    _using_sdk = False
    app = FastAPI(
        title="GraphStrike — OpenEnv",
        description="RL environment for detecting coordinated fake account rings in social networks.",
        version="1.0.0",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-session environment (for local dev / demo)
# For concurrent sessions, use a session manager keyed by session ID.
_env = FakeGangEnvironment()


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    done: bool
    reward: Optional[float]
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest) -> StepResponse:
    obs = _env.reset(task=req.task, seed=req.seed, episode_id=req.episode_id)
    return StepResponse(
        observation=obs.model_dump(),
        done=obs.done,
        reward=obs.reward,
        message=obs.message,
    )


@app.post("/step", response_model=StepResponse)
def step(action: FakeGangAction) -> StepResponse:
    obs = _env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        done=obs.done,
        reward=obs.reward,
        message=obs.message,
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    return _env.state.model_dump()


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": ["easy", "medium", "hard"],
        "descriptions": {
            "easy": "50 accounts, 10 fakes, no evasion, 30 steps",
            "medium": "200 accounts, 10 fakes + 20 decoys, evasion at step 20, 50 steps",
            "hard": "1000 accounts, 10 fakes + 50 decoys, recurring evasion, 80 steps",
        },
        "action_schema": {
            "action_type": ["inspect", "investigate_network", "flag", "unflag", "submit"],
            "account_id": "string (required for all actions except submit)",
        },
        "score_range": [0.0, 1.0],
    }


@app.get("/grader")
def grader() -> Dict[str, Any]:
    if not _env._done:
        raise HTTPException(status_code=400, detail="Episode not complete. Call SUBMIT first.")
    return {
        "score": _env._last_grader_score,
        "task": _env._task,
        "episode_id": _env._episode_id,
    }


@app.post("/baseline")
def baseline() -> Dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference import run_rule_based_episode  # type: ignore[import]
    scores: Dict[str, float] = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_rule_based_episode(_env, task=task, seed=0)
    return {"scores": scores, "agent": "rule_based"}


# ---------------------------------------------------------------------------
# Entry points (called by `uv run server` via pyproject.toml scripts)
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the environment server. PORT env var controls the port (default 7860)."""
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
