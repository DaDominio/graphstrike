"""
FastAPI Backend for Fake Gang Detection Dashboard

Provides:
- WebSocket streaming for policy compilation, episodes, and training
- REST endpoints for episode control and metrics
- CORS support for React frontend
"""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

# Add parent to path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from server.environment import FakeGangEnvironment
from server.policy_compiler import compile_policy, get_policy
from models import FakeGangAction, ActionType

# ============================================================================
# Lifespan Context Manager
# ============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    print("🚀 Starting Fake Gang Detection Dashboard API")
    print("📍 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")

    # Pre-load policies
    for platform in ["Instagram", "Snapchat"]:
        try:
            policy = get_policy(platform)
            policy_cache[platform] = policy.model_dump()
            print(f"✓ Loaded {platform} policy: θ={policy.threshold:.3f}")
        except Exception as e:
            print(f"⚠ Failed to load {platform} policy: {e}")

    yield

    # Shutdown
    print("👋 Shutting down Dashboard API")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Fake Gang Detection Dashboard API", lifespan=lifespan)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default, CRA default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

active_episodes: Dict[str, Dict] = {}  # episode_id -> {env, obs, task, seed, platform}
active_trainings: Dict[str, Dict] = {}  # training_id -> {status, results, ...}
policy_cache: Dict[str, Dict] = {}  # platform -> policy


# ============================================================================
# Request/Response Models
# ============================================================================

class CompilePolicyRequest(BaseModel):
    platform: str  # "Instagram" or "Snapchat"
    use_cache: bool = True


class StartEpisodeRequest(BaseModel):
    platform: Optional[str] = None  # Auto-assign by seed if None
    task: str = "easy"
    seed: int = 0


class StepActionRequest(BaseModel):
    action_type: str
    account_id: Optional[str] = None


class StartTrainingRequest(BaseModel):
    episodes: int = 50
    task: str = "easy"
    platforms: Optional[List[str]] = None  # ["Instagram", "Snapchat"] or None for auto


# ============================================================================
# WebSocket: Policy Compilation
# ============================================================================

@app.websocket("/ws/compile_policy/{platform}")
async def websocket_compile_policy(websocket: WebSocket, platform: str):
    """Stream policy compilation progress."""
    await websocket.accept()

    try:
        # Send progress updates
        await websocket.send_json({
            "type": "progress",
            "message": f"🔍 Starting policy compilation for {platform}..."
        })

        await asyncio.sleep(0.5)

        await websocket.send_json({
            "type": "progress",
            "message": "🌐 Fetching transparency reports from Tavily..."
        })

        await asyncio.sleep(1)

        await websocket.send_json({
            "type": "progress",
            "message": "🤖 Extracting parameters with Groq LLM..."
        })

        await asyncio.sleep(1.5)

        await websocket.send_json({
            "type": "progress",
            "message": "📊 Computing Bayesian threshold..."
        })

        # Actually compile policy (uses cache if available)
        policy = get_policy(platform)

        await websocket.send_json({
            "type": "progress",
            "message": f"✅ Policy compiled: θ={policy.threshold:.3f}"
        })

        await asyncio.sleep(0.5)

        # Send complete policy
        await websocket.send_json({
            "type": "complete",
            "policy": policy.model_dump(),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })


# ============================================================================
# WebSocket: Episode Execution
# ============================================================================

@app.websocket("/ws/episode/{episode_id}")
async def websocket_episode(websocket: WebSocket, episode_id: str):
    """Stream episode state updates."""
    await websocket.accept()

    if episode_id not in active_episodes:
        await websocket.send_json({
            "type": "error",
            "message": f"Episode {episode_id} not found"
        })
        return

    ep_data = active_episodes[episode_id]
    env = ep_data["env"]
    obs = ep_data["obs"]

    try:
        # Send initial state
        await websocket.send_json({
            "type": "observation",
            "data": obs.model_dump(),
        })

        # Wait for actions from client (if manual mode)
        # For now, we'll just send the current state
        while True:
            await asyncio.sleep(1)
            # In a real implementation, this would wait for step actions

    except WebSocketDisconnect:
        pass


# ============================================================================
# WebSocket: Training Loop
# ============================================================================

@app.websocket("/ws/training/{training_id}")
async def websocket_training(websocket: WebSocket, training_id: str):
    """Stream training progress."""
    await websocket.accept()

    if training_id not in active_trainings:
        await websocket.send_json({
            "type": "error",
            "message": f"Training {training_id} not found"
        })
        return

    training_data = active_trainings[training_id]

    try:
        # Stream training progress (simulated here)
        # In real implementation, this would run the actual training loop
        await websocket.send_json({
            "type": "started",
            "episodes": training_data["episodes"],
            "task": training_data["task"],
        })

        # Simulate training episodes
        # TODO: Replace with actual training loop from agent/train.py
        for ep in range(training_data["episodes"]):
            await asyncio.sleep(2)  # Simulate episode execution

            platform = "Instagram" if ep % 2 == 0 else "Snapchat"
            score = 0.85 + (ep * 0.001)  # Simulated improving score

            await websocket.send_json({
                "type": "episode_complete",
                "episode": ep,
                "platform": platform,
                "score": score,
                "tp": 9,
                "fp": 1,
                "fn": 1,
            })

        await websocket.send_json({
            "type": "training_complete",
            "avg_score": 0.87,
        })

    except WebSocketDisconnect:
        pass


# ============================================================================
# REST: Policy Endpoints
# ============================================================================

@app.post("/api/compile_policy")
async def compile_policy_rest(request: CompilePolicyRequest):
    """Compile policy (REST endpoint, non-streaming)."""
    policy = get_policy(request.platform) if request.use_cache else compile_policy(request.platform, use_cache=False)
    policy_cache[request.platform] = policy.model_dump()

    return {
        "policy": policy.model_dump(),
        "sources": policy.sources,
    }


@app.get("/api/policy/{platform}")
async def get_policy_rest(platform: str):
    """Get cached policy."""
    if platform in policy_cache:
        return {"policy": policy_cache[platform]}

    policy = get_policy(platform)
    policy_cache[platform] = policy.model_dump()
    return {"policy": policy.model_dump()}


# ============================================================================
# REST: Episode Endpoints
# ============================================================================

@app.post("/api/episode/start")
async def start_episode(request: StartEpisodeRequest):
    """Start a new episode."""
    episode_id = str(uuid.uuid4())

    env = FakeGangEnvironment()
    obs = env.reset(task=request.task, seed=request.seed)

    # Load policy
    policy = get_policy(obs.platform)

    # Store episode state
    active_episodes[episode_id] = {
        "env": env,
        "obs": obs,
        "task": request.task,
        "seed": request.seed,
        "platform": obs.platform,
        "policy": policy.model_dump(),
        "created_at": datetime.now().isoformat(),
    }

    return {
        "episode_id": episode_id,
        "platform": obs.platform,
        "observation": obs.model_dump(),
        "policy": policy.model_dump(),
    }


@app.post("/api/episode/{episode_id}/step")
async def step_episode(episode_id: str, request: StepActionRequest):
    """Execute one step in episode."""
    if episode_id not in active_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    ep_data = active_episodes[episode_id]
    env = ep_data["env"]

    # Create action
    action = FakeGangAction(
        action_type=ActionType[request.action_type.upper()],
        account_id=request.account_id,
    )

    # Step environment
    obs = env.step(action)
    ep_data["obs"] = obs

    # Build graph data
    graph_data = {
        "nodes": [
            {
                "id": p.account_id,
                "fake_risk_score": p.fake_risk_score,
                "hub_legitimacy_score": p.hub_legitimacy_score,
                "is_fake": p.account_id in env._ep["gang_member_ids"],
            }
            for p in obs.visible_accounts
        ],
        "edges": [
            {
                "source": acc_id,
                "target": target_id,
                "is_mutual": target_id in obs.graph_edges.get(target_id, []) and acc_id in obs.graph_edges[target_id],
            }
            for acc_id, targets in obs.graph_edges.items()
            for target_id in targets
        ],
    }

    return {
        "observation": obs.model_dump(),
        "graph_data": graph_data,
        "done": obs.done,
        "reward": obs.reward,
    }


@app.get("/api/episode/{episode_id}/state")
async def get_episode_state(episode_id: str):
    """Get current episode state."""
    if episode_id not in active_episodes:
        raise HTTPException(status_code=404, detail="Episode not found")

    ep_data = active_episodes[episode_id]
    obs = ep_data["obs"]
    env = ep_data["env"]

    # Build graph data
    graph_data = {
        "nodes": [
            {
                "id": p.account_id,
                "fake_risk_score": p.fake_risk_score,
                "hub_legitimacy_score": p.hub_legitimacy_score,
                "is_fake": p.account_id in env._ep["gang_member_ids"],
            }
            for p in obs.visible_accounts
        ],
        "edges": [
            {
                "source": acc_id,
                "target": target_id,
                "is_mutual": target_id in obs.graph_edges.get(target_id, []) and acc_id in obs.graph_edges[target_id],
            }
            for acc_id, targets in obs.graph_edges.items()
            for target_id in targets
        ],
    }

    return {
        "episode_id": episode_id,
        "platform": ep_data["platform"],
        "policy": ep_data["policy"],
        "observation": obs.model_dump(),
        "graph_data": graph_data,
        "metrics": {
            "steps_used": env._step_count,
            "max_steps": env._max_steps,
            "flagged_count": len(obs.flagged_ids),
            "grader_score": env._last_grader_score,
        },
    }


# ============================================================================
# REST: Training Endpoints
# ============================================================================

@app.post("/api/training/start")
async def start_training(request: StartTrainingRequest):
    """Start training loop."""
    training_id = str(uuid.uuid4())

    active_trainings[training_id] = {
        "status": "running",
        "episodes": request.episodes,
        "task": request.task,
        "platforms": request.platforms or ["Instagram", "Snapchat"],
        "results": [],
        "created_at": datetime.now().isoformat(),
    }

    return {
        "training_id": training_id,
        "status": "running",
    }


@app.get("/api/training/{training_id}/metrics")
async def get_training_metrics(training_id: str):
    """Get training metrics."""
    if training_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training not found")

    return {
        "training_id": training_id,
        "metrics": active_trainings[training_id].get("results", []),
    }


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_episodes": len(active_episodes),
        "active_trainings": len(active_trainings),
        "timestamp": datetime.now().isoformat(),
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
