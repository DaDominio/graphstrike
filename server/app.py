"""FastAPI + Gradio server for the GraphStrike OpenEnv environment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from models import FakeGangAction, FakeGangObservation, FakeGangState, ActionType
from environment import FakeGangEnvironment

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GraphStrike — OpenEnv",
    description="RL environment for detecting coordinated fake account rings in social networks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_env = FakeGangEnvironment()


# ---------------------------------------------------------------------------
# Schemas
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
# OpenEnv API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest):
    obs = _env.reset(task=req.task, seed=req.seed, episode_id=req.episode_id)
    return StepResponse(observation=obs.model_dump(), done=obs.done, reward=obs.reward, message=obs.message)

@app.post("/step", response_model=StepResponse)
def step(action: FakeGangAction):
    obs = _env.step(action)
    return StepResponse(observation=obs.model_dump(), done=obs.done, reward=obs.reward, message=obs.message)

@app.get("/state")
def state():
    return _env.state.model_dump()

@app.get("/tasks")
def list_tasks():
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
        "strategy_guide": (
            "Goal: FLAG exactly 10 coordinated fake accounts hidden in the network. "
            "1) INSPECT accounts to reveal profiles (costs 1 step). "
            "2) Look for fake signals: shared_ip_count >= 5 (gang shares one IP), "
            "photo_reuse_score >= 0.50, bio_template_score >= 0.40, comment_repeat_score >= 0.50. "
            "3) FLAG suspicious accounts (FREE — costs 0 steps). "
            "4) After FLAG, check suspect_ids in observation — these are auto-elevated neighbors, likely more gang members. "
            "5) INSPECT suspects first (highest priority). "
            "6) SUBMIT when you have 10 flags. "
            "Do NOT flag accounts with hub_legitimacy_score > 0.70 (celebrities). "
            "The observation 'message' field contains actionable HINT lines."
        ),
    }

@app.get("/grader")
def grader():
    if not _env._done:
        raise HTTPException(status_code=400, detail="Episode not complete. Call SUBMIT first.")
    return {"score": _env._last_grader_score, "task": _env._task, "episode_id": _env._episode_id}

@app.get("/metadata")
def metadata():
    return {
        "name": "graphstrike",
        "description": "RL environment for detecting coordinated fake account rings in social networks.",
        "version": "1.0.0", "author": "Pandago",
        "tags": ["social-network", "fraud-detection", "graph", "rl"],
    }

@app.get("/schema")
def schema():
    return {
        "action": FakeGangAction.model_json_schema(),
        "observation": FakeGangObservation.model_json_schema(),
        "state": FakeGangState.model_json_schema(),
    }

@app.post("/mcp")
def mcp(body: Dict[str, Any] = {}):
    method = body.get("method", "")
    req_id = body.get("id", 1)
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": [
            {"name": "reset", "description": "Reset the environment",
             "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}, "seed": {"type": "integer"}}}},
            {"name": "step", "description": "Take an action", "inputSchema": FakeGangAction.model_json_schema()},
            {"name": "state", "description": "Get episode state", "inputSchema": {"type": "object", "properties": {}}},
        ]}}
    return {"jsonrpc": "2.0", "id": req_id, "result": {"name": "graphstrike", "version": "1.0.0", "protocolVersion": "2024-11-05"}}

@app.post("/baseline")
def baseline():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference import run_rule_based_episode
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_rule_based_episode(_env, task=task, seed=0)
    return {"scores": scores, "agent": "rule_based"}


# HF Spaces probes GET /web to detect if a web UI exists.
# Must return 200 BEFORE Gradio mount (Gradio's catch-all would shadow it).
@app.get("/web", response_class=HTMLResponse)
def web_view():
    return """<!DOCTYPE html>
<html><head><meta http-equiv="refresh" content="0;url=/"><title>GraphStrike</title></head>
<body><p>Loading <a href="/">GraphStrike</a>...</p></body></html>"""


# ---------------------------------------------------------------------------
# Gradio web interface — mounted at /
# ---------------------------------------------------------------------------

try:
    import gradio as gr

    def _fmt_obs(d: dict) -> str:
        lines = []
        lines.append(f"**Task:** {d.get('task','?')}  |  **Done:** {d.get('done',False)}  |  **Steps remaining:** {d.get('steps_remaining','?')}")
        if d.get('reward') is not None:
            lines.append(f"**Reward:** {d['reward']:.2f}")
        fl = d.get('flagged_ids', [])
        lines.append(f"**Flagged ({len(fl)}/10):** {fl}")
        su = d.get('suspect_ids', [])
        lines.append(f"**Suspects ({len(su)}):** {su}")
        lines.append(f"**Visible:** {len(d.get('visible_account_ids',[]))} IDs  |  **Inspected:** {len(d.get('inspected_ids',[]))} accounts")
        if d.get('evasion_triggered'):
            lines.append(f"**Evasion events:** {d.get('evasion_count',0)}")
        lines.append(f"**Message:** {d.get('message','')}")
        return "\n\n".join(lines)

    def _fmt_profiles(d: dict) -> str:
        accs = d.get("visible_accounts", [])
        if not accs:
            return "No accounts inspected yet. Use **INSPECT** to reveal profiles."
        rows = ["| Account | Status | Risk | Node | Beh | Graph | Hub | Photo | Bio | F.Nbrs |",
                "|---------|--------|------|------|-----|-------|-----|-------|-----|--------|"]
        for a in sorted(accs, key=lambda x: x.get("fake_risk_score",0), reverse=True)[:25]:
            rows.append(f"| {a.get('account_id','')} | {a.get('status','?')} | {a.get('fake_risk_score',0):.3f} "
                        f"| {a.get('node_risk',0):.2f} | {a.get('behavior_risk',0):.2f} | {a.get('graph_risk',0):.2f} "
                        f"| {a.get('hub_legitimacy_score',0):.2f} | {a.get('photo_reuse_score',0):.2f} "
                        f"| {a.get('bio_template_score',0):.2f} | {a.get('flagged_neighbor_count',0)} |")
        return "\n".join(rows)

    def gr_reset(task, seed):
        try:
            obs = _env.reset(task=task, seed=int(seed))
            d = obs.model_dump()
            return _fmt_obs(d), _fmt_profiles(d), json.dumps(d, indent=2, default=str)
        except Exception as e:
            return f"**Error:** {e}", "", "{}"

    def gr_step(action_type, account_id):
        try:
            acc = account_id.strip() if action_type != "submit" else None
            action = FakeGangAction(action_type=ActionType(action_type), account_id=acc)
            obs = _env.step(action)
            d = obs.model_dump()
            return _fmt_obs(d), _fmt_profiles(d), json.dumps(d, indent=2, default=str)
        except Exception as e:
            return f"**Error:** {e}", "", "{}"

    def gr_grader():
        if not _env._done:
            return "Episode not complete. Call SUBMIT first."
        return json.dumps({"score": _env._last_grader_score, "task": _env._task, "episode_id": _env._episode_id}, indent=2)

    def gr_baseline():
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import run_rule_based_episode
        scores = {t: run_rule_based_episode(_env, task=t, seed=0) for t in ["easy", "medium", "hard"]}
        return json.dumps({"scores": scores, "agent": "rule_based"}, indent=2)

    with gr.Blocks(title="GraphStrike") as demo:
        gr.Markdown(
            "# GraphStrike\n"
            "### Coordinated Fake Account Ring Detection — OpenEnv RL Environment\n\n"
            "Detect all 10 members of a coordinated fake account ring hidden in a social network.\n"
            "Use **INSPECT** to reveal profiles, **FLAG** to mark fakes, **SUBMIT** to end.\n\n"
            "`/reset` `/step` `/state` `/grader` `/baseline` `/tasks` `/health` — [Swagger](/docs)"
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 1. Start Episode")
                task_dd = gr.Dropdown(["easy","medium","hard"], value="easy", label="Task")
                seed_in = gr.Number(value=0, label="Seed", precision=0)
                reset_btn = gr.Button("Reset Episode", variant="primary", size="lg")
            with gr.Column():
                gr.Markdown("#### 2. Take Actions")
                action_dd = gr.Dropdown(["inspect","investigate_network","flag","unflag","submit"], value="inspect", label="Action Type")
                acc_in = gr.Textbox(label="Account ID", placeholder="e.g. acc_0012")
                step_btn = gr.Button("Step", variant="primary", size="lg")

        obs_md = gr.Markdown(value="*Click 'Reset Episode' to begin.*")
        with gr.Accordion("Account Profiles (sorted by risk)", open=True):
            prof_md = gr.Markdown(value="")
        with gr.Row():
            grader_btn = gr.Button("Get Grader Score")
            baseline_btn = gr.Button("Run Baseline (all 3 tasks)")
        result_box = gr.Textbox(label="Result", lines=5, interactive=False)
        with gr.Accordion("Raw JSON", open=False):
            raw_json = gr.Textbox(label="Raw JSON", lines=15, interactive=False)

        reset_btn.click(gr_reset, [task_dd, seed_in], [obs_md, prof_md, raw_json])
        step_btn.click(gr_step, [action_dd, acc_in], [obs_md, prof_md, raw_json])
        grader_btn.click(gr_grader, [], result_box)
        baseline_btn.click(gr_baseline, [], result_box)

    app = gr.mount_gradio_app(app, demo, path="/")
    print("[GraphStrike] Gradio UI mounted at /", flush=True)

except Exception as exc:
    print(f"[GraphStrike] Gradio unavailable: {exc}", flush=True)

    @app.get("/", response_class=HTMLResponse)
    def root_fallback():
        return "<html><body><h1>GraphStrike</h1><p>API-only mode. <a href='/docs'>Swagger</a></p></body></html>"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"[GraphStrike] Starting on port {port}", flush=True)
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, log_level="info", workers=1)

if __name__ == "__main__":
    main()
