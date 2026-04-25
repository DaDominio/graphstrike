"""FastAPI + Gradio server for the GraphStrike OpenEnv environment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, Optional

from models import FakeGangAction, FakeGangObservation, FakeGangState, ActionType
from environment import FakeGangEnvironment

# ---------------------------------------------------------------------------
# App + environment
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GraphStrike — OpenEnv",
    description="RL environment for detecting coordinated fake account rings in social networks.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve project images at /static/* and /img/* (NOT /assets/ — Gradio uses that path for its own JS/CSS)
_PROJECT_ROOT = Path(__file__).parent.parent
_ASSETS_DIR = _PROJECT_ROOT / "assets"
_IMAGES_DIR = _PROJECT_ROOT / "images"
if _ASSETS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_ASSETS_DIR)), name="static")
if _IMAGES_DIR.exists():
    app.mount("/img", StaticFiles(directory=str(_IMAGES_DIR)), name="img")

_env = FakeGangEnvironment()

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    platform: Optional[str] = None

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    done: bool
    reward: Optional[float]
    message: str
    # Round 2: top-level fields surfaced after submit so callers don't have to
    # parse the message string. Both are None for non-terminal steps.
    decision_package: Optional[Dict[str, Any]] = None
    grader_score: Optional[float] = None

# ---------------------------------------------------------------------------
# OpenEnv API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}

def _build_step_response(obs) -> StepResponse:
    decision = _env._decision_package if obs.done and _env._decision_package else None
    # Always source grader_score from the decision package when present so the
    # top-level field and decision_package["grader_score"] cannot disagree.
    if decision and "grader_score" in decision:
        grader = decision["grader_score"]
    elif obs.done:
        grader = _env._last_grader_score
    else:
        grader = None
    return StepResponse(
        observation=obs.model_dump(),
        done=obs.done,
        reward=obs.reward,
        message=obs.message,
        decision_package=decision,
        grader_score=grader,
    )

@app.post("/reset", response_model=StepResponse)
def reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()
    obs = _env.reset(task=req.task, seed=req.seed, episode_id=req.episode_id, platform=req.platform)
    return _build_step_response(obs)

@app.post("/step", response_model=StepResponse)
def step(action: FakeGangAction):
    obs = _env.step(action)
    return _build_step_response(obs)

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
            "action_type": [
                "inspect", "investigate_network", "flag", "unflag", "submit",
                "get_policy", "reverse_image_search", "analyze_bio", "check_ip",
            ],
            "account_id": "string (required for all actions except submit and get_policy)",
        },
        "platforms": ["Instagram", "Snapchat"],
        "score_range": [0.0, 1.0],
    }

@app.get("/grader")
def grader():
    if not _env._done:
        raise HTTPException(status_code=400, detail="Episode not complete. Call SUBMIT first.")
    return {"score": _env._last_grader_score, "task": _env._task, "episode_id": _env._episode_id}

@app.get("/decision")
def decision():
    """Round 2: moderation decision package built at SUBMIT."""
    if not _env._done:
        raise HTTPException(status_code=400, detail="Episode not complete. Call SUBMIT first.")
    return _env._decision_package

@app.get("/metadata")
def metadata():
    return {
        "name": "graphstrike", "version": "1.0.0", "author": "Pandago",
        "description": "RL environment for detecting coordinated fake account rings in social networks.",
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

@app.api_route("/baseline", methods=["GET", "POST"])
def baseline():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from inference import run_rule_based_episode
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_rule_based_episode(_env, task=task, seed=0)
    return {"scores": scores, "agent": "rule_based"}


# HF Spaces probes /web — redirect to root (must be on FastAPI before Gradio mount)
@app.get("/web", response_class=RedirectResponse)
def web_redirect():
    return RedirectResponse(url="/")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

import pandas as pd

# ── Benchmark data ───────────────────────────────────────────────────────────

BENCH_SEED0 = [
    # [Model, Params, Easy, Medium, Hard, Mean]  — sorted by Mean desc
    ["Llama 4 Scout 17B",  "17B",  0.960, 0.979, 0.976, 0.972],
    ["Ministral 3 8B",     "8B",   0.967, 0.964, 0.964, 0.965],
    ["DeepSeek V3.2",      "685B", 0.967, 0.960, 0.933, 0.953],
    ["Nemotron Super 3",   "49B",  0.930, 0.941, 0.964, 0.945],
    ["Rule-Based Baseline","—",    0.910, 0.906, 0.904, 0.907],
    ["Gemma 3 12B",        "12B",  0.900, 0.908, 0.908, 0.905],
]

BENCH_VARIANCE = [
    # [Model, Easy mean, Easy var, Med mean, Med var, Hard mean, Hard var]
    ["Llama 4 Scout 17B", 0.960, 0.000007, 0.979, 0.000001, 0.976, 0.000063],
    ["Nemotron Super 3",  0.957, 0.000,    0.957, 0.000,    0.645, 0.208],
    ["Ministral 3 8B",    0.958, 0.000,    0.645, 0.208,    0.623, 0.195],
    ["DeepSeek V3.2",     0.640, 0.205,    0.957, 0.000,    0.645, 0.208],
    ["Gemma 3 12B",       0.912, 0.000,    0.917, 0.000,    0.603, 0.182],
]

PROFILE_HEADERS = ["Account", "Status", "Risk", "Node", "Beh", "Graph", "Hub", "Photo", "Bio", "IP", "F.Nbrs"]

# Long-format DataFrame for BarPlot
_bench_long_rows = []
for _r in BENCH_SEED0:
    _bench_long_rows += [
        {"Model": _r[0], "Task": "Easy",   "Score": _r[2]},
        {"Model": _r[0], "Task": "Medium", "Score": _r[3]},
        {"Model": _r[0], "Task": "Hard",   "Score": _r[4]},
    ]
BENCH_LONG_DF = pd.DataFrame(_bench_long_rows)


# ── HTML table builders ──────────────────────────────────────────────────────

def _score_color(s: float) -> str:
    if s >= 0.960: return "#22c55e"
    if s >= 0.930: return "#86efac"
    if s >= 0.910: return "#facc15"
    return "#f97316"

def _var_color(v: float) -> str:
    if v < 0.001:  return "#22c55e"
    if v < 0.05:   return "#facc15"
    return "#f87171"

_TH = "padding:11px 16px;font-weight:600;white-space:nowrap;"
_TD = "padding:10px 16px;white-space:nowrap;"
_TABLE_WRAP = (
    "overflow-x:auto;border-radius:10px;border:1px solid #1e3a5f;"
    "font-family:'IBM Plex Mono',monospace;font-size:13.5px;"
)
_THEAD_BG = "background:#0c2340;"

def _leaderboard_html() -> str:
    header = (
        f"<thead><tr style='{_THEAD_BG}'>"
        f"<th style='{_TH}color:#64748b;'>#</th>"
        f"<th style='{_TH}color:#e2e8f0;text-align:left;'>Model</th>"
        f"<th style='{_TH}color:#94a3b8;text-align:center;'>Params</th>"
        f"<th style='{_TH}color:#4ade80;text-align:center;'>Easy</th>"
        f"<th style='{_TH}color:#facc15;text-align:center;'>Medium</th>"
        f"<th style='{_TH}color:#f87171;text-align:center;'>Hard</th>"
        f"<th style='{_TH}color:#c084fc;text-align:center;'>Mean</th>"
        f"</tr></thead>"
    )
    rows = ""
    for i, r in enumerate(BENCH_SEED0):
        bg  = "#162032" if i % 2 == 0 else "#0f172a"
        is_base = r[0] == "Rule-Based Baseline"
        name_cell = (
            f"{r[0]} <span style='color:#64748b;font-size:11px;'>(baseline)</span>"
            if is_base else r[0]
        )
        name_color = "#94a3b8" if is_base else "#e2e8f0"
        rows += (
            f"<tr style='background:{bg};'>"
            f"<td style='{_TD}color:#475569;text-align:center;'>{i+1}</td>"
            f"<td style='{_TD}color:{name_color};'>{name_cell}</td>"
            f"<td style='{_TD}color:#64748b;text-align:center;'>{r[1]}</td>"
            + "".join(
                f"<td style='{_TD}color:{_score_color(r[j])};font-weight:700;"
                f"text-align:center;'>{r[j]:.3f}</td>"
                for j in (2, 3, 4)
            )
            + f"<td style='{_TD}color:{_score_color(r[5])};font-weight:800;"
              f"font-size:14px;text-align:center;'>{r[5]:.3f}</td>"
            f"</tr>"
        )
    return f"<div style='{_TABLE_WRAP}'><table style='width:100%;border-collapse:collapse;'>{header}<tbody>{rows}</tbody></table></div>"


def _variance_html() -> str:
    header = (
        f"<thead><tr style='{_THEAD_BG}'>"
        f"<th style='{_TH}color:#e2e8f0;text-align:left;'>Model</th>"
        f"<th style='{_TH}color:#4ade80;text-align:center;'>Easy — mean / var</th>"
        f"<th style='{_TH}color:#facc15;text-align:center;'>Medium — mean / var</th>"
        f"<th style='{_TH}color:#f87171;text-align:center;'>Hard — mean / var</th>"
        f"</tr></thead>"
    )
    rows = ""
    for i, r in enumerate(BENCH_VARIANCE):
        bg = "#162032" if i % 2 == 0 else "#0f172a"
        def cell(mean, var):
            return (
                f"<td style='{_TD}text-align:center;'>"
                f"<span style='color:#e2e8f0;font-weight:600;'>{mean:.3f}</span>"
                f" <span style='color:{_var_color(var)};font-size:11px;'>/ {var:.1e}</span>"
                f"</td>"
            )
        rows += (
            f"<tr style='background:{bg};'>"
            f"<td style='{_TD}color:#e2e8f0;font-weight:500;'>{r[0]}</td>"
            + cell(r[1], r[2]) + cell(r[3], r[4]) + cell(r[5], r[6])
            + "</tr>"
        )
    return f"<div style='{_TABLE_WRAP};margin-top:20px;'><table style='width:100%;border-collapse:collapse;'>{header}<tbody>{rows}</tbody></table></div>"


def _baseline_html() -> str:
    rows_data = [
        ("Easy",   0.9100, "100%", "#4ade80"),
        ("Medium", 0.9060, "84%",  "#facc15"),
        ("Hard",   0.9038, "52%",  "#f87171"),
    ]
    header = (
        f"<thead><tr style='{_THEAD_BG}'>"
        f"<th style='{_TH}color:#e2e8f0;'>Task</th>"
        f"<th style='{_TH}color:#e2e8f0;text-align:center;'>Score (seed=0)</th>"
        f"<th style='{_TH}color:#e2e8f0;text-align:center;'>Win Rate (50 seeds)</th>"
        f"</tr></thead>"
    )
    rows = ""
    for i, (task, score, wr, col) in enumerate(rows_data):
        bg = "#162032" if i % 2 == 0 else "#0f172a"
        rows += (
            f"<tr style='background:{bg};'>"
            f"<td style='{_TD}color:{col};font-weight:600;'>{task}</td>"
            f"<td style='{_TD}color:#e2e8f0;font-weight:700;text-align:center;'>{score:.4f}</td>"
            f"<td style='{_TD}color:{col};font-weight:600;text-align:center;'>{wr}</td>"
            f"</tr>"
        )
    return f"<div style='{_TABLE_WRAP};margin-top:4px;'><table style='width:100%;border-collapse:collapse;'>{header}<tbody>{rows}</tbody></table></div>"


try:
    import gradio as gr

    # ── Observation / profile helpers ─────────────────────────────────────────

    def _fmt_obs(d: dict) -> str:
        lines = []
        task  = d.get('task', '?').upper()
        done  = d.get('done', False)
        steps = d.get('steps_remaining', '?')
        state_label = "Done" if done else "In Progress"
        lines.append(f"### Task: **{task}**  |  Steps remaining: **{steps}**  |  {state_label}")
        if d.get('reward') is not None:
            lines.append(f"**Final Reward:** `{d['reward']:.2f}`")
        fl = d.get('flagged_ids', [])
        lines.append(f"**Flagged ({len(fl)}/10):** " + (" ".join(f"`{f}`" for f in fl) if fl else "*none*"))
        su  = d.get('suspect_ids', [])
        ins = set(d.get('inspected_ids', []))
        uninspected_sus = [s for s in su if s not in ins]
        if uninspected_sus:
            lines.append(f"**Suspects — uninspected ({len(uninspected_sus)}):** " + " ".join(f"`{s}`" for s in uninspected_sus))
        lines.append(f"**Visible:** {len(d.get('visible_account_ids',[]))} IDs  |  **Inspected:** {len(d.get('inspected_ids',[]))} accounts")
        if d.get('evasion_triggered'):
            lines.append(f"**Evasion events fired:** {d.get('evasion_count', 0)}")
        lines.append(f"\n> {d.get('message', '')}")
        return "\n\n".join(lines)

    def _profile_rows(d: dict) -> list:
        accs = d.get("visible_accounts", [])
        if not accs:
            return []
        STATUS_MAP = {
            "confirmed_fake": "confirmed_fake [flagged]",
            "suspect":        "suspect",
            "normal":         "normal",
        }
        rows = []
        for a in sorted(accs, key=lambda x: x.get("fake_risk_score", 0), reverse=True)[:40]:
            rows.append([
                a.get("account_id", ""),
                STATUS_MAP.get(a.get("status", ""), a.get("status", "")),
                round(a.get("fake_risk_score", 0), 3),
                round(a.get("node_risk", 0), 3),
                round(a.get("behavior_risk", 0), 3),
                round(a.get("graph_risk", 0), 3),
                round(a.get("hub_legitimacy_score", 0), 3),
                round(a.get("photo_reuse_score", 0), 3),
                round(a.get("bio_template_score", 0), 3),
                a.get("shared_ip_count", 0),
                a.get("flagged_neighbor_count", 0),
            ])
        return rows

    def _fmt_visible_ids(d: dict) -> str:
        ins      = set(d.get('inspected_ids', []))
        suspects = set(d.get('suspect_ids', []))
        flagged  = set(d.get('flagged_ids', []))
        visible  = d.get('visible_account_ids', [])
        if not visible:
            return "*No visible accounts yet.*"
        parts = []
        for vid in visible:
            if vid in flagged:
                parts.append(f"**[F]** `{vid}`")
            elif vid in suspects and vid not in ins:
                parts.append(f"**[S]** `{vid}`")
            elif vid in ins:
                parts.append(f"`{vid}`")
            else:
                parts.append(f"`{vid}`")
        return "  ".join(parts)

    # ── Playground callbacks ──────────────────────────────────────────────────

    def gr_reset(task, seed):
        try:
            obs = _env.reset(task=task, seed=int(seed))
            d   = obs.model_dump()
            return _fmt_obs(d), _profile_rows(d), _fmt_visible_ids(d), json.dumps(d, indent=2, default=str)
        except Exception as e:
            return f"**Error:** {e}", [], "", "{}"

    def gr_step(action_type, account_id):
        try:
            acc    = account_id.strip() if action_type not in ("submit", "get_policy") else None
            action = FakeGangAction(action_type=ActionType(action_type), account_id=acc)
            obs    = _env.step(action)
            d      = obs.model_dump()
            return _fmt_obs(d), _profile_rows(d), _fmt_visible_ids(d), json.dumps(d, indent=2, default=str)
        except Exception as e:
            return f"**Error:** {e}", [], "", "{}"

    def gr_grader():
        if not _env._done:
            return "Episode not complete — call SUBMIT first."
        return (
            f"**Score:** `{_env._last_grader_score:.4f}`  |  "
            f"**Task:** {_env._task}  |  "
            f"**Episode:** `{_env._episode_id}`"
        )

    def gr_baseline():
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference import run_rule_based_episode
        scores = {t: run_rule_based_episode(_env, task=t, seed=0) for t in ["easy", "medium", "hard"]}
        mean = sum(scores.values()) / 3
        return (
            f"**Baseline (rule-based, seed=0)**\n\n"
            f"Easy: `{scores['easy']:.4f}`  |  Medium: `{scores['medium']:.4f}`  |  "
            f"Hard: `{scores['hard']:.4f}`  |  Mean: `{mean:.4f}`"
        )

    # ── Build Gradio UI ───────────────────────────────────────────────────────

    # ── README content (rendered as styled HTML) ─────────────────────────────

    _README_HTML = """
<style>
.gs-readme { font-family: 'Inter', system-ui, sans-serif; color: #cbd5e1; line-height: 1.7; max-width: 960px; margin: 0 auto; padding: 8px 4px 32px; }
.gs-readme h2 { color: #e2e8f0; font-size: 1.12em; font-weight: 700; border-bottom: 1px solid #1e3a5f; padding-bottom: 8px; margin: 32px 0 14px; letter-spacing: -0.2px; }
.gs-readme h3 { color: #7dd3fc; font-size: 0.97em; font-weight: 600; margin: 20px 0 8px; }
.gs-readme p  { margin: 0 0 10px; font-size: 0.92em; }
.gs-readme code { background: #0c2340; color: #7dd3fc; padding: 2px 7px; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.84em; }
.gs-readme pre { background: #0a1628; border: 1px solid #1e3a5f; border-radius: 8px; padding: 14px 18px; overflow-x: auto; margin: 10px 0 16px; }
.gs-readme pre code { background: none; padding: 0; color: #93c5fd; font-size: 0.82em; }
.gs-table { width: 100%; border-collapse: collapse; margin: 10px 0 18px; font-size: 0.86em; }
.gs-table th { background: #0c2340; color: #94a3b8; font-weight: 600; padding: 9px 14px; text-align: left; border-bottom: 1px solid #1e3a5f; }
.gs-table td { padding: 8px 14px; border-bottom: 1px solid #0f1e30; color: #cbd5e1; }
.gs-table tr:nth-child(even) td { background: #060e1a; }
.gs-badge { display:inline-block; padding: 2px 9px; border-radius: 4px; font-size: 0.78em; font-weight: 700; }
.gs-badge-easy   { background:#052e16; color:#4ade80; border:1px solid #166534; }
.gs-badge-medium { background:#2d1f00; color:#facc15; border:1px solid #92400e; }
.gs-badge-hard   { background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d; }
.gs-card { background: #0a1628; border: 1px solid #1e3a5f; border-radius: 10px; padding: 16px 20px; margin: 10px 0; }
.gs-card h3 { margin-top: 0; }
.gs-formula { background: #050d18; border-left: 3px solid #3b82f6; padding: 12px 18px; border-radius: 0 8px 8px 0; margin: 12px 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.83em; color: #93c5fd; white-space: pre; overflow-x: auto; }
.gs-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 14px 0; }
.gs-stat { background: #0a1628; border: 1px solid #1e3a5f; border-radius: 8px; padding: 14px 16px; text-align: center; }
.gs-stat-val { font-size: 1.7em; font-weight: 800; color: #38bdf8; font-family: 'IBM Plex Mono', monospace; display: block; }
.gs-stat-lbl { font-size: 0.77em; color: #64748b; margin-top: 4px; display: block; }
.gs-img { width: 100%; border-radius: 10px; border: 1px solid #1e3a5f; margin: 14px 0; display: block; background: #0a1628; }
.gs-img-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 14px 0; }
.gs-img-caption { font-size: 0.78em; color: #475569; text-align: center; margin-top: -8px; margin-bottom: 12px; font-style: italic; }
.gs-divider { border: none; border-top: 1px solid #0f1e30; margin: 28px 0; }
</style>

<div class="gs-readme">

<!-- OVERVIEW -->
<div class="gs-card" style="border-color:#2563eb;margin-bottom:20px;border-width:1px 1px 1px 3px;">
  <h3 style="color:#7dd3fc;font-size:1.05em;">What is GraphStrike?</h3>
  <p>An <strong style="color:#e2e8f0;">OpenEnv-compatible</strong> reinforcement learning environment where an LLM agent
  must identify all 10 members of a coordinated fake account ring hidden inside a synthetic social network.
  The agent learns via <strong>Reflexion</strong> and a <strong>dynamic hybrid rule/LLM policy</strong> — no gradient
  updates, no fine-tuning required.</p>
  <p style="margin:0;">Submitted to the <strong style="color:#e2e8f0;">OpenEnv Hackathon × SCALER School of Technology</strong>.
  Judges deploy this container, run their own LLM agent against it, and score on task quality, environment design,
  code quality, creativity, and domain quality.</p>
</div>

<!-- KEY STATS -->
<div class="gs-grid">
  <div class="gs-stat"><span class="gs-stat-val">10</span><span class="gs-stat-lbl">Gang members to find per episode</span></div>
  <div class="gs-stat"><span class="gs-stat-val">3</span><span class="gs-stat-lbl">Difficulty tiers (easy / medium / hard)</span></div>
  <div class="gs-stat"><span class="gs-stat-val">150</span><span class="gs-stat-lbl">Pre-generated episodes (50 per task)</span></div>
  <div class="gs-stat"><span class="gs-stat-val">24</span><span class="gs-stat-lbl">Automated validator checks</span></div>
</div>

<!-- SYSTEM ARCHITECTURE -->
<h2>System Architecture</h2>
<img src="/static/sys arch.png" class="gs-img" alt="System Architecture" onerror="this.style.display='none'">
<p class="gs-img-caption">End-to-end pipeline: episode generation → environment server → hybrid agent → reflexion memory</p>

<!-- DIFFICULTY -->
<h2>Task Difficulty Tiers</h2>
<table class="gs-table">
  <tr><th>Task</th><th>Network Size</th><th>Gang</th><th>Decoys</th><th>Max Steps</th><th>Win Condition</th><th>Baseline Score</th></tr>
  <tr><td><span class="gs-badge gs-badge-easy">Easy</span></td><td>50 accounts</td><td>10</td><td>0</td><td>30</td><td>Recall ≥ 0.8, Precision ≥ 0.7</td><td>0.910</td></tr>
  <tr><td><span class="gs-badge gs-badge-medium">Medium</span></td><td>200 accounts</td><td>10</td><td>20</td><td>50</td><td>Recall ≥ 0.8, Precision ≥ 0.7</td><td>0.906</td></tr>
  <tr><td><span class="gs-badge gs-badge-hard">Hard</span></td><td>1000 accounts</td><td>10</td><td>50</td><td>80</td><td>Recall ≥ 0.9, Precision ≥ 0.8</td><td>0.904</td></tr>
</table>
<p style="font-size:0.84em;color:#64748b;margin-top:-8px;">Hard mode fires 4 evasion events (steps 15, 30, 45, 60) that drop intra-gang follow edges mid-investigation, destroying graph signals.</p>

<hr class="gs-divider">

<!-- DETECTION SIGNALS -->
<h2>Detection Signal Hierarchy</h2>
<img src="/static/gs.png" class="gs-img" alt="Signal Hierarchy" onerror="this.style.display='none'">
<p class="gs-img-caption">Node signals (offline) → Behavioral signals (temporal/device) → Graph signals (live at INSPECT) → False-positive control via hub legitimacy</p>

<h3>Node Signals (pre-computed offline)</h3>
<table class="gs-table">
  <tr><th>Feature</th><th>Fake Range</th><th>Real Range</th><th>What it measures</th></tr>
  <tr><td><code>photo_reuse_score</code></td><td>0.30 – 0.95</td><td>0.00 – 0.15</td><td>Stolen celebrity photos via pHash fingerprint matching</td></tr>
  <tr><td><code>bio_template_score</code></td><td>0.20 – 0.90</td><td>0.00 – 0.12</td><td>Cosine similarity to known fake bio templates</td></tr>
  <tr><td><code>comment_repeat_score</code></td><td>0.60 – 0.90</td><td>0.00 – 0.08</td><td>Fraction of copy-pasted spam comments across accounts</td></tr>
</table>

<h3>Behavioral Signals (temporal + device)</h3>
<table class="gs-table">
  <tr><th>Feature</th><th>Fake Pattern</th></tr>
  <tr><td><code>avg_post_hour</code></td><td>All 10 gang members post within ±0.5h of each other (coordinated scheduling)</td></tr>
  <tr><td><code>account_age_days</code></td><td>Created same week — base_age ± 7 days</td></tr>
  <tr><td><code>shared_ip_count</code></td><td>= 9 for all gang members (one IP subnet per episode, unique seed)</td></tr>
</table>

<h3>Graph Signals (computed live at INSPECT)</h3>
<table class="gs-table">
  <tr><th>Feature</th><th>Fake Pattern</th></tr>
  <tr><td><code>mutual_follow_rate</code></td><td>0.6 – 0.9 (dense intra-gang mutual follows)</td></tr>
  <tr><td><code>flagged_neighbor_count</code></td><td>Grows as investigation proceeds — strongest late-game signal</td></tr>
  <tr><td><code>avg_neighbor_photo_reuse</code></td><td>High when cluster shares stolen content</td></tr>
</table>

<hr class="gs-divider">

<!-- EPISODE FLOW -->
<h2>Episode Lifecycle &amp; Action Mechanics</h2>
<img src="/static/episode.png" class="gs-img" alt="Episode Flow" onerror="this.style.display='none'">
<p class="gs-img-caption">Episode flow: reset → inspect/flag/investigate loop → dual SUSPECT cascade → submit → grader score</p>

<h3>Action Space</h3>
<table class="gs-table">
  <tr><th>Action</th><th>Step Cost</th><th>Effect</th></tr>
  <tr><td><code>INSPECT acc_XXXX</code></td><td>1 step</td><td>Reveals full AccountProfile + follow list; adds 1-hop neighbors to visible set</td></tr>
  <tr><td><code>INVESTIGATE_NETWORK acc_XXXX</code></td><td>2 steps</td><td>Bidirectional 2-hop expansion (outgoing + incoming edges); re-cascades SUSPECT</td></tr>
  <tr><td><code>FLAG acc_XXXX</code></td><td>FREE</td><td>Marks as fake; triggers dual SUSPECT cascade (follow-graph + IP cluster)</td></tr>
  <tr><td><code>UNFLAG acc_XXXX</code></td><td>FREE</td><td>Removes flag; clears CONFIRMED_FAKE status</td></tr>
  <tr><td><code>SUBMIT</code></td><td>FREE</td><td>Ends episode; triggers grader scoring</td></tr>
</table>

<h3>Dual SUSPECT Cascade (triggered by FLAG)</h3>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:10px 0;">
  <div class="gs-card">
    <h3 style="color:#4ade80;margin-top:0;">Cascade 1 — Follow-Graph</h3>
    <p style="margin:0;font-size:0.88em;">Every account the flagged member <em>follows</em> (<code>_live_edges</code>) becomes SUSPECT if visible and NORMAL. Gang follow density is 0.70+ so this is high-precision.</p>
  </div>
  <div class="gs-card">
    <h3 style="color:#facc15;margin-top:0;">Cascade 2 — IP Cluster</h3>
    <p style="margin:0;font-size:0.88em;">Every visible account sharing the same <code>ip_cluster_id</code> becomes SUSPECT. Gang shares <code>ip_gang_&lt;seed&gt;</code>; real accounts have unique IPs. <strong>Zero false positives.</strong></p>
  </div>
</div>

<hr class="gs-divider">

<!-- RISK SCORING -->
<h2>Risk Scoring Mathematics</h2>
<img src="/img/big.png" class="gs-img" alt="Risk Scoring Overview" onerror="this.style.display='none'">
<p class="gs-img-caption">All scoring functions are stateless and deterministic — called inside _build_profile() at every INSPECT</p>

<div class="gs-img-pair">
  <div>
    <img src="/static/formulas-1.png" class="gs-img" alt="Risk Formulas Part 1" onerror="this.style.display='none'">
    <p class="gs-img-caption">Node risk, Behavior risk, Graph risk components</p>
  </div>
  <div>
    <img src="/static/formulas-2.png" class="gs-img" alt="Risk Formulas Part 2" onerror="this.style.display='none'">
    <p class="gs-img-caption">Hub legitimacy, Composite fake_risk_score formula</p>
  </div>
</div>

<div class="gs-formula">fake_risk = clip(
  0.30 × node_risk        ← content signals (photo reuse, bio templates)
+ 0.25 × behavior_risk   ← temporal + age clustering
+ 0.45 × graph_risk      ← structural coordination (highest weight — hardest to fake)
− 0.25 × hub_legitimacy, ← subtractive: celebrities score ≈ 0 before clip
0.0, 1.0)</div>

<h3>Grader Score Formula</h3>
<div class="gs-formula">recall    = tp / 10
precision = tp / max(tp + fp, 1)
efficiency = max(0, (max_steps − steps_used) / max_steps)

if recall ≥ 0.8 and precision ≥ 0.7:
    score = 0.55 + 0.20×recall + 0.15×precision + 0.10×efficiency
else:
    score = 0.30×recall + 0.10×precision

# Maximum possible: 1.00  |  Win threshold: ~0.815</div>

<hr class="gs-divider">

<!-- REFLEXION -->
<h2>Reflexion Learning</h2>
<img src="/static/reflexion.png" class="gs-img" alt="Reflexion Learning Loop" onerror="this.style.display='none'">
<p class="gs-img-caption">Post-episode lessons injected into every future prompt — learning without weight updates</p>

<p>The LLM (Qwen3-80B via AWS Bedrock) cannot be fine-tuned — it is a black-box API.
Instead, a separate Qwen3 call generates a 2–3 sentence lesson after each episode.
The best winning trajectory is stored as a few-shot example injected into all future prompts.</p>

<pre><code>Episode N:
  LLM acts using: system_prompt + reflections[last 4] + best_trajectory
  Episode ends → WIN or LOSS
  LOSS → generate_reflection(action_log, outcome) → lesson stored
  WIN  → save trajectory if better reward + generate_success_reflection

Episode N+1:
  last 4 reflections + best win trajectory injected into prompt
  → LLM has learned from its past without any weight updates</code></pre>

<hr class="gs-divider">

<!-- HYBRID POLICY -->
<h2>Hybrid Policy — The Novel Contribution</h2>
<img src="/static/hybrid.png" class="gs-img" alt="Hybrid Policy Architecture" onerror="this.style.display='none'">
<p class="gs-img-caption">Dynamic alpha-weighted blend: rules dominate early, LLM earns trust through wins and reflections</p>

<p>A <strong>dynamic α-weighted blend</strong> of a deterministic rule engine and the LLM. α represents trust in the LLM —
starts at 0.20 (rules dominate), climbs as the LLM wins consistently and accumulates reflections, capped per task
to prevent the LLM from overriding correct high-confidence rule decisions.</p>

<div class="gs-formula">reflection_factor = min(1.0, n_reflections / 4.0)
raw   = 0.20 + reflection_factor × (0.80 × recent_win_rate + 0.12)
alpha = clamp(raw, 0.20, task_cap)

Per-task caps:  easy → 0.50  |  medium → 0.70  |  hard → 0.85</div>

<img src="/img/plot.png" class="gs-img" alt="Alpha progression over training" onerror="this.style.display='none'">
<p class="gs-img-caption">Alpha progression: rule-dominated early training → LLM earns authority through wins</p>

<h3>Rule Confidence Levels</h3>
<table class="gs-table">
  <tr><th>Situation</th><th>Rule Action</th><th>Confidence</th></tr>
  <tr><td>Steps remaining = 0</td><td>SUBMIT</td><td>1.00</td></tr>
  <tr><td>Uninspected SUSPECT accounts exist</td><td>INSPECT suspects[0]</td><td>0.95</td></tr>
  <tr><td><code>fake_risk ≥ 0.85</code></td><td>FLAG that account</td><td>0.95</td></tr>
  <tr><td><code>fake_risk</code> in [threshold, 0.85)</td><td>FLAG that account</td><td>0.70 – 0.94</td></tr>
  <tr><td>10 flags placed</td><td>SUBMIT</td><td>0.85</td></tr>
  <tr><td>Steps remaining ≤ 3</td><td>SUBMIT</td><td>0.90</td></tr>
  <tr><td>Uninspected accounts available</td><td>INSPECT top candidate</td><td>0.30</td></tr>
</table>
<p style="font-size:0.85em;color:#64748b;">When <code>rule_confidence ≥ alpha</code> the rule engine overrides. At easy cap (0.50), the LLM controls only exploratory INSPECT decisions. At hard cap (0.85), the LLM controls most decisions except forced submits and suspect cascade.</p>

</div>
"""

    _HEADER_HTML = """
<style>
.gr-dataframe th { background:#0c2340!important;color:#94a3b8!important;font-weight:700!important;font-size:12px!important;padding:10px 12px!important;border-bottom:1px solid #1e3a5f!important; }
.gr-dataframe td { font-size:12.5px!important;padding:8px 12px!important; }
</style>
<div style="background:linear-gradient(135deg,#050d1a 0%,#0b1f3a 50%,#060f1e 100%);
            padding:24px 32px 20px;border-radius:12px;
            border:1px solid #1e3a5f;margin-bottom:2px;
            box-shadow:0 4px 24px rgba(0,0,0,0.5);">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
    <div>
      <h1 style="color:#e2e8f0;margin:0;font-size:1.9em;font-weight:800;letter-spacing:-0.5px;
                  font-family:'Inter',system-ui,sans-serif;">GraphStrike</h1>
      <p style="color:#475569;margin:3px 0 0;font-size:0.88em;letter-spacing:0.3px;font-family:'IBM Plex Mono',monospace;">
        COORDINATED FAKE ACCOUNT RING DETECTION &mdash; OPENENV RL ENVIRONMENT
      </p>
    </div>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px;">
    <span style="background:#052e16;color:#4ade80;padding:3px 10px;border-radius:20px;font-size:0.78em;font-weight:600;border:1px solid #166534;">OpenEnv Hackathon</span>
    <span style="background:#0c1a2e;color:#7dd3fc;padding:3px 10px;border-radius:20px;font-size:0.78em;font-weight:600;border:1px solid #1e40af;">Reinforcement Learning</span>
    <span style="background:#1c0533;color:#c084fc;padding:3px 10px;border-radius:20px;font-size:0.78em;font-weight:600;border:1px solid #6b21a8;">Hybrid Policy</span>
    <span style="background:#2d1f00;color:#fbbf24;padding:3px 10px;border-radius:20px;font-size:0.78em;font-weight:600;border:1px solid #92400e;">Reflexion Learning</span>
    <span style="background:#1a0505;color:#f87171;padding:3px 10px;border-radius:20px;font-size:0.78em;font-weight:600;border:1px solid #7f1d1d;">Fraud Detection</span>
  </div>
</div>"""

    _FOOTER_HTML = """
<div style="text-align:center;padding:24px 0 8px;color:#1e3a5f;font-size:12px;
            border-top:1px solid #0f1e30;margin-top:28px;font-family:'IBM Plex Mono',monospace;">
  GraphStrike &mdash; OpenEnv Hackathon &times; SCALER School of Technology &nbsp;|&nbsp;
  <a href="/docs" style="color:#334155;text-decoration:none;">API Docs</a>
</div>"""

    with gr.Blocks(title="GraphStrike") as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Tabs():

            # ══════════════ TAB 1: README ══════════════
            with gr.Tab("Overview"):
                gr.HTML(_README_HTML)

            # ══════════════ TAB 2: PLAYGROUND ══════════════
            with gr.Tab("Playground"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("**1 — Episode**")
                        task_dd   = gr.Dropdown(["easy","medium","hard"], value="easy", label="Task")
                        seed_in   = gr.Number(value=0, label="Seed", precision=0)
                        reset_btn = gr.Button("Reset", variant="primary")

                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("**2 — Action**")
                        action_dd = gr.Dropdown(
                            ["inspect","investigate_network","flag","unflag","submit",
                             "get_policy","reverse_image_search","analyze_bio","check_ip"],
                            value="inspect", label="Action")
                        acc_in   = gr.Textbox(label="Account ID", placeholder="acc_0012")
                        step_btn = gr.Button("Step", variant="primary")

                    with gr.Column(scale=1, min_width=180):
                        gr.Markdown("**3 — Score**")
                        gr.Markdown("<br>", container=False)
                        grader_btn   = gr.Button("Grader Score",   size="sm")
                        baseline_btn = gr.Button("Baseline Agent", size="sm")
                        gr.Button("API Docs (Swagger)", size="sm", link="/docs", link_target="_blank")

                obs_md = gr.Markdown(value="*Reset an episode to begin.*")

                gr.Markdown("**Account Profiles** — sorted by fake risk score (highest first)")
                prof_table = gr.Dataframe(
                    headers=PROFILE_HEADERS,
                    datatype=["str","str","number","number","number","number",
                               "number","number","number","number","number"],
                    value=[],
                    interactive=False,
                    wrap=False,
                    column_widths=["110px","160px","70px","70px","70px",
                                   "70px","70px","70px","70px","55px","70px"],
                )

                result_md = gr.Markdown(value="")

                with gr.Accordion("All Visible IDs", open=False):
                    vis_md = gr.Markdown(value="")
                with gr.Accordion("Raw JSON", open=False):
                    raw_json = gr.Textbox(lines=20, interactive=False)

                reset_btn.click(gr_reset,      [task_dd, seed_in],  [obs_md, prof_table, vis_md, raw_json])
                step_btn.click( gr_step,       [action_dd, acc_in], [obs_md, prof_table, vis_md, raw_json])
                grader_btn.click(gr_grader,    [],                   result_md)
                baseline_btn.click(gr_baseline,[],                   result_md)

            # ══════════════ TAB 2: BENCHMARKS ══════════════
            with gr.Tab("Benchmarks"):
                gr.Markdown(
                    "### LLM Agent Evaluation — GraphStrike Environment\n"
                    "Agents evaluated with identical system prompts and structured inference. "
                    "Grader score range: **0.0 – 1.0** (win threshold ≥ 0.815). "
                    "Score colours: "
                    "<span style='color:#22c55e'>■</span> ≥0.960 &nbsp; "
                    "<span style='color:#86efac'>■</span> ≥0.930 &nbsp; "
                    "<span style='color:#facc15'>■</span> ≥0.910 &nbsp; "
                    "<span style='color:#f97316'>■</span> below",
                    sanitize_html=False,
                )

                gr.Markdown("#### Leaderboard — Single Seed (seed=0)")
                gr.HTML(_leaderboard_html())

                gr.Markdown("#### Score Distribution by Task")
                gr.BarPlot(
                    value=BENCH_LONG_DF,
                    x="Model", y="Score", color="Task",
                    title="Agent Scores by Task (seed=0)",
                    color_map={"Easy": "#4ade80", "Medium": "#facc15", "Hard": "#f87171"},
                    y_lim=[0.50, 1.0],
                    x_label_angle=-25,
                    height=340,
                )

                gr.Markdown(
                    "#### Stability — 3-Seed Variance Check (seeds 0, 1, 2)\n"
                    "Variance colour: "
                    "<span style='color:#22c55e'>■</span> stable (&lt;0.001) &nbsp; "
                    "<span style='color:#facc15'>■</span> moderate &nbsp; "
                    "<span style='color:#f87171'>■</span> high",
                    sanitize_html=False,
                )
                gr.HTML(_variance_html())

                gr.Markdown("#### Rule-Based Baseline (no LLM, deterministic)")
                gr.HTML(_baseline_html())

                gr.Markdown(
                    "#### Key Observations\n"
                    "- Hard task is the real differentiator — evasion events destroy graph signals "
                    "mid-investigation, requiring adaptive reasoning beyond memorised patterns.\n"
                    "- Llama 4 Scout 17B achieves the lowest variance on hard (6e-5), "
                    "outperforming models with 40× more parameters.\n"
                    "- The rule-based baseline is competitive at mean 0.907, confirming "
                    "the environment's signal quality. LLM value is in evasion adaptation.\n"
                    "- All frontier models exceed 0.93 on easy/medium — cascade mechanics "
                    "are learnable from the structured observation format."
                )

        gr.HTML(_FOOTER_HTML)

    app = gr.mount_gradio_app(app, demo, path="/")
    print("[GraphStrike] Gradio UI mounted at /", flush=True)

except Exception as exc:
    import traceback
    print(f"[GraphStrike] Gradio unavailable: {exc}", flush=True)
    traceback.print_exc()

    @app.get("/", response_class=HTMLResponse)
    def root_fallback():
        return "<html><body><h1>GraphStrike</h1><p>API mode. <a href='/docs'>Swagger</a></p></body></html>"

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
