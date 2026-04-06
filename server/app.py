"""FastAPI + Gradio server for the GraphStrike OpenEnv environment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
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

_env = FakeGangEnvironment()

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
    }

@app.get("/grader")
def grader():
    if not _env._done:
        raise HTTPException(status_code=400, detail="Episode not complete. Call SUBMIT first.")
    return {"score": _env._last_grader_score, "task": _env._task, "episode_id": _env._episode_id}

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

@app.post("/baseline")
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
            acc    = account_id.strip() if action_type != "submit" else None
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

    _HEADER_HTML = """
<div style="background:linear-gradient(135deg,#0c1a2e 0%,#0f2d4a 60%,#0a1f3c 100%);
            padding:22px 28px;border-radius:10px;border:1px solid #1e3a5f;margin-bottom:4px;">
  <h1 style="color:#e2e8f0;margin:0 0 4px;font-size:1.75em;letter-spacing:-0.3px;">GraphStrike</h1>
  <p style="color:#64748b;margin:0;font-size:0.95em;">
    Coordinated Fake Account Ring Detection &mdash; OpenEnv RL Environment
  </p>
</div>"""

    _FOOTER_HTML = """
<div style="text-align:center;padding:28px 0 10px;color:#334155;font-size:12.5px;
            border-top:1px solid #1e3a5f;margin-top:32px;">
  Built by team <strong style="color:#60a5fa;letter-spacing:0.3px;">computeXor</strong>
</div>"""

    with gr.Blocks(title="GraphStrike", css="""
        .gr-tab-item { font-size: 14px; }
        .gr-dataframe th { background: #0c2340 !important; color: #94a3b8 !important; font-weight: 600; }
    """) as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Tabs():

            # ══════════════ TAB 1: PLAYGROUND ══════════════
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
                            ["inspect","investigate_network","flag","unflag","submit"],
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
