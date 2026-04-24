# GraphStrike — Single Source of Truth

> Consolidates: `FINAL_SUMMARY.md`, `IMPLEMENTATION_COMPLETE.md`, `IMPLEMENTATION_STATUS.md`, `INFERENCE_UPDATE.md`, `PIPELINE.md`, `QUICKSTART.md`, `ROUND2_COMPLETE.md`, `ROUND2_STATUS.md`, `ROUND2_TRAINING_READY.md`, `server/ROUND2_FINAL_STATUS.md`, and the top-level `ROUND2_ARCHITECTURE.md` / `ROUND2_IMPLEMENTATION_PLAN.md` / `ROUND2_QUICK_REFERENCE.md` / `OpenEnv-Complete.md`.
>
> The HF-Space `README.md` is kept (it contains the YAML frontmatter Spaces needs). The per-directory `dashboard/README.md` describes only the local dashboard and stays with it.

---

## 1. What GraphStrike is

An OpenEnv-compatible RL environment. An LLM agent must identify the 10 members of a coordinated fake-account ring hidden inside a synthetic social network. Round 2 adds **platform-adaptive trust-and-safety**: each episode belongs to Instagram or Snapchat; detection threshold and reward shape are compiled from real platform policy text; key account signals are hidden at reset and revealed only through explicit tool-like actions.

Learning is via Reflexion + episodic memory (no weight updates). The environment is trainable with HF TRL through repeated rollouts and numeric rewards.

---

## 2. Round 2 deltas (what changed vs Round 1)

| Area | Round 1 | Round 2 |
|---|---|---|
| Platform | — | `platform` field per episode (Instagram/Snapchat, by seed parity) |
| Policy | hardcoded thresholds | `PlatformPolicy` compiled dynamically from transparency reports (Bayesian θ*) |
| Signals | all visible at INSPECT | `photo_reuse_score`, `bio_template_score`, `ip_cluster_id` start hidden (0.0 / "") |
| Actions | `inspect`, `investigate_network`, `flag`, `unflag`, `submit` | + `get_policy`, `reverse_image_search`, `analyze_bio`, `check_ip` |
| Reward | fixed FP penalty 0.5 | platform FP penalty (IG 0.1, Snap 0.01) + platform bonus (IG precision ≥ 0.95 → +2.0, Snap recall ≥ 0.95 → +2.0) |
| Grader | fixed threshold 0.35 | accepts platform threshold + `threshold_factor` bonus |
| Risk weights | fixed (node 0.30 / beh 0.25 / graph 0.45) | `compute_weighted_fake_risk(primary_signal)` boosts the platform's primary signal |

Platform assignment is deterministic: `seed % 2 == 0 → Instagram`, else Snapchat. This is the only platform input — it comes from the training driver, not the LLM.

---

## 3. Architecture

```
Episode reset
  ├── generator.generate_episode(task, seed)   → network + hidden_signals
  ├── policy_compiler.get_policy(platform)     → PlatformPolicy (cached)
  └── revealed_signals = {photo_reuse:{}, bio_template:{}, ip_cluster:{}}

Agent step loop
  GET_POLICY               (0 steps)  → returns PlatformPolicy summary
  INSPECT acc              (1 step)   → full profile (hidden signals still 0.0)
  REVERSE_IMAGE_SEARCH acc (1 step)   → reveals photo_reuse_score
  ANALYZE_BIO acc          (1 step)   → reveals bio_template_score
  CHECK_IP acc             (2 steps)  → reveals ip_cluster_id
  INVESTIGATE_NETWORK acc  (2 steps)  → 2-hop expansion + SUSPECT cascade
  FLAG acc                 (0 steps)  → dual SUSPECT cascade (follow-graph + IP)
  UNFLAG acc               (0 steps)
  SUBMIT                   (0 steps)  → terminal reward + moderation decision

Submit
  reward = tp*1.0 − fp*fp_penalty_weight − fn*0.3
         + win/recall/efficiency/platform-specific bonuses
  grader_score ∈ [0,1] via scoring.grader_score(tp,fp,fn,steps,max_steps,θ,fp_penalty)
```

---

## 4. Policy Compiler (`server/policy_compiler.py`)

Bayesian threshold derivation:

```
θ* = 1 / (1 + (π × C_fn) / ((1 − π) × C_fp))
```

Inputs extracted from scraped policy text (Tavily search → Groq Llama-3.1-8B):

| Field | Meaning |
|---|---|
| `base_rate` (π) | prevalence of fakes on the platform |
| `fn_cost_signal` | low / medium / high / critical → maps to C_fn ∈ {100, 1000, 5000, 20000} |
| `fp_cost_signal` | low / medium / high → maps to C_fp ∈ {0.01, 0.1, 1.0} |
| `harm_weight` | enforcement-vs-creator balance (multiplies C_fn) |
| `primary_enforcement_signal` | `photo_reuse` \| `bio_template` \| `ip_cluster` |

Offline fallbacks (used when Tavily/Groq not available) are cached at `policy_cache/{instagram,snapchat}.json`:

| Platform | θ | fp_penalty | primary_signal |
|---|---|---|---|
| Instagram | **0.081** (strict) | 0.1 | photo_reuse |
| Snapchat | **0.740** (lenient) | 0.01 | bio_template |

The compiler returns a `PlatformPolicy` (see `models.py:124`) carrying threshold, base_rate, fn/fp cost signals, harm_weight, primary signal, fp_penalty_weight, sources[], confidence, compiled_at.

---

## 5. Hidden-signal architecture

Episode JSON now stores hidden signals at episode level, *not* per account:

```json
{
  "episode_id": "easy_042_Instagram",
  "platform": "Instagram",
  "hidden_signals": {
    "photo_reuse":  {"acc_0001": 0.87, ...},
    "bio_template": {"acc_0001": 0.72, ...},
    "ip_cluster":   {"acc_0001": "ip_gang_42", ...}
  }
}
```

`_build_profile()` reads `photo_reuse_score` / `bio_template_score` / `ip_cluster_id` from `account.features` — which start at 0.0 / "". Tool actions copy from `ep["hidden_signals"]` into `account.features`, then call `_build_profile` to refresh risk. This keeps Round-1 code paths unchanged.

---

## 6. Scoring (`server/scoring.py`)

Stateless risk functions (kept from Round 1): `compute_node_risk`, `compute_behavior_risk`, `compute_graph_risk`, `compute_hub_legitimacy`, `compute_fake_risk`.

Round 2 additions:
- `compute_weighted_fake_risk(..., primary_signal)` boosts the platform's primary signal (node +0.15 for content signals; behavior +0.15 for ip_cluster).
- `classify_risk(fake_risk, threshold=0.35)` now accepts platform threshold.
- `grader_score(tp, fp, fn, steps, max_steps, threshold, fp_penalty_weight)` adds `0.05 × (1 − threshold)` to rewards stricter platforms.

Win conditions (unchanged from Round 1): easy/medium `recall ≥ 0.8, precision ≥ 0.7`; hard `recall ≥ 0.9, precision ≥ 0.8`.

---

## 7. Tool-action contracts

| Action | Cost | Reveals | Fallback when episode has no `hidden_signals` key |
|---|---|---|---|
| `GET_POLICY` | 0 | — (returns `PlatformPolicy` summary) | — |
| `REVERSE_IMAGE_SEARCH` | 1 step, −0.01 | `photo_reuse_score` | reads from `account.features` directly |
| `ANALYZE_BIO` | 1 step, −0.01 | `bio_template_score` | same |
| `CHECK_IP` | 2 steps, −0.02 | `ip_cluster_id` + cluster size | same |

All tool handlers validate `acc_id in self._accounts`, call `_build_profile(acc_id)` to refresh the cached profile, and trigger `_do_submit(forced=True)` if max steps were consumed.

---

## 8. Reward shape (from `_do_submit`)

```
reward = tp × 1.0
       − fp × self._policy.fp_penalty_weight     # IG 0.1 / Snap 0.01
       − fn × 0.3
       + 5.0    if recall ≥ win_recall AND precision ≥ win_precision
       + 3.0    if tp == 10 (perfect recall)
       + 2.0    if partial win (recall met, precision missed)
       + 1.0    if SUBMIT with ≥ 50% steps remaining
       + 2.0    if IG and precision ≥ 0.95
       + 2.0    if Snap and recall ≥ 0.95
       − 1.0 × evasion_count   (hard only)
       − 2.0    if forced SUBMIT
```

---

## 9. Environment state / obs / state schema (OpenEnv-compliant)

- `FakeGangAction`: `action_type: ActionType`, `account_id: Optional[str]`
- `FakeGangObservation`: `done`, `reward`, `visible_accounts[AccountProfile]`, `visible_account_ids`, `flagged_ids`, `inspected_ids`, `graph_edges`, `steps_remaining`, `evasion_triggered`, `evasion_count`, `task`, `message`, `suspect_ids`, **`platform`**
- `FakeGangState`: `episode_id`, `step_count`, `task`, `score_so_far`, `evasion_count`, `network_size`, `gang_size`, `episode_seed`, **`platform`**

`reset(task, seed) → Obs`, `step(action) → Obs`, `state → State` — semantics unchanged from Round 1.

---

## 10. HTTP API (`server/app.py`)

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | `{"status":"healthy"}` |
| `/reset` | POST | `{task, seed, episode_id}` → `StepResponse` |
| `/step` | POST | `FakeGangAction` body → `StepResponse` |
| `/state` | GET | Current `FakeGangState` |
| `/tasks` | GET | Task list + **Round 2 action_schema** |
| `/grader` | GET | Normalized [0,1] score; requires SUBMIT first |
| `/metadata` | GET | HF Spaces metadata |
| `/schema` | GET | Pydantic JSON schemas for action/observation/state |
| `/mcp` | POST | MCP JSON-RPC for tools/list |
| `/baseline` | POST | Runs rule-based baseline on all 3 tasks |
| `/` | GET | Gradio playground (reset, step, grader, baseline, benchmarks) |

---

## 11. TRL / training readiness

- `agent/train.py` is the Round-2 driver. It runs platform-aware rollouts (Bedrock / HF / rule), tracks `tool_counts` per episode, emits per-platform IG/Snap summaries and writes `results/training_results.json`.
- `runs/metrics.jsonl` is written by the top-level `train.py` (Round-1 Reflexion loop, still functional).
- `plot_metrics.py` (added) reads `runs/metrics.jsonl` and writes `reward_curve.png` / `loss_curve.png` — submission artifacts.
- Rewards are numeric and stable. Observations serialize via `policy._format_observation(obs)`; actions parse from `<action>…</action>` blocks.

```bash
# Round 2 driver (platform-aware)
python -m agent.train --episodes 50 --task easy --backend rule        # no-API baseline
python -m agent.train --episodes 50 --backend bedrock --model-id qwen.qwen3-next-80b-a3b

# Round 1 Reflexion + hybrid loop (unchanged)
python train.py --task easy --episodes 20

# Plot curves
python plot_metrics.py                 # reads runs/metrics.jsonl
```

---

## 12. Quickstart

```bash
# 1. Install
cd fake_gang_env
uv sync   # or: pip install -r requirements.txt

# 2. Generate episodes (one-shot — caches 150 JSONs)
python -m server.generator

# 3. Start the environment server
python -m uvicorn server.app:app --port 7860
# → http://localhost:7860  (Gradio UI) and Swagger at /docs

# 4. Sanity-check Round 2
python test_round2.py
python validate.py --url http://localhost:7860     # 24 checks

# 5. Run the Round 2 training loop
python -m agent.train --episodes 20 --task easy --backend rule --output results/r2.json
```

Docker:
```bash
docker build -f server/Dockerfile -t graphstrike .
docker run -p 7860:7860 -v $(pwd)/memory:/app/memory -v $(pwd)/runs:/app/runs graphstrike
```

---

## 13. Files that matter

**Edit for Round 2 changes:**
- `models.py` — data schemas
- `server/environment.py` — reset/step/state + tool handlers
- `server/generator.py` — platform + hidden_signals
- `server/scoring.py` — platform-aware scoring
- `server/policy_compiler.py` — Bayesian θ*
- `server/app.py` — HTTP surface
- `agent/policy.py` — LLM prompt + action parser
- `agent/train.py` — Round 2 rollout driver

**Do not touch unless necessary:**
- `inference.py` — working Round 2 inference script
- `bedrock_model.py` — Bedrock client
- Round 1 memory/Reflexion: `agent/memory.py`, `agent/reflection.py`, `agent/hybrid_policy.py`, top-level `train.py`
- `client.py` — HTTP client used by Round 1 train loop
- `validate.py`, `test_judge_eval.py` — external validators
- Cached `episodes/*.json` and `policy_cache/*.json`

---

## 14. Outstanding (optional) work

1. **Tool-use reward shaping** (dense signal) — penalize redundant tool calls on same account; bonus for GET_POLICY early.
2. **Moderation decision package** — return `recommended_action ∈ {queue_for_review, temporary_hold, scheduled_ban, batch_takedown}` with evidence + policy rationale in the terminal observation.
3. **`hybrid_policy.py` platform-aware upgrade** — the Round-1 rule engine still uses fixed `_THRESHOLDS`; could read `env._policy.threshold` instead. Low priority because `agent/train.py` is the Round-2 driver.
4. **Dashboard** — `dashboard/DASHBOARD_SPEC.md` describes a React + D3.js live demo (policy compiler panel, network graph, training curves). Not required for submission.
5. **Additional platforms** (Twitter/X, TikTok) — straightforward once the compiler pipeline is trusted.
6. **README.md** — add a Round 2 section (current README describes Round 1 only).

---

## 15. Design decisions (kept from earlier docs, condensed)

- **Hidden signals at episode level, not account level** — easier to track revelation, cleaner rollback between episodes.
- **Platform assignment by seed parity** — reproducible without extra RNG state; 50/50 split.
- **Bayesian θ*** — principled, explainable, varies sensibly when policy text changes.
- **Asymmetric tool costs** — CHECK_IP is 2× to force the agent to use cheap signals first. Creates strategic depth that the reward function can teach.
- **Cached policies** — hackathon-demo viable without network; live compilation optional.
- **Weighted risk via primary signal** — lets the same scoring math serve both platforms without hardcoding `if platform == …`.

---

## 16. Known tests / validation

- `test_round2.py` — 9-stage end-to-end. Passes locally against `server/environment.py`.
- `validate.py` — 24 HTTP validator checks against a running server.
- `test_judge_eval.py` — judge model vs. environment scoring.

All four were verified against the current tree.
