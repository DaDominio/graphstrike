# GraphStrike — Single Source of Truth

> Consolidates: `FINAL_SUMMARY.md`, `IMPLEMENTATION_COMPLETE.md`, `IMPLEMENTATION_STATUS.md`, `INFERENCE_UPDATE.md`, `PIPELINE.md`, `QUICKSTART.md`, `ROUND2_COMPLETE.md`, `ROUND2_STATUS.md`, `ROUND2_TRAINING_READY.md`, `server/ROUND2_FINAL_STATUS.md`, and the top-level `ROUND2_ARCHITECTURE.md` / `ROUND2_IMPLEMENTATION_PLAN.md` / `ROUND2_QUICK_REFERENCE.md` / `OpenEnv-Complete.md`.
>
> The HF-Space `README.md` is kept (it contains the YAML frontmatter Spaces needs). The per-directory `dashboard/README.md` describes only the local dashboard and stays with it.

---

## 1. What GraphStrike is

An OpenEnv-compatible RL environment. An LLM agent must identify the 10 members of a coordinated fake-account ring hidden inside a synthetic social network. Round 2 makes detection **platform-adaptive**:

- Each episode belongs to a platform (Instagram, Snapchat, X, LinkedIn, Reddit, … any name).
- A `PlatformPolicy` is **compiled from real transparency-report text** via a Bayesian threshold formula and cached per-platform.
- The high-signal account fields (`photo_reuse_score`, `bio_template_score`, `ip_cluster_id`) start hidden and are revealed only by explicit tool actions.
- Reward shape, FP penalty, grader score, and the moderation-decision package are all derived from the compiled policy rather than hardcoded.

A separate **shared evaluation runner** drives episodes deterministically and consults the LLM at exactly two decision points per suspicious account; six thin model shims plug in HF-router or Bedrock models against that runner.

---

## 2. Round 2 deltas (what changed vs Round 1)

| Area | Round 1 | Round 2 |
|---|---|---|
| Platform | — | `platform` field per episode; any name supported (env defaults to seed-parity Instagram/Snapchat) |
| Policy | hardcoded thresholds | `PlatformPolicy` compiled dynamically from transparency reports (Bayesian θ\*) with 30-day cache freshness and sanity checks |
| Signals | all visible at INSPECT | `photo_reuse_score`, `bio_template_score`, `ip_cluster_id` start at `0.0 / ""` and are revealed only by tool actions |
| Visible accounts | populated only on INSPECT | populated for every visible account from reset; tool reveals propagate immediately |
| Per-step reward | `null` for non-terminal steps | float delta of `self._score` returned every step |
| Actions | `inspect`, `investigate_network`, `flag`, `unflag`, `submit` | + `get_policy`, `reverse_image_search`, `analyze_bio`, `check_ip` |
| Reward shaping | terminal only | + `+0.20` first-action GET\_POLICY bonus, redundant-tool penalties, no-evidence flag deny |
| Submit response | `{observation, done, reward, message}` | + top-level `decision_package` and `grader_score` |
| Eval | one monolithic `qwen_test_judge_eval.py` per model | shared `_round2_runner.py` + 6 thin shims, two LLM decision points per account |

Platform assignment is deterministic in the env: `seed % 2 == 0 → Instagram`, else Snapchat. The eval runner remaps seeds so any requested platform actually fires (`--platform Instagram` forces even seeds, `--platform Snapchat` forces odd).

---

## 3. End-to-end policy flow (from transparency report to gradient signal)

This is the spine of Round 2. Every other component reads from this pipeline.

```
                       (ONE-TIME / OFFLINE)
   transparency-report URLs                             policy_cache/
   ────────────────────────                             ─────────────
       │                                                  │
       ▼                                                  ▲
   Tavily search                                          │
   query: "{platform} fake account content                │
           policy enforcement 2024 2025"                  │
       │                                                  │
       ▼                                                  │
   Groq Llama-3.1-8B extraction                           │
   → {base_rate π, fn_cost_signal, fp_cost_signal,        │
      harm_weight, primary_signal, confidence}            │
       │                                                  │
       ▼                                                  │
   sanitize_pi()  — clamp [0.0005, 0.05]                  │
   (>0.05 ⇒ "enforcement rate misread", clamp + warn)     │
       │                                                  │
       ▼                                                  │
   compute_threshold(π, fn_signal, fp_signal, hw)         │
   ────────────────────────────────────────────────       │
   C_fn = FN_COST_MAP[fn_signal]                          │
   C_fp = FP_COST_MAP[fp_signal]                          │
   θ_raw = C_fn·π / [C_fn·π + C_fp·(1−π)]                 │
   θ*    = clamp(θ_raw / harm_weight, 0.01, 0.95)         │
   fp_penalty_weight = C_fp                               │
       │                                                  │
       ▼                                                  │
   PlatformPolicy(threshold=θ*, base_rate=π,              │
                  fn/fp_cost_signal, harm_weight,         │
                  primary_enforcement_signal,             │
                  fp_penalty_weight=C_fp,                 │
                  confidence, sources, used_fallback) ────┘
       │
       ▼ sanity_check_policy() — surfaces warnings
       ▼  (high θ*, suspicious π, low confidence, bad signal name)
       ▼
       cached to policy_cache/{platform}.json
                     │
=====================│=====================
                     │ (PER EPISODE — RUNTIME)
                     ▼
   client.reset(task, seed)
   env.platform = "Instagram"
   env._policy = get_policy("Instagram")    ◄── reads cached JSON
                     │                          (recompiles if >30 days old)
                     ▼
   deterministic step 0:  GET_POLICY  (free, +0.20 first-action bonus)
   message:  "Policy compiled: Platform: Instagram |
             Threshold: 0.369 | Primary Signal: photo_reuse | FP Penalty: 0.1x | …"
                     │
                     ▼
   runner._policy_from_message() → policy dict {threshold, primary_signal, fp_weight}
                     │
                     ▼
   per suspicious account, sorted by risk_score desc:
     INSPECT (deterministic)
     INVESTIGATE_NETWORK if risk ≥ 0.80 (deterministic, once)

     ┌─ DP1 (LLM) ─────────────────────────────┐
     │ prompt includes platform, primary_signal,│
     │ θ*, revealed-vs-None signals, budget     │
     │ → "reverse_image_search" / "analyze_bio" │
     │ / "check_ip" / "done"                    │
     └──────────────────────────────────────────┘
     ↓ (loop until "done" or signals sufficient)

     ┌─ DP2 (LLM) ─────────────────────────────┐
     │ prompt includes revealed signals,        │
     │ θ*, fp_penalty=C_fp, running tp/fp count │
     │ → "flag" / "skip"                        │
     └──────────────────────────────────────────┘
                     │
                     ▼
   SUBMIT (deterministic)
   reward = tp·1.0 − fp·C_fp − fn·0.3 + bonuses − penalties
                                  ▲
                                  └── platform-specific via fp_penalty_weight
   grader_score and decision_package surfaced at top level of /step response.
```

**Two views of the same policy:**
- `θ*` is in the **prompt** at DP1/DP2 → the LLM conditions on it.
- `C_fp` (= `fp_penalty_weight`) is in the **terminal reward** → the LLM is graded against it.

Both come from the same compile-time computation; they cannot drift apart.

---

## 4. Policy Compiler (`server/policy_compiler.py`)

### 4.1 Formula

```
θ_raw = C_fn · π / [C_fn · π + C_fp · (1 − π)]
θ*    = clamp(θ_raw / harm_weight, 0.01, 0.95)
fp_penalty_weight = C_fp
```

Action rule the threshold serves: **FLAG if `risk_score ≥ θ*`**.

`θ_raw` is the share of expected cost coming from missed fakes. Higher `C_fn` or higher base rate → higher `θ_raw` → lower threshold (the agent should flag more aggressively when misses are expensive).

`harm_weight > 1` strict (lowers θ\*); `harm_weight < 1` lenient (raises θ\*).

> **History note.** The original spec used `θ_raw = C_fp(1−π) / [C_fp(1−π) + C_fn·π]` — the *complementary* probability. With small π that formula collapses to `≈ 1` for every platform (π is the bottleneck, not the costs). Audit on 2026-04-25 confirmed this was a formula-direction error; the orientation above is correct for our action rule.

### 4.2 Cost maps

```python
FN_COST_MAP = {"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 4.0}
FP_COST_MAP = {"low": 0.1, "medium": 0.5, "high": 1.5}
```

Signals are extracted from policy text by an LLM and constrained to these keys (defaults `high` / `medium` if absent or invalid).

### 4.3 Extraction inputs

| Field | Source | Sanitization |
|---|---|---|
| `base_rate` (π) | LLM extraction from transparency report | `sanitize_pi`: clamp to `[0.0005, 0.05]`; >0.05 logs *"likely enforcement rate misread, clamped"*. The prompt also instructs the LLM to return `0.005` if it sees an enforcement rate or no prevalence figure. |
| `fn_cost_signal` | LLM extraction | invalid → `high` |
| `fp_cost_signal` | LLM extraction | invalid → `medium` |
| `harm_weight` | LLM extraction | non-numeric → `1.0` |
| `primary_enforcement_signal` | LLM extraction | None / blank / non-string → `photo_reuse` |
| `confidence` | LLM extraction | non-numeric → `0.0` |

### 4.4 Tavily query (generic, platform-agnostic)

```python
query = f"{platform} fake account content policy enforcement 2024 2025"
```

The previous query was Meta/Instagram-specific; the generic form works for **any** platform name. Domain filtering (`is_high_signal_source`) was removed for the same reason — it gated to meta.com/snap.com domains.

### 4.5 Caching & freshness

- Cached at `policy_cache/{platform_lowercase}.json`.
- Entries older than `CACHE_TTL_DAYS = 30` are treated as stale and recompiled.
- `compile_policy(platform, use_cache=True)` is the runtime entry; `--use-cache` flag controls CLI behavior (default re-compile when invoked from CLI).

### 4.6 Fallbacks

- `FALLBACK_POLICIES` provides hardcoded params for Instagram / Snapchat. Any other platform falls back to `GENERIC_FALLBACK` (π=0.005, fn=high, fp=medium, hw=1.0).
- Fallback policies set `used_fallback=True` (a new field on `PlatformPolicy`).
- The **threshold value** in fallbacks is computed via the same formula — there is no hardcoded threshold in the policy compiler anymore.

### 4.7 Sanity check (`sanity_check_policy`)

After every compile, the compiler prints warnings for any of:

| Trigger | Meaning |
|---|---|
| `θ* > 0.90` | agent will almost never flag — check fn_cost extraction |
| `θ* < 0.005` | agent will flag nearly everything — check fp_cost extraction |
| `base_rate > 0.05` | likely enforcement-rate misread |
| `confidence < 0.60` | low extraction quality; consider falling back |
| `primary_signal ∉ {photo_reuse, bio_template, ip_cluster, behavior}` | not a known tool action |

Sanity check **does not block compilation**; it surfaces issues so an operator can review before running eval.

### 4.8 CLI

```bash
python -m server.policy_compiler --platform <Name>          # always recompile
python -m server.policy_compiler --platform <Name> --use-cache
```

### 4.9 Currently compiled policies

| Platform   | π     | fn_signal | fp_signal | hw  | θ\*   | C_fp | confidence | used_fallback |
|------------|------:|-----------|-----------|----:|------:|-----:|-----------:|--------------:|
| X          | 0.005 | high      | low       | 1.0 | 0.091 | 0.10 | 0.80       | False |
| Instagram  | 0.030 | critical  | low       | 1.5 | 0.369 | 0.10 | 0.80       | False |
| Snapchat   | 0.005 | low       | low       | 1.0 | 0.025 | 0.10 | 0.50 ⚠     | False |
| LinkedIn   | 0.005 | critical  | low       | 1.0 | 0.167 | 0.10 | 0.80       | False |
| Reddit     | 0.005 | low       | low       | 1.0 | 0.025 | 0.10 | 0.50 ⚠     | False |

Snapchat and Reddit currently raise the *low confidence* sanity warning — extraction is noisy on those transparency reports. Consider forcing the fallback path before training on them.

---

## 5. Hidden-signal architecture

Episode JSON stores hidden signals at episode level, not per account:

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

`account.features` start with `photo_reuse_score = 0.0`, `bio_template_score = 0.0`, `ip_cluster_id = ""`. Tool handlers copy from `ep["hidden_signals"]` into `account.features` and refresh the cached profile so subsequent observations carry the revealed value.

> **Known limitation.** `generator.py` accepts a `platform` arg but currently produces identical hidden-signal distributions for every platform. Platform conditioning is therefore purely *prompt-side* — the LLM learns to read θ\* and C_fp from the prompt and reward, not to recognize platform-specific data shape. Parametrizing the generator by platform is a separate follow-up.

---

## 6. Scoring (`server/scoring.py`)

Stateless risk functions (kept from Round 1): `compute_node_risk`, `compute_behavior_risk`, `compute_graph_risk`, `compute_hub_legitimacy`, `compute_fake_risk`.

Round 2 additions:
- `compute_weighted_fake_risk(..., primary_signal)` boosts the platform's primary signal (node risk +0.15 for content signals; behavior risk +0.15 for `ip_cluster`).
- `classify_risk(fake_risk, threshold)` accepts platform threshold.
- `grader_score(tp, fp, fn, steps, max_steps, threshold, fp_penalty_weight)` adds `0.05 × (1 − threshold)` to reward stricter platforms.

Win conditions (unchanged from Round 1): easy/medium `recall ≥ 0.8, precision ≥ 0.7`; hard `recall ≥ 0.9, precision ≥ 0.8`.

---

## 7. Tool-action contracts

| Action | Step cost | Score delta | Reveals | Notes |
|---|---|---|---|---|
| `GET_POLICY` | 0 | `+0.20` once (first action) | — (returns `PlatformPolicy` summary in `message`) | Free; bonus only fires on `_action_count == 1` |
| `INSPECT` | 1 | `−0.01` | full profile, edges | needed before any DP1/DP2 logic |
| `REVERSE_IMAGE_SEARCH` | 1 | `−0.01` (`−0.05` if redundant) | `photo_reuse_score` | sets `account.features.photo_reuse_score` |
| `ANALYZE_BIO` | 1 | `−0.01` (`−0.05` if redundant) | `bio_template_score` | sets `account.features.bio_template_score` |
| `CHECK_IP` | 2 | `−0.02` (`−0.10` if redundant) | `ip_cluster_id` + cluster-size message | heaviest tool; only worth it for shared_ip ≥ 5 |
| `INVESTIGATE_NETWORK` | 2 | `−0.02` | 2-hop expansion + SUSPECT cascade | unchanged from Round 1 |
| `FLAG` | 0 | `−0.15` if no evidence (deny) | dual SUSPECT cascade (follow-graph + IP) | "no evidence" = not inspected AND no tool used on the account |
| `UNFLAG` | 0 | 0 | — | unchanged |
| `SUBMIT` | 0 | terminal-reward formula (§ 8) | end episode | also surfaces `decision_package` and `grader_score` at top level |

All tool handlers validate `acc_id in self._accounts`, refresh the cached profile, and force `_do_submit(forced=True)` if max steps were consumed.

---

## 8. Reward shape (per-step deltas + terminal)

Per-step delta is now visible on every `/step` response: it is `round(self._score - self._last_score, 4)`. `terminal_reward` overrides the delta on the SUBMIT step so the caller sees the full episode reward there.

### 8.1 Per-step shaping (visible immediately)

```
+0.20    GET_POLICY as first action              (once per episode)
-0.01    per inspect / reverse_image_search / analyze_bio  (time cost)
-0.02    per check_ip / investigate_network                (time cost)
-0.05    per redundant reverse_image_search / analyze_bio
-0.10    per redundant check_ip
-0.15    blind FLAG (no inspect, no tool used on account)  ← deny + penalty
```

### 8.2 Terminal reward at SUBMIT

```
reward = tp · 1.0
       − fp · self._policy.fp_penalty_weight        (= C_fp; varies per platform)
       − fn · 0.3
       + 5.0   if recall ≥ win_recall AND precision ≥ win_precision
       + 3.0   if tp == 10 (perfect recall)
       + 2.0   if partial win (recall met, precision missed)
       + 1.0   if SUBMIT with ≥ 50% steps remaining
       + 2.0   if Instagram and precision ≥ 0.95
       + 2.0   if Snapchat   and recall    ≥ 0.95
       − 1.0 × evasion_count   (hard task only)
       − 2.0   if forced SUBMIT (ran out of steps)
       − 0.15 × |unsupported_flags|   (flags with no revealed signals at submit time)
```

Note: `fp_penalty_weight` is platform-specific and is the principal lever the policy compiler pulls. Same FP behavior costs more on X (1.5) than on Instagram/Snapchat (0.1).

---

## 9. Schemas (OpenEnv-compliant)

### 9.1 Models

- `FakeGangAction`: `action_type: ActionType`, `account_id: Optional[str]`
- `FakeGangObservation`: `done`, `reward` (per-step delta or terminal), `visible_accounts[AccountProfile]` (now populated for every visible id), `visible_account_ids`, `flagged_ids`, `inspected_ids`, `graph_edges`, `steps_remaining`, `evasion_triggered`, `evasion_count`, `task`, `message`, `suspect_ids`, **`platform`**
- `FakeGangState`: `episode_id`, `step_count`, `task`, `score_so_far`, `evasion_count`, `network_size`, `gang_size`, `episode_seed`, **`platform`**
- `PlatformPolicy`: `platform`, `threshold`, `base_rate`, `fn_cost_signal`, `fp_cost_signal`, `harm_weight`, `primary_enforcement_signal`, `fp_penalty_weight`, `sources`, `confidence`, `compiled_at`, **`used_fallback`**

### 9.2 `StepResponse` (HTTP)

```json
{
  "observation": { ... },
  "done": <bool>,
  "reward": <float | null>,
  "message": "...",
  "decision_package": { ... } | null,   // populated after SUBMIT
  "grader_score":     <float> | null     // populated after SUBMIT, sourced from decision_package
}
```

`decision_package` (after SUBMIT) carries:
- `platform`, `flagged_accounts[]`, `recommended_action ∈ {queue_for_review, temporary_hold, scheduled_ban, batch_takedown}`
- `evidence_summary`: `flagged`, `revealed_photo_reuse`, `revealed_bio_template`, `revealed_ip_cluster`, `unsupported_flags[]`
- `policy_rationale`: textual explanation including θ\*, primary signal, FP penalty, observed precision/recall
- `tp`, `fp`, `fn`, `precision`, `recall`, `reward`, `grader_score`

The terminal `message` also embeds the keywords `flagged_accounts`, `evidence_summary`, `policy_rationale`, `grader_score` for callers that grep the message string.

---

## 10. HTTP API (`server/app.py`)

| Endpoint | Method | Notes |
|---|---|---|
| `/health` | GET | `{"status":"healthy"}` |
| `/reset` | POST | `{task, seed, episode_id}` → `StepResponse` |
| `/step` | POST | `FakeGangAction` body → `StepResponse` (per-step reward delta + decision_package + grader_score on SUBMIT) |
| `/state` | GET | Current `FakeGangState` |
| `/tasks` | GET | Task list + Round 2 action_schema (9 actions) |
| `/grader` | GET | Normalized [0,1] score; requires SUBMIT first |
| `/metadata` | GET | HF Spaces metadata |
| `/schema` | GET | Pydantic JSON schemas |
| `/mcp` | POST | MCP JSON-RPC for tools/list |
| `/baseline` | POST | Runs rule-based baseline on all 3 tasks |
| `/` | GET | Gradio playground |

`openenv.yaml` action schema mirrors all nine action types (Round 1 five plus the four Round 2 tools).

---

## 11. Evaluation runner (`eval-models/_round2_runner.py`)

### 11.1 Outer loop (deterministic)

```
reset(task, seed)
  ↓
GET_POLICY (step 0)   ← always; bonus +0.20
  ↓
loop over visible accounts sorted by (suspect_flag, risk_score) desc:
    INSPECT if not yet inspected
    INVESTIGATE_NETWORK if risk_score ≥ 0.80 (once per account, ≥5 steps left)

    DP1 loop (LLM) — pick a tool or "done"
        reverse_image_search | analyze_bio | check_ip | done
        stops on "done", missing budget, or photo + bio both revealed

    DP2 (LLM) — flag-or-skip
        flag → env.step(FLAG)
        skip → leave alone, move to next account
  ↓
SUBMIT
```

Stops early when `done` is signaled, `steps_remaining ≤ 1`, or `max_accounts_per_episode = 15` accounts have been processed.

### 11.2 The two LLM decision points

**DP1 — tool selection** prompt includes:
- `platform`, `primary_signal`, `θ*`
- `account_id`, `risk_score`, `hub_legitimacy`
- Each revealed signal value or `None`
- `steps_remaining`, tool costs

**DP2 — flag decision** prompt includes:
- All revealed signals for the account
- `θ*`, `fp_penalty = C_fp`
- Running `flagged / 10`, `steps_remaining`

Each prompt asks for **exactly one token** so parsing is robust. Invalid completions are counted in `dp1_invalid` / `dp2_invalid` for QA.

### 11.3 Per-episode JSONL log

`eval-models/results/{model}_{platform}_results.jsonl` — one line per episode:

```json
{
  "model": "Bedrock/qwen.qwen3-next-80b-a3b",
  "platform": "Instagram",
  "task": "easy", "seed": 0,
  "episode_id": "easy_000_Instagram",
  "threshold": 0.369, "primary_signal": "photo_reuse",
  "steps_taken": 14, "inspected": 5,
  "tool_calls": {"reverse_image_search": 5, "analyze_bio": 4, "check_ip": 1,
                 "get_policy": 1, "investigate_network": 1},
  "flagged": 7,
  "dp1_calls": 12, "dp2_calls": 5, "dp1_invalid": 0, "dp2_invalid": 0,
  "reward": 4.32, "grader_score": 0.71,
  "final_message": "...", "wall_seconds": 23.4
}
```

### 11.4 Public entry point

```python
from _round2_runner import run_evaluation
run_evaluation(
    model_name="qwen-72b",
    call_llm=lambda prompt: ...,   # injectable adapter
    platform="Instagram",
    base_url="http://localhost:7860",
    tasks=["easy", "medium", "hard"],
    seeds=[0, 1, 2],
)
```

The runner remaps requested seeds to the env's parity rule so `--platform Instagram` actually runs Instagram episodes (`even`) and `--platform Snapchat` runs Snapchat (`odd`). Other platform names pass seeds through unmodified (env then falls back to its parity default for that seed).

### 11.5 Import-order safety

The runner unconditionally inserts the project root at `sys.path[0]` and evicts any cached `models` / `client` modules so a stale copy in `~/.local/lib/python3.12/site-packages` cannot win. If your shim raises `ActionType has no attribute GET_POLICY`, that means the safety insert was skipped — verify you are running today's runner.

---

## 12. Model shims (`eval-models/{qwen,gemma,deepseek,llama,mistral,nvidia}_test_judge_eval.py`)

Each shim is ~30 lines. It declares the model identifiers and delegates to the runner via `_llm_adapters.make_caller`:

| Shim | HF model | Bedrock model |
|---|---|---|
| qwen | `Qwen/Qwen2.5-72B-Instruct` | `qwen.qwen3-next-80b-a3b` |
| gemma | `google.gemma-3-12b-it` | same |
| deepseek | `deepseek.v3.2` | same |
| llama | `meta.llama4-scout-17b-instruct-v1:0` | same |
| mistral | `mistral.ministral-3-8b-instruct` | same |
| nvidia | `nvidia.nemotron-super-3-120b` | same |

`_llm_adapters.py` exposes `make_hf_caller(model)`, `make_bedrock_caller(model_id)`, and a unified `make_caller(backend, hf_model, bedrock_model)`. Both backends strip `<think>...</think>` reasoning blocks and retry up to 3× with exponential backoff.

### Usage

```bash
# HF router (needs HF_TOKEN)
python eval-models/qwen_test_judge_eval.py --url http://localhost:7860 --platform Instagram

# AWS Bedrock (needs AWS_* env vars)
python eval-models/qwen_test_judge_eval.py --bedrock --url http://localhost:7860 --platform Snapchat \
    --tasks easy medium --seeds 0 1 2
```

---

## 13. Files that matter

**Source of truth (read first):**
- `reference.md` — this file
- `models.py` — data schemas (`PlatformPolicy.used_fallback` is new)
- `server/policy_compiler.py` — Bayesian θ\*, sanity check, generic Tavily, 30-day cache
- `server/environment.py` — reset/step/state, tool handlers, per-step reward delta, no-evidence flag deny, decision package
- `server/app.py` — `StepResponse` with top-level `decision_package` and `grader_score`
- `server/scoring.py` — risk/grader math
- `server/generator.py` — episode generation, `hidden_signals`
- `eval-models/_round2_runner.py` — deterministic loop + DP1/DP2
- `eval-models/_llm_adapters.py` — HF + Bedrock callers
- `eval-models/{model}_test_judge_eval.py` — six thin shims
- `openenv.yaml` — action schema mirrors all 9 actions
- `check.sh` — 12-step Round 2 system check (server side)

**Operational:**
- `policy_cache/{platform}.json` — compiled policies (delete to force recompile)
- `episodes/{task}_{seed}.json` — generated episodes (regenerate with `python -m server.generator`)
- `eval-models/results/{model}_{platform}_results.jsonl` — per-episode eval logs

**Round 1 still functional:**
- `agent/train.py`, `agent/policy.py`, `agent/memory.py`, `agent/reflection.py`, `agent/hybrid_policy.py`
- `inference.py`, `bedrock_model.py`, `client.py`
- `validate.py`, `test_round2.py`

---

## 14. Quickstart

```bash
# 1. Install
cd fake_gang_env
uv sync   # or: pip install -r requirements.txt

# 2. Compile / refresh platform policies (one-time, then per ≥30 days)
python -m server.policy_compiler --platform Instagram
python -m server.policy_compiler --platform Snapchat
python -m server.policy_compiler --platform X
python -m server.policy_compiler --platform LinkedIn

# 3. (Re)generate episodes
python -m server.generator

# 4. Start the env server
python -m uvicorn server.app:app --port 7860

# 5. End-to-end system check (12 verifications)
bash check.sh

# 6. Run a model shim against the live server
export HF_TOKEN=...                              # or AWS_*
python eval-models/qwen_test_judge_eval.py \
    --url http://localhost:7860 \
    --platform Instagram \
    --tasks easy medium hard \
    --seeds 0 1 2
# Logs: eval-models/results/Qwen_Qwen2.5-72B-Instruct_instagram_results.jsonl
```

Docker:
```bash
docker build -f server/Dockerfile -t graphstrike .
docker run -p 7860:7860 -v $(pwd)/memory:/app/memory -v $(pwd)/runs:/app/runs graphstrike
```

---

## 15. System check (`check.sh`)

Twelve numbered checks against a running server at `http://localhost:7860`:

| # | Check | Pass criterion |
|---|---|---|
| 1–4 | health, /tasks, /reset, /step GET_POLICY | endpoints respond; action schema lists 9 types; threshold appears in message |
| 5 | INSPECT first visible account | profile returned; account_id is real (extracted from `visible_account_ids` *before* inspect) |
| 6 | REVERSE_IMAGE_SEARCH | `photo_reuse_score > 0` for that account in `observation.visible_accounts[*]` |
| 7 | ANALYZE_BIO | `bio_template_score > 0` |
| 8 | CHECK_IP | message reports cluster, `shared_ip_count` populated |
| 9 | GET_POLICY first-action bonus | per-step `reward ≥ 0.15` |
| 10 | redundant tool penalty | second `reverse_image_search` reward < first |
| 11 | blind FLAG penalty | flag without prior inspect/tool → reward `≤ −0.10` |
| 12 | full episode | submit response carries the four decision-package keywords + non-null `grader_score` |

CHECK 5–8 read from `observation.visible_accounts[*]` rather than a non-existent top-level `profile` field — the prior version of `check.sh` had that bug.

---

## 16. Bug fixes shipped 2026-04-25

| # | File | Symptom | Root cause | Fix |
|---|---|---|---|---|
| 1 | `server/environment.py` | `reward: null` on every non-terminal step | `_make_observation` only set `terminal_reward` | Track `_last_score`; return `score - _last_score` as per-step delta |
| 2 | `server/environment.py` | `visible_accounts: []` until INSPECT | observation included only `_profiled` | Build a profile for every `_visible_id` (cached for inspected, fresh otherwise). Tool reveals propagate because `_build_profile` reads from `account.features` which the tool handlers update. |
| 3 | `server/environment.py` | Tool reveals invisible to caller | covered by Bug 2 | — |
| 4 | `server/environment.py` | GET_POLICY +0.20 not visible | accumulated into `_score` but `_make_observation` never returned it | covered by Bug 1 |
| 5 | `server/environment.py`, `server/app.py` | submit response missing decision-package keywords + `grader_score` | message lacked the literal keywords; StepResponse only had four fields | enrich submit message; add `decision_package` and `grader_score` as top-level fields on StepResponse |
| 6 | `server/environment.py` | blind FLAG (no inspect, no tool) returned 0 reward | submit-time `unsupported_flags` only fires at SUBMIT | `_do_flag` now denies blind flags immediately with `−0.15` |
| 7 | `eval-models/_round2_runner.py` | `ActionType has no attribute GET_POLICY` when running shims | `if _PARENT not in sys.path` guard skipped the insert because path was already present at lower priority; site-packages `models.py` won | Insert `_PARENT` at index 0 unconditionally; evict cached `models`/`client` from `sys.modules` |
| 8 | `check.sh` | acc\_000 hardcoded; profile field read from wrong path | script bugs | extract real `account_id` from `observation.visible_account_ids` *before* CHECK 5; read profiles from `observation.visible_accounts[*]` |
| — | `server/policy_compiler.py` | θ\* always ≈ 0.95 | formula direction inverted (computed FP-cost share, not FN-cost share) | `θ_raw = C_fn·π / [C_fn·π + C_fp·(1−π)]` |
| — | `server/policy_compiler.py` | enforcement-rate misreads (e.g. Snap π=0.262) | LLM confusion between "% removed" and "% prevalence" | `sanitize_pi` clamp `[0.0005, 0.05]` + warning; extraction prompt explicitly disambiguates |
| — | `server/policy_compiler.py` | crash on Pydantic validation when LLM returned `None` for `primary_enforcement_signal` | strict typing | coerce None / blank to `photo_reuse`; same for `confidence` |

---

## 17. Sanity rules for adding a new platform

After running `python -m server.policy_compiler --platform <Name>`:

| Property | Acceptable range | Action if outside |
|---|---|---|
| `threshold` | `[0.005, 0.90]` | review — likely cost-signal extraction issue |
| `base_rate` | `[0.0005, 0.05]` | review — likely enforcement-rate misread |
| `confidence` | `≥ 0.60` | force fallback or improve sources |
| `primary_signal` | one of `{photo_reuse, bio_template, ip_cluster, behavior}` | coerced to `photo_reuse` |
| `used_fallback` | match expectation | ensure Tavily/Groq keys are set if False expected |

Cross-platform ordering is **not** an invariant. Any platform may land anywhere on the [0.01, 0.95] θ\* scale depending on its actual policy.

---

## 18. Outstanding (optional) work

1. **Platform-specific episode generation** — `generate_episode` accepts a `platform` arg but produces identical hidden-signal distributions. Parametrize π, signal strengths, and evasion behavior per platform for richer training data.
2. **TRL/GRPO trainer wrapper** — runner produces `(prompt, completion)` pairs at DP1/DP2 and per-step rewards. Threading these into a TRL `DataCollator` is the next step (training-side scope, not part of this readiness pass).
3. **Force-fallback flag on the CLI** — convenient way to ignore Tavily and use hardcoded params when sanity check raises low-confidence warnings.
4. **`hybrid_policy.py` platform-aware upgrade** — Round-1 rule engine still uses fixed `_THRESHOLDS`; could read `env._policy.threshold`. Low priority since `agent/train.py` and the eval runner are independent.
5. **Dashboard** — `dashboard/DASHBOARD_SPEC.md` describes a React + D3 demo; not required.

---

## 19. Design decisions (kept from earlier docs, condensed)

- **Hidden signals at episode level, not account level** — easier to track revelation, cleaner rollback between episodes.
- **Platform assignment by seed parity (env)** — reproducible without extra RNG state; eval runner remaps seeds when `--platform` is requested.
- **Bayesian θ\*** — principled, explainable, varies sensibly when policy text changes. Action rule is `FLAG if risk ≥ θ*`.
- **Asymmetric tool costs** — CHECK_IP is 2× to force the agent to use cheap signals first.
- **Cached policies + 30-day TTL** — hackathon-demo viable without network; live recompile on staleness.
- **Two LLM decision points** — keeps the LLM's job focused (tool-pick + flag/skip) and makes (prompt, completion, reward) tuples cleanly attributable for future RL training.
- **Top-level `decision_package` + `grader_score`** — callers shouldn't have to grep the message string for the four submission fields.

---

## 20. Known tests / validation

- `bash check.sh` — 12-step end-to-end against a running server (Round 2 system check).
- `test_round2.py` — 9-stage Python test against `server/environment.py`.
- `validate.py` — 24 HTTP validator checks against a running server.
- `eval-models/{model}_test_judge_eval.py` — judge model vs. environment scoring with two-decision-point loop.

All four were verified against the current tree on 2026-04-25.
