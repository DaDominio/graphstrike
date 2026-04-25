---
title: GraphStrike
emoji: 🕵️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - reinforcement-learning
  - social-network
  - fraud-detection
  - openenv
  - llm-agent
---
<br>

<p align="center">
<img src="assets/logo.png" width="600"/>
</p>

<br>

<p align="center">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/HF%20Spaces-FFBF00?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=gradio&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenEnv-4B5563?style=for-the-badge&logo=envato&logoColor=white"/>
  <img src="https://img.shields.io/badge/Amazon%20Bedrock-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
</p>
<br>

<h1 align="center">
</h1>
  <p align="center">
    An OpenEnv-compatible reinforcement learning environment where an LLM agent must identify all 10 members of a coordinated fake account network hidden inside a synthetic social network. The agent learns via Reflexion and a dynamic hybrid rule/LLM policy , not via gradient updates or fine-tuning.
    <br />
    </p>
</p>

<br>

## Round 2 — Platform-Adaptive Trust & Safety

Round 2 makes detection **platform-aware end-to-end**. Each episode runs on a named platform (Instagram, Snapchat, X, LinkedIn, Reddit, …); a `PlatformPolicy` is compiled offline from real transparency-report text and cached per platform; the high-signal account fields start hidden and are revealed only by explicit tool actions; and a shared evaluation runner consults the LLM at exactly two decision points per suspicious account.

### How the policy lives end-to-end

```
transparency reports                        per-episode runtime
────────────────────                        ───────────────────
Tavily search ──► Groq Llama-3.1            client.reset(task, seed)
extracts {π, fn_cost, fp_cost,                       │
          harm_weight, primary_signal}      env loads policy_cache/{platform}.json
              │                                      │
              ▼                              GET_POLICY (step 0, +0.20 bonus)
sanitize_pi()  clamp π to [5e-4, 0.05]               │
compute_threshold:                          DP1 (LLM): pick tool
  θ_raw = C_fn·π / [C_fn·π + C_fp·(1−π)]      reverse_image_search / analyze_bio
  θ*    = clamp(θ_raw / harm_weight, .01, .95)  / check_ip / done
  fp_penalty_weight = C_fp                          │
              │                              DP2 (LLM): flag / skip
              ▼                                      │
sanity_check_policy() warns on outliers     SUBMIT
              │                              reward = tp − fp·C_fp − fn·0.3 + bonuses
policy_cache/{platform}.json (30-day TTL)    decision_package + grader_score returned
```

The threshold `θ*` is read by the LLM in DP1/DP2 prompts; the FP-penalty `C_fp` is paid in the terminal reward at SUBMIT. Both come from the same compile-time computation — they cannot drift apart.

### What's new in this Round 2 cut

- **9 actions** including `get_policy`, `reverse_image_search`, `analyze_bio`, `check_ip`. `openenv.yaml` action_schema mirrors all nine.
- **Per-step reward delta** is returned on every `/step` (not only at SUBMIT), so per-action shaping like the GET_POLICY bonus and tool penalties are immediately visible.
- **`visible_accounts` is populated for every visible id at reset**, with hidden signals at `0.0 / ""` until tools reveal them.
- **`StepResponse` carries top-level `decision_package` and `grader_score`** after SUBMIT — callers no longer need to grep the message.
- **Blind FLAG is denied at flag time with `−0.15`** when the agent has neither inspected the account nor used any tool on it.
- **Generic platform support**: the policy compiler uses the platform-agnostic Tavily query `"{platform} fake account content policy enforcement 2024 2025"` and falls back to a generic policy when no hardcoded entry exists.
- **Two-decision-point eval runner** (`eval-models/_round2_runner.py`) drives episodes deterministically and exposes the LLM only at DP1 (tool pick) and DP2 (flag/skip). Six thin model shims (qwen, gemma, deepseek, llama, mistral, nvidia) plug in HF-router or Bedrock backends through a shared `_llm_adapters.py`.
- **`bash check.sh`** runs a 12-step system check covering health, all 9 actions, reward shaping, and the decision package.

Full architecture, formula audit, end-to-end policy lifecycle, sanity rules, scoring math, tool contracts, eval-runner internals, and quickstart commands live in **[`reference.md`](reference.md)** (single source of truth).

## Theme

**SUPPORT**

### Customer Service Agents

Complex environment where agents resolve multi-step queries using external tools and APIs.

## Problem Statement

**The task:** A social network contains fake accounts organised into a single coordinated ring of 10. The ring behaves in a coordinated way — same posting hour, same IP subnet, stolen celebrity photos, copy-paste bios. The agent must find all 10 by navigating a limited step budget, inspecting accounts, and flagging suspects.

## Proposed Solution

An OpenEnv-compatible reinforcement learning environment where an LLM agent must identify all 10 members of a coordinated fake account ring hidden inside a synthetic social network. The agent learns via **Reflexion** and a **dynamic hybrid rule/LLM policy** — not via gradient updates or fine-tuning.

---
## Novelty Highlights

- **Adaptive Hybrid Intelligence (Rules + LLM):** Unlike static ensembles, GraphStrike dynamically blends deterministic rules and LLM reasoning using a trust gate, shifting control as performance improves.
- **Learning Without Fine-Tuning:** Instead of updating model weights, the agent learns through Reflexion lessons and best-trajectory memory injected into future prompts.
- **Graph-First Detection Pipeline:** Detection is not account-by-account only; it uses cascade effects, neighbor propagation, and multi-hop graph expansion to uncover coordinated rings.
- **Math-Grounded Decision Control:** Risk composition, trust calibration, and grader alignment are formula-driven, making behavior interpretable and reproducible.
- **Adversarial Evasion Benchmarking:** Hard-mode includes timed evasion events, so success reflects robustness under disruption rather than overfitting to static patterns.
- **Safety-Net by Design:** High-confidence rule overrides prevent catastrophic LLM errors while preserving LLM flexibility for strategic exploration.
---

## Performance Summary

We evaluate GraphStrike's hybrid rule/LLM policy across multiple *frontier models to measure how well each model handles the investigation task. All runs use
the same inference pipeline (`inference.py`) with identical system prompts and structured logging. Each model ran: (1) seed=0 on all 3 tasks, and
(2) seeds 0-2 on all 3 tasks for variance measurement.*

**Seed=0 scores (single episode per task):**

<p align="center">
  <img src="images/table1.png" alt="Model Performance Table" width="1600"/>
</p>
<br>

**3-seed variance scores (mean across seeds 0, 1, 2):**

<p align="center">
  <img src="images/table2.png" alt="Model Performance Table" width="1600"/>
</p>
<br>

 **Rule-Based Baseline (no LLM, deterministic)**

<p align="center">
  <img src="images/table3.png" alt="Model Performance Table" width="1600"/>
</p>
<br>

---
## Table of Contents

1. [What This Is](#1-what-this-is)
2. [The Problem: How Fake Detection Actually Works](#2-the-problem-how-fake-detection-actually-works)
3. [Synthetic Data Generation](#3-synthetic-data-generation)
4. [Data Model](#4-data-model)
5. [The RL Environment](#5-the-rl-environment)
6. [Risk Scoring Mathematics](#6-risk-scoring-mathematics)
8. [The LLM Policy (Qwen3 via Bedrock)](#8-the-llm-policy-qwen3-via-bedrock)
9. [Reflexion — How the Agent Learns](#9-reflexion--how-the-agent-learns)
10. [Hybrid Policy — The Novel Contribution](#10-hybrid-policy--the-novel-contribution)
11. [Training Loop End-to-End](#11-training-loop-end-to-end)
12. [API Reference](#12-api-reference)
13. [Docker Deployment](#13-docker-deployment)
14. [Submission Requirements](#14-submission-requirements)
15. [Verification & Validation](#15-verification--validation)

---

## 1. What is this !?

This is an **OpenEnv hackathon** submission. OpenEnv is a framework for building RL environments with a standard microservice interface (`/reset`, `/step`, `/state`) so that any agent implementation can plug in.

**What makes this non-trivial:**

- The network is large (50–1000 accounts depending on difficulty).
- Fake accounts are mixed with innocent high-signal "decoy" accounts.
- In hard mode, the gang actively evades — dropping intra-gang follows, renaming profiles — while the agent is mid-investigation.
- The agent cannot see the full network upfront: it must explore via INSPECT and INVESTIGATE_NETWORK actions, spending steps to reveal information.

**What makes the learning novel:**

- The LLM (inference via AWS Bedrock) cannot be fine-tuned — it is a black-box API.
- The agent learns via **Reflexion**: post-episode lessons are written back into memory and injected into every future prompt.
- A **dynamic hybrid policy** (α-weighted) blends the LLM with a deterministic rule engine, with the blend weight α updating based on recent win rate. Rules dominate early; the LLM takes over as it proves itself.

### System Architecture

![System Architecture](assets/sys%20arch.png)

---

## 2. The Problem: How Fake Detection Actually Works

A real-world fake account detector does **not** read post content. Detection relies on three categories of signals computed from metadata:

### Signal Hierarchy (Node -> Behavioral -> Graph)

![Signal Hierarchy](assets/gs.png)

- **Node signals (offline):** content fingerprints like photo reuse, bio-template similarity, and comment repetition provide the first suspicion layer.
- **Behavioral signals (temporal/device):** coordinated posting hour, account-age clustering, and shared IP subnet add stronger gang-level evidence.
- **Graph signals (live at INSPECT):** mutual follows, flagged-neighbor growth, and cluster alignment are hardest to evade, so they carry the highest weight in risk scoring.
- **False-positive control:** high-legitimacy hubs (for example celebrities) are down-weighted through hub-legitimacy discounting.

---

## 3. Synthetic Data Generation

**File:** `server/generator.py`

Episodes are generated deterministically by seed. 150 episodes are pre-generated (50 per task) and cached as JSON files in `episodes/`.

### Network Composition

| Task | Network size | Gang | Decoys | Real | Max steps |
|---|---|---|---|---|---|
| easy | 50 | 10 | 0 | 40 | 30 |
| medium | 200 | 10 | 20 | 170 | 50 |
| hard | 1000 | 10 | 50 | 940 | 80 |

- **Gang accounts:** All 10 share `base_age` (same creation week), tightly clustered `avg_post_hour`, high `photo_reuse_score`/`bio_template_score`, `comment_repeat_score` in [0.60, 0.90], `ip_cluster_id = "ip_gang_{seed}"`, and dense intra-gang follow edges (density 0.60–0.80).
- **Real accounts:** Log-normal follower distributions, unique IP clusters, low fake scores.
- **Decoy accounts** (medium/hard): Real accounts with elevated fraud scores (0.20–0.40 range) — they look suspicious but are NOT gang members and penalise reckless flagging.
- **Celebrity accounts** (2 per episode): 100k–5M followers, very low fake scores, high `hub_legitimacy_score`.
- **Zero-edge isolates** (2 per episode): No edges — test whether the agent wastes steps on disconnected nodes.

---

## 4. Data Model

**File:** `models.py`

### ActionType

| Value | Cost | Effect |
|---|---|---|
| `inspect` | 1 step | Reveals full `AccountProfile` + follow list |
| `investigate_network` | 2 steps | Expands 2 hops; reveals account IDs only |
| `flag` | 0 steps | Marks account as gang member; triggers SUSPECT cascade |
| `unflag` | 0 steps | Removes flag; clears CONFIRMED_FAKE status |
| `submit` | 0 steps | Ends episode; triggers scoring |

### AccountProfile — key fields

| Category | Fields |
|---|---|
| Raw counts | `follower_count`, `following_count`, `post_count` |
| Temporal | `avg_post_hour`, `account_age_days` |
| Content pipeline (0–1) | `photo_reuse_score`, `bio_template_score`, `comment_repeat_score` |
| IP/device | `shared_ip_count`, `ip_cluster_id` |
| Graph (live at INSPECT) | `mutual_follow_rate`, `flagged_neighbor_count`, `avg_neighbor_photo_reuse`, `post_hour_cluster_score` |
| Risk breakdown | `fake_risk_score`, `node_risk`, `behavior_risk`, `graph_risk`, `hub_legitimacy_score` |
| Evasion/status | `name_change_count`, `status` (NORMAL/SUSPECT/CONFIRMED_FAKE) |

### FakeGangObservation — what the agent sees each step

`done`, `reward`, `visible_accounts`, `visible_account_ids`, `flagged_ids`, `inspected_ids`, `suspect_ids`, `graph_edges`, `steps_remaining`, `evasion_triggered`, `evasion_count`, `task`, `message`

---

## 5. The RL Environment

**File:** `server/environment.py`

### Episode Lifecycle & Action Mechanics

![Episode Flow](assets/episode.png)

**FLAG cascade (dual):** When FLAG(X) is called — (1) every visible account that X follows becomes SUSPECT via the follow-graph, and (2) every visible account sharing X's `ip_cluster_id` becomes SUSPECT. Gang members share `ip_gang_{seed}`; real accounts have unique IPs → zero false positives.

### Reward Function

```
base_reward = tp×1.0 − fp×0.5 − fn×0.3

Win condition:
  easy/medium:  recall ≥ 0.8 AND precision ≥ 0.7
  hard:         recall ≥ 0.9 AND precision ≥ 0.8

Bonuses:
  +5.0   full win
  +3.0   perfect recall
  +2.0   partial win (high recall, low precision)
  +1.0   efficiency (SUBMIT with ≥50% steps remaining)
  −1.0   per evasion event (hard mode)
  −2.0   forced submit (ran out of steps)
```

### Evasion (hard mode)

- **`unfollow_intragang`:** 30% of intra-gang edges randomly removed mid-investigation — destroys graph signal. Fires 4 times (steps 15, 30, 45, 60).
- **`rename_count`:** Random gang members get `name_change_count += 1` — a visual evasion signal.

---

## 6. Risk Scoring Mathematics

**File:** `server/scoring.py` — all functions are stateless and deterministic.

![pipw](images/big.png)

![Risk Scoring Formulas 1](assets/formulas-1.png)

![Risk Scoring Formulas 2](assets/formulas-2.png)

---

## 8. The LLM Policy (Qwen3 via Bedrock)

**File:** `agent/policy.py`

**Model:** `qwen.qwen3-next-80b-a3b` via AWS Bedrock Converse API (`maxTokens=512, temperature=0.4`)

### Prompt Structure

Every step, the policy builds a prompt from three components:

```
[reflections from past episodes]       ← grows richer every episode
[best trajectory few-shot example]     ← best win ever, showing the full action log
━━━ CURRENT CASE ━━━
[formatted observation]                ← status badges, risk scores, suspect list
What is your next action?
```

Accounts in the observation are **sorted by `fake_risk_score` descending**, with status badges prepended. `fnbr=N(!)` highlights when `flagged_neighbor_count > 0`; `[HUB?]` warns the LLM not to flag high-legitimacy accounts.

### Required Response Format

```xml
<thinking>
Reasoning — which account is most suspicious and why.
</thinking>
<action>
INSPECT acc_0041
</action>
```

If parsing fails, a heuristic fallback inspects the highest-scored uninspected account. Retries use exponential backoff (1s, 2s, 4s) up to 3 attempts.

---

## 9. Reflexion — How the Agent Learns

**Files:** `agent/reflection.py`, `agent/memory.py`

The agent **cannot** update Qwen3's weights — Bedrock is a black-box API. Instead, it learns via **Reflexion**: post-episode lessons are written as text and injected into future prompts.

### Reflexion Learning Loop

![Reflexion Learning Loop](assets/reflexion.png)



```
Episode N:
  1. LLM acts using: system_prompt + reflections[last 4] + best_trajectory
  2. Episode ends → WIN or LOSS
  3. Post-episode:
     LOSS → generate_reflection(action_log, outcome) → lesson stored
     WIN  → save trajectory if better reward + generate_success_reflection

Episode N+1:
  → last 4 reflections + best win trajectory injected into prompt
  → LLM has learned from its past
```

**Example generated reflection:**
> *"The starting accounts were all real; I wasted 8 steps inspecting low-signal nodes before pivoting. When photo_reuse and bio_template are both below 0.3 after 3 inspections, immediately use INVESTIGATE_NETWORK to jump to a different graph region."*

All memory persists in a Docker volume (`memory/`) across container restarts — reflections, best trajectories, win history, and α values per task.

---

## 10. Hybrid Policy — The Novel Contribution

**File:** `agent/hybrid_policy.py`

**Key insight:** A new LLM agent starts dumb but improves over time. A rule engine is always consistent but cannot adapt. The hybrid policy exploits both — rules provide a safety net early while the LLM builds its track record; once the LLM proves itself, rules step back.

### Architecture

![Hybrid Policy Architecture](assets/hybrid.png)

### Alpha (α): The Trust Weight

α is a per-task value in [0.20, cap] representing current trust in the LLM:

```
reflection_factor = min(1.0, n_reflections / 4.0)
raw = 0.20 + reflection_factor × (0.80 × recent_win_rate + 0.12)
α = clamp(raw, 0.20, cap)
```

| Task | α cap | Rationale |
|---|---|---|
| easy | 0.50 | Rule engine alone achieves ~91% — LLM should assist, not override |
| medium | 0.70 | Decoys require some LLM judgment, but cascade must stay |
| hard | 0.85 | LLM needs latitude for evasion adaptation, but safety rules remain |

**Alpha trajectory over training (easy task, cap=0.50):**

| Episode | Win rate | Reflections | α (capped) |
|---|---|---|---|
| 1 | 0% | 0 | 0.20 |
| 5 | 20% | 4 | 0.48 |
| 10 | 50% | 9 | **0.50** |
| 20 | 80% | 19 | **0.50** |

<br>

![System Architecture](images/plot.png)

### Rule Confidence Levels

| Situation | Action | Confidence |
|---|---|---|
| Steps remaining = 0 | SUBMIT | 1.00 |
| Uninspected SUSPECT accounts exist | INSPECT suspects[0] | 0.95 |
| `fake_risk ≥ 0.85` | FLAG that account | 0.95 |
| `fake_risk` in [threshold, 0.85) | FLAG that account | 0.70+ |
| 10 accounts already flagged | SUBMIT | 0.85 |
| Steps remaining ≤ 3 | SUBMIT | 0.90 |
| Uninspected accounts available | INSPECT top candidate | 0.30 |

At **α=0.20** (early): rules dominate (~90% of decisions). At **α=0.50** (moderate): LLM controls exploration; rules control safety. At **α=0.85** (high): LLM controls most decisions; rules only override forced submits and uninspected suspects.

α is saved to `memory/alpha_{task}.json` and persists across Docker restarts — the agent doesn't reset to 0.20 every time.

---

## 11. Training Loop End-to-End

**File:** `train.py`

### Curriculum

| Phase | Episodes | Task | Goal |
|---|---|---|---|
| 1 | 1–20 | easy | Learn basic signal thresholds, build first reflections |
| 2 | 21–35 | medium | Handle decoys, learn evasion response |
| 3 | 36–50 | hard | Feature-only detection, persistent evasion |

Seeds rotate deterministically: `seed = (episode_num + task_offset) % 50`

### Per-Episode Flow

```
for ep in range(n_episodes):

  1. DETERMINE TASK      curriculum_task(ep) or fixed task
  2. COMPUTE ALPHA       compute_alpha(win_rate, n_reflections, task)
  3. LOAD CONTEXT        last 4 reflections + best win trajectory
  4. RUN EPISODE         while not obs.done:
                           blend(rule_action, llm_action, rule_conf, α)
                           → obs = env.step(final)
  5. POST-EPISODE        record_win → update α → generate reflection
  6. LOG                 task | win/loss | reward | recall | precision | α | modes
```

Episode metrics (flushed to `runs/metrics.jsonl` every 5 episodes) include: `episode`, `task`, `won`, `reward`, `recall`, `precision`, `steps_used`, `alpha_used`, `mode_agree`, `mode_rule`, `mode_llm`, `n_reflections_used`.

You can watch the transition: early episodes have high `rule` counts; later episodes have high `agree` counts (LLM learned to make the same decisions as the rules, but also brings strategic reasoning the rules can't).

---

## 12. API Reference

**File:** `server/app.py`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{"status": "healthy"}` |
| `/tasks` | GET | Task list + `action_schema` + `score_range: [0.0, 1.0]` |
| `/reset` | POST | Accepts `{task, seed}` → returns initial observation |
| `/step` | POST | Accepts any `FakeGangAction` → returns updated observation |
| `/state` | GET | Current episode metadata (step count, task, score) |
| `/grader` | GET | Normalised [0.0, 1.0] score after SUBMIT |
| `/baseline` | POST | Runs rule-based agent on all 3 tasks, returns scores |

**Baseline performance:**

| Task | Seed=0 score | Win rate (50 seeds) | Mean score (50 seeds) |
|---|---|---|---|
| easy | 0.91 | 100% | ~0.91 |
| medium | 0.906 | 84% | ~0.77 |
| hard | 0.9038 | 52% | ~0.47 |

---

## 13. Docker Deployment

```bash
# Build
docker build -f server/Dockerfile -t graphstrike .

# Run
docker run -it \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -v $(pwd)/memory:/app/memory \
  -v $(pwd)/runs:/app/runs \
  -p 8000:8000 \
  graphstrike
```

The `memory/` and `runs/` volumes preserve all learning between container restarts.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | (required) | For Bedrock/Qwen3 access |
| `AWS_SECRET_ACCESS_KEY` | (required) | For Bedrock/Qwen3 access |
| `AWS_DEFAULT_REGION` | `us-east-1` | Bedrock region |
| `TRAIN_TASK` | (curriculum) | Fix to `easy`/`medium`/`hard` |
| `TRAIN_EPISODES` | `50` | Total training episodes |
| `TRAIN_TEMP` | `0.4` | LLM sampling temperature |
| `TRAIN_VERBOSE` | `0` | Set `1` for per-step action logging |
| `SERVER_PORT` | `8000` | FastAPI port |

### Startup Sequence (`run.sh`)

```
1. Validate AWS credentials
2. python server/generator.py    → generates 150 episode JSON files
3. uvicorn server.app:app        → starts the environment server
4. Health check polling          → waits until /health responds
5. python train.py               → runs the full training loop
```

---


### Full HTTP validation

```bash
python3 -m uvicorn server.app:app --port 8001 &
sleep 3
python3 validate.py --url http://localhost:8001
# Expected: Results: 24/24 passed — all OK
```

### Deployed Endpoint Verification

```bash
curl https://pandago-graphstrike.hf.space/health
# → {"status": "healthy"}

curl https://pandago-graphstrike.hf.space/tasks
# → {"tasks": ["easy","medium","hard"], "action_schema": {...}, "score_range": [0.0, 1.0]}

curl -X POST https://pandago-graphstrike.hf.space/baseline
# → {"scores": {"easy": 0.91, "medium": 0.906, "hard": 0.9038}, "agent": "rule_based"}
```

---

![Material wave loading](https://github.com/user-attachments/assets/a08255eb-9647-471d-9881-61871332249f)

## Developed with ❤️ by Team ComputeXOR

### {

### [Sai Nivedh](https://github.com/SaiNivedh26) ,

### [Charuvarthan](https://github.com/Charuvarthan-T) ,

### [Sajeev](https://github.com/SajeevSenthil)

### }
