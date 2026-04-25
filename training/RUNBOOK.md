# GRPO training runbook

The training Space is **GPU-only**. The env runs on a separate Space and is hit
over HTTP — we never run the env server inside the training container.

- env Space (HTTP API):  `https://pandago-graphstrike-model-training.hf.space`
- training Space (this): created by `training/deploy_hf.sh`

## 0. Local prereqs (one-time)

```bash
pip install -r training/requirements.txt
huggingface-cli login        # for hf upload + gated models
wandb login                  # optional but recommended
```

## 1. Local smoke test (do this BEFORE deploying)

```bash
./training/test_local.sh                # stages 1-3, ~30s
./training/test_local.sh --with-grpo    # stages 1-4, ~5min, needs torch
```

What the four stages prove:

| Stage | Proves                                                        |
|-------|---------------------------------------------------------------|
| 1     | `parse.py` + `rewards.py` are correct on golden inputs        |
| 2     | The remote env Space is reachable + has the expected schema   |
| 3     | The refactored runner emits per-decision tuples end-to-end    |
| 4     | TRL `GRPOTrainer` boots and the env-grounded reward fn returns |

If stage 2 fails, the env Space is sleeping — open it once in the browser to
wake it, then re-run.

## 2. Phase 0 — baseline (no training)

Local, against the remote env, with the real model:

```bash
python -m training.train_grpo --phase phase0 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url https://pandago-graphstrike-model-training.hf.space \
    --platform Instagram --tasks easy medium --seeds 0 1 2 3 4 5 6 7
```

**Save these two numbers** — phase 2's gate references them:
- `format_ok_rate` (phase 1 gate floor)
- `mean_grader`    (phase 2 gate floor)

## 3. Phases 1-3 — running ON the GPU Space

The training Space starts automatically when it boots. You set a Space
**variable** to pick which phase, then push or restart:

| variable        | value                                    |
|-----------------|------------------------------------------|
| `PHASE`         | `phase0` / `phase1` / `phase2` / `phase3`|
| `MODEL`         | `Qwen/Qwen2.5-1.5B-Instruct`             |
| `PLATFORM`      | `Instagram`                              |
| `ENV_BASE_URL`  | `https://pandago-graphstrike-model-training.hf.space` |
| `WANDB_PROJECT` | `fakegang-grpo`                          |
| `PUSH_TO_HUB`   | `1` to push checkpoint after training    |
| `PUSH_REPO_ID`  | `<user>/fakegang-grpo-ckpt`              |
| `WANDB_API_KEY` | (secret)                                 |
| `HF_TOKEN`      | (secret, needed for gated models / push) |

`entrypoint.sh` will:
1. Probe `${ENV_BASE_URL}/health` (60s budget) — fails fast if unreachable.
2. Run `python -m training.train_grpo --phase $PHASE --model $MODEL ...`.
3. Optionally `hf upload` the checkpoint dir.
4. `tail -f /dev/null` to keep the Space alive for log inspection.

**Switching phases**: change `PHASE`, then **Settings → Restart Space**.

**W&B watchlist:**
- phase 1 (10 steps):  `train/loss` finite; `train/kl` < 0.1; format rate stable.
- phase 2 (50 steps):  `train/reward` trends up; entropy not collapsing (>~1.0).
- phase 3 (1000 steps): mean reward keeps climbing; held-out grader > teacher.

## 4. Held-out evaluation

Run from your laptop against the remote env, pointing at the trained ckpt:

```bash
python -m training.eval --gate phase2 \
    --model <user>/fakegang-grpo-ckpt \
    --base-url https://pandago-graphstrike-model-training.hf.space \
    --platform LinkedIn --seeds 100 101 102 103 104 \
    --baseline-grader <PHASE_0_MEAN_GRADER>
```

LinkedIn (θ\*=0.167) is the recommended held-out platform — it sits between X
(0.091) and Instagram (0.553). TikTok came out identical to X so it's a poor
generalization test.

## 5. Deploy / re-deploy the training Space

```bash
./training/deploy_hf.sh <hf-username> fakegang-grpo a10g-small
```

The script:
1. Creates the Space (Docker SDK) — idempotent.
2. Copies our Dockerfile to repo root (HF requires it there).
3. Writes a Space `README.md` with `sdk: docker`.
4. Uploads the working copy (excludes `runs/`, `__pycache__/`, `eval-models/results/`).
5. Requests `a10g-small` GPU.

Pause when not training (stops billing):

```bash
python -c "from huggingface_hub import HfApi; HfApi().pause_space('<user>/fakegang-grpo')"
```

## 6. Env-grounded reward (how it works)

For each generated completion, the reward fn:
1. Looks up the source `(task, seed, decision_index, decision_type)` from the
   row metadata that TRL passes through as kwargs.
2. Calls `training.grounded_reward.score_completion`, which replays the entire
   episode through the **remote env** using the heuristic teacher for every
   turn EXCEPT turn `decision_index`, where it injects the candidate completion.
3. Returns `1.0·grader + 0.3·clip(step_reward·0.1, ±0.5) + 0.2·format_ok`.

Cost: each completion = 1 env episode. With group size K=4 and 4 prompts/step,
that's 16 episodes/optimizer-step. Phase 2 (50 steps × 16 = 800 episodes)
takes ~10–20 min wall clock against an A10G-small. Phase 3 (1000 steps) is
several hours — plan accordingly.
