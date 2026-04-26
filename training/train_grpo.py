"""GRPO training entry point for the fake-gang policy.

Phases (sanity gate):
  phase0  collect baseline rollouts only (no training)
  phase1  smoke    — 10 optimizer steps, 1 episode/step, learning_rate=1e-6
  phase2  signal   — 50 optimizer steps, 4 episodes/step, learning_rate=1e-6
  phase3  full     — 1000 steps, 8 episodes/step, learning_rate=5e-7

The TRL `GRPOTrainer` API is used. Each "prompt" comes from the env (via the
runner returning per-decision tuples); rewards are computed by `rewards.py`.

Usage:
  python -m training.train_grpo --phase phase1 \
         --model Qwen/Qwen2.5-1.5B-Instruct \
         --base-url http://localhost:8000 --platform Instagram \
         --wandb-project fakegang-grpo

Note: this is a scaffold. The TRL GRPOTrainer integration here uses a custom
reward_fn that re-runs an episode per generated batch and maps completions →
scalar rewards. Adapt to your TRL version (tested against trl>=0.11).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from training.prompts import build_chat  # noqa: E402
from training.rollout import HFPolicy, rollout_batch, rollout_batch_multiplatform, reward_for_tuple  # noqa: E402


PHASE_CONFIG = {
    "phase0": dict(steps=0,    eps_per_step=8, lr=0.0,  num_gen=2, desc="baseline only, no training"),
    "phase1": dict(steps=10,   eps_per_step=1, lr=1e-6, num_gen=2, desc="smoke: gradients flow, no NaN"),
    "phase2": dict(steps=25,   eps_per_step=4, lr=1e-6, num_gen=2, desc="signal: reward should trend up"),
    "phase3": dict(steps=26,   eps_per_step=6, lr=1e-6, num_gen=4, desc="multi-platform demo run"),
}

TRAIN_PLATFORMS = ["Instagram", "X", "Snapchat"]
EVAL_PLATFORM   = "LinkedIn"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=list(PHASE_CONFIG), required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--base-url", default=os.getenv("API_BASE_URL_ENV", "http://localhost:8000"))
    # Single-platform mode (phase0/1/2). Ignored in phase3 (multi-platform).
    p.add_argument("--platform", default="Instagram")
    # Multi-platform overrides (phase3).
    p.add_argument("--platforms", nargs="+", default=None,
                   help="Training platforms (phase3). Defaults to Instagram,X,Snapchat")
    p.add_argument("--eval-platform", default=EVAL_PLATFORM,
                   help="Held-out eval platform (default: LinkedIn)")
    p.add_argument("--tasks", nargs="+", default=["easy", "medium"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    p.add_argument("--out-dir", default=str(_HERE / "runs"))
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--num-generations", type=int, default=None,
                   help="GRPO group size (default: from PHASE_CONFIG)")
    return p.parse_args()


def baseline_only(args):
    """Phase 0: just collect rollouts with the un-trained policy and report metrics."""
    print(f"[phase0] loading policy {args.model}")
    policy = HFPolicy(args.model)
    rows = rollout_batch(
        policy.generate,
        base_url=args.base_url,
        platform=args.platform,
        tasks=args.tasks,
        seeds=args.seeds,
        model_tag=f"baseline:{args.model}",
    )
    if not rows:
        print("[phase0] no rows produced — env unreachable?")
        return
    n = len(rows)
    fmt = sum(r["format_ok"] for r in rows) / n
    grader = sum(r["r_grader"] for r in rows) / n
    total = sum(r["r_total"] for r in rows) / n
    print(f"[phase0] n={n} format_ok_rate={fmt:.3f} mean_grader={grader:.3f} mean_r_total={total:.3f}")

    out = Path(args.out_dir) / "phase0_rollouts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[phase0] wrote {out}")


def _build_seed_dataset(args, platforms, tok):
    """Run heuristic teacher across all training platforms, return HF Dataset."""
    from datasets import Dataset
    from training.build_dataset import heuristic_teacher

    if len(platforms) > 1:
        seed_rows = rollout_batch_multiplatform(
            heuristic_teacher(),
            base_url=args.base_url,
            platforms=platforms,
            tasks=args.tasks,
            seeds=args.seeds,
            model_tag="seed:heuristic",
        )
    else:
        seed_rows = rollout_batch(
            heuristic_teacher(),
            base_url=args.base_url,
            platform=platforms[0],
            tasks=args.tasks,
            seeds=args.seeds,
            model_tag="seed:heuristic",
        )
    if not seed_rows:
        raise SystemExit("Seed rollouts produced 0 rows — is the env server running?")

    prompts = [
        tok.apply_chat_template(build_chat(r["prompt"]), tokenize=False, add_generation_prompt=True)
        for r in seed_rows
    ]
    return Dataset.from_dict({
        "prompt":         prompts,
        "decision_type":  [r["decision_type"]  for r in seed_rows],
        "decision_index": [r["decision_index"] for r in seed_rows],
        "task":           [r["task"]           for r in seed_rows],
        "seed":           [str(r["seed"])      for r in seed_rows],
        "platform":       [r.get("platform") or platforms[0] for r in seed_rows],
    })


def evaluate_linkedin(trainer, tok, args):
    """Zero-shot evaluation on held-out platform (LinkedIn) after training."""
    import json
    from training.rollout import rollout_batch

    eval_plat = args.eval_platform
    print(f"\n[eval] zero-shot evaluation on held-out platform: {eval_plat}")

    policy = HFPolicy(args.model, temperature=0.0)
    # Point to the trained checkpoint weights if available
    ckpt_dir = str(Path(args.out_dir) / args.phase)
    if Path(ckpt_dir).exists():
        try:
            policy._tok   = trainer.processing_class
            policy._model = trainer.model
            policy._model.eval()
        except Exception:
            pass  # fall back to original model if patching fails

    rows = rollout_batch(
        policy.generate,
        base_url=args.base_url,
        platform=eval_plat,
        tasks=args.tasks,
        seeds=[100, 101, 102, 103, 104],
        model_tag=f"eval:{eval_plat}",
    )
    if not rows:
        print(f"[eval] no rows for {eval_plat} — skipping")
        return

    n = len(rows)
    fmt    = sum(r["format_ok"] for r in rows) / n
    grader = sum(r["r_grader"]  for r in rows) / n
    total  = sum(r["r_total"]   for r in rows) / n
    print(f"[eval] {eval_plat} | n={n} format_ok={fmt:.3f} grader={grader:.3f} r_total={total:.3f}")

    try:
        import wandb
        if wandb.run:
            wandb.log({
                f"eval/{eval_plat}/format_ok":  fmt,
                f"eval/{eval_plat}/grader":     grader,
                f"eval/{eval_plat}/r_total":    total,
            })
    except Exception:
        pass

    out = Path(args.out_dir) / f"eval_{eval_plat.lower()}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[eval] wrote {out}")


def train(args):
    """Phases 1/2/3: GRPO training loop using TRL."""
    cfg = PHASE_CONFIG[args.phase]

    # Phase3 uses all training platforms; earlier phases use single platform.
    if args.phase == "phase3":
        platforms = args.platforms or TRAIN_PLATFORMS
    else:
        platforms = [args.platform]

    print(f"[{args.phase}] {cfg['desc']} | steps={cfg['steps']} lr={cfg['lr']} platforms={platforms}")

    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = _build_seed_dataset(args, platforms, tok)
    print(f"[{args.phase}] seed dataset: {len(ds)} prompts across {platforms}")

    from training.grounded_reward import score_completion
    default_platform = platforms[0]

    # Consecutive all-zero-reward step counter. Shared via closure.
    _zero_streak = {"n": 0}
    _ZERO_WARN   = 5   # warn after this many consecutive dead steps
    _ZERO_ABORT  = 20  # give up if env has been down this long

    def reward_fn(prompts, completions, **kw):
        try:
            decisions = kw.get("decision_type", ["dp1"] * len(completions))
            idxs      = kw.get("decision_index", [0]    * len(completions))
            tasks     = kw.get("task",           ["easy"]* len(completions))
            seeds     = kw.get("seed",           [0]    * len(completions))
            plats     = kw.get("platform", [default_platform] * len(completions))
            out = []
            n_zero = 0
            for i, (comp, dt, di, tk, sd, pl) in enumerate(
                    zip(completions, decisions, idxs, tasks, seeds, plats)):
                if i > 0:
                    time.sleep(1.0)
                rb, dbg = score_completion(
                    base_url=args.base_url, platform=pl, task=tk, seed=int(sd),
                    decision_index=int(di), decision_type=dt, completion=comp,
                )
                out.append(rb.total)
                if dbg.get("timeout") or rb.total == 0.0:
                    n_zero += 1
            if n_zero:
                print(f"  [reward_fn] {n_zero}/{len(completions)} zero-reward this step")

            # Tiny jitter on non-zero rewards so identical episodes don't collapse
            # to std=0 and kill the GRPO gradient signal (frac_reward_zero_std=1).
            out = [r + random.gauss(0, 0.005) if r > 0 else r for r in out]

            # Track consecutive dead steps.
            if all(r == 0.0 for r in out):
                _zero_streak["n"] += 1
                streak = _zero_streak["n"]
                if streak == _ZERO_WARN:
                    print(f"  [reward_fn] WARNING: {streak} consecutive all-zero steps — env may be down")
                elif streak >= _ZERO_ABORT:
                    raise RuntimeError(
                        f"Env appears down: {streak} consecutive all-zero reward steps. "
                        "Check env Space health and restart training."
                    )
            else:
                _zero_streak["n"] = 0  # reset on any non-zero step

            return out
        except RuntimeError:
            raise  # let the abort propagate — it's intentional
        except Exception as e:
            print(f"  [reward_fn] fatal error (returning all zeros): {e}")
            _zero_streak["n"] += 1
            return [0.0] * len(completions)

    num_gen = args.num_generations or cfg["num_gen"]
    grpo_cfg = GRPOConfig(
        output_dir=str(Path(args.out_dir) / args.phase),
        per_device_train_batch_size=num_gen,
        num_generations=num_gen,
        gradient_accumulation_steps=2,
        max_steps=cfg["steps"],
        learning_rate=cfg["lr"],
        logging_steps=1,
        save_steps=max(int(cfg["steps"]) // 2, 1),  # checkpoint at halfway + end
        save_total_limit=1,                           # keep only the latest checkpoint
        report_to=["wandb"] if args.wandb_project else [],
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_completion_length=128,
        temperature=0.9,
        top_p=0.95,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model()

    # Held-out LinkedIn eval — runs automatically after phase3 training.
    if args.phase == "phase3":
        evaluate_linkedin(trainer, tok, args)


def main():
    args = parse_args()
    if args.phase == "phase0":
        baseline_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
