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
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from training.prompts import build_chat, SYSTEM_PROMPT  # noqa: E402
from training.rollout import HFPolicy, rollout_batch, reward_for_tuple  # noqa: E402


PHASE_CONFIG = {
    "phase0": dict(steps=0,    eps_per_step=8, lr=0.0,  num_gen=2, desc="baseline only, no training"),
    "phase1": dict(steps=10,   eps_per_step=1, lr=1e-6, num_gen=2, desc="smoke: gradients flow, no NaN"),
    "phase2": dict(steps=50,   eps_per_step=4, lr=1e-6, num_gen=4, desc="signal: reward should trend up"),
    "phase3": dict(steps=1000, eps_per_step=8, lr=5e-7, num_gen=4, desc="full run"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=list(PHASE_CONFIG), required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--base-url", default=os.getenv("API_BASE_URL_ENV", "http://localhost:8000"))
    p.add_argument("--platform", default="Instagram")
    p.add_argument("--tasks", nargs="+", default=["easy", "medium"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--out-dir", default=str(_HERE / "runs"))
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--num-generations", type=int, default=None,
                   help="GRPO group size (default: from PHASE_CONFIG per phase)")
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


def train(args):
    """Phases 1/2/3: GRPO training loop using TRL."""
    cfg = PHASE_CONFIG[args.phase]
    print(f"[{args.phase}] {cfg['desc']} | steps={cfg['steps']} eps/step={cfg['eps_per_step']} lr={cfg['lr']}")

    # Lazy imports — heavy deps only loaded when actually training.
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Build a prompt distribution by running one heuristic-teacher pass.
    # GRPO will resample completions from the trainee for each prompt.
    from training.build_dataset import heuristic_teacher
    seed_rows = rollout_batch(
        heuristic_teacher(),
        base_url=args.base_url,
        platform=args.platform,
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
    # Per-prompt metadata threaded through to the reward fn via TRL's
    # extra-column kwarg passthrough. Required to replay the correct turn.
    ds = Dataset.from_dict({
        "prompt":         prompts,
        "decision_type":  [r["decision_type"]  for r in seed_rows],
        "decision_index": [r["decision_index"] for r in seed_rows],
        "task":           [r["task"]           for r in seed_rows],
        "seed":           [r["seed"]           for r in seed_rows],
        "platform":       [r["platform"] or args.platform for r in seed_rows],
    })

    from training.grounded_reward import score_completion

    def reward_fn(prompts, completions, **kw):
        """Env-grounded GRPO reward. For each generated completion, replay one
        episode through the env injecting this completion at the matching turn,
        and compose the three-component reward.

        TRL passes extra dataset columns as parallel lists in **kw.
        """
        decisions = kw.get("decision_type", ["dp1"] * len(completions))
        idxs      = kw.get("decision_index", [0] * len(completions))
        tasks     = kw.get("task",     ["easy"] * len(completions))
        seeds     = kw.get("seed",     [0] * len(completions))
        plats     = kw.get("platform", [args.platform] * len(completions))
        out = []
        for comp, dt, di, tk, sd, pl in zip(completions, decisions, idxs, tasks, seeds, plats):
            try:
                rb, _ = score_completion(
                    base_url=args.base_url, platform=pl, task=tk, seed=int(sd),
                    decision_index=int(di), decision_type=dt, completion=comp,
                )
                out.append(rb.total)
            except Exception as e:
                # Don't crash the trainer on a flaky env call — penalize and log.
                print(f"  [reward_fn] env error: {e}")
                out.append(0.0)
        return out

    # num_gen comes from PHASE_CONFIG so phase1 always uses 2.
    # TRL's find_executable_batch_size may halve per_device_train_batch_size at
    # runtime; setting num_generations <= that halved floor prevents the
    # "generation_batch_size must be divisible by num_generations" error.
    num_gen = args.num_generations or cfg["num_gen"]
    grpo_cfg = GRPOConfig(
        output_dir=str(Path(args.out_dir) / args.phase),
        per_device_train_batch_size=num_gen,
        num_generations=num_gen,
        max_steps=cfg["steps"],
        learning_rate=cfg["lr"],
        logging_steps=1,
        save_steps=max(1, int(cfg["steps"]) // 4) if int(cfg["steps"]) else 1,
        report_to=["wandb"] if args.wandb_project else [],
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model()


def main():
    args = parse_args()
    if args.phase == "phase0":
        baseline_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
