"""Main training loop for the Fake Gang Detection LLM Agent.

Learning mechanism (Reflexion + Episodic Memory):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Episode N                                                          │
  │    1. LLM receives: system_prompt + reflections + few_shot + obs   │
  │    2. LLM reasons and outputs one action                           │
  │    3. Action is sent to OpenEnv HTTP server → new obs + reward     │
  │    4. Repeat until done                                             │
  │    5. Post-episode:                                                 │
  │       • If FAIL → Qwen generates a reflection → stored to disk     │
  │       • If WIN  → trajectory saved as few-shot example             │
  │    6. Both are injected into Episode N+1's prompt → better policy  │
  └─────────────────────────────────────────────────────────────────────┘

Curriculum:
  Phase 1 (episodes 1-20) : easy   — learn basic signal detection
  Phase 2 (episodes 21-35): medium — learn to handle evasion
  Phase 3 (episodes 36-50): hard   — feature-only detection

Usage:
  python train.py                         # defaults: 50 episodes, curriculum
  python train.py --task easy --episodes 20
  python train.py --env-url http://localhost:8000 --episodes 50 --log-dir runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve imports
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "server"))

from client import FakeGangEnvClient, StepResult
from models import FakeGangAction, FakeGangObservation, ActionType
from agent.memory import AgentMemory
from agent.hybrid_policy import get_hybrid_action, compute_alpha
from agent.reflection import generate_reflection, generate_success_reflection

# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------

def _curriculum_task(episode_num: int, override_task: Optional[str]) -> str:
    if override_task:
        return override_task
    if episode_num < 20:
        return "easy"
    if episode_num < 35:
        return "medium"
    return "hard"


def _episode_seed(episode_num: int, task: str) -> int:
    """Rotate through pre-generated seeds so each task sees varied episodes."""
    offsets = {"easy": 0, "medium": 50, "hard": 100}
    return (episode_num + offsets.get(task, 0)) % 50


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: FakeGangEnvClient,
    task: str,
    seed: int,
    memory: AgentMemory,
    episode_num: int,
    alpha: float = 0.20,
    temperature: float = 0.4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one full episode using the hybrid policy.  Returns a metrics dict.

    alpha — current LLM trust weight (0.20 = rules dominate, 1.00 = pure LLM).
    """
    result = env.reset(task=task, seed=seed)
    obs: FakeGangObservation = result.observation

    # Fetch learning context from memory
    reflections = memory.get_reflections(task, n=4)
    few_shot = memory.get_best_trajectory(task)

    action_log: List[str] = []
    mode_log: List[str] = []
    step_num = 0

    while not obs.done:
        action, raw_llm, mode = get_hybrid_action(
            obs=obs,
            reflections=reflections,
            few_shot_example=few_shot,
            alpha=alpha,
            temperature=temperature,
        )

        # Build a human-readable action string for the log
        action_str = action.action_type.value.upper()
        if action.account_id:
            action_str += f" {action.account_id}"
        action_log.append(action_str)
        mode_log.append(mode)

        if verbose:
            mode_tag = mode.split("(")[0]  # strip params for brevity
            print(f"    Step {step_num+1:2d}: {action_str:35s} [{mode_tag}]")

        result = env.step(action)
        obs = result.observation
        step_num += 1

        if obs.done:
            break

    # Parse final message for TP/FP/FN
    final_msg = obs.message
    won = "[WIN]" in final_msg
    reward = result.reward if result.reward is not None else obs.reward or 0.0
    steps_used = ({"easy": 30, "medium": 50, "hard": 80}.get(task, 30)) - obs.steps_remaining

    # Extract recall/precision from message if present
    recall = _extract_float(final_msg, "Recall=")
    precision = _extract_float(final_msg, "Precision=")

    # Summarise mode distribution for the episode
    agree_count = sum(1 for m in mode_log if m == "agree")
    rule_count = sum(1 for m in mode_log if m.startswith("rule_override"))
    llm_count = sum(1 for m in mode_log if m.startswith("llm"))

    return {
        "episode": episode_num,
        "task": task,
        "seed": seed,
        "won": won,
        "reward": round(reward, 3),
        "steps_used": steps_used,
        "recall": recall,
        "precision": precision,
        "action_log": action_log,
        "final_message": final_msg,
        "n_reflections_used": len(reflections),
        "had_few_shot": few_shot is not None,
        "alpha_used": round(alpha, 3),
        "mode_agree": agree_count,
        "mode_rule": rule_count,
        "mode_llm": llm_count,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _extract_float(text: str, key: str) -> float:
    import re
    m = re.search(re.escape(key) + r"([0-9.]+)", text)
    return float(m.group(1)) if m else 0.0


# ---------------------------------------------------------------------------
# Learning step (post-episode)
# ---------------------------------------------------------------------------

def learning_step(
    metrics: Dict[str, Any],
    memory: AgentMemory,
) -> str:
    """Generate reflection or save trajectory. Returns a short status string."""
    task = metrics["task"]
    won = metrics["won"]
    action_log = metrics["action_log"]
    final_msg = metrics["final_message"]
    steps_used = metrics["steps_used"]
    max_steps = {"easy": 30, "medium": 50, "hard": 80}.get(task, 30)
    ep = metrics["episode"]

    if won:
        # Save trajectory as few-shot example
        saved = memory.add_trajectory(
            task=task,
            action_log=action_log,
            final_message=final_msg,
            reward=metrics["reward"],
            episode_num=ep,
        )
        # Also generate a success reflection occasionally
        if saved or memory.reflection_count(task) == 0:
            ref = generate_success_reflection(task, action_log, final_msg, steps_used, max_steps, ep)
            memory.add_reflection(task, ref, ep, metrics["reward"])
            return f"new best trajectory saved + success reflection generated"
        return "trajectory kept (not better than current best)"
    else:
        # Generate failure reflection
        ref = generate_reflection(
            task=task,
            action_log=action_log,
            final_message=final_msg,
            won=False,
            steps_used=steps_used,
            max_steps=max_steps,
            episode_num=ep,
        )
        memory.add_reflection(task, ref, ep, metrics["reward"])
        return f"reflection generated: \"{ref[:80]}…\""


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self) -> None:
        self.by_task: Dict[str, List[Dict]] = defaultdict(list)
        self.all: List[Dict] = []

    def record(self, m: Dict[str, Any]) -> None:
        self.by_task[m["task"]].append(m)
        self.all.append(m)

    def win_rate(self, task: Optional[str] = None, last_n: int = 10) -> float:
        records = self.by_task[task] if task else self.all
        if not records:
            return 0.0
        window = records[-last_n:]
        return sum(1 for r in window if r["won"]) / len(window)

    def print_episode(self, m: Dict[str, Any], learn_status: str) -> None:
        wr = self.win_rate(m["task"], last_n=10)
        tag = "WIN " if m["won"] else "LOSS"
        alpha = m.get("alpha_used", 0.20)
        agree = m.get("mode_agree", 0)
        rule = m.get("mode_rule", 0)
        llm = m.get("mode_llm", 0)
        total_steps = agree + rule + llm
        mode_str = (
            f"agree={agree} rule={rule} llm={llm}"
            if total_steps > 0
            else "n/a"
        )
        print(
            f"  Ep {m['episode']:3d} | {m['task']:6s} | {tag} | "
            f"reward={m['reward']:+7.2f} | "
            f"recall={m['recall']:.2f} prec={m['precision']:.2f} | "
            f"steps={m['steps_used']:2d} | "
            f"wr={wr:.0%} | α={alpha:.2f} | {mode_str}"
        )
        print(f"         └─ learn: {learn_status}")

    def print_summary(self, phase: str) -> None:
        print(f"\n{'━'*70}")
        print(f"  {phase} SUMMARY")
        print(f"{'━'*70}")
        for task in ["easy", "medium", "hard"]:
            records = self.by_task[task]
            if not records:
                continue
            wins = sum(1 for r in records if r["won"])
            avg_r = sum(r["reward"] for r in records) / len(records)
            avg_recall = sum(r["recall"] for r in records) / len(records)
            print(
                f"  {task:6s}: {wins}/{len(records)} wins ({100*wins/len(records):.0f}%) | "
                f"avg reward={avg_r:+.2f} | avg recall={avg_recall:.2f}"
            )
        total = len(self.all)
        total_wins = sum(1 for r in self.all if r["won"])
        if total:
            print(f"  {'TOTAL':6s}: {total_wins}/{total} wins ({100*total_wins/total:.0f}%)")
        print(f"{'━'*70}\n")


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------

def save_metrics(metrics_list: List[Dict], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "metrics.jsonl"
    with open(path, "a") as f:
        for m in metrics_list:
            f.write(json.dumps(m) + "\n")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    env_url: str = "http://localhost:8000",
    task: Optional[str] = None,
    n_episodes: int = 50,
    temperature: float = 0.4,
    verbose: bool = False,
    log_dir: Path = Path("runs"),
) -> None:
    print(f"\n{'━'*70}")
    print(f"  Fake Gang Detection — Hybrid Policy Training")
    print(f"  OpenEnv server : {env_url}")
    print(f"  Episodes       : {n_episodes}")
    print(f"  Task schedule  : {task or 'curriculum (easy→medium→hard)'}")
    print(f"  LLM            : Qwen3 via AWS Bedrock")
    print(f"  Policy         : Hybrid (rules ↔ LLM, dynamic α)")
    print(f"  Learning       : Reflexion (reflections + few-shot in prompt)")
    print(f"{'━'*70}\n")

    memory = AgentMemory()
    tracker = ProgressTracker()
    pending_metrics: List[Dict] = []

    print(memory.summary())
    print()

    with FakeGangEnvClient(base_url=env_url) as env:
        for ep in range(n_episodes):
            current_task = _curriculum_task(ep, task)
            seed = _episode_seed(ep, current_task)

            # --- Phase announcements ---
            if ep == 0:
                print("=== Phase 1: Easy — learning basic signal detection ===")
            elif ep == 20 and task is None:
                print("\n=== Phase 2: Medium — handling evasion ===")
            elif ep == 35 and task is None:
                print("\n=== Phase 3: Hard — feature-only detection ===")

            # --- Compute α for this episode ---
            # Load persisted α and update with latest win rate + reflection count
            n_refs = memory.reflection_count(current_task)
            wr = memory.recent_win_rate(current_task, n=10)
            alpha = compute_alpha(recent_win_rate=wr, n_reflections=n_refs)

            # --- Run episode ---
            metrics = run_episode(
                env=env,
                task=current_task,
                seed=seed,
                memory=memory,
                episode_num=ep + 1,
                alpha=alpha,
                temperature=temperature,
                verbose=verbose,
            )

            # --- Post-episode: record win, update + persist α ---
            memory.record_win(current_task, metrics["won"], ep + 1)
            # Recompute with the just-added result for next episode's alpha
            new_wr = memory.recent_win_rate(current_task, n=10)
            new_alpha = compute_alpha(recent_win_rate=new_wr, n_reflections=n_refs)
            memory.save_alpha(current_task, new_alpha)

            # --- Learning step ---
            learn_status = learning_step(metrics, memory)

            # --- Log ---
            tracker.record(metrics)
            pending_metrics.append(metrics)
            tracker.print_episode(metrics, learn_status)

            # Flush metrics every 5 episodes
            if len(pending_metrics) >= 5:
                save_metrics(pending_metrics, log_dir)
                pending_metrics.clear()

            # Phase summary every 10 episodes
            if (ep + 1) % 10 == 0:
                tracker.print_summary(f"Episodes 1–{ep+1}")

    # Final flush and summary
    if pending_metrics:
        save_metrics(pending_metrics, log_dir)

    tracker.print_summary("FINAL")
    print(memory.summary())
    print(f"\nMetrics saved to: {log_dir / 'metrics.jsonl'}")
    print(f"Memory saved to: {memory.memory_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Fake Gang Detection LLM agent (Reflexion + Episodic Memory)."
    )
    parser.add_argument("--env-url", default="http://localhost:8000",
                        help="OpenEnv server URL (default: http://localhost:8000)")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default=None,
                        help="Fix task (default: curriculum easy→medium→hard)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Total training episodes (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="LLM temperature (default: 0.4)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each action step")
    parser.add_argument("--log-dir", default="runs",
                        help="Directory for metrics output (default: runs/)")
    args = parser.parse_args()

    train(
        env_url=args.env_url,
        task=args.task,
        n_episodes=args.episodes,
        temperature=args.temperature,
        verbose=args.verbose,
        log_dir=Path(args.log_dir),
    )
