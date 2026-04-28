"""Build a JSONL dataset of (prompt, completion, reward) tuples by running a
seed/teacher policy through the env and collecting per-decision tuples.

This drives Phase 0 baseline + supplies the prompt distribution for GRPO
rollouts (we replay env states from these episodes during training).

Usage:
  python -m training.build_dataset \
      --base-url http://localhost:8000 \
      --platform Instagram \
      --seeds 0 1 2 3 4 \
      --tasks easy medium hard \
      --out training/data/baseline.jsonl

The "teacher" call_llm is a deterministic stub (heuristic) by default; pass
--teacher openai:<model> to use a real API teacher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))
sys.path.insert(0, str(_PARENT / "eval-models"))

from _round2_runner import _run_episode, _seeds_for_platform  # noqa: E402
from client import FakeGangEnvClient  # noqa: E402

from training.parse import parse_completion  # noqa: E402
from training.rewards import compute_reward  # noqa: E402


def heuristic_teacher() -> Callable[[str], str]:
    """Deterministic teacher producing JSON-shaped completions matching prompts.py.

    Tool selection: prefer reverse_image_search → analyze_bio → check_ip → done.
    Flag decision: flag if 'risk:' >= threshold else skip (quick text scrape).
    """
    state = {"i": 0}
    rotation = ["reverse_image_search", "analyze_bio", "check_ip", "done"]

    def call(prompt: str) -> str:
        if "[DP2" in prompt or "flag decision" in prompt:
            try:
                risk = float(prompt.split("risk:")[1].split("|")[0].strip())
                thr = float(prompt.split("threshold:")[1].split("|")[0].strip())
                action = "flag" if risk >= thr else "skip"
            except Exception:
                action = "skip"
            return json.dumps({"action": action, "reason": "heuristic"})
        choice = rotation[state["i"] % len(rotation)]
        state["i"] += 1
        return json.dumps({"action": choice, "reason": "heuristic"})

    return call


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.getenv("API_BASE_URL_ENV", "http://localhost:8000"))
    p.add_argument("--platform", default="Instagram")
    p.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=str(_HERE / "data" / "baseline.jsonl"))
    p.add_argument("--teacher", default="heuristic", help="heuristic | openai:<model>")
    args = p.parse_args()

    if args.teacher != "heuristic":
        raise SystemExit("Only the heuristic teacher is wired in this scaffold.")
    call_llm = heuristic_teacher()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = _seeds_for_platform(args.seeds, args.platform)
    client = FakeGangEnvClient(base_url=args.base_url)

    n_rows = 0
    with open(out_path, "w") as fout:
        for task in args.tasks:
            for seed in seeds:
                log, tuples = _run_episode(
                    client, "teacher:heuristic", args.platform, task, seed, call_llm,
                    collect_tuples=True,
                )
                for di, t in enumerate(tuples):
                    _, _, format_ok = parse_completion(t["completion"], t["decision_type"])
                    rb = compute_reward(t.get("grader_score"), t.get("step_reward"), format_ok)
                    row = {
                        "prompt": t["prompt"],
                        "completion": t["completion"],
                        "decision_type": t["decision_type"],
                        "platform": t["platform"],
                        "threshold": t["threshold"],
                        "fp_penalty": t["fp_penalty"],
                        "step_index": t["step_index"],
                        "decision_index": di,
                        "episode_id": t.get("episode_id"),
                        "task": task,
                        "seed": seed,
                        **rb.as_dict(),
                        "format_ok": format_ok,
                        "raw_step_reward": t.get("step_reward"),
                        "grader_score": t.get("grader_score"),
                    }
                    fout.write(json.dumps(row) + "\n")
                    n_rows += 1
                print(f"  task={task} seed={seed} grader={log.grader_score} flagged={log.flagged} tuples={len(tuples)}")

    print(f"\nWrote {n_rows} rows → {out_path}")


if __name__ == "__main__":
    main()
