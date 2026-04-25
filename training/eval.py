"""Phase-gate evaluation for the GRPO policy.

Runs the trained model against held-out (task, seed) pairs, prints aggregate
metrics, and emits a JSONL log compatible with `eval-models/results/*.jsonl`.

Sanity-gate thresholds (configurable):
  phase1 (smoke):  format_ok_rate ≥ 0.50
  phase2 (signal): mean_grader strictly higher than phase0 baseline
  phase3 (full):   mean_grader ≥ best teacher baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from training.rollout import HFPolicy, rollout_batch  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model path or hub id (trained checkpoint)")
    p.add_argument("--base-url", default=os.getenv("API_BASE_URL_ENV", "http://localhost:8000"))
    p.add_argument("--platform", default="LinkedIn", help="Held-out platform (default LinkedIn)")
    p.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    p.add_argument("--out", default=str(_HERE / "runs" / "eval.jsonl"))
    p.add_argument("--gate", choices=["phase1", "phase2", "phase3"], default=None)
    p.add_argument("--baseline-grader", type=float, default=None,
                   help="Mean grader from phase0 baseline, required when --gate=phase2")
    args = p.parse_args()

    policy = HFPolicy(args.model, temperature=0.0)
    rows = rollout_batch(
        policy.generate,
        base_url=args.base_url, platform=args.platform,
        tasks=args.tasks, seeds=args.seeds,
        model_tag=f"eval:{args.model}",
    )
    if not rows:
        raise SystemExit("No rows produced — env unreachable?")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    fmt = sum(r["format_ok"] for r in rows) / n
    grader = sum(r["r_grader"] for r in rows) / n
    total = sum(r["r_total"] for r in rows) / n
    print(f"\n=== eval @ {args.platform} (n={n} decisions) ===")
    print(f"  format_ok_rate: {fmt:.3f}")
    print(f"  mean_grader:    {grader:.3f}")
    print(f"  mean_r_total:   {total:.3f}")
    print(f"  log:            {args.out}")

    if args.gate == "phase1":
        ok = fmt >= 0.50
        print(f"  gate phase1: format_ok_rate >= 0.50 → {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    if args.gate == "phase2":
        if args.baseline_grader is None:
            raise SystemExit("--baseline-grader required for phase2 gate")
        ok = grader > args.baseline_grader
        print(f"  gate phase2: grader {grader:.3f} > baseline {args.baseline_grader:.3f} → {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    if args.gate == "phase3":
        ok = grader >= 0.50  # tune to teacher's mean
        print(f"  gate phase3: grader >= 0.50 → {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
