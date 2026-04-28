"""Plot training curves from runs/metrics.jsonl.

Outputs:
  runs/reward_curve.png  - reward per episode + rolling mean
  runs/loss_curve.png    - 1.0 - grader_score (proxy "loss") per episode

Usage:
  python plot_metrics.py [--input runs/metrics.jsonl] [--out runs/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rolling(xs, w):
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > w:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="runs/metrics.jsonl")
    p.add_argument("--out", default="runs")
    p.add_argument("--window", type=int, default=10)
    args = p.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"No rows in {src}")

    eps = [r.get("episode", i + 1) for i, r in enumerate(rows)]
    rewards = [float(r.get("reward", 0.0)) for r in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, rewards, alpha=0.4, label="reward")
    plt.plot(eps, _rolling(rewards, args.window), linewidth=2, label=f"rolling mean (w={args.window})")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.xlabel("episode"); plt.ylabel("reward"); plt.title("Training reward")
    plt.legend()
    plt.savefig(out / "reward_curve.png", dpi=120)
    plt.close()

    # Proxy loss = 1 - normalized grader score; if missing, derive from recall+precision
    losses = []
    for r in rows:
        g = r.get("grader_score")
        if g is None:
            recall = float(r.get("recall", 0.0))
            precision = float(r.get("precision", 0.0))
            g = 0.55 + 0.20 * recall + 0.15 * precision if (recall >= 0.8 and precision >= 0.7) else 0.30 * recall + 0.10 * precision
        losses.append(max(0.0, 1.0 - float(g)))

    plt.figure(figsize=(8, 4.5))
    plt.plot(eps, losses, alpha=0.4, label="loss (1 - grader)")
    plt.plot(eps, _rolling(losses, args.window), linewidth=2, label=f"rolling mean (w={args.window})")
    plt.xlabel("episode"); plt.ylabel("loss"); plt.title("Training loss proxy")
    plt.legend()
    plt.savefig(out / "loss_curve.png", dpi=120)
    plt.close()

    print(f"wrote {out / 'reward_curve.png'}")
    print(f"wrote {out / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
