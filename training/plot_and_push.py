"""
Run after training to pull W&B metrics, generate plots, and push to HF dataset repo.
Usage: python -m training.plot_and_push --run-id <wandb_run_id> --plots-repo <hf-username>/graphstrike-grpo-plots
"""
import argparse, os
import wandb
import matplotlib.pyplot as plt
from huggingface_hub import HfApi
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--phase", default="phase2")
    p.add_argument("--plots-repo", required=True)
    p.add_argument("--wandb-project", default="fakegang-grpo")
    p.add_argument("--wandb-entity", default=None)
    args = p.parse_args()

    api_wb = wandb.Api()
    run = api_wb.run(f"{args.wandb_entity or ''}/{args.wandb_project}/{args.run_id}".lstrip("/"))
    history = run.history(samples=2000)

    out = Path("/tmp/grpo_plots")
    out.mkdir(exist_ok=True)

    METRICS = [
        ("reward",                    "Mean Reward per Step"),
        ("reward_std",                "Reward Std Dev"),
        ("rewards/reward_fn/mean",    "Reward Fn Mean"),
        ("train/grad_norm",           "Gradient Norm"),
        ("train/entropy",             "Entropy"),
        ("train/num_tokens",          "Tokens Processed"),
    ]

    for col, title in METRICS:
        if col not in history.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history["_step"], history[col], linewidth=1.5)
        ax.set_title(f"{title} — {args.phase}")
        ax.set_xlabel("Step")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = out / f"{args.phase}_{col.replace('/', '_')}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  saved {fname}")

    # Push all plots to HF dataset repo
    hf = HfApi(token=os.environ["HF_TOKEN"])
    hf.upload_folder(
        folder_path=str(out),
        repo_id=args.plots_repo,
        repo_type="dataset",
        path_in_repo=args.phase,
        commit_message=f"plots: {args.phase} run {args.run_id}",
    )
    print(f"[plot_and_push] pushed plots to {args.plots_repo}/{args.phase}")

if __name__ == "__main__":
    main()
