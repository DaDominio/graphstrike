"""Persistent episodic memory for the LLM detective agent.

Stores:
  - Reflections: short lessons the agent generates after each episode
  - Best trajectories: full action logs from high-reward episodes (used as few-shot examples)

All data is written to disk so learning persists across container restarts
when the memory/ directory is mounted as a Docker volume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MEMORY_DIR = Path(__file__).parent.parent / "memory"


class AgentMemory:
    """Disk-backed memory for reflections and successful trajectories."""

    def __init__(self, memory_dir: Path = MEMORY_DIR) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reflections  (one JSONL file per task)
    # ------------------------------------------------------------------

    def _reflections_path(self, task: str) -> Path:
        return self.memory_dir / f"reflections_{task}.jsonl"

    def add_reflection(
        self,
        task: str,
        text: str,
        episode_num: int,
        reward: float,
    ) -> None:
        entry = {
            "episode": episode_num,
            "reward": round(reward, 3),
            "reflection": text.strip(),
        }
        with open(self._reflections_path(task), "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_reflections(self, task: str, n: int = 4) -> List[str]:
        """Return the n most recent reflection texts for a task."""
        path = self._reflections_path(task)
        if not path.exists():
            return []
        lines = path.read_text().strip().splitlines()
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        # Return the last n reflections
        return [e["reflection"] for e in entries[-n:]]

    def reflection_count(self, task: str) -> int:
        path = self._reflections_path(task)
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text().splitlines() if line.strip())

    # ------------------------------------------------------------------
    # Best trajectory  (one JSON file per task — stores single best run)
    # ------------------------------------------------------------------

    def _trajectory_path(self, task: str) -> Path:
        return self.memory_dir / f"best_trajectory_{task}.json"

    def add_trajectory(
        self,
        task: str,
        action_log: List[str],
        final_message: str,
        reward: float,
        episode_num: int,
    ) -> bool:
        """Save trajectory if it's better than the current best. Returns True if saved."""
        path = self._trajectory_path(task)
        current_best_reward = -999.0
        if path.exists():
            try:
                current_best_reward = json.loads(path.read_text()).get("reward", -999.0)
            except (json.JSONDecodeError, KeyError):
                pass

        if reward > current_best_reward:
            data = {
                "task": task,
                "episode": episode_num,
                "reward": round(reward, 3),
                "action_log": action_log,
                "final_message": final_message,
            }
            path.write_text(json.dumps(data, indent=2))
            return True
        return False

    def get_best_trajectory(self, task: str) -> Optional[Dict[str, Any]]:
        """Return best saved trajectory for task, or None."""
        path = self._trajectory_path(task)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Win history + alpha persistence
    # ------------------------------------------------------------------

    def _wins_path(self, task: str) -> Path:
        return self.memory_dir / f"wins_{task}.jsonl"

    def _alpha_path(self, task: str) -> Path:
        return self.memory_dir / f"alpha_{task}.json"

    def record_win(self, task: str, won: bool, episode_num: int) -> None:
        """Append an episode outcome to the win history for this task."""
        entry = {"episode": episode_num, "won": won}
        with open(self._wins_path(task), "a") as f:
            f.write(json.dumps(entry) + "\n")

    def recent_win_rate(self, task: str, n: int = 10) -> float:
        """Return win rate over the last n episodes for this task."""
        path = self._wins_path(task)
        if not path.exists():
            return 0.0
        entries = []
        for line in path.read_text().strip().splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        window = entries[-n:]
        if not window:
            return 0.0
        return sum(1 for e in window if e["won"]) / len(window)

    def save_alpha(self, task: str, alpha: float) -> None:
        """Persist the current α (LLM trust weight) for a task."""
        self._alpha_path(task).write_text(json.dumps({"alpha": round(alpha, 3)}))

    def load_alpha(self, task: str, default: float = 0.20) -> float:
        """Load persisted α, or return default if not saved yet."""
        path = self._alpha_path(task)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text()).get("alpha", default)
        except (json.JSONDecodeError, KeyError):
            return default

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["=== Agent Memory ==="]
        for task in ["easy", "medium", "hard"]:
            n_ref = self.reflection_count(task)
            best = self.get_best_trajectory(task)
            best_r = f"{best['reward']:+.2f}" if best else "none"
            alpha = self.load_alpha(task)
            wr = self.recent_win_rate(task, n=10)
            lines.append(
                f"  {task:6s}: {n_ref:3d} reflections | best reward: {best_r} "
                f"| α={alpha:.2f} | wr(last10)={wr:.0%}"
            )
        return "\n".join(lines)
