# log_collector/trajectory_logger.py

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from env.memory.history import EpisodeHistory
from env.config import LOG_DIR


# ─── Trajectory Logger ──────────────────────────────────────
class TrajectoryLogger:

    def __init__(self):
        self.log_dir = LOG_DIR / "trajectories"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ─── Log Episode ────────────────────────────────────────
    def log_episode(self, history: EpisodeHistory) -> str:
        """
        Save full episode trajectory to JSON file.
        Returns path of saved file.
        """

        trajectory = self._build_trajectory(history)
        file_path = self._get_file_path(history.episode_id)

        with open(file_path, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)

        return str(file_path)

    # ─── Build Trajectory ───────────────────────────────────
    def _build_trajectory(
        self,
        history: EpisodeHistory
    ) -> Dict[str, Any]:
        """Build trajectory dictionary from episode history."""

        steps = []
        for step in history.steps:
            steps.append({
                "step_number": step.step_number,
                "email_id": step.email_id,
                "action": step.action.action_type.value,
                "reward": step.reward,
                "is_correct": step.result.is_correct,
                "is_partial": step.result.is_partial,
                "message": step.result.message,
                "timestamp": str(step.timestamp),
            })

        return {
            "episode_id": history.episode_id,
            "task_level": history.task_level,
            "total_steps": len(history.steps),
            "total_reward": history.total_reward,
            "is_complete": history.is_complete,
            "start_time": str(history.start_time),
            "end_time": str(history.end_time),
            "steps": steps,
        }

    # ─── Get File Path ──────────────────────────────────────
    def _get_file_path(self, episode_id: str) -> Path:
        """Generate file path for episode."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_dir / f"{episode_id}_{timestamp}.json"

    # ─── Load Trajectory ────────────────────────────────────
    def load_trajectory(self, episode_id: str) -> Dict[str, Any]:
        """Load a saved trajectory by episode ID."""
        for file in self.log_dir.glob(f"{episode_id}_*.json"):
            with open(file, "r") as f:
                return json.load(f)
        return {}

    # ─── Load All Trajectories ──────────────────────────────
    def load_all(self) -> List[Dict[str, Any]]:
        """Load all saved trajectories."""
        trajectories = []
        for file in self.log_dir.glob("*.json"):
            with open(file, "r") as f:
                trajectories.append(json.load(f))
        return trajectories

    # ─── Get Stats ──────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        """Get stats across all saved trajectories."""
        trajectories = self.load_all()

        if not trajectories:
            return {"total_episodes": 0}

        total = len(trajectories)
        passed = sum(
            1 for t in trajectories
            if t.get("total_reward", 0) > 0
        )
        avg_reward = sum(
            t.get("total_reward", 0)
            for t in trajectories
        ) / total

        return {
            "total_episodes": total,
            "passed_episodes": passed,
            "avg_reward": round(avg_reward, 3),
            "pass_rate": round(passed / total, 3),
        }