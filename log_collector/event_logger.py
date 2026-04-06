# log_collector/event_logger.py

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from env.config import LOG_DIR


# ─── Event Logger ───────────────────────────────────────────
class EventLogger:

    def __init__(self):
        self.log_dir = LOG_DIR / "events"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "events.jsonl"

    # ─── Log Event ──────────────────────────────────────────
    def log(self, event_type: str, data: Dict[str, Any]):
        """Log a single event."""

        event = {
            "timestamp": str(datetime.now()),
            "event_type": event_type,
            "data": data,
        }

        # Append to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    # ─── Specific Event Loggers ─────────────────────────────
    def log_reset(self, episode_id: str, task_level: str):
        """Log environment reset."""
        self.log("reset", {
            "episode_id": episode_id,
            "task_level": task_level,
        })

    def log_step(
        self,
        episode_id: str,
        step: int,
        action: str,
        reward: float,
        is_correct: bool,
    ):
        """Log a step."""
        self.log("step", {
            "episode_id": episode_id,
            "step": step,
            "action": action,
            "reward": reward,
            "is_correct": is_correct,
        })

    def log_episode_end(
        self,
        episode_id: str,
        total_reward: float,
        total_steps: int,
    ):
        """Log episode end."""
        self.log("episode_end", {
            "episode_id": episode_id,
            "total_reward": total_reward,
            "total_steps": total_steps,
        })

    def log_grade(
        self,
        episode_id: str,
        score: float,
        passed: bool,
    ):
        """Log grading result."""
        self.log("grade", {
            "episode_id": episode_id,
            "score": score,
            "passed": passed,
        })

    def log_error(self, episode_id: str, error: str):
        """Log an error."""
        self.log("error", {
            "episode_id": episode_id,
            "error": error,
        })

    # ─── Load Events ────────────────────────────────────────
    def load_events(self, event_type: str = None):
        """Load all events optionally filtered by type."""
        if not self.log_file.exists():
            return []

        events = []
        with open(self.log_file, "r") as f:
            for line in f:
                event = json.loads(line.strip())
                if event_type is None or event["event_type"] == event_type:
                    events.append(event)

        return events




## What each file does

### `trajectory_logger.py` — Full Episode Recorder
# Saves the **complete story** of every episode to a JSON file:
# ```
# log_collector/
# └── trajectories/
#     ├── ep_a3f2b1_20240115_103022.json
#     ├── ep_b4c2d1_20240115_103522.json
#     └── ep_c5d3e2_20240115_104022.json