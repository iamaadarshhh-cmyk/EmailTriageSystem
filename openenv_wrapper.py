# openenv_wrapper.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple
from tasks.corpus import TASKS
from grader.grader import EpisodeGrader
import uuid


class EmailTriageEnv(gym.Env):
    """
    OpenEnv/Gymnasium compliant Email Triage Environment.
    Judges will programmatically check this interface.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ─── Valid Actions ──────────────────────────────────────
    VALID_ACTIONS = [
        "classify",
        "reply",
        "escalate",
        "archive",
        "flag",
        "delete",
        "skip",
        "mark_spam",
    ]

    def __init__(
        self,
        task_id: str = "task1_easy",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.task_id = task_id
        self.render_mode = render_mode
        self.task_config = TASKS.get(task_id, TASKS["task1_easy"])
        self.grader = EpisodeGrader()

        # ─── Observation Space ──────────────────────────────
        self.observation_space = spaces.Dict({
            "inbox_remaining": spaces.Discrete(100),
            "emails_processed": spaces.Discrete(100),
            "step_count": spaces.Discrete(100),
            "session_score": spaces.Box(
                low=-10.0, high=10.0,
                shape=(1,), dtype=np.float32
            ),
            "spam_score": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            ),
            "urgency_score": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,), dtype=np.float32
            ),
            "is_done": spaces.Discrete(2),
        })

        # ─── Action Space ───────────────────────────────────
        self.action_space = spaces.Discrete(len(self.VALID_ACTIONS))

        # ─── State ──────────────────────────────────────────
        self._session = None
        self._current_index = 0

    # ─── Reset ──────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset environment — returns (observation, info)."""

        super().reset(seed=seed)

        emails = self.task_config["emails"]

        self._session = {
            "task_id": self.task_id,
            "task_config": self.task_config,
            "emails": emails,
            "current_index": 0,
            "actions": [],
            "step_count": 0,
            "session_score": 0.0,
            "triage_history": [],
            "is_done": False,
            "episode_id": str(uuid.uuid4())[:8],
        }

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    # ─── Step ───────────────────────────────────────────────
    def step(
        self,
        action: int,
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take one step.
        Returns: (observation, reward, terminated, truncated, info)
        """

        assert self._session is not None, "Call reset() first!"

        # Convert int action to string
        action_str = self.VALID_ACTIONS[action]

        emails = self._session["emails"]
        current_index = self._session["current_index"]

        # ─── Get Current Email ──────────────────────────────
        if current_index >= len(emails):
            obs = self._get_obs()
            return obs, 0.0, True, False, self._get_info()

        current_email = emails[current_index]

        # ─── Build Action Dict ──────────────────────────────
        action_dict = {
            "action_type": action_str,
            "triage": {
                "priority": self._infer_priority(current_email),
                "category": self._infer_category(current_email),
                "confidence": 0.7,
            },
            "escalate_to": self._infer_escalation(current_email),
            "reply_body": "",
            "reason": f"Gymnasium agent selected {action_str}",
        }

        # ─── Grade Step ─────────────────────────────────────
        step_score = self.grader._grade_single(
            action_dict,
            current_email,
            self.task_config.get("difficulty", "easy"),
        )
        reward = float(step_score["total"])

        # ─── Update State ───────────────────────────────────
        self._session["actions"].append(action_dict)
        self._session["step_count"] += 1
        self._session["session_score"] += reward
        self._session["current_index"] += 1
        self._session["triage_history"].append({
            "email_id": current_email["email_id"],
            "action": action_str,
            "reward": round(reward, 3),
        })

        # ─── Check Terminated ───────────────────────────────
        terminated = (
            self._session["current_index"] >= len(emails) or
            self._session["step_count"] >= self.task_config["max_steps"]
        )
        self._session["is_done"] = terminated

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, False, info

    # ─── Get Observation ────────────────────────────────────
    def _get_obs(self) -> Dict:
        """Return flat observation dict."""

        if self._session is None:
            return {
                "inbox_remaining": np.int64(0),
                "emails_processed": np.int64(0),
                "step_count": np.int64(0),
                "session_score": np.array([0.0], dtype=np.float32),
                "spam_score": np.array([0.0], dtype=np.float32),
                "urgency_score": np.array([0.0], dtype=np.float32),
                "is_done": np.int64(1),
            }

        emails = self._session["emails"]
        idx = self._session["current_index"]
        current_email = emails[idx] if idx < len(emails) else None

        spam_score = 0.0
        urgency_score = 0.0

        if current_email:
            spam_score = self._compute_spam_score(current_email)
            urgency_score = self._compute_urgency_score(current_email)

        return {
            "inbox_remaining": np.int64(
                len(emails) - self._session["current_index"]
            ),
            "emails_processed": np.int64(self._session["current_index"]),
            "step_count": np.int64(self._session["step_count"]),
            "session_score": np.array(
                [self._session["session_score"]], dtype=np.float32
            ),
            "spam_score": np.array([spam_score], dtype=np.float32),
            "urgency_score": np.array([urgency_score], dtype=np.float32),
            "is_done": np.int64(1 if self._session["is_done"] else 0),
        }

    # ─── Get Info ───────────────────────────────────────────
    def _get_info(self) -> Dict:
        """Return info dict."""
        if self._session is None:
            return {}

        emails = self._session["emails"]
        idx = self._session["current_index"]
        current_email = emails[idx] if idx < len(emails) else None

        return {
            "task_id": self.task_id,
            "episode_id": self._session.get("episode_id"),
            "step_count": self._session["step_count"],
            "session_score": self._session["session_score"],
            "current_email": {
                "email_id": current_email["email_id"],
                "subject": current_email["subject"],
                "sender": current_email["sender"],
                "body": current_email["body"],
                "attachments": current_email.get("attachments", []),
            } if current_email else None,
            "triage_history": self._session["triage_history"][-3:],
        }

    # ─── Grade Episode ───────────────────────────────────────
    def grade(self) -> Dict:
        """Grade completed episode."""
        if not self._session:
            return {}
        return self.grader.grade(
            task_id=self.task_id,
            actions=self._session["actions"],
            emails=self._session["emails"][:len(self._session["actions"])],
            task_config=self.task_config,
        )

    # ─── Render ─────────────────────────────────────────────
    def render(self):
        """Render current state."""
        if self._session is None:
            return "Environment not initialized."

        emails = self._session["emails"]
        idx = self._session["current_index"]
        email = emails[idx] if idx < len(emails) else None

        output = (
            f"\n{'='*60}\n"
            f"Task     : {self.task_id}\n"
            f"Step     : {self._session['step_count']}/{self.task_config['max_steps']}\n"
            f"Score    : {self._session['session_score']:.3f}\n"
            f"{'─'*60}\n"
        )

        if email:
            output += (
                f"Email    : {email['email_id']}\n"
                f"Subject  : {email['subject']}\n"
                f"From     : {email['sender']}\n"
                f"{'─'*60}\n"
                f"Remaining: {len(emails) - idx}\n"
            )
        else:
            output += "Inbox is empty.\n"

        output += f"{'='*60}\n"

        if self.render_mode == "human":
            print(output)
        return output

    # ─── Close ──────────────────────────────────────────────
    def close(self):
        """Clean up."""
        self._session = None

    # ─── Helpers ────────────────────────────────────────────
    def _compute_spam_score(self, email: Dict) -> float:
        """Compute spam signal score."""
        spam_words = [
            "prize", "free", "click", "verify",
            "suspended", "winner", "claim", "urgent",
        ]
        text = (email["subject"] + " " + email["body"]).lower()
        hits = sum(1 for w in spam_words if w in text)
        return min(hits / len(spam_words), 1.0)

    def _compute_urgency_score(self, email: Dict) -> float:
        """Compute urgency signal score."""
        urgent_words = [
            "urgent", "critical", "asap", "immediately",
            "down", "breach", "outage", "emergency",
        ]
        text = (email["subject"] + " " + email["body"]).lower()
        hits = sum(1 for w in urgent_words if w in text)
        return min(hits / len(urgent_words), 1.0)

    def _infer_priority(self, email: Dict) -> str:
        """Infer priority from email signals."""
        urgency = self._compute_urgency_score(email)
        spam = self._compute_spam_score(email)
        if spam > 0.4:
            return "spam"
        if urgency > 0.3:
            return "urgent"
        return "medium"

    def _infer_category(self, email: Dict) -> str:
        """Infer category from email signals."""
        text = (email["subject"] + " " + email["body"]).lower()
        if any(w in text for w in ["invoice", "billing", "payment"]):
            return "billing"
        if any(w in text for w in ["bug", "error", "down", "crash"]):
            return "bug_report"
        if any(w in text for w in ["prize", "free", "winner"]):
            return "spam"
        return "general_inquiry"

    def _infer_escalation(self, email: Dict) -> str:
        """Infer escalation target from email signals."""
        text = (email["subject"] + " " + email["body"]).lower()
        if any(w in text for w in ["database", "db", "postgres"]):
            return "on-call-dba"
        if any(w in text for w in ["security", "breach", "hack"]):
            return "security-team"
        if any(w in text for w in ["invoice", "billing", "payment"]):
            return "finance-team"
        if any(w in text for w in ["legal", "lawsuit", "terminate"]):
            return "legal"
        return ""


# ─── Register with Gymnasium ────────────────────────────────
gym.register(
    id="EmailTriage-v0",
    entry_point="openenv_wrapper:EmailTriageEnv",
    kwargs={"task_id": "task1_easy"},
)

gym.register(
    id="EmailTriage-easy-v0",
    entry_point="openenv_wrapper:EmailTriageEnv",
    kwargs={"task_id": "task1_easy"},
)

gym.register(
    id="EmailTriage-medium-v0",
    entry_point="openenv_wrapper:EmailTriageEnv",
    kwargs={"task_id": "task2_medium"},
)

gym.register(
    id="EmailTriage-hard-v0",
    entry_point="openenv_wrapper:EmailTriageEnv",
    kwargs={"task_id": "task3_hard"},
)


# ─── Quick Test ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing OpenEnv Gymnasium wrapper...\n")

    for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
        env = EmailTriageEnv(task_id=task_id)
        obs, info = env.reset(seed=42)

        print(f"Task: {task_id}")
        print(f"Observation: {obs}")

        total_reward = 0.0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        grade = env.grade()
        print(f"Score : {grade['final_score']:.4f}")
        print(f"Passed: {grade['passed']}")
        print(f"{'─'*40}")
        env.close()

    print("\n✅ Gymnasium wrapper working!")