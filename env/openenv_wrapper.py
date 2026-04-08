# env/openenv_wrapper.py
"""
OpenEnv-compatible Gymnasium wrapper for EmailTriageEnv.
Judges will programmatically check this interface.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple

from env.core.environment import EmailTriageEnvironment
from env.models import Action, ActionType
from env.config import VALID_ACTIONS, MAX_STEPS


ACTIONS = [
    "read", "reply", "forward", "delete", "archive",
    "label", "mark_spam", "escalate", "defer", "summarize",
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


class EmailTriageGymEnv(gym.Env):
    """
    Gymnasium-compatible Email Triage RL Environment.

    Observation space:
        spam_score    : float [0, 1]
        urgency       : float [0, 1]
        inbox_remaining: int  [0, MAX_EMAILS]
        step_count    : int   [0, MAX_STEPS]

    Action space:
        Discrete(10) — one per VALID_ACTIONS entry

    Reward:
        +1.0 correct action
        +0.3 partial credit
        -0.5 wrong action
        -0.01 step penalty (efficiency)
        safety bonuses / penalties applied on top
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        task_level: str = "easy",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.task_level = task_level
        self.render_mode = render_mode

        from tasks.task_factory import TaskFactory
        task = TaskFactory.create(task_level)
        self._env = EmailTriageEnvironment(task=task)

        self.observation_space = spaces.Dict({
            "spam_score":       spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "urgency":          spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "inbox_remaining":  spaces.Discrete(MAX_STEPS + 1),
            "step_count":       spaces.Discrete(MAX_STEPS + 1),
        })
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._current_email_id: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        super().reset(seed=seed)
        raw_obs = self._env.reset(user_id="gym_agent")
        self._current_email_id = raw_obs.get("email_id")
        obs = self._convert_obs(raw_obs)
        info = {"episode_id": self._env.episode_id}
        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        action_str = IDX_TO_ACTION[action]
        if not self._current_email_id:
            return self._empty_step()

        env_action = Action(
            action_type=ActionType(action_str),
            email_id=self._current_email_id,
        )
        raw_obs, reward, done, info = self._env.step(env_action)
        self._current_email_id = raw_obs.get("email_id")
        obs = self._convert_obs(raw_obs)
        terminated = done
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode in ("human", "ansi"):
            print(self._env.render())

    def close(self):
        self._env.close()

    def _convert_obs(self, raw: Dict) -> Dict[str, Any]:
        signals = raw.get("signals", {})
        return {
            "spam_score":      np.array([signals.get("spam_score", 0.0)], dtype=np.float32),
            "urgency":         np.array([signals.get("urgency", 0.0)],    dtype=np.float32),
            "inbox_remaining": int(raw.get("unread_count", 0)),
            "step_count":      int(raw.get("step_count", 0)),
        }

    def _empty_step(self):
        obs = {
            "spam_score":      np.array([0.0], dtype=np.float32),
            "urgency":         np.array([0.0], dtype=np.float32),
            "inbox_remaining": 0,
            "step_count":      0,
        }
        return obs, 0.0, True, False, {}