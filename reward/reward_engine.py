# reward/engine.py

from typing import Dict, Any
from env.models import ActionResult, ActionType
from env.models import Email
from reward.components.correctness import CorrectnessReward
from reward.components.efficiency import EfficiencyReward
from reward.components.safety import SafetyReward
from env.config import (
    REWARD_CORRECT_ACTION,
    REWARD_WRONG_ACTION,
    REWARD_PARTIAL_CREDIT,
    REWARD_STEP_PENALTY,
)


# ─── Reward Engine ──────────────────────────────────────────
class RewardEngine:

    def __init__(self):
        self.correctness = CorrectnessReward()
        self.efficiency = EfficiencyReward()
        self.safety = SafetyReward()

    def calculate(
        self,
        result: ActionResult,
        email: Email,
        step_count: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Calculate total reward from all components.
        Returns reward breakdown.
        """

        # ─── Component Rewards ──────────────────────────────
        correctness_score = self.correctness.calculate(result)
        efficiency_score = self.efficiency.calculate(step_count, max_steps)
        safety_score = self.safety.calculate(result, email)

        # ─── Total Reward ───────────────────────────────────
        total_reward = (
            correctness_score +
            efficiency_score +
            safety_score
        )

        # ─── Reward Breakdown ───────────────────────────────
        breakdown = {
            "total_reward": round(total_reward, 3),
            "correctness": round(correctness_score, 3),
            "efficiency": round(efficiency_score, 3),
            "safety": round(safety_score, 3),
            "is_correct": result.is_correct,
            "is_partial": result.is_partial,
        }

        return breakdown