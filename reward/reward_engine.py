# reward/reward_engine.py

# reward/reward_engine.py

from typing import Dict, Any
from env.models import ActionResult, ActionType
from env.models import Email
from reward.components.correctness import CorrectnessReward
from reward.components.efficiency import EfficiencyReward
from reward.components.safety import SafetyReward


class RewardEngine:

    def __init__(self):
        self.correctness = CorrectnessReward()
        self.efficiency = EfficiencyReward()
        self.safety = SafetyReward()

    def calculate(
        self,
        result: ActionResult,
        email: Email,
        action: ActionType,          # 🔥 NEW
        step_count: int,
        max_steps: int,
        difficulty: str = "easy",    # 🔥 NEW
    ) -> Dict[str, Any]:
        """
        Context-aware reward calculation with curriculum scaling.
        """

        # ─────────────────────────────────────────────
        # 🧠 1. CORRECTNESS (NOW CONTEXT-AWARE)
        # ─────────────────────────────────────────────
        correctness_score = self.correctness.calculate(
            result,
            email,
            action,
            difficulty
        )

        # ─────────────────────────────────────────────
        # ⚡ 2. EFFICIENCY
        # ─────────────────────────────────────────────
        efficiency_score = self.efficiency.calculate(
            step_count,
            max_steps
        )

        # ─────────────────────────────────────────────
        # 🛡️ 3. SAFETY
        # ─────────────────────────────────────────────
        safety_score = self.safety.calculate(
            result,
            email
        )

        # ─────────────────────────────────────────────
        # 💸 4. ACTION COST (NEW)
        # ─────────────────────────────────────────────
        action_cost = 0.0

        if action == ActionType.ESCALATE:
            action_cost -= 0.1  # escalation is expensive

        if action == ActionType.DELETE and email.category != "spam":
            action_cost -= 0.3  # dangerous delete

        # ─────────────────────────────────────────────
        # 🎯 5. TOTAL REWARD
        # ─────────────────────────────────────────────
        total_reward = (
            correctness_score +
            efficiency_score +
            safety_score +
            action_cost
        )

        # ─────────────────────────────────────────────
        # 📊 6. BREAKDOWN (IMPORTANT FOR DEBUGGING)
        # ─────────────────────────────────────────────
        breakdown = {
            "total_reward": round(total_reward, 3),
            "correctness": round(correctness_score, 3),
            "efficiency": round(efficiency_score, 3),
            "safety": round(safety_score, 3),
            "action_cost": round(action_cost, 3),  # 🔥 NEW
            "is_correct": result.is_correct,
            "is_partial": result.is_partial,
            "difficulty": difficulty,              # 🔥 NEW
        }

        return breakdown