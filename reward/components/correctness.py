# reward/components/correctness.py

from env.models import ActionResult
from env.config import (
    REWARD_CORRECT_ACTION,
    REWARD_WRONG_ACTION,
    REWARD_PARTIAL_CREDIT,
)


class CorrectnessReward:

    def calculate(self, result: ActionResult) -> float:
        """Reward based on whether action was correct."""

        if result.is_correct:
            return REWARD_CORRECT_ACTION    # +1.0

        elif result.is_partial:
            return REWARD_PARTIAL_CREDIT    # +0.3

        else:
            return REWARD_WRONG_ACTION      # -0.5