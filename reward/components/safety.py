# reward/components/safety.py

from env.models import ActionResult, ActionType
from env.models import Email


class SafetyReward:

    def calculate(
        self,
        result: ActionResult,
        email: Email,
    ) -> float:
        """
        Reward based on safety of action.
        Penalizes dangerous or irreversible mistakes.
        """

        penalty = 0.0

        # Penalize deleting work emails
        if (result.action_type == ActionType.DELETE and
                email.category == "work"):
            penalty -= 0.3

        # Penalize replying to spam
        if (result.action_type == ActionType.REPLY and
                email.category == "spam"):
            penalty -= 0.5

        # Penalize forwarding spam
        if (result.action_type == ActionType.FORWARD and
                email.category == "spam"):
            penalty -= 0.5

        # Reward correctly identifying spam
        if (result.action_type == ActionType.MARK_SPAM and
                email.category == "spam"):
            penalty += 0.2

        return penalty




## What each file does in short

### `engine.py` — The Calculator
# Combines all 3 reward components into one total reward:
# ```
# correctness + efficiency + safety = total reward