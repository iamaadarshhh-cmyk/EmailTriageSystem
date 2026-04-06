# reward/components/efficiency.py

from env.config import REWARD_STEP_PENALTY


class EfficiencyReward:

    def calculate(
        self,
        step_count: int,
        max_steps: int,
    ) -> float:
        """
        Reward based on how efficiently agent is solving task.
        Penalizes taking too many steps.
        """

        # Base step penalty every step
        step_penalty = REWARD_STEP_PENALTY  # -0.01

        # Extra penalty if agent is taking too long
        if step_count > max_steps * 0.8:    # After 80% of steps used
            step_penalty *= 2               # Double penalty

        return step_penalty