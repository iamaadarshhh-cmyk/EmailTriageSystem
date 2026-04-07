# reward/components/correctness.py

from env.models import ActionType, Priority


class CorrectnessReward:

    def calculate(
        self,
        taken_action: str,
        expected_action: str,
        email: dict,
        difficulty: str = "easy",
    ) -> float:
        """
        Shaped reward based on action correctness.
        Rewards scale with difficulty level.
        """

        ground_truth = email.get("ground_truth", {})
        is_spam = ground_truth.get("is_spam", False)
        expected_priority = ground_truth.get(
            "expected_priority", "medium"
        )

        # ─── Difficulty Multiplier ──────────────────────────
        multiplier = {
            "easy": 1.0,
            "medium": 1.2,
            "hard": 1.5,
        }.get(difficulty, 1.0)

        # ─── Critical Safety Violations ────────────────────
        if taken_action == "reply" and is_spam:
            return -1.5 * multiplier  # Replied to phishing!

        if taken_action == "forward" and is_spam:
            return -1.5 * multiplier  # Forwarded phishing!

        if taken_action == "archive" and expected_priority == "urgent":
            return -0.8 * multiplier  # Ignored urgent email!

        if taken_action == "delete" and not is_spam:
            return -0.8 * multiplier  # Deleted legit email!

        # ─── Correct Action ─────────────────────────────────
        if taken_action == expected_action:
            return 1.0 * multiplier

        # ─── Partial Credit ─────────────────────────────────
        partial_matches = {
            "delete": ["mark_spam", "archive"],
            "mark_spam": ["delete"],
            "archive": ["flag"],
            "reply": ["escalate"],
            "escalate": ["flag", "reply"],
            "flag": ["escalate"],
        }
        if taken_action in partial_matches.get(expected_action, []):
            return 0.3 * multiplier

        # ─── Non-urgent escalated (noise penalty) ───────────
        if taken_action == "escalate" and expected_priority in ["low", "medium"]:
            return -0.2 * multiplier

        # ─── Wrong action ───────────────────────────────────
        return -0.5 * multiplier