# tasks/definitions/easy.py

import uuid
from typing import Optional
from env.models.state import Email
from env.models.actions import ActionType


class EasyTask:

    level = "easy"

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id or f"easy_{uuid.uuid4().hex[:6]}"
        self.description = "Classify emails correctly"

    def get_expected_action(self, email: Email) -> ActionType:
        if email.category == "spam":
            return ActionType.MARK_SPAM
        elif email.category == "work":
            return ActionType.READ
        elif email.category == "personal":
            return ActionType.READ
        return ActionType.READ

    # ✅ IMPORTANT: correct indentation
    def evaluate_action(self, state, action):
        email = state.current_email

        if email is None:
            return 0.0

        expected = self.get_expected_action(email)

        if action.action_type == expected:
            return 1.0

        return 0.0