# tasks/definitions/medium.py

import uuid
from typing import Optional
from env.models.state import Email
from env.models.actions import ActionType


class MediumTask:

    level = "medium"

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id or f"medium_{uuid.uuid4().hex[:6]}"
        self.description = "Handle emails with context awareness"

    def get_expected_action(self, email: Email) -> ActionType:
        """Return expected action for given email."""

        if email.category == "spam":
            return ActionType.MARK_SPAM

        elif email.category == "work":
            # Check subject for urgency
            subject = email.subject.lower()
            if any(word in subject for word in
                   ["urgent", "asap", "immediately", "critical"]):
                return ActionType.ESCALATE
            elif any(word in subject for word in
                     ["meeting", "call", "presentation"]):
                return ActionType.REPLY
            elif any(word in subject for word in
                     ["invoice", "payment", "due"]):
                return ActionType.FORWARD
            else:
                return ActionType.ARCHIVE

        elif email.category == "personal":
            subject = email.subject.lower()
            if any(word in subject for word in
                   ["birthday", "congratulations", "thank"]):
                return ActionType.REPLY
            else:
                return ActionType.READ

        return ActionType.READ  # default
    
    def evaluate_action(self, state, action):
        email = state.current_email

        if email is None:
            return 0.0

        expected = self.get_expected_action(email)

        if action.action_type == expected:
            return 1.0

        return 0.0