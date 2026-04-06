# tasks/definitions/hard.py

import uuid
from typing import Optional
from env.models.state import Email
from env.models.actions import ActionType


class HardTask:

    level = "hard"

    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id or f"hard_{uuid.uuid4().hex[:6]}"
        self.description = "Handle complex multi step email workflows"

    def get_expected_action(self, email: Email) -> ActionType:
        """Return expected action for given email."""

        if email.category == "spam":
            return ActionType.MARK_SPAM

        elif email.category == "work":
            subject = email.subject.lower()
            body = email.body.lower()

            if any(word in subject for word in
                   ["urgent", "critical", "asap"]):
                return ActionType.ESCALATE

            elif any(word in body for word in
                     ["summarize", "summary", "brief", "overview"]):
                return ActionType.SUMMARIZE

            elif any(word in subject for word in
                     ["deadline", "overdue", "pending"]):
                return ActionType.DEFER

            elif any(word in subject for word in
                     ["meeting", "presentation", "review"]):
                return ActionType.REPLY

            else:
                return ActionType.FORWARD

        elif email.category == "personal":
            subject = email.subject.lower()
            body = email.body.lower()

            if any(word in subject for word in
                   ["urgent", "emergency", "help"]):
                return ActionType.REPLY

            elif any(word in body for word in
                     ["summary", "update", "news"]):
                return ActionType.SUMMARIZE

            else:
                return ActionType.DEFER

        return ActionType.READ  # default
    
    def evaluate_action(self, state, action):
        email = state.current_email

        if email is None:
            return 0.0

        subject = email.subject.lower()
        body = email.body.lower()

        score = 0.0

        # ─── Spam handling ─────────────────────────────
        if email.category == "spam":
            if action.action_type.name == "MARK_SPAM":
                return 1.0
            return 0.0

        # ─── Urgent / critical emails ──────────────────
        if any(word in subject for word in ["urgent", "asap", "immediately", "critical"]):
            if action.action_type.name == "ESCALATE":
                score = 1.0
            elif action.action_type.name == "REPLY":
                score = 0.6

        # ─── Meeting / coordination ───────────────────
        elif any(word in subject for word in ["meeting", "call", "schedule"]):
            if action.action_type.name == "REPLY":
                score = 1.0

        # ─── Finance / invoice ────────────────────────
        elif any(word in subject for word in ["invoice", "payment", "due"]):
            if action.action_type.name == "FORWARD":
                score = 1.0

        # ─── Informational emails ─────────────────────
        elif any(word in subject for word in ["report", "update", "newsletter"]):
            if action.action_type.name in ["READ", "ARCHIVE"]:
                score = 0.7

        # ─── Default handling ─────────────────────────
        else:
            if action.action_type.name in ["READ", "ARCHIVE"]:
                score = 0.5

        return score