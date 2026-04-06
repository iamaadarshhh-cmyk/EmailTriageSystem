# env/core/transition.py

from typing import Tuple, Optional
from env.models import AgentState, Email
from env.models import Action, ActionResult, ActionType, ActionValidator
from env.config import MAX_STEPS
from reward.reward_engine import RewardEngine


# ─── Transition Engine ──────────────────────────────────────
class TransitionEngine:

    def __init__(self):
        self.reward_engine = RewardEngine()

    def step(
        self,
        state: AgentState,
        action: Action,
        task,
    ) -> Tuple[AgentState, ActionResult]:
        """
        Take one step in the environment.
        Returns updated state and action result.
        """

        # ─── Validate Action ────────────────────────────────
        if not ActionValidator.is_valid(action.action_type.value):
            state.step_count += 1
            result = self._invalid_action_result(action)
            return state, result

        if not ActionValidator.validate_parameters(action):
            state.step_count += 1
            result = self._missing_params_result(action)
            return state, result

        # ─── Execute Action ─────────────────────────────────
        state, result = self._execute_action(state, action, task)

        # ─── Update State ───────────────────────────────────
        state.step_count += 1
        state.total_reward += result.reward
        state.action_history.append(action.action_type.value)

        # ─── Check if Done ──────────────────────────────────
        if state.step_count >= MAX_STEPS:
            state.is_done = True

        if state.inbox.unread_count <= 0:
            state.is_done = True

        return state, result

    # ─── Execute Action ─────────────────────────────────────
    def _execute_action(
        self,
        state: AgentState,
        action: Action,
        task,
    ) -> Tuple[AgentState, ActionResult]:
        """Execute action and return updated state + result."""

        email = state.current_email

        # FIX: guard against task=None (environment defaults task to None)
        if task is None:
            score = 0.5
        else:
            score = task.evaluate_action(state, action)

        is_correct = score >= 0.9
        is_partial = 0.3 <= score < 0.9

        # ─── Update Email State ─────────────────────────────
        if action.action_type == ActionType.READ:
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        elif action.action_type == ActionType.REPLY:
            email.is_replied = True
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        elif action.action_type == ActionType.DELETE:
            email.is_deleted = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        elif action.action_type == ActionType.ARCHIVE:
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        elif action.action_type == ActionType.MARK_SPAM:
            email.is_deleted = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        elif action.action_type == ActionType.FORWARD:
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        elif action.action_type == ActionType.LABEL:
            label = action.parameters.get("label_name", "unlabeled")
            email.labels.append(label)
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        elif action.action_type == ActionType.ESCALATE:
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1
            email.labels.append("escalated")

        elif action.action_type == ActionType.DEFER:
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1
            email.labels.append("deferred")

        elif action.action_type == ActionType.SUMMARIZE:
            email.is_read = True
            email.is_handled = True
            state.inbox.unread_count -= 1

        # ─── Calculate Reward BEFORE advancing to next email ────
        # FIX: reward engine must run on the current email, not the next one.
        # Previously _next_email() was called first, making the ordering fragile.
        temp_result = ActionResult(
            success=True,
            action_type=action.action_type,
            email_id=action.email_id,
            is_correct=is_correct,
            is_partial=is_partial,
            metadata={"task_score": score},
        )

        reward_data = self.reward_engine.calculate(
            result=temp_result,
            email=email,
            step_count=state.step_count,
            max_steps=MAX_STEPS,
        )

        # FIX: removed external `score * 0.5` boost — task score is already
        # factored into is_correct/is_partial which the reward engine uses.
        # Double-counting inflated rewards and made the scale inconsistent.
        total_reward = reward_data["total_reward"]

        # ─── Move to Next Email (after reward is calculated) ────
        state.current_email = self._next_email(state)

        # ─── Final Result ────────────────────────────────────────
        result = ActionResult(
            success=True,
            action_type=action.action_type,
            email_id=action.email_id,
            reward=round(total_reward, 3),
            message=f"{action.action_type.value} executed successfully",
            is_correct=is_correct,
            is_partial=is_partial,
            metadata={
                **reward_data,
                "task_score": score,
            },
        )

        return state, result

    # ─── Next Email ─────────────────────────────────────────
    def _next_email(self, state: AgentState) -> Optional[Email]:
        """Move to next unhandled email in inbox."""
        for email in state.inbox.emails:
            if not email.is_handled:
                return email
        return None

    # ─── Invalid Action Result ──────────────────────────────
    def _invalid_action_result(self, action: Action) -> ActionResult:
        return ActionResult(
            success=False,
            action_type=action.action_type,
            email_id=action.email_id,
            reward=-0.5,
            message="Invalid action type",
            is_correct=False,
        )

    # ─── Missing Params Result ──────────────────────────────
    def _missing_params_result(self, action: Action) -> ActionResult:
        return ActionResult(
            success=False,
            action_type=action.action_type,
            email_id=action.email_id,
            reward=-0.5,
            message="Missing required parameters",
            is_correct=False,
        )