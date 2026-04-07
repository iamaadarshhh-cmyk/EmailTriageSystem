# tests/test_reward.py

from reward.reward_engine import RewardEngine
from env.models import ActionResult, ActionType
from env.models import Email
from datetime import datetime


def mock_email(category="work"):
    return Email(
        email_id="test",
        subject="Test",
        sender="a",
        recipient="b",
        body="Test email",
        timestamp=datetime.now(),
        category=category,
    )


def test_correct_reward():
    engine = RewardEngine()

    result = ActionResult(
        success=True,
        action_type=ActionType.READ,
        email_id="1",
        is_correct=True,
    )

    reward = engine.calculate(result, mock_email(), 1, 20)

    assert reward["total_reward"] > 0


def test_wrong_reward():
    engine = RewardEngine()

    result = ActionResult(
        success=True,
        action_type=ActionType.DELETE,
        email_id="1",
        is_correct=False,
    )

    reward = engine.calculate(result, mock_email(), 1, 20)

    assert reward["total_reward"] < 0 or reward["total_reward"] <= 0


def test_safety_penalty():
    engine = RewardEngine()

    result = ActionResult(
        success=True,
        action_type=ActionType.REPLY,
        email_id="1",
        is_correct=False,
    )

    spam_email = mock_email(category="spam")

    reward = engine.calculate(result, spam_email, 1, 20)

    assert reward["safety"] <= 0