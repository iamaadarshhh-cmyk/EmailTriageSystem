# tests/test_env.py

from env.core.environment import EmailTriageEnvironment
from env.models.actions import Action, ActionType
from tasks.task_factory import TaskFactory


def test_environment_reset():
    task = TaskFactory.create("easy")
    env = EmailTriageEnvironment(task=task)

    obs = env.reset(user_id="test_user")

    assert obs is not None
    assert "email_id" in obs
    assert env.state is not None


def test_environment_step():
    task = TaskFactory.create("easy")
    env = EmailTriageEnvironment(task=task)

    obs = env.reset(user_id="test_user")

    action = Action(
        action_type=ActionType.READ,
        email_id=obs["email_id"],
    )

    new_obs, reward, done, info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "is_correct" in info


def test_environment_done():
    task = TaskFactory.create("easy")
    env = EmailTriageEnvironment(task=task)

    obs = env.reset(user_id="test_user")

    # Force steps until done
    for _ in range(25):
        if obs["is_done"]:
            break

        action = Action(
            action_type=ActionType.READ,
            email_id=obs["email_id"],
        )

        obs, _, _, _ = env.step(action)

    assert env.state.is_done is True