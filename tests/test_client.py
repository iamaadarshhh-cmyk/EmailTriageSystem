# tests/test_client.py

from client.client import EmailTriageClient


def test_client_health():
    client = EmailTriageClient()
    assert isinstance(client.health_check(), bool)


def test_client_reset_and_step():
    client = EmailTriageClient()

    if not client.health_check():
        return  # skip if server not running

    data = client.reset(task_level="easy")
    obs = data["observation"]

    result = client.step(
        action_type="read",
        email_id=obs["email_id"],
    )

    assert "reward" in result
    assert "observation" in result

    client.close()