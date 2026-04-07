# tests/test_server.py

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_reset():
    response = client.post("/env/reset", json={
        "user_id": "test",
        "task_level": "easy"
    })

    assert response.status_code == 200
    data = response.json()

    assert "episode_id" in data
    assert "observation" in data


def test_step():
    reset = client.post("/env/reset", json={
        "user_id": "test",
        "task_level": "easy"
    }).json()

    episode_id = reset["episode_id"]
    obs = reset["observation"]

    response = client.post(
        f"/env/step?episode_id={episode_id}",
        json={
            "action_type": "read",
            "email_id": obs["email_id"],
            "parameters": {}
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert "reward" in data
    assert "observation" in data