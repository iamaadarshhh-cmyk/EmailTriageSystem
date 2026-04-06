# client/client.py

import httpx
from typing import Dict, Any, Optional
from env.config import SERVER_HOST, SERVER_PORT


# ─── Email Triage Client ────────────────────────────────────
class EmailTriageClient:

    def __init__(
        self,
        host: str = SERVER_HOST,
        port: int = SERVER_PORT,
    ):
        self.base_url = f"http://{host}:{port}"
        self.client = httpx.Client(timeout=30.0)
        self.episode_id: Optional[str] = None

    # ─── Reset ──────────────────────────────────────────────
    def reset(
        self,
        user_id: str = "agent",
        task_level: str = "easy",
    ) -> Dict[str, Any]:
        """Reset environment and start new episode."""

        response = self.client.post(
            f"{self.base_url}/env/reset",
            json={
                "user_id": user_id,
                "task_level": task_level,
            }
        )
        response.raise_for_status()
        data = response.json()

        # Store episode ID
        self.episode_id = data["episode_id"]

        return data

    # ─── Step ───────────────────────────────────────────────
    def step(
        self,
        action_type: str,
        email_id: str,
        parameters: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Take one step in the environment."""

        if not self.episode_id:
            raise ValueError("No active episode. Call reset() first.")

        response = self.client.post(
            f"{self.base_url}/env/step",
            params={"episode_id": self.episode_id},
            json={
                "action_type": action_type,
                "email_id": email_id,
                "parameters": parameters,
            }
        )
        response.raise_for_status()
        return response.json()

    # ─── Render ─────────────────────────────────────────────
    def render(self) -> str:
        """Render current environment state."""

        if not self.episode_id:
            raise ValueError("No active episode. Call reset() first.")

        response = self.client.get(
            f"{self.base_url}/env/render/{self.episode_id}"
        )
        response.raise_for_status()
        return response.json()["render"]

    # ─── Grade ──────────────────────────────────────────────
    def grade(self) -> Dict[str, Any]:
        """Get final grade for current episode."""

        if not self.episode_id:
            raise ValueError("No active episode. Call reset() first.")

        response = self.client.get(
            f"{self.base_url}/grader/grade/{self.episode_id}"
        )
        response.raise_for_status()
        return response.json()

    # ─── Get Task Info ──────────────────────────────────────
    def get_task_info(self, level: str) -> Dict[str, Any]:
        """Get info about a specific task level."""

        response = self.client.get(
            f"{self.base_url}/tasks/{level}"
        )
        response.raise_for_status()
        return response.json()

    # ─── Health Check ───────────────────────────────────────
    def health_check(self) -> bool:
        """Check if server is running."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    # ─── Close ──────────────────────────────────────────────
    def close(self):
        """Close the client connection."""
        self.client.close()
        self.episode_id = None