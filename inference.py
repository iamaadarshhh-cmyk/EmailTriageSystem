import os
import json
import httpx
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Config ─────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://adarsh706-email-triage-openenv.hf.space"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # REQUIRED

MODEL_NAME = "gpt-4o-mini"
TASKS = ["task1_easy", "task2_medium", "task3_hard"]

# ─── Logging (MANDATORY FORMAT) ─────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ─── OpenAI Client (SAFE) ───────────────────────────────────
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# ─── Agent ──────────────────────────────────────────────────
class LLMAgent:

    def __init__(self):
        self.model = MODEL_NAME

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        email = observation.get("current_email")
        if not email:
            return self._fallback_action("no email")

        # fallback if no API key
        if not OPENAI_API_KEY or openai_client is None:
            return self._rule_based_action(observation)

        prompt = f"""
        Email:
        Subject: {email.get('subject','')}
        Body: {email.get('body','')}

        Choose one action:
        classify, reply, escalate, archive, delete

        Return JSON with action_type only.
        """

        try:
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )

            content = response.choices[0].message.content.strip()

            if "escalate" in content:
                return {"action_type": "escalate"}
            elif "reply" in content:
                return {"action_type": "reply"}
            elif "delete" in content:
                return {"action_type": "delete"}
            else:
                return {"action_type": "archive"}

        except Exception as e:
            return self._fallback_action(str(e))

    def _rule_based_action(self, observation):
        email = observation.get("current_email", {})
        text = (email.get("subject", "") + " " + email.get("body", "")).lower()

        if "urgent" in text or "error" in text:
            return {"action_type": "escalate"}
        elif "invoice" in text or "payment" in text:
            return {"action_type": "reply"}
        elif "spam" in text or "offer" in text:
            return {"action_type": "delete"}
        else:
            return {"action_type": "archive"}

    def _fallback_action(self, reason):
        return {"action_type": "archive"}

# ─── Env Client ─────────────────────────────────────────────
class EnvClient:

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.session_id = None

    def reset(self, task_id: str):
        r = self.client.post(f"{self.base_url}/reset/{task_id}")
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        return data

    def step(self, action: Dict):
        r = self.client.post(
            f"{self.base_url}/step",
            params={"session_id": self.session_id},
            json=action,
        )
        r.raise_for_status()
        return r.json()

    def grade(self):
        r = self.client.get(
            f"{self.base_url}/grade",
            params={"session_id": self.session_id},
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()

# ─── Run ────────────────────────────────────────────────────
def run_task(task_id: str):

    env = EnvClient(API_BASE_URL)
    agent = LLMAgent()

    rewards = []
    steps = 0
    success = False

    log_start(task_id, "email_triage", MODEL_NAME)

    try:
        data = env.reset(task_id)
        obs = data["observation"]
        done = False

        while not done:
            try:
                action = agent.select_action(obs)
                result = env.step(action)

                reward = result.get("reward", {}).get("immediate", 0.0)
                done = result.get("done", False)

                rewards.append(reward)
                steps += 1

                log_step(steps, action["action_type"], reward, done, None)

                obs = result["observation"]

            except Exception as e:
                log_step(steps, "error", 0.0, True, str(e))
                break

        try:
            grade = env.grade()
            score = grade.get("final_score", 0.0)
            success = grade.get("passed", False)
        except Exception:
            score = sum(rewards)
            success = False

    finally:
        env.close()
        log_end(success, steps, score, rewards)

# ─── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    for task in TASKS:
        run_task(task)