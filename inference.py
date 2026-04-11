import os
import json
import httpx
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Config ─────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://adarsh706-email-triage-openenv.hf.space"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # REQUIRED

TASKS = ["task1_easy", "task2_medium", "task3_hard"]

if not API_BASE_URL:
    print("❌ API_BASE_URL not set")
    exit(1)

# ─── Setup OpenAI (SAFE) ────────────────────────────────────
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None


# ─── LLM Agent ──────────────────────────────────────────────
class LLMAgent:

    def __init__(self):
        self.model = "gpt-4o-mini"

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        email = observation.get("current_email")
        if not email:
            return self._fallback_action("no email")

        # 🔥 fallback if no API key
        if not OPENAI_API_KEY or openai_client is None:
            return self._rule_based_action(observation)

        prompt = self._build_prompt(observation)

        try:
            response = openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert email triage assistant. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=400,
                temperature=0.1,
            )

            content = response.choices[0].message.content
            return self._parse_response(content)

        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            return self._fallback_action(str(e))

    def _build_prompt(self, observation: Dict) -> str:
        email = observation["current_email"]
        history = observation.get("triage_history", [])
        remaining = observation.get("inbox_remaining", 0)

        return f"""Analyze this email and decide the best action.

EMAIL:
Subject: {email.get('subject', '')}
From: {email.get('sender_name', '')} <{email.get('sender', '')}>
Body: {email.get('body', '')}

INBOX STATUS:
- Emails remaining: {remaining}
- Recent actions: {json.dumps(history[-3:], indent=2)}

AVAILABLE ACTIONS:
- classify
- reply
- escalate
- archive
- delete

Respond ONLY with JSON:
{{
    "action_type": "archive",
    "triage": {{
        "priority": "low",
        "category": "general_inquiry",
        "confidence": 0.8
    }},
    "reason": "Short explanation"
}}"""

    def _parse_response(self, content: str) -> Dict:
        try:
            import re

            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                if "action_type" not in parsed:
                    parsed["action_type"] = "archive"

                if "triage" not in parsed:
                    parsed["triage"] = {
                        "priority": "low",
                        "category": "general_inquiry",
                        "confidence": 0.5,
                    }

                return parsed

        except Exception as e:
            print(f"⚠️ Parse error: {e}")

        return self._fallback_action("parse error")

    # 🔥 Improved rule-based fallback
    def _rule_based_action(self, observation):
        email = observation.get("current_email", {})
        subject = email.get("subject", "").lower()
        body = email.get("body", "").lower()

        text = subject + " " + body

        if "urgent" in text or "error" in text:
            return {
                "action_type": "escalate",
                "triage": {
                    "priority": "urgent",
                    "category": "bug_report",
                    "confidence": 0.7,
                },
                "reason": "Detected urgency/error keywords",
            }
        elif "invoice" in text or "payment" in text:
            return {
                "action_type": "reply",
                "triage": {
                    "priority": "high",
                    "category": "billing",
                    "confidence": 0.7,
                },
                "reason": "Billing related content",
            }
        elif "spam" in text or "offer" in text:
            return {
                "action_type": "delete",
                "triage": {
                    "priority": "spam",
                    "category": "spam",
                    "confidence": 0.8,
                },
                "reason": "Spam detected",
            }
        else:
            return self._fallback_action("default rule")

    def _fallback_action(self, reason: str) -> Dict:
        return {
            "action_type": "archive",
            "triage": {
                "priority": "low",
                "category": "general_inquiry",
                "confidence": 0.5,
            },
            "reason": f"Fallback: {reason}",
        }


# ─── Environment Client ──────────────────────────────────────
class EnvClient:

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.session_id = None

    def reset(self, task_id: str) -> Dict:
        response = self.client.post(f"{self.base_url}/reset/{task_id}")
        response.raise_for_status()
        data = response.json()
        self.session_id = data["session_id"]
        return data

    def step(self, action: Dict) -> Dict:
        response = self.client.post(
            f"{self.base_url}/step",
            params={"session_id": self.session_id},
            json=action,
        )
        response.raise_for_status()
        return response.json()

    def grade(self) -> Dict:
        response = self.client.get(
            f"{self.base_url}/grade",
            params={"session_id": self.session_id},
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception:
            return False

    def close(self):
        self.client.close()


# ─── Run ─────────────────────────────────────────────────────
def run_baseline(task_id: str):

    env = EnvClient(API_BASE_URL)
    agent = LLMAgent()

    if not env.health():
        print("❌ Server not reachable!")
        return {}

    try:
        data = env.reset(task_id)
        observation = data["observation"]

        done = False

        while not done:
            try:
                action = agent.select_action(observation)
                result = env.step(action)

                observation = result["observation"]
                done = result["done"]

            except Exception as e:
                print(f"⚠️ Step error: {e}")
                break

        try:
            grade = env.grade()
            print(f"✅ Score: {grade['final_score']} | Passed: {grade['passed']}")
            return grade
        except Exception as e:
            print(f"⚠️ Grading failed: {e}")
            return {}

    finally:
        env.close()


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    for task_id in TASKS:
        print(f"\n🚀 Running {task_id}...")
        run_baseline(task_id)