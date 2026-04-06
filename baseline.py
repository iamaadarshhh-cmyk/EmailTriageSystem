# baseline.py

import os
import json
import httpx
from groq import Groq
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# ─── Config ─────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

TASKS = ["task1_easy", "task2_medium", "task3_hard"]

# ─── Setup Groq ─────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)


# ─── LLM Agent ──────────────────────────────────────────────
class LLMAgent:

    def __init__(self):
        self.model = "llama-3.1-8b-instant"  # Free Groq model

    def select_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Use Groq LLaMA to select action."""

        email = observation.get("current_email")
        if not email:
            return self._fallback_action("no email")

        prompt = self._build_prompt(observation)

        try:
            response = groq_client.chat.completions.create(
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
                max_tokens=500,
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
Subject: {email['subject']}
From: {email['sender_name']} <{email['sender']}>
Body: {email['body']}
Attachments: {email.get('attachments', [])}

INBOX STATUS:
- Emails remaining: {remaining}
- Recent actions: {json.dumps(history[-3:], indent=2)}

AVAILABLE ACTIONS:
- classify: Categorize the email
- reply: Send a reply
- escalate: Route to a team
- archive: Archive for reference
- flag: Flag for follow-up
- delete: Delete (spam only!)
- skip: Skip this email (penalty)

VALID ESCALATION TARGETS:
on-call-dba, finance-team, security-team, engineering, billing-team,
account-manager, legal, data-privacy-team, ciso, hr, fraud-review-team,
safety-team, communications, ceo

CATEGORIES:
bug_report, feature_request, billing, general_inquiry,
complaint, praise, internal, spam, work, personal

PRIORITIES: urgent, high, medium, low, spam

DECISION RULES:
- Production outages → escalate to on-call-dba, priority=urgent
- Security/breach → escalate to security-team, priority=urgent
- Legal threats → escalate to ceo,legal,account-manager
- Billing/invoice → escalate to finance-team, priority=high
- Obvious spam/phishing → delete, priority=spam
- Customer praise → reply, priority=low
- Feature requests → reply, priority=medium
- Internal FYI → archive, priority=low
- GDPR requests → escalate to data-privacy-team

Respond ONLY with this exact JSON format, no other text:
{{
    "action_type": "escalate",
    "triage": {{
        "priority": "urgent",
        "category": "bug_report",
        "confidence": 0.95
    }},
    "escalate_to": "on-call-dba",
    "reply_body": null,
    "reason": "Production database down causing $12k/min revenue loss"
}}"""

    def _parse_response(self, content: str) -> Dict:
        try:
            import re
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Ensure required fields exist
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

    def _fallback_action(self, reason: str) -> Dict:
        """Safe fallback action that won't crash server."""
        return {
            "action_type": "archive",
            "triage": {
                "priority": "low",
                "category": "general_inquiry",
                "confidence": 0.5,
            },
            "escalate_to": "",
            "reply_body": "",
            "reason": f"Fallback: {reason}",
        }


# ─── Environment Client ──────────────────────────────────────
class EnvClient:

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.session_id = None

    def reset(self, task_id: str) -> Dict:
        response = self.client.post(
            f"{self.base_url}/reset/{task_id}"
        )
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
        if not response.is_success:
            print(f"⚠️ Server error: {response.status_code}")
            print(f"Detail: {response.text}")
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


# ─── Run Baseline ────────────────────────────────────────────
def run_baseline(
    task_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:

    env = EnvClient()
    agent = LLMAgent()

    if not env.health():
        print("❌ Server not running! Start with: python server.py")
        return {}

    try:
        data = env.reset(task_id)
        observation = data["observation"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task     : {task_id}")
            print(f"Session  : {env.session_id}")
            print(f"Emails   : {observation['inbox_remaining']}")
            print(f"Model    : llama-3.1-8b (Groq FREE)")
            print(f"{'='*60}")

        step = 0
        total_reward = 0.0
        done = False

        while not done:
            if observation.get("is_done"):
                break

            email = observation.get("current_email")
            if not email:
                break

            if verbose:
                print(f"\n📧 Step {step + 1}")
                print(f"Subject : {email['subject']}")
                print(f"From    : {email['sender']}")

            # Agent selects action
            action = agent.select_action(observation)

            if verbose:
                print(f"Action  : {action.get('action_type')}")
                print(f"Priority: {action.get('triage', {}).get('priority')}")
                print(f"Reason  : {str(action.get('reason', ''))[:80]}")

            # Take step
            try:
                result = env.step(action)
                observation = result["observation"]
                reward = result["reward"]["immediate"]
                done = result["done"]
                total_reward += reward
                step += 1

                if verbose:
                    print(f"Reward  : {reward:+.3f}")
                    print(f"Feedback: {result['reward']['feedback']}")

            except Exception as e:
                print(f"⚠️ Step error: {e}")
                break

        # Grade
        grade = env.grade()

        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Episode Complete!")
            print(f"Steps      : {step}")
            print(f"Reward     : {total_reward:+.3f}")
            print(f"Score      : {grade['final_score']:.4f}")
            print(f"Passed     : {'✅' if grade['passed'] else '❌'} {grade['passed']}")
            print(f"Threshold  : {grade['threshold']}")
            print(f"{'='*60}\n")

        return grade

    finally:
        env.close()


# ─── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, help="Specific task to run")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else TASKS
    results = []

    for task_id in tasks_to_run:
        print(f"\n🚀 Running {task_id}...")
        result = run_baseline(task_id, verbose=not args.quiet)
        if result:
            results.append(result)

    if results:
        print("\n📊 FINAL SUMMARY")
        print(f"{'─'*50}")
        print(f"{'Task':<25} {'Score':<10} {'Passed'}")
        print(f"{'─'*50}")
        for r in results:
            status = "✅" if r["passed"] else "❌"
            print(f"{r['task_id']:<25} {r['final_score']:<10.4f} {status}")
        avg = sum(r["final_score"] for r in results) / len(results)
        print(f"{'─'*50}")
        print(f"{'Average':<25} {avg:<10.4f}")

        with open("baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to baseline_results.json")