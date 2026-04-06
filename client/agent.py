# client/agent.py

import random
from typing import Dict, Any
from client.client import EmailTriageClient

# NEW: import heuristic
from utils.heuristics import guess_action

# Needed to build Email object
from env.models import Email


# ─── Base Agent ─────────────────────────────────────────────
class BaseAgent:

    def __init__(self, client: EmailTriageClient):
        self.client = client

    def select_action(self, observation: Dict[str, Any]) -> str:
        raise NotImplementedError


# ─── Random Agent ───────────────────────────────────────────
class RandomAgent(BaseAgent):
    """Agent that selects random actions."""

    def select_action(self, observation: Dict[str, Any]) -> str:
        available = observation.get("available_actions", [])
        if not available:
            return "read"
        return random.choice(available)


# ─── Heuristic Rule-Based Agent (UPGRADED) ──────────────────
class RuleBasedAgent(BaseAgent):
    """Agent that uses NLP heuristics to select actions."""

    def select_action(self, observation: Dict[str, Any]) -> str:

        available = observation.get("available_actions", [])

        # 🚨 Safety fallback
        if not available:
            return "read"

        # 🔥 NEW: use signals instead of heuristics
        signals = observation.get("signals", {})
        urgency = signals.get("urgency", 0.0)
        spam = signals.get("spam_score", 0.0)

        # 🔥 Decision logic
        if spam > 0.6 and "mark_spam" in available:
            return "mark_spam"
        if spam >= 0.3 and "mark_spam" in available:
            return "mark_spam"
        if urgency > 0.7 and "escalate" in available:
            return "escalate"

        if "read" in available:
            return "read"

        # fallback
        return available[0]

# ─── Agent Runner ───────────────────────────────────────────
class AgentRunner:

    def __init__(
        self,
        agent: BaseAgent,
        client: EmailTriageClient,
    ):
        self.agent = agent
        self.client = client

    def run(
        self,
        task_level: str = "easy",
        user_id: str = "agent",
        verbose: bool = True,
    ) -> Dict[str, Any]:

        # ─── Reset Environment ──────────────────────────────
        data = self.client.reset(
            user_id=user_id,
            task_level=task_level,
        )

        observation = data["observation"]
        episode_id = data["episode_id"]

        if verbose:
            print(f"\n{'='*50}")
            print(f"Episode  : {episode_id}")
            print(f"Task     : {task_level}")
            print(f"{'='*50}")

        total_reward = 0.0
        step = 0
        done = False

        # ─── Run Episode ────────────────────────────────────
        while not done:

            # ✅ FIX: stop if environment says done
            if observation.get("is_done"):
                break

            email_id = observation.get("email_id")

            # ✅ FIX: stop if no email
            if not email_id:
                break

            # Select action
            action_type = self.agent.select_action(observation)

            if verbose:
                print(f"\nStep {step + 1}")
                print(f"Email   : {observation.get('subject', '')}")
                print(f"Signals : {observation.get('signals', {})}")
                print(f"Action  : {action_type}")

            # Take step
            result = self.client.step(
                action_type=action_type,
                email_id=email_id,
            )

            observation = result["observation"]
            reward = result["reward"]
            done = result["done"]
            info = result["info"]

            total_reward += reward
            step += 1

            if verbose:
                print(f"Reward  : {reward:.3f}")
                print(f"Correct : {info.get('is_correct', False)}")

        # ─── Get Final Grade ────────────────────────────────
        grade = self.client.grade()

        if verbose:
            print(f"\n{'='*50}")
            print(f"Episode Complete!")
            print(f"Steps        : {step}")
            print(f"Total Reward : {total_reward:.3f}")
            print(f"Final Score  : {grade['final_score']}")
            print(f"Passed       : {grade['passed']}")
            print(f"{'='*50}\n")

        return grade