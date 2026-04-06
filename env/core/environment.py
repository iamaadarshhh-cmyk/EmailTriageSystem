# env/core/environment.py

import uuid
from typing import Dict, Any, Optional, Tuple
from env.models import AgentState, Inbox
from env.models import Action, ActionResult, ActionType
from env.core.observation import ObservationBuilder
from env.core.transition import TransitionEngine
from env.memory.history import EpisodeHistory, StepRecord
from env.memory.user_memory import UserMemory, MemoryEntry
from env.simulator import EmailSimulator
from env.config import MAX_STEPS


# ─── Main Environment ───────────────────────────────────────
class EmailTriageEnvironment:

    def __init__(self, task=None):
        self.task = task                          # Current task
        self.state: Optional[AgentState] = None  # Current state
        self.history: Optional[EpisodeHistory] = None  # Episode history
        self.memory: Optional[UserMemory] = None  # Agent memory
        self.simulator = EmailSimulator()         # Email generator
        self.transition = TransitionEngine()      # Step engine
        self.episode_id = None                    # Current episode ID

    # ─── Reset ──────────────────────────────────────────────
    def reset(self, user_id: str = "agent") -> Dict[str, Any]:
        """Reset environment and start a new episode."""

        # Generate new episode ID
        self.episode_id = f"ep_{uuid.uuid4().hex[:8]}"

        # Generate new inbox using simulator
        inbox = self.simulator.generate_inbox(owner=user_id)

        # Set first email as current
        first_email = inbox.emails[0] if inbox.emails else None

        # Create fresh agent state
        self.state = AgentState(
            inbox=inbox,
            current_email=first_email,
        )

        # Create fresh episode history
        self.history = EpisodeHistory(
            episode_id=self.episode_id,
            task_level=self.task.level if self.task else "easy",
        )

        # Create fresh user memory
        self.memory = UserMemory(user_id=user_id)

        # Return initial observation
        return ObservationBuilder.build(self.state)

    # ─── Step ───────────────────────────────────────────────
    def step(
        self,
        action: Action,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take one step in the environment.
        Returns: observation, reward, done, info
        """

        if self.state is None:
            raise ValueError("Environment not reset. Call reset() first.")

       

        # Execute transition
        self.state, result = self.transition.step(
            self.state, action, self.task
        )

        # 🚨 HARD SAFETY STOP
        if self.state.step_count >= MAX_STEPS:
            self.state.is_done = True
            self.state.current_email = None

        # Record step in history
        step_record = StepRecord(
            step_number=self.state.step_count,
            email_id=action.email_id,
            action=action,
            result=result,
            reward=result.reward,   #added this line to fix error
        )
        self.history.add_step(step_record)

        # Add to user memory
        memory_entry = MemoryEntry(
            email_id=action.email_id,
            action_taken=action.action_type,
            outcome="success" if result.is_correct else "failure",
            reward=result.reward,
        )
        self.memory.add_entry(memory_entry)

        # Build new observation
        observation = ObservationBuilder.build(self.state)

        # Info dictionary
        info = {
            "episode_id": self.episode_id,
            "is_correct": result.is_correct,
            "is_partial": result.is_partial,
            "message": result.message,
            "step_count": self.state.step_count,
            "memory_summary": self.memory.summary(),
        }

        # Check if done
        if self.state.is_done:
            self.history.complete()

        return observation, result.reward, self.state.is_done, info

   

    # ─── Render ─────────────────────────────────────────────
    def render(self) -> str:
        """Print current state of environment."""
        if self.state is None:
            return "Environment not initialized."

        email = self.state.current_email
        if email is None:
            return "Inbox is empty."

        return (
            f"\n{'='*50}\n"
            f"Episode  : {self.episode_id}\n"
            f"Step     : {self.state.step_count}/{MAX_STEPS}\n"
            f"Reward   : {self.state.total_reward:.2f}\n"
            f"{'─'*50}\n"
            f"Email ID : {email.email_id}\n"
            f"Subject  : {email.subject}\n"
            f"Sender   : {email.sender}\n"
            f"Category : {email.category}\n"
            f"{'─'*50}\n"
            f"Unread   : {self.state.inbox.unread_count}\n"
            f"{'='*50}\n"
        )

    # ─── Close ──────────────────────────────────────────────
    def close(self):
        """Clean up environment."""
        self.state = None
        self.history = None
        self.memory = None





# ## What it does in one line
# > `EmailTriageEnvironment` is the **main controller** — it connects all the pieces together into one complete environment. 🎮

# ---

# ## How all files connect here
# ```
# EmailTriageEnvironment
#         │
#         ├── EmailSimulator     → generates inbox + emails
#         ├── TransitionEngine   → executes actions + rewards
#         ├── ObservationBuilder → builds what agent sees
#         ├── EpisodeHistory     → records all steps
#         └── UserMemory         → stores agent experiences