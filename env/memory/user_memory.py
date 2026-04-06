# env/memory/user_memory.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from env.models import ActionType
from env.config import MAX_MEMORY_ENTRIES


# ─── Single Memory Entry ────────────────────────────────────
@dataclass
class MemoryEntry:
    email_id: str                       # Which email this memory is about
    action_taken: ActionType            # What action was taken
    # FIX: outcome is now a Literal type — prevents silent mismatches
    # from callers passing "Success", "correct", "1", etc.
    outcome: Literal["success", "failure"]
    reward: float                       # Reward earned
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)  # e.g. ["spam", "urgent"]


# ─── User Memory ────────────────────────────────────────────
@dataclass
class UserMemory:
    user_id: str                        # Whose memory this is
    entries: List[MemoryEntry] = field(default_factory=list)
    action_counts: Dict[str, int] = field(default_factory=dict)
    success_counts: Dict[str, int] = field(default_factory=dict)
    # FIX: removed total_entries manual counter — it could drift from
    # len(entries) if anything appended directly. Computed on demand instead.

    def add_entry(self, entry: MemoryEntry):
        """Add a new memory entry, enforcing a rolling size cap."""
        self.entries.append(entry)

        # FIX: cap memory size to prevent unbounded growth in long benchmarks
        if len(self.entries) > MAX_MEMORY_ENTRIES:
            self.entries.pop(0)

        # Track action counts
        action = entry.action_taken.value
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        # Track success counts
        if entry.outcome == "success":
            self.success_counts[action] = self.success_counts.get(action, 0) + 1

    def get_success_rate(self, action_type: ActionType) -> float:
        """Get success rate for a specific action."""
        action = action_type.value
        total = self.action_counts.get(action, 0)
        success = self.success_counts.get(action, 0)
        if total == 0:
            return 0.0
        return round(success / total, 2)

    def get_most_used_action(self) -> Optional[str]:
        """Get the action used most often."""
        if not self.action_counts:
            return None
        return max(self.action_counts, key=self.action_counts.get)

    def get_recent_entries(self, n: int) -> List[MemoryEntry]:
        """Get last n memory entries."""
        return self.entries[-n:]

    def get_entries_by_tag(self, tag: str) -> List[MemoryEntry]:
        """Get all entries with a specific tag."""
        return [e for e in self.entries if tag in e.tags]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of user memory."""
        return {
            "user_id": self.user_id,
            # FIX: use len(entries) as single source of truth
            "total_entries": len(self.entries),
            "action_counts": self.action_counts,
            # FIX: expose computed success rates instead of raw counts
            # so consumers don't have to recompute manually
            "success_rates": {
                action: self.get_success_rate(ActionType(action))
                for action in self.action_counts
            },
            "most_used_action": self.get_most_used_action(),
        }

## What each part does

### 🧠 `MemoryEntry`
# One **single memory** the agent stores about a past action:
# ```
# email_001 → READ → success → reward 1.0 → tagged ["work", "urgent"]
# ```

# ---

# ### 💾 `UserMemory`
# The agent's **long term memory** — stores all past experiences and tracks patterns.

# It has 5 helper methods:

# | Method | What it does |
# |---|---|
# | `add_entry()` | Adds a memory and updates action/success counts |
# | `get_success_rate()` | How often a specific action succeeded |
# | `get_most_used_action()` | Which action agent uses most |
# | `get_recent_entries(n)` | Last n memories |
# | `get_entries_by_tag()` | Find memories by tag e.g. "spam" |
# | `summary()` | Overall stats of agent's memory |

# ---

# ## How it connects
# ```
# Agent takes action
#       ↓
# ActionResult returned
#       ↓
# MemoryEntry created
#       ↓
# UserMemory.add_entry() called
#       ↓
# Agent uses memory to make
# better decisions next time