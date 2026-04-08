# env/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = DATA_DIR / "templates"

MAX_STEPS = 20
MAX_EMAILS_PER_INBOX = 10
OBSERVATION_HISTORY = 5
MAX_MEMORY_ENTRIES = 100        # ← was missing, caused ImportError in user_memory.py

TASK_LEVELS = ["easy", "medium", "hard"]

VALID_ACTIONS = [
    "read", "reply", "forward", "delete", "archive",
    "label", "mark_spam", "escalate", "defer", "summarize",
]

REWARD_CORRECT_ACTION = 1.0
REWARD_WRONG_ACTION = -0.5
REWARD_PARTIAL_CREDIT = 0.3
REWARD_STEP_PENALTY = -0.01

PASS_THRESHOLD = 0.7

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

LOG_DIR = BASE_DIR / "log_collector"
LOG_LEVEL = "DEBUG"