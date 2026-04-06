# env/config.py

from pathlib import Path

# ─── Project Root ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ─── Data Paths ─────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = DATA_DIR / "templates"

# ─── Environment Settings ───────────────────────────────────
MAX_STEPS = 20               # Max steps per episode
MAX_EMAILS_PER_INBOX = 10   # Emails per simulated inbox
OBSERVATION_HISTORY = 5     # How many past steps to include

# ─── Task Difficulty Levels ─────────────────────────────────
TASK_LEVELS = ["easy", "medium", "hard"]

# ─── Action Space ───────────────────────────────────────────
VALID_ACTIONS = [
    "read",
    "reply",
    "forward",
    "delete",
    "archive",
    "label",
    "mark_spam",
    "escalate",
    "defer",
    "summarize",
]

# ─── Reward Settings ────────────────────────────────────────
REWARD_CORRECT_ACTION = 1.0
REWARD_WRONG_ACTION = -0.5
REWARD_PARTIAL_CREDIT = 0.3
REWARD_STEP_PENALTY = -0.01   # Small penalty per step (efficiency)

# ─── Grader Settings ────────────────────────────────────────
PASS_THRESHOLD = 0.7          # Score >= 0.7 considered passing

# ─── Server Settings ────────────────────────────────────────
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# ─── Logging ────────────────────────────────────────────────
LOG_DIR = BASE_DIR / "log_collector"
LOG_LEVEL = "DEBUG"