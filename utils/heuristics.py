# utils/heuristics.py

from typing import Optional
from env.models.state import Email
from env.models.actions import ActionType
from utils.text_processing import extract_keywords, is_spam_like


def guess_action(email: Email) -> ActionType:
    """
    Guess the best action for an email
    using simple heuristics.
    """

    subject = email.subject.lower()
    body = email.body.lower()
    combined = subject + " " + body

    # ─── Spam Detection ─────────────────────────────────────
    if is_spam_like(combined):
        return ActionType.MARK_SPAM

    # ─── Urgency Detection ──────────────────────────────────
    urgent_words = ["urgent", "asap", "critical", "immediately"]
    if any(word in combined for word in urgent_words):
        return ActionType.ESCALATE

    # ─── Meeting Detection ──────────────────────────────────
    meeting_words = ["meeting", "call", "presentation", "conference"]
    if any(word in combined for word in meeting_words):
        return ActionType.REPLY

    # ─── Invoice Detection ──────────────────────────────────
    invoice_words = ["invoice", "payment", "due", "bill"]
    if any(word in combined for word in invoice_words):
        return ActionType.FORWARD

    # ─── Summary Detection ──────────────────────────────────
    summary_words = ["summary", "summarize", "overview", "brief"]
    if any(word in combined for word in summary_words):
        return ActionType.SUMMARIZE

    # ─── Default ────────────────────────────────────────────
    return ActionType.READ


def get_email_priority(email: Email) -> str:
    """Get priority level of email."""
    subject = email.subject.lower()
    body = email.body.lower()
    combined = subject + " " + body

    if any(w in combined for w in ["urgent", "critical", "asap"]):
        return "high"
    elif any(w in combined for w in ["meeting", "deadline", "invoice"]):
        return "medium"
    else:
        return "low"




## What each file does

### `logger.py` — Logging System
# Sets up **3 loggers**:
# ```
# Console logger → prints colored logs to terminal
# app.log        → saves all logs to file (rotates at 10MB)
# error.log      → saves only errors (kept 30 days)
# ```

# Console output looks like:
# ```
# 2024-01-15 10:30:22 | INFO     | env:45 | Episode started | ep_a3f2b1
# 2024-01-15 10:30:23 | DEBUG    | env:67 | Step | ep_a3f2b1 | step=1
# 2024-01-15 10:30:24 | ERROR    | env:89 | Error occurred
# ```

### `text_processing.py` — Text Utilities
# 5 helper functions:

# | Function | What it does |
# |---|---|
# | `clean_text()` | Removes extra spaces and special characters |
# | `get_preview()` | Gets first 200 chars of email body |
# | `extract_keywords()` | Finds important words in email |
# | `is_spam_like()` | Checks if text looks like spam |
# | `truncate_email_body()` | Cuts email to max 100 words |

### `heuristics.py` — Smart Guessing
# Uses text analysis to **guess the best action**:
# ```
# spam words    → MARK_SPAM
# urgent words  → ESCALATE
# meeting words → REPLY
# invoice words → FORWARD
# summary words → SUMMARIZE
# default       → READ