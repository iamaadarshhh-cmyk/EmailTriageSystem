# env/models.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ─── Enums ──────────────────────────────────────────────────
class Priority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPAM = "spam"


class Category(str, Enum):
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    BILLING = "billing"
    GENERAL_INQUIRY = "general_inquiry"
    COMPLAINT = "complaint"
    PRAISE = "praise"
    INTERNAL = "internal"
    SPAM = "spam"
    WORK = "work"
    PERSONAL = "personal"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    REPLY = "reply"
    ESCALATE = "escalate"
    ARCHIVE = "archive"
    FLAG = "flag"
    SKIP = "skip"
    DELETE = "delete"
    MARK_SPAM = "mark_spam"


# ─── Email Model ────────────────────────────────────────────
class Email(BaseModel):
    email_id: str
    subject: str
    sender: str
    sender_name: str = ""
    body: str
    timestamp: str
    thread_id: Optional[str] = None
    attachments: List[str] = []
    is_read: bool = False
    is_flagged: bool = False


# ─── Triage Decision ────────────────────────────────────────
class TriageDecision(BaseModel):
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ─── Action Model ───────────────────────────────────────────
class Action(BaseModel):
    action_type: ActionType
    triage: Optional[TriageDecision] = None
    reply_body: Optional[str] = None
    escalate_to: Optional[str] = None
    reason: Optional[str] = None


# ─── Reward Breakdown ───────────────────────────────────────
class RewardBreakdown(BaseModel):
    priority_accuracy: float = 0.0
    category_accuracy: float = 0.0
    action_correctness: float = 0.0
    escalation_routing: float = 0.0
    reply_quality: float = 0.0
    penalties: float = 0.0
    total: float = 0.0
    feedback: str = ""


# ─── Observation Model ──────────────────────────────────────
class Observation(BaseModel):
    current_email: Optional[Email] = None
    inbox_remaining: int = 0
    emails_processed: int = 0
    session_score: float = 0.0
    triage_history: List[Dict[str, Any]] = []
    task_description: str = ""
    available_actions: List[str] = []
    step_count: int = 0
    max_steps: int = 50
    is_done: bool = False


# ─── Ground Truth (hidden from agent) ───────────────────────
class GroundTruth(BaseModel):
    expected_priority: Priority
    expected_category: Category
    expected_action: ActionType
    escalate_to: List[str] = []
    reply_keywords: List[str] = []
    forbidden_phrases: List[str] = []
    notes: str = ""