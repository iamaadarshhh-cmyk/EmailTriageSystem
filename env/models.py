# env/models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


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
    READ = "read"
    FORWARD = "forward"
    LABEL = "label"
    DEFER = "defer"
    SUMMARIZE = "summarize"


# ─── Email ───────────────────────────────────────────────────
class Email(BaseModel):
    email_id: str
    subject: str
    sender: str
    sender_name: str = ""
    recipient: str = "user@example.com"
    body: str
    timestamp: Any                       # accepts str or datetime
    thread_id: Optional[str] = None
    attachments: List[str] = []
    category: str = "work"               # spam / work / personal
    is_read: bool = False
    is_flagged: bool = False
    is_replied: bool = False             # ← transition.py references this
    is_deleted: bool = False             # ← transition.py references this
    is_handled: bool = False             # ← transition.py references this
    labels: List[str] = []              # ← transition.py appends to this


# ─── Inbox ───────────────────────────────────────────────────
class Inbox(BaseModel):
    inbox_id: str
    owner: str
    emails: List[Email]
    total_emails: int
    unread_count: int


# ─── AgentState ──────────────────────────────────────────────
class AgentState(BaseModel):
    inbox: Inbox
    current_email: Optional[Email] = None
    step_count: int = 0
    total_reward: float = 0.0
    action_history: List[str] = []
    is_done: bool = False


# ─── ActionValidator ─────────────────────────────────────────
class ActionValidator:
    VALID = {a.value for a in ActionType}

    @staticmethod
    def is_valid(action_type: str) -> bool:
        return action_type in ActionValidator.VALID

    @staticmethod
    def validate_parameters(action: "Action") -> bool:
        if action.action_type == ActionType.LABEL:
            return bool(action.parameters.get("label_name"))
        return True


# ─── Action ──────────────────────────────────────────────────
class Action(BaseModel):
    action_type: ActionType
    email_id: str
    parameters: Dict[str, Any] = {}
    triage: Optional[Dict[str, Any]] = None
    reply_body: Optional[str] = None
    escalate_to: Optional[str] = None
    reason: Optional[str] = None


# ─── ActionResult ────────────────────────────────────────────
class ActionResult(BaseModel):
    success: bool
    action_type: ActionType
    email_id: str
    reward: float = 0.0
    message: str = ""
    is_correct: bool = False
    is_partial: bool = False
    metadata: Dict[str, Any] = {}


# ─── TriageDecision ──────────────────────────────────────────
class TriageDecision(BaseModel):
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ─── GroundTruth ─────────────────────────────────────────────
class GroundTruth(BaseModel):
    expected_priority: Priority
    expected_category: Category
    expected_action: ActionType
    escalate_to: List[str] = []
    reply_keywords: List[str] = []
    forbidden_phrases: List[str] = []
    notes: str = ""


# ─── RewardBreakdown ─────────────────────────────────────────
class RewardBreakdown(BaseModel):
    priority_accuracy: float = 0.0
    category_accuracy: float = 0.0
    action_correctness: float = 0.0
    escalation_routing: float = 0.0
    reply_quality: float = 0.0
    penalties: float = 0.0
    total: float = 0.0
    feedback: str = ""