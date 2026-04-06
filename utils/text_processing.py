# utils/text_processing.py

import re
from typing import List, Optional


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip().lower()


def get_preview(text: str, max_chars: int = 200) -> str:
    """Get preview of text."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text."""
    urgent_words = [
        "urgent", "asap", "immediately", "critical",
        "important", "priority", "deadline", "overdue",
    ]
    spam_words = [
        "win", "winner", "prize", "free", "click",
        "offer", "limited", "claim", "verify", "suspended",
    ]
    work_words = [
        "meeting", "project", "deadline", "report",
        "invoice", "presentation", "review", "update",
    ]

    text_lower = text.lower()
    found = []

    for word in urgent_words + spam_words + work_words:
        if word in text_lower:
            found.append(word)

    return found


def is_spam_like(text: str) -> bool:
    """Check if text looks like spam."""
    spam_indicators = [
        "click here", "win", "free", "prize",
        "verify your account", "suspended",
        "limited time", "act now",
    ]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in spam_indicators)


def truncate_email_body(body: str, max_words: int = 100) -> str:
    """Truncate email body to max words."""
    words = body.split()
    if len(words) <= max_words:
        return body
    return " ".join(words[:max_words]) + "..."