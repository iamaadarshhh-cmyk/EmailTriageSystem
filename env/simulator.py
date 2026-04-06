# env/simulator.py

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from env.models import Email, Inbox
from env.config import (
    MAX_EMAILS_PER_INBOX,
    TEMPLATES_DIR,
)


# ─── Email Simulator ────────────────────────────────────────
class EmailSimulator:

    def __init__(self):
        self.templates = self._load_templates()

    # ─── Load Templates ─────────────────────────────────────
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """Load email templates from JSON files."""
        templates = {}
        categories = ["spam", "work", "personal"]

        for category in categories:
            path = TEMPLATES_DIR / f"{category}.json"
            if path.exists():
                with open(path, "r") as f:
                    templates[category] = json.load(f)
            else:
                templates[category] = self._default_templates(category)

        return templates

    # ─── Default Templates ──────────────────────────────────
    def _default_templates(self, category: str) -> List[Dict]:
        """Fallback templates if JSON files are missing."""
        defaults = {
            "spam": [
                {
                    "subject": "You have won a prize!",
                    "sender": "noreply@spam.com",
                    "body": "Click here to claim your prize worth $1000!"
                },
                {
                    "subject": "Urgent: Verify your account",
                    "sender": "security@fakebank.com",
                    "body": "Your account will be suspended. Click here now!"
                },
            ],
            "work": [
                {
                    "subject": "Meeting at 3pm today",
                    "sender": "manager@company.com",
                    "body": "Please join the team meeting at 3pm in conference room B."
                },
                {
                    "subject": "Project update required",
                    "sender": "lead@company.com",
                    "body": "Can you send me the latest update on the project by EOD?"
                },
            ],
            "personal": [
                {
                    "subject": "Weekend plans?",
                    "sender": "friend@gmail.com",
                    "body": "Hey! Are you free this weekend? Let's catch up!"
                },
                {
                    "subject": "Happy Birthday!",
                    "sender": "family@gmail.com",
                    "body": "Wishing you a wonderful birthday! Hope you have a great day!"
                },
            ],
        }
        return defaults[category]

    # ─── Generate Single Email ──────────────────────────────
    def generate_email(self, category: str = None) -> Email:
        """Generate one random email."""
        if category is None:
            category = random.choice(["spam", "work", "personal"])

        templates = self.templates.get(category, [])
        template = random.choice(templates)

        return Email(
            email_id=f"email_{uuid.uuid4().hex[:8]}",
            subject=template["subject"],
            sender=template["sender"],
            recipient="user@example.com",
            body=template["body"],
            timestamp=self._random_timestamp(),
            category=category,
        )

    # ─── Generate Inbox ─────────────────────────────────────
    def generate_inbox(
        self,
        owner: str = "user",
        n_emails: int = None,
        mix: Dict[str, int] = None
    ) -> Inbox:
        """Generate a full inbox with mixed emails."""

        if n_emails is None:
            n_emails = random.randint(3, MAX_EMAILS_PER_INBOX)

        emails = []

        if mix:
            # Generate specific mix e.g. {"spam": 2, "work": 3, "personal": 1}
            for category, count in mix.items():
                for _ in range(count):
                    emails.append(self.generate_email(category))
        else:
            # Generate random mix
            for _ in range(n_emails):
                emails.append(self.generate_email())

        # Shuffle so emails aren't grouped by category
        random.shuffle(emails)

        return Inbox(
            inbox_id=f"inbox_{uuid.uuid4().hex[:8]}",
            owner=owner,
            emails=emails,
            total_emails=len(emails),
            unread_count=len(emails),
        )

    # ─── Random Timestamp ───────────────────────────────────
    def _random_timestamp(self) -> datetime:
        """Generate a random timestamp within the last 7 days."""
        days_ago = random.randint(0, 7)
        hours_ago = random.randint(0, 23)
        return datetime.now() - timedelta(days=days_ago, hours=hours_ago)