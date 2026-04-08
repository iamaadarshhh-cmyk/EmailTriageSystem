# env/simulator.py
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

from env.models import Email, Inbox
from env.config import MAX_EMAILS_PER_INBOX, TEMPLATES_DIR


class EmailSimulator:

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, List[Dict]]:
        templates = {}
        for category in ["spam", "work", "personal"]:
            path = TEMPLATES_DIR / f"{category}.json"
            if path.exists():
                with open(path, "r") as f:
                    templates[category] = json.load(f)
            else:
                templates[category] = self._default_templates(category)
        return templates

    def _default_templates(self, category: str) -> List[Dict]:
        defaults = {
            "spam": [
                {"subject": "You have won a prize!", "sender": "noreply@spam.com",
                 "body": "Click here to claim your prize worth $1000!"},
            ],
            "work": [
                {"subject": "Meeting at 3pm today", "sender": "manager@company.com",
                 "body": "Please join the team meeting at 3pm in conference room B."},
            ],
            "personal": [
                {"subject": "Weekend plans?", "sender": "friend@gmail.com",
                 "body": "Hey! Are you free this weekend? Let's catch up!"},
            ],
        }
        return defaults[category]

    def generate_email(self, category: str = None) -> Email:
        if category is None:
            category = random.choice(["spam", "work", "personal"])
        templates = self.templates.get(category, [])
        template = random.choice(templates)
        return Email(
            email_id=f"email_{uuid.uuid4().hex[:8]}",
            subject=template["subject"],
            sender=template["sender"],
            recipient="user@example.com",   # ← fixed (was missing)
            body=template["body"],
            timestamp=str(self._random_timestamp()),
            category=category,
        )

    def generate_inbox(
        self,
        owner: str = "user",
        n_emails: int = None,
        mix: Dict[str, int] = None,
    ) -> Inbox:
        if n_emails is None:
            n_emails = random.randint(3, MAX_EMAILS_PER_INBOX)
        emails = []
        if mix:
            for category, count in mix.items():
                for _ in range(count):
                    emails.append(self.generate_email(category))
        else:
            for _ in range(n_emails):
                emails.append(self.generate_email())
        random.shuffle(emails)
        return Inbox(
            inbox_id=f"inbox_{uuid.uuid4().hex[:8]}",
            owner=owner,
            emails=emails,
            total_emails=len(emails),
            unread_count=len(emails),
        )

    def _random_timestamp(self) -> datetime:
        return datetime.now() - timedelta(
            days=random.randint(0, 7),
            hours=random.randint(0, 23),
        )