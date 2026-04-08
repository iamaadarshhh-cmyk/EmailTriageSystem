# utils/noise_injector.py
"""
Noise injection for harder task difficulty.
Adds misleading signals to emails so agents can't rely on
simple keyword matching — forces genuine reasoning.
"""
import random
import copy
from env.models import Email


URGENCY_NOISE = [
    "Please note this is for your records.",
    "No action required at this time.",
    "FYI only.",
]

SPAM_NOISE_PHRASES = [
    "Click here to unsubscribe.",
    "This is a limited time offer for our members.",
    "Win a chance to earn rewards.",
]

LEGITIMATE_URGENCY_PHRASES = [
    "This is time-sensitive.",
    "Please respond ASAP.",
    "Critical update required.",
]


class NoiseInjector:
    """
    Injects controlled noise into emails to increase task difficulty.

    noise_level=0.0 → no change (easy tasks)
    noise_level=0.3 → moderate noise (medium tasks)
    noise_level=0.6 → heavy noise (hard tasks)
    """

    def __init__(self, noise_level: float = 0.3, seed: int = None):
        self.noise_level = noise_level
        if seed is not None:
            random.seed(seed)

    def inject(self, email: Email) -> Email:
        """Return a noisy copy of the email. Does not mutate original."""
        if random.random() > self.noise_level:
            return email

        noisy = email.model_copy(deep=True)
        strategy = random.choice([
            self._add_spam_noise_to_legit,
            self._add_urgency_noise_to_spam,
            self._truncate_body,
            self._swap_sender_domain,
        ])
        return strategy(noisy)

    def inject_batch(self, emails: list) -> list:
        return [self.inject(e) for e in emails]

    # ── Strategies ─────────────────────────────────────────

    def _add_spam_noise_to_legit(self, email: Email) -> Email:
        """Add spam-looking phrases to a legitimate email."""
        if email.category != "spam":
            phrase = random.choice(SPAM_NOISE_PHRASES)
            email.body = email.body + f"\n\n{phrase}"
        return email

    def _add_urgency_noise_to_spam(self, email: Email) -> Email:
        """Add urgency phrases to spam to trick escalation logic."""
        if email.category == "spam":
            phrase = random.choice(LEGITIMATE_URGENCY_PHRASES)
            email.subject = phrase + " — " + email.subject
        return email

    def _truncate_body(self, email: Email) -> Email:
        """Simulate partial email body — harder to classify."""
        words = email.body.split()
        cutoff = max(10, len(words) // 2)
        email.body = " ".join(words[:cutoff]) + "... [message truncated]"
        return email

    def _swap_sender_domain(self, email: Email) -> Email:
        """Change sender domain to look legitimate or suspicious."""
        if email.category == "spam":
            # Make spam look legit
            email.sender = email.sender.replace(
                email.sender.split("@")[-1], "company-notifications.com"
            )
        else:
            # Make legit look suspicious
            original_domain = email.sender.split("@")[-1]
            email.sender = email.sender.replace(
                original_domain, original_domain.replace(".", "-") + ".xyz"
            )
        return email