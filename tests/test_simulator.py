# tests/test_simulator.py

from env.simulator import EmailSimulator
from env.models import Email, Inbox


def test_generate_email():
    simulator = EmailSimulator()

    email = simulator.generate_email()

    assert isinstance(email, Email)
    assert email.email_id is not None
    assert email.subject != ""
    assert email.sender != ""
    assert email.body != ""
    assert email.category in ["spam", "work", "personal"]


def test_generate_email_with_category():
    simulator = EmailSimulator()

    email = simulator.generate_email(category="spam")

    assert email.category == "spam"


def test_generate_inbox_default():
    simulator = EmailSimulator()

    inbox = simulator.generate_inbox(owner="test_user")

    assert isinstance(inbox, Inbox)
    assert inbox.owner == "test_user"
    assert len(inbox.emails) > 0
    assert inbox.total_emails == len(inbox.emails)
    assert inbox.unread_count == len(inbox.emails)


def test_generate_inbox_with_count():
    simulator = EmailSimulator()

    inbox = simulator.generate_inbox(owner="test_user", n_emails=5)

    assert len(inbox.emails) == 5
    assert inbox.total_emails == 5


def test_generate_inbox_with_mix():
    simulator = EmailSimulator()

    mix = {
        "spam": 2,
        "work": 2,
        "personal": 1,
    }

    inbox = simulator.generate_inbox(owner="test_user", mix=mix)

    categories = [email.category for email in inbox.emails]

    assert categories.count("spam") == 2
    assert categories.count("work") == 2
    assert categories.count("personal") == 1


def test_random_timestamp():
    simulator = EmailSimulator()

    email = simulator.generate_email()

    assert email.timestamp is not None