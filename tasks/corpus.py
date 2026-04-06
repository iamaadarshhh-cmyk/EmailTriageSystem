# tasks/corpus.py

# tasks/corpus.py

from env.models import Priority, Category, ActionType, GroundTruth
# ─── Task 1 — Easy (5 emails) ───────────────────────────────
TASK1_EMAILS = [
    {
        "email_id": "t1_e1",
        "subject": "URGENT: Production database is down",
        "sender": "ops-alert@company.com",
        "sender_name": "Ops Alert System",
        "body": (
            "CRITICAL ALERT: Our production PostgreSQL database has gone offline at 14:32 UTC. "
            "All customer-facing services are impacted. Revenue loss is approximately $12,000/minute. "
            "On-call DBA needs to be paged immediately. "
            "Error logs: Connection refused on port 5432. Host: prod-db-01.internal"
        ),
        "timestamp": "2024-01-15T14:32:00Z",
        "attachments": ["error_logs.txt"],
        "ground_truth": GroundTruth(
            expected_priority=Priority.URGENT,
            expected_category=Category.BUG_REPORT,
            expected_action=ActionType.ESCALATE,
            escalate_to=["on-call-dba"],
            notes="Production outage — must escalate immediately",
        )
    },
    {
        "email_id": "t1_e2",
        "subject": "Please unsubscribe me from your mailing list",
        "sender": "customer123@gmail.com",
        "sender_name": "John Customer",
        "body": (
            "Hi, I've been receiving your newsletters for the past few months "
            "but I'm no longer interested. Could you please remove me from your mailing list? "
            "Thank you for your understanding."
        ),
        "timestamp": "2024-01-15T10:15:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.LOW,
            expected_category=Category.GENERAL_INQUIRY,
            expected_action=ActionType.REPLY,
            reply_keywords=["unsubscribed", "removed", "apologies"],
            notes="Simple unsubscribe — reply and process",
        )
    },
    {
        "email_id": "t1_e3",
        "subject": "Congratulations! You've won $1,000,000",
        "sender": "winner@prize-claim-now.com",
        "sender_name": "Prize Committee",
        "body": (
            "Dear Lucky Winner! You have been selected as the grand prize winner of $1,000,000! "
            "To claim your prize, click the link below and provide your bank details. "
            "This offer expires in 24 hours! Act now! Click here: http://suspicious-link.xyz/claim"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.SPAM,
            expected_category=Category.SPAM,
            expected_action=ActionType.DELETE,
            notes="Obvious phishing spam — delete immediately",
        )
    },
    {
        "email_id": "t1_e4",
        "subject": "Invoice #INV-2024-0892 — 30 Days Overdue",
        "sender": "billing@vendor-corp.com",
        "sender_name": "Vendor Corp Billing",
        "body": (
            "Dear Accounts Payable Team, "
            "This is a formal notice that Invoice #INV-2024-0892 for $47,500 "
            "is now 30 days past due. The original due date was December 15, 2023. "
            "Please arrange payment immediately to avoid service interruption and late fees. "
            "Contact: billing@vendor-corp.com | +1-800-555-0100"
        ),
        "timestamp": "2024-01-15T11:30:00Z",
        "attachments": ["invoice_INV-2024-0892.pdf"],
        "ground_truth": GroundTruth(
            expected_priority=Priority.HIGH,
            expected_category=Category.BILLING,
            expected_action=ActionType.ESCALATE,
            escalate_to=["finance-team"],
            notes="Overdue invoice — escalate to finance",
        )
    },
    {
        "email_id": "t1_e5",
        "subject": "Thank you for the amazing support!",
        "sender": "happy.customer@gmail.com",
        "sender_name": "Happy Customer",
        "body": (
            "Hi team! I just wanted to reach out and say how incredibly helpful "
            "your support team has been over the past week. "
            "Sarah in particular went above and beyond to resolve my issue. "
            "You've earned a customer for life! Keep up the great work!"
        ),
        "timestamp": "2024-01-15T13:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.LOW,
            expected_category=Category.PRAISE,
            expected_action=ActionType.REPLY,
            reply_keywords=["thank", "appreciate", "forward"],
            notes="Customer praise — reply warmly",
        )
    },
]

# ─── Task 2 — Medium (7 emails) ─────────────────────────────
TASK2_EMAILS = [
    {
        "email_id": "t2_e1",
        "subject": "Possible data breach — need immediate guidance",
        "sender": "dev.lead@company.com",
        "sender_name": "Dev Lead",
        "body": (
            "Hi, I've noticed some unusual access patterns in our S3 bucket logs. "
            "It looks like an external IP has been pulling customer PII data for the past 48 hours. "
            "I'm not sure if this is a breach or a misconfigured integration. "
            "We need someone from security to take a look ASAP before we notify customers."
        ),
        "timestamp": "2024-01-15T08:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.URGENT,
            expected_category=Category.BUG_REPORT,
            expected_action=ActionType.ESCALATE,
            escalate_to=["security-team"],
            notes="Potential data breach — security team only",
        )
    },
    {
        "email_id": "t2_e2",
        "subject": "API rate limits hitting our integration",
        "sender": "tech@partner-company.com",
        "sender_name": "Partner Tech Team",
        "body": (
            "Hello, we've been experiencing consistent 429 errors when hitting your API "
            "since yesterday afternoon. Our integration pulls data every 5 minutes "
            "and we're well within the documented 1000 req/hr limit. "
            "This is blocking our nightly batch job. Can someone investigate?"
        ),
        "timestamp": "2024-01-15T09:30:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.HIGH,
            expected_category=Category.BUG_REPORT,
            expected_action=ActionType.ESCALATE,
            escalate_to=["engineering"],
            notes="Platform bug — engineering team",
        )
    },
    {
        "email_id": "t2_e3",
        "subject": "Board deck review — need feedback by Friday",
        "sender": "cfo@company.com",
        "sender_name": "CFO",
        "body": (
            "Team, please review the attached Q4 board presentation by Friday EOD. "
            "Pay particular attention to slides 12-18 (financials) and 23-27 (roadmap). "
            "This goes to the board on Monday morning — no extensions. "
        ),
        "timestamp": "2024-01-15T10:00:00Z",
        "attachments": ["Q4_Board_Deck_v3.pptx"],
        "ground_truth": GroundTruth(
            expected_priority=Priority.HIGH,
            expected_category=Category.INTERNAL,
            expected_action=ActionType.FLAG,
            notes="Internal — flag for follow up, don't escalate",
        )
    },
    {
        "email_id": "t2_e4",
        "subject": "Your subscription renews in 3 days",
        "sender": "noreply@saas-tool.com",
        "sender_name": "SaaS Tool",
        "body": (
            "Hi there! Just a friendly reminder that your Pro subscription "
            "renews automatically in 3 days on January 18th for $299/month. "
            "No action needed if you wish to continue. "
            "To cancel or change your plan, visit your account settings."
        ),
        "timestamp": "2024-01-15T08:30:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.LOW,
            expected_category=Category.BILLING,
            expected_action=ActionType.ARCHIVE,
            notes="Routine billing notice — archive",
        )
    },
    {
        "email_id": "t2_e5",
        "subject": "Feature request: bulk export to CSV",
        "sender": "poweruser@enterprise-client.com",
        "sender_name": "Enterprise User",
        "body": (
            "Hi Product Team, our analytics team spends hours manually exporting data "
            "from your dashboard. A bulk CSV export with date range filters would save us "
            "at least 10 hours per week. We'd love to see this on the roadmap. "
            "Happy to jump on a call to discuss requirements!"
        ),
        "timestamp": "2024-01-15T11:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.MEDIUM,
            expected_category=Category.FEATURE_REQUEST,
            expected_action=ActionType.REPLY,
            reply_keywords=["thank", "roadmap", "consider", "feedback"],
            notes="Feature request — acknowledge and reply",
        )
    },
    {
        "email_id": "t2_e6",
        "subject": "RE: RE: RE: FWD: Meeting notes",
        "sender": "colleague@company.com",
        "sender_name": "Colleague",
        "body": (
            "Hey, I'm forwarding the notes from last Tuesday's all-hands. "
            "Nothing urgent here, just FYI. "
            "Let me know if you have questions! "
            "---- Original Message ---- "
            "From: admin@company.com: Meeting notes attached."
        ),
        "timestamp": "2024-01-15T12:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.LOW,
            expected_category=Category.INTERNAL,
            expected_action=ActionType.ARCHIVE,
            notes="Internal FYI — archive",
        )
    },
    {
        "email_id": "t2_e7",
        "subject": "Your account has been compromised — act now",
        "sender": "security-alert@paypa1.com",
        "sender_name": "PayPal Security",
        "body": (
            "We detected unusual activity on your account. "
            "Your account has been temporarily limited. "
            "Please verify your identity immediately: http://paypa1-secure.xyz/verify "
            "Failure to verify within 24 hours will result in permanent account closure."
        ),
        "timestamp": "2024-01-15T07:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.SPAM,
            expected_category=Category.SPAM,
            expected_action=ActionType.DELETE,
            notes="Phishing — note misspelled paypa1.com",
        )
    },
]

# ─── Task 3 — Hard (5 emails) ───────────────────────────────
TASK3_EMAILS = [
    {
        "email_id": "t3_e1",
        "subject": "Enterprise contract termination + legal action notice",
        "sender": "legal@big-enterprise.com",
        "sender_name": "Legal Department",
        "body": (
            "Dear Sir/Madam, "
            "We are writing to formally notify you of our decision to terminate "
            "our enterprise contract (Contract #ENT-2022-0445, value: $2.4M annually) "
            "effective 30 days from this notice, citing repeated SLA breaches. "
            "Furthermore, we are instructing our legal team to pursue damages. "
            "We expect a response from your executive team within 48 hours."
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "attachments": ["termination_notice.pdf", "sla_breach_evidence.xlsx"],
        "ground_truth": GroundTruth(
            expected_priority=Priority.URGENT,
            expected_category=Category.COMPLAINT,
            expected_action=ActionType.ESCALATE,
            escalate_to=["ceo", "legal", "account-manager"],
            reply_keywords=["acknowledge", "executive", "48 hours", "resolve"],
            forbidden_phrases=["ignore", "reject", "impossible"],
            notes="Multi-team escalation + careful reply required",
        )
    },
    {
        "email_id": "t3_e2",
        "subject": "GDPR Article 17 — Right to Erasure Request",
        "sender": "data.rights@eu-customer.de",
        "sender_name": "EU Customer",
        "body": (
            "To Whom It May Concern, "
            "Under GDPR Article 17, I am formally requesting the erasure of all "
            "personal data you hold about me. My registered email is: data.rights@eu-customer.de "
            "Please confirm receipt of this request and provide a timeline for completion. "
            "I understand this must be completed within 30 days per regulation."
        ),
        "timestamp": "2024-01-15T10:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.HIGH,
            expected_category=Category.GENERAL_INQUIRY,
            expected_action=ActionType.ESCALATE,
            escalate_to=["data-privacy-team"],
            reply_keywords=["30 days", "GDPR", "erasure", "confirm"],
            forbidden_phrases=["cannot", "impossible", "decline"],
            notes="GDPR request — must mention 30 days and GDPR in reply",
        )
    },
    {
        "email_id": "t3_e3",
        "subject": "Concerned about AI safety in your product",
        "sender": "researcher@ai-safety-institute.org",
        "sender_name": "AI Safety Researcher",
        "body": (
            "Hello, I'm a researcher at the AI Safety Institute. "
            "We've been studying your AI product and have identified several behaviors "
            "that may pose safety risks to vulnerable users, including potential for "
            "manipulation and misinformation generation. "
            "We'd like to schedule a responsible disclosure meeting before we publish "
            "our findings in 60 days."
        ),
        "timestamp": "2024-01-15T11:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.URGENT,
            expected_category=Category.COMPLAINT,
            expected_action=ActionType.ESCALATE,
            escalate_to=["safety-team", "legal", "communications"],
            reply_keywords=["meeting", "discuss", "take seriously"],
            forbidden_phrases=["dismiss", "reject", "unfounded"],
            notes="AI safety disclosure — handle very carefully",
        )
    },
    {
        "email_id": "t3_e4",
        "subject": "Suspicious activity from employee account — insider threat?",
        "sender": "security@company.com",
        "sender_name": "Security Team",
        "body": (
            "CONFIDENTIAL — We've detected unusual data access patterns from "
            "employee account john.doe@company.com over the past 72 hours. "
            "This includes bulk downloads of customer PII and IP documents at odd hours. "
            "We recommend immediate account suspension pending investigation. "
            "Please advise on next steps — time is critical."
        ),
        "timestamp": "2024-01-15T07:30:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.URGENT,
            expected_category=Category.INTERNAL,
            expected_action=ActionType.ESCALATE,
            escalate_to=["ciso", "hr", "legal"],
            reply_keywords=["suspend", "investigation", "confidential"],
            notes="Insider threat — CISO + HR + Legal",
        )
    },
    {
        "email_id": "t3_e5",
        "subject": "Payroll processing blocked — 500 employees affected",
        "sender": "payroll@company.com",
        "sender_name": "Payroll Team",
        "body": (
            "URGENT: Our payroll processing for this Friday has been blocked "
            "by what appears to be a fraud detection false positive in our payment gateway. "
            "500 employees will not receive their salaries if this is not resolved in the next 6 hours. "
            "We need immediate escalation to the payment provider and account manager."
        ),
        "timestamp": "2024-01-15T06:00:00Z",
        "ground_truth": GroundTruth(
            expected_priority=Priority.URGENT,
            expected_category=Category.BILLING,
            expected_action=ActionType.ESCALATE,
            escalate_to=["fraud-review-team", "account-manager"],
            reply_keywords=["urgent", "6 hours", "escalated", "payroll"],
            notes="Time critical — payroll blocked",
        )
    },
]

# ─── Task Registry ──────────────────────────────────────────
TASKS = {
    "task1_easy": {
        "emails": TASK1_EMAILS,
        "description": "Basic inbox triage — classify and handle 5 clear emails",
        "max_steps": 20,
        "pass_threshold": 0.70,
        "difficulty": "easy",
    },
    "task2_medium": {
        "emails": TASK2_EMAILS,
        "description": "Mixed priority inbox — requires contextual reasoning",
        "max_steps": 35,
        "pass_threshold": 0.70,
        "difficulty": "medium",
    },
    "task3_hard": {
        "emails": TASK3_EMAILS,
        "description": "High stakes escalations — all require multi-team routing + replies",
        "max_steps": 25,
        "pass_threshold": 0.65,
        "difficulty": "hard",
    },
}