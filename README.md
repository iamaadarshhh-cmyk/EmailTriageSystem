---
title: Email Triage OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Email Triage RL Environment


An **OpenEnv-compatible reinforcement learning environment** where AI agents learn
to triage an inbox — classifying emails, routing escalations, and avoiding costly mistakes.

Built for the Meta × SST OpenEnv Hackathon.

---

## Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `spam_score` | float32 | [0, 1] | Likelihood email is spam |
| `urgency` | float32 | [0, 1] | Urgency signal from subject + body |
| `inbox_remaining` | int | [0, 50] | Emails left in inbox |
| `step_count` | int | [0, 20] | Steps taken this episode |

## Action Space

`Discrete(10)` — one per valid action:

| Index | Action | Description |
|---|---|---|
| 0 | read | Mark as read, no triage |
| 1 | reply | Send a reply |
| 2 | forward | Forward to team |
| 3 | delete | Delete (spam only) |
| 4 | archive | Archive for reference |
| 5 | label | Apply a label |
| 6 | mark_spam | Mark as spam |
| 7 | escalate | Route to specialist team |
| 8 | defer | Defer for later |
| 9 | summarize | Summarize content |

## Reward Function

| Component | Weight | Description |
|---|---|---|
| Correctness | +1.0 / +0.3 / -0.5 | Correct / partial / wrong action |
| Step penalty | -0.01 | Applied every step (efficiency) |
| Safety | +0.2 | Correctly identified spam |
| Safety | -0.3 | Deleted a work email |
| Safety | -0.5 | Replied to spam |

## Task Difficulty

| Level | Emails | Max Steps | Pass Threshold | Notes |
|---|---|---|---|---|
| easy | 3–10 random | 20 | 0.70 | Clear spam/work/personal categories |
| medium | 3–10 random | 20 | 0.70 | Urgency + meeting detection required |
| hard | 3–10 random | 20 | 0.70 | Multi-signal noise injected |

## Quickstart

```bash
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run agents
python scripts/run_easy_task.py
python scripts/run_medium_task.py
python scripts/run_hard_task.py
python scripts/run_benchmark.py
```

## Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

## Gymnasium API

```python
from env.openenv_wrapper import EmailTriageGymEnv

env = EmailTriageGymEnv(task_level="medium")
obs, info = env.reset()

for _ in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.close()
```

## Project Structure

```
email_triage_env/
├── env/           # Core environment, models, simulator
├── reward/        # Reward engine + components
├── tasks/         # Task definitions + corpus
├── server/        # FastAPI REST server
├── grader/        # Episode grader
├── client/        # HTTP client + agents
├── utils/         # Heuristics, noise, text processing
├── evaluation/    # Benchmark + metrics
└── tests/         # Full test suite
```