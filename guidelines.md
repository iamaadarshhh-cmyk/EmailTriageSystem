# 🏆 Hackathon Guidelines & Evaluation Strategy

This document outlines how the project aligns with hackathon expectations and how to maximize scoring.

---

## 🎯 1. Problem Clarity (Must Have)

✔ Clearly define:
- What problem you are solving
- Why it matters

✅ In this project:
- Problem: AI agents lack real-world decision environments
- Solution: RL-based email simulation system

---

## 💡 2. Innovation (High Weightage)

Judges look for:
- Unique idea
- Not a basic CRUD app

✅ Your Advantage:
- RL + Email Simulation (rare combo)
- Decision-based environment (not just prediction)

💡 Tip:
Highlight:
> “This is not a chatbot — it's a decision-making evaluation system.”

---

## ⚙️ 3. Technical Complexity

✔ Show depth:
- Modular architecture
- Reward system
- State transitions

✅ Must emphasize:
- Reward engineering
- Task difficulty scaling
- Agent-environment interaction

---

## 🧠 4. AI Usage (Critical)

✔ Must NOT look like:
- Simple API wrapper

✔ Must look like:
- System-level AI integration

✅ Show:
- Agent reasoning
- Action selection
- Evaluation metrics

---

## 📊 5. Scalability

Judges love scalable systems.

✔ Mention:
- Can add new tasks easily
- Can plug in different agents
- Can extend to other domains

---

## 🎨 6. UI / UX (Bonus Points)

Even if CLI-based:

✔ Improve:
- Tabular outputs
- Clean logs
- Structured display

💡 Optional:
- Add Streamlit / dashboard

---

## ⚡ 7. Performance & Efficiency

✔ Avoid:
- Hardcoded logic everywhere

✔ Show:
- Generalization
- Clean abstractions

---

## 🧪 8. Demonstration (VERY IMPORTANT)

Your demo should:

1. Show task running
2. Show agent making decisions
3. Show reward calculation
4. Show final score

Real-world task simulation

The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.

OpenEnv spec compliance

Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.

Minimum 3 tasks with agent graders

Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.

Meaningful reward function

Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).

Baseline inference script

Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.