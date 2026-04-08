# 🚀 RL-Based Email Environment (Autonomous Agent Evaluation System)

## 📌 Overview

This project is an **Open Reinforcement Learning Environment** designed to evaluate and train AI agents on realistic email-handling tasks.

The system simulates a dynamic inbox where an agent must:
- Read emails
- Classify intent
- Take appropriate actions

It serves as a **benchmark environment** for testing decision-making, reasoning, and automation capabilities of AI systems.

---

## 🎯 Problem Statement

Modern AI agents struggle with:
- Real-world ambiguity
- Multi-step reasoning
- Contextual decision-making

This project solves that by creating a **structured yet flexible environment** where:
- Tasks vary in difficulty (easy → hard)
- Rewards are dynamically calculated
- Actions have consequences

---

## 🧠 Core Idea

We treat email handling as a **Reinforcement Learning problem**:

- **State** → Email content + metadata
- **Action** → Reply / Archive / Label / Escalate
- **Reward** → Based on correctness, timing, and relevance

---

## 🏗️ Architecture

### 1. Environment Layer
- Simulates inbox
- Generates tasks
- Maintains state transitions

### 2. Agent Layer
- Takes actions based on email input
- Can be rule-based or AI-powered (LLM / policy model)

### 3. Reward Engine
- Evaluates agent performance
- Modular scoring system:
  - Correctness
  - Efficiency
  - Relevance

### 4. Task System
- Predefined scenarios:
  - `easy`
  - `medium`
  - `hard`

---

## 🔄 Workflow

1. Load task
2. Present emails sequentially
3. Agent processes each email
4. Action is evaluated
5. Reward is assigned
6. Episode ends → total score computed

---

## 📊 Features

- ✅ Modular reward system
- ✅ Difficulty scaling
- ✅ Realistic email simulation
- ✅ Pluggable agents (LLM / custom policy)
- ✅ CLI-based execution
- ✅ Easy extensibility

---

## 📁 Project Structure

env/
├── core/
│   ├── __init__.py
│   ├── environment.py
│   ├── transition.py
│   └── observation.py
│
├── models.py
│   
│
├── memory/
│   ├── __init__.py
│   ├── user_memory.py
│   └── history.py
│
├── simulator.py
└── config.py
tasks/
├── definitions/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
└── corpus.py
|
server.py
|
client/
├── __init__.py
├── client.py
├── agent.py
└── examples/
    ├── run_easy.py
    ├── run_medium.py
    └── run_hard.py
reward/
├── reward_engine.py
└── components/
    ├── correctness.py
    ├── efficiency.py
    └── safety.py
evaluation/
├── __init__.py
├── metrics.py
└── benchmark.py
grader/
└── grader.py
data/
├── templates/
│   ├── spam.json
│   ├── work.json
│   └── personal.json
│
└── generators/
    ├── email_generator.py
    └── noise_injector.py
utils/
├── __init__.py
├── text_processing.py
├── heuristics.py
└── logger.py
tests/
├── test_env.py
├── test_reward.py
├── test_server.py
└── test_client.py
scripts/
├── run_server.py
├── debug_env.py
└── run_benchmark.py
log_collector/
├── __init__.py
├── trajectory_logger.py
└── event_logger.py
|
baseline.py
|
Dockerfile
|
test_env.py
|
openenv.yaml
|
openenv_wrapper.py