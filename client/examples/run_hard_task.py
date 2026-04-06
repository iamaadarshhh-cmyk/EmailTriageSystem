# client/examples/run_hard.py

from client.client import EmailTriageClient
from client.agent import RuleBasedAgent, RandomAgent, AgentRunner


def main():
    # ─── Setup ──────────────────────────────────────────────
    client = EmailTriageClient()

    # Check server is running
    if not client.health_check():
        print("❌ Server is not running!")
        print("Run: uvicorn server.app:app --reload --port 8000")
        return

    print("✅ Server is running!")

    # ─── Get Task Info ──────────────────────────────────────
    info = client.get_task_info("hard")
    print(f"\n📋 Task Info:")
    print(f"Description : {info['description']}")
    print(f"Actions     : {info['actions']}")
    print(f"Threshold   : {info['reward_threshold']}")

    # ─── Run Rule Based Agent ────────────────────────────────
    print("\n🤖 Running Rule Based Agent...")
    agent = RuleBasedAgent(client=client)
    runner = AgentRunner(agent=agent, client=client)

    rule_grade = runner.run(
        task_level="hard",
        user_id="rule_agent",
        verbose=True,
    )

    # ─── Run Random Agent ────────────────────────────────────
    print("\n🎲 Running Random Agent...")
    random_agent = RandomAgent(client=client)
    random_runner = AgentRunner(agent=random_agent, client=client)

    random_grade = random_runner.run(
        task_level="hard",
        user_id="random_agent",
        verbose=True,
    )

    # ─── Compare Results ─────────────────────────────────────
    print("\n📊 Comparison:")
    print(f"{'─'*40}")
    print(f"{'Agent':<20} {'Score':<10} {'Passed'}")
    print(f"{'─'*40}")
    print(f"{'Rule Based':<20} {rule_grade['final_score']:<10} {rule_grade['passed']}")
    print(f"{'Random':<20} {random_grade['final_score']:<10} {random_grade['passed']}")
    print(f"{'─'*40}")

    # ─── Close ──────────────────────────────────────────────
    client.close()


if __name__ == "__main__":
    main()


## What each file does

### `run_easy.py` — Level 1

# 1. Check server running
# 2. Run RuleBasedAgent on easy task
# 3. Print final score
# ```

# ### `run_medium.py` — Level 2
# ```
# 1. Check server running
# 2. Print task info
# 3. Run RuleBasedAgent on medium task
# 4. Print detailed score breakdown
# ```

# ### `run_hard.py` — Level 3
# ```
# 1. Check server running
# 2. Print task info
# 3. Run RuleBasedAgent on hard task
# 4. Run RandomAgent on hard task
# 5. Compare both agents side by side