# client/examples/run_medium.py

from client.client import EmailTriageClient
from client.agent import RuleBasedAgent, AgentRunner


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
    info = client.get_task_info("medium")
    print(f"\n📋 Task Info:")
    print(f"Description : {info['description']}")
    print(f"Actions     : {info['actions']}")
    print(f"Threshold   : {info['reward_threshold']}")

    # ─── Run Medium Task ─────────────────────────────────────
    agent = RuleBasedAgent(client=client)
    runner = AgentRunner(agent=agent, client=client)

    grade = runner.run(
        task_level="medium",
        user_id="agent_medium",
        verbose=True,
    )

    # ─── Print Results ──────────────────────────────────────
    print("\n📊 Final Results:")
    print(f"Score      : {grade['final_score']}")
    print(f"Passed     : {grade['passed']}")
    print(f"Accuracy   : {grade['breakdown']['accuracy']}")
    print(f"Efficiency : {grade['breakdown']['efficiency']}")
    print(f"Safety     : {grade['breakdown']['safety']}")

    # ─── Close ──────────────────────────────────────────────
    client.close()


if __name__ == "__main__":
    main()