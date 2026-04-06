# client/examples/run_easy.py

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

    # ─── Run Easy Task ──────────────────────────────────────
    agent = RuleBasedAgent(client=client)
    runner = AgentRunner(agent=agent, client=client)

    grade = runner.run(
        task_level="easy",
        user_id="agent_easy",
        verbose=True,
    )

    # ─── Print Results ──────────────────────────────────────
    print("\n📊 Final Results:")
    print(f"Score    : {grade['final_score']}")
    print(f"Passed   : {grade['passed']}")
    print(f"Accuracy : {grade['breakdown']['accuracy']}")

    # ─── Close ──────────────────────────────────────────────
    client.close()


if __name__ == "__main__":
    main()