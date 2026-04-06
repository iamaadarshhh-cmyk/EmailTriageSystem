# scripts/debug_env.py

from env.core.environment import EmailTriageEnvironment
from env.models.actions import Action, ActionType
from tasks.task_factory import TaskFactory


def main():
    print("\nDebugging Environment\n")

    # ─── Create Task ───────────────────────────────────────
    task = TaskFactory.create("medium")

    # ─── Create Environment ────────────────────────────────
    env = EmailTriageEnvironment(task=task)

    # ─── Reset ─────────────────────────────────────────────
    obs = env.reset(user_id="debug_user")

    print("Initial Observation:")
    print(obs)

    done = False
    step = 0

    # ─── Manual Loop ───────────────────────────────────────
    while not done:

        print("\n" + "="*50)
        print(f"Step {step + 1}")
        print(f"Email   : {obs.get('subject')}")
        print(f"Hint    : {obs.get('category_hint')}")
        print(f"Actions : {obs.get('available_actions')}")

        action_input = input("Enter action: ").strip().lower()

        # Validate action
        try:
            action_type = ActionType(action_input)
        except ValueError:
            print("❌ Invalid action. Try again.")
            continue

        action = Action(
            action_type=action_type,
            email_id=obs.get("email_id"),
        )

        obs, reward, done, info = env.step(action)

        print(f"Reward  : {reward}")
        print(f"Correct : {info.get('is_correct')}")
        print(f"Message : {info.get('message')}")

        step += 1

    print("\nEpisode Finished")
    print("Final Reward:", env.state.total_reward)

    env.close()


if __name__ == "__main__":
    main()