# scripts/run_benchmark.py

from evaluation.benchmark import Benchmark


def main():
    print("\n📊 Running Benchmark Suite\n")

    benchmark = Benchmark()

    results = []

    for level in ["easy", "medium", "hard"]:
        for agent in ["rule", "random"]:

            print(f"\n🔹 Agent: {agent} | Task: {level}")

            result = benchmark.run(
                agent_type=agent,
                task_level=level,
                episodes=5,
            )

            metrics = result["metrics"]

            print("Episodes   :", metrics["episodes"])
            print("Avg Score  :", metrics["avg_score"])
            print("Pass Rate  :", metrics["pass_rate"])
            print("Avg Steps  :", metrics["avg_steps"])

            results.append((agent, level, metrics))

    print("\n✅ Benchmark Complete\n")


if __name__ == "__main__":
    main()