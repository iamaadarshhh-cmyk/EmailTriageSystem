# evaluation/benchmark.py

from typing import Dict, Any, List
from client.client import EmailTriageClient
from client.agent import RuleBasedAgent, RandomAgent, AgentRunner
from evaluation.metrics import Metrics


class Benchmark:

    def __init__(self):
        self.client = EmailTriageClient()

    def run(
        self,
        agent_type: str,
        task_level: str,
        episodes: int = 5,
    ) -> Dict[str, Any]:

        # Select agent
        if agent_type == "rule":
            agent = RuleBasedAgent(self.client)
        elif agent_type == "random":
            agent = RandomAgent(self.client)
        else:
            raise ValueError("Invalid agent type")

        runner = AgentRunner(agent, self.client)

        results: List[Dict[str, Any]] = []

        for _ in range(episodes):
            result = runner.run(task_level=task_level, verbose=False)
            results.append(result)

        metrics = Metrics.compute(results)

        return {
            "agent": agent_type,
            "task": task_level,
            "metrics": metrics,
        }