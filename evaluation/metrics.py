# evaluation/metrics.py

from typing import List, Dict, Any


class Metrics:

    @staticmethod
    def compute(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate metrics across episodes."""

        if not episodes:
            return {
                "avg_score": 0.0,
                "pass_rate": 0.0,
                "avg_steps": 0.0,
            }

        total = len(episodes)

        avg_score = sum(e["final_score"] for e in episodes) / total
        pass_rate = sum(1 for e in episodes if e["passed"]) / total
        avg_steps = sum(e["stats"]["total_steps"] for e in episodes) / total

        return {
            "episodes": total,
            "avg_score": round(avg_score, 3),
            "pass_rate": round(pass_rate, 3),
            "avg_steps": round(avg_steps, 2),
        }