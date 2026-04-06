# graders/graders.py

from typing import Dict, Any, List
from env.models import Action, ActionType, Priority, GroundTruth


class EpisodeGrader:

    def grade(
        self,
        task_id: str,
        actions: List[Dict],
        emails: List[Dict],
        task_config: Dict,
    ) -> Dict[str, Any]:
        """Grade full episode deterministically."""

        if not actions:
            return self._empty_grade(task_id)

        difficulty = task_config.get("difficulty", "easy")
        threshold = task_config.get("pass_threshold", 0.70)

        scores = []
        for action_data, email_data in zip(actions, emails):
            score = self._grade_single(action_data, email_data, difficulty)
            scores.append(score)

        # ─── Aggregate ──────────────────────────────────────
        avg_score = sum(s["total"] for s in scores) / len(scores)

        # ─── Episode Bonuses ────────────────────────────────
        bonus = 0.0
        urgent_emails = [e for e in emails
                         if e["ground_truth"].expected_priority == Priority.URGENT]
        urgent_correct = sum(
            1 for a, e in zip(actions, emails)
            if e["ground_truth"].expected_priority == Priority.URGENT
            and a.get("triage", {}).get("priority") == "urgent"
        )
        if urgent_emails and urgent_correct == len(urgent_emails):
            bonus += 0.10  # All urgents caught

        skips = sum(1 for a in actions if a.get("action_type") == "skip")
        if skips == 0:
            bonus += 0.05  # No skips

        final_score = min(avg_score + bonus, 1.0)

        return {
            "task_id": task_id,
            "final_score": round(final_score, 4),
            "passed": final_score >= threshold,
            "threshold": threshold,
            "breakdown": {
                "avg_per_email": round(avg_score, 4),
                "bonus": round(bonus, 4),
                "per_email_scores": scores,
            },
            "stats": {
                "total_emails": len(emails),
                "actions_taken": len(actions),
                "skips": skips,
                "urgents_caught": urgent_correct,
            }
        }

    def _grade_single(
        self,
        action: Dict,
        email: Dict,
        difficulty: str,
    ) -> Dict[str, float]:
        """Grade one action against one email's ground truth."""

        truth: GroundTruth = email["ground_truth"]
        score = {}

        # ─── Priority Accuracy (0.30) ────────────────────────
        taken_priority = action.get("triage", {}).get("priority")
        if taken_priority == truth.expected_priority.value:
            score["priority"] = 0.30
        elif self._adjacent_priority(taken_priority, truth.expected_priority.value):
            score["priority"] = 0.15
        else:
            # Miss urgent = heavy penalty
            if truth.expected_priority == Priority.URGENT:
                score["priority"] = -0.20
            else:
                score["priority"] = 0.0

        # ─── Category Accuracy (0.20) ────────────────────────
        taken_category = action.get("triage", {}).get("category")
        if taken_category == truth.expected_category.value:
            score["category"] = 0.20
        else:
            score["category"] = 0.0

        # ─── Action Correctness (0.25) ───────────────────────
        taken_action = action.get("action_type")
        if taken_action == truth.expected_action.value:
            score["action"] = 0.25
        elif taken_action == "skip":
            score["action"] = -0.05
        else:
            score["action"] = 0.0

        # ─── Escalation Routing (0.10) ───────────────────────
        if truth.expected_action == ActionType.ESCALATE:
            taken_teams = action.get("escalate_to", "")
            if isinstance(taken_teams, str):
                taken_teams = [t.strip() for t in taken_teams.split(",")]
            expected_teams = set(truth.escalate_to)
            taken_set = set(taken_teams)
            if expected_teams:
                jaccard = len(expected_teams & taken_set) / len(expected_teams | taken_set)
                score["escalation"] = round(0.10 * jaccard, 4)
            else:
                score["escalation"] = 0.10
        else:
            score["escalation"] = 0.10  # full credit if no escalation needed

        # ─── Reply Quality (0.15) ────────────────────────────
        reply_body = action.get("reply_body", "") or ""
        if truth.expected_action == ActionType.REPLY and reply_body:
            keywords_hit = sum(
                1 for kw in truth.reply_keywords
                if kw.lower() in reply_body.lower()
            )
            forbidden_hit = sum(
                1 for ph in truth.forbidden_phrases
                if ph.lower() in reply_body.lower()
            )
            keyword_score = keywords_hit / max(len(truth.reply_keywords), 1)
            length_ok = len(reply_body) > 50
            score["reply"] = round(
                0.15 * keyword_score * (1.0 if length_ok else 0.5)
                - (0.05 * forbidden_hit), 4
            )
        else:
            score["reply"] = 0.10  # neutral if reply not needed

        # ─── Destructive Penalty ────────────────────────────
        penalty = 0.0
        if taken_action == "delete" and truth.expected_category.value != "spam":
            penalty -= 0.30  # Deleted non-spam!
        if taken_action in ["reply", "escalate"] and truth.expected_category == "spam":
            penalty -= 0.10  # Replied to spam!

        score["penalties"] = penalty
        score["total"] = round(
            sum(v for k, v in score.items() if k != "total"), 4
        )

        return score

    def _adjacent_priority(self, taken: str, expected: str) -> bool:
        """Check if priorities are adjacent levels."""
        order = ["low", "medium", "high", "urgent"]
        if taken not in order or expected not in order:
            return False
        return abs(order.index(taken) - order.index(expected)) == 1

    def _empty_grade(self, task_id: str) -> Dict:
        return {
            "task_id": task_id,
            "final_score": 0.0,
            "passed": False,
            "threshold": 0.70,
            "breakdown": {},
            "stats": {"total_emails": 0, "actions_taken": 0},
        }