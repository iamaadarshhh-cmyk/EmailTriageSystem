# server.py


import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from env.models import Action, Email, GroundTruth
from tasks.corpus import TASKS
from grader.grader import EpisodeGrader


# ─── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Email Triage OpenEnv",
    description="A real-world email triage environment for AI agent evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session Storage ────────────────────────────────────────
sessions: Dict[str, Any] = {}
grader = EpisodeGrader()


# ─── Request Models ─────────────────────────────────────────
class StepRequest(BaseModel):
    action_type: str
    triage: Optional[Dict[str, Any]] = None
    reply_body: Optional[str] = None
    escalate_to: Optional[str] = None
    reason: Optional[str] = None


# ─── Root ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "reset": "POST /reset/{task_id}",
            "step": "POST /step",
            "state": "GET /state",
            "grade": "GET /grade",
            "tasks": "GET /tasks",
            "validate": "GET /openenv/validate",
            "docs": "/docs",
        }
    }


# ─── Health Check ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset_default():
    """Default reset endpoint required by validator."""
    
    # Pick any default task (first one)
    task_id = list(TASKS.keys())[0]
    
    return reset(task_id)

# ─── Reset ──────────────────────────────────────────────────
@app.post("/reset/{task_id}")
def reset(task_id: str, session_id: Optional[str] = None, seed: Optional[int] = None):
    """Reset environment and start new episode."""

    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Available: {list(TASKS.keys())}"
        )

    # Generate session ID
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    task_config = TASKS[task_id]
    emails = task_config["emails"]

    # Store session
    sessions[session_id] = {
        "task_id": task_id,
        "task_config": task_config,
        "emails": emails,
        "current_index": 0,
        "actions": [],
        "step_count": 0,
        "session_score": 0.0,
        "triage_history": [],
        "is_done": False,
        "seed": seed,
    }

    # Build first observation
    observation = _build_observation(sessions[session_id])

    return {
        "session_id": session_id,
        "task_id": task_id,
        "observation": observation,
        "message": f"Episode started — {len(emails)} emails to triage",
    }


# ─── Step ───────────────────────────────────────────────────
@app.post("/step")
def step(request: StepRequest, session_id: str):
    """Take one step — submit action for current email."""

    session = sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first."
        )

    if session["is_done"]:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call /reset to start a new one."
        )

    emails = session["emails"]
    current_index = session["current_index"]

    if current_index >= len(emails):
        session["is_done"] = True
        raise HTTPException(status_code=400, detail="No more emails to process.")

    # ─── Get Current Email ──────────────────────────────────
    current_email = emails[current_index]
    ground_truth = current_email["ground_truth"]

    # ─── Build Action Dict ──────────────────────────────────────
    action_dict = {
        "action_type": request.action_type.lower(),
        "triage": request.triage or {
            "priority": "low",
            "category": "general_inquiry",
            "confidence": 0.5,
        },
        "reply_body": request.reply_body or "",
        "escalate_to": request.escalate_to or "",
        "reason": request.reason or "",
    }

    # ─── Calculate Step Reward ──────────────────────────────
    step_score = grader._grade_single(
        action_dict, current_email,
        session["task_config"].get("difficulty", "easy")
    )
    reward = step_score["total"]

    # ─── Update Session ─────────────────────────────────────
    session["actions"].append(action_dict)
    session["step_count"] += 1
    session["session_score"] += reward
    session["current_index"] += 1
    session["triage_history"].append({
        "email_id": current_email["email_id"],
        "action": request.action_type,
        "reward": round(reward, 3),
    })

    # ─── Check Done ─────────────────────────────────────────
    done = (
        session["current_index"] >= len(emails) or
        session["step_count"] >= session["task_config"]["max_steps"]
    )
    session["is_done"] = done

    # ─── Build New Observation ──────────────────────────────
    observation = _build_observation(session)

    return {
        "observation": observation,
        "reward": {
            "immediate": round(reward, 3),
            "components": step_score,
            "feedback": _generate_feedback(step_score, ground_truth),
        },
        "done": done,
        "info": {
            "email_id": current_email["email_id"],
            "step": session["step_count"],
            "session_score": round(session["session_score"], 3),
        }
    }


# ─── State ──────────────────────────────────────────────────
@app.get("/state")
def state(session_id: str):
    """Return full serializable episode state."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "step_count": session["step_count"],
        "session_score": round(session["session_score"], 3),
        "current_index": session["current_index"],
        "total_emails": len(session["emails"]),
        "is_done": session["is_done"],
        "triage_history": session["triage_history"],
    }


# ─── Grade ──────────────────────────────────────────────────
@app.get("/grade")
def grade(session_id: str):
    """Grade the completed episode."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    result = grader.grade(
        task_id=session["task_id"],
        actions=session["actions"],
        emails=session["emails"][:len(session["actions"])],
        task_config=session["task_config"],
    )

    return result


# ─── Tasks ──────────────────────────────────────────────────
@app.get("/tasks")
def get_tasks():
    """Return all available tasks."""
    return {
        task_id: {
            "description": config["description"],
            "difficulty": config["difficulty"],
            "n_emails": len(config["emails"]),
            "max_steps": config["max_steps"],
            "pass_threshold": config["pass_threshold"],
        }
        for task_id, config in TASKS.items()
    }


# ─── OpenEnv Validate ───────────────────────────────────────
@app.get("/openenv/validate")
def validate():
    """Self-test all tasks for OpenEnv compliance."""
    results = {}

    for task_id in TASKS:
        try:
            # Test reset
            session_id = f"test_{task_id}"
            reset_result = reset(task_id, session_id=session_id)
            assert "observation" in reset_result

            # Test step
            test_request = StepRequest(
                action_type="archive",
                triage={"priority": "low", "category": "general_inquiry"},
                reason="Test step",
            )
            step_result = step(test_request, session_id=session_id)
            assert "reward" in step_result
            assert "observation" in step_result

            results[task_id] = {"status": "ok"}

            # Cleanup
            del sessions[session_id]

        except Exception as e:
            results[task_id] = {"status": "error", "detail": str(e)}

    return {
        "valid": all(r["status"] == "ok" for r in results.values()),
        "tasks": results,
    }




# ─── Helpers ────────────────────────────────────────────────
def _build_observation(session: Dict) -> Dict:
    """Build observation dict from current session."""
    emails = session["emails"]
    current_index = session["current_index"]
    task_config = session["task_config"]

    current_email = None
    if current_index < len(emails):
        e = emails[current_index]
        current_email = {
            "email_id": e["email_id"],
            "subject": e["subject"],
            "sender": e["sender"],
            "sender_name": e.get("sender_name", ""),
            "body": e["body"],
            "timestamp": e["timestamp"],
            "attachments": e.get("attachments", []),
            "thread_id": e.get("thread_id"),
        }

    return {
        "current_email": current_email,
        "inbox_remaining": len(emails) - current_index,
        "emails_processed": current_index,
        "session_score": round(session["session_score"], 3),
        "triage_history": session["triage_history"][-3:],
        "task_description": task_config["description"],
        "available_actions": [
            "classify", "reply", "escalate",
            "archive", "flag", "skip", "delete", "mark_spam"
        ],
        "step_count": session["step_count"],
        "max_steps": task_config["max_steps"],
        "is_done": session["is_done"],
    }


def _generate_feedback(score: Dict, ground_truth) -> str:
    """Generate human readable feedback for the action."""
    total = score.get("total", 0)
    if total >= 0.7:
        return "✅ Excellent action!"
    elif total >= 0.4:
        return "⚠️ Partially correct — check priority and category."
    elif total < 0:
        return "❌ Incorrect — critical mistake detected."
    else:
        return "❌ Incorrect action for this email."

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

# ─── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    main()