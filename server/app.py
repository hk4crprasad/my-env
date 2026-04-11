"""
FastAPI application for the Email Triage Environment.

Exposes the environment via HTTP endpoints compatible with the OpenEnv spec:
  POST /reset  — start a new episode
  POST /step   — execute an agent action
  GET  /state  — retrieve current environment state
  GET  /health — health check
  GET  /schema — action/observation JSON schemas
  GET  /tasks  — list available tasks

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import sys
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailAction, EmailObservation
from server.environment import EmailTriageEnvironment
from server.tasks import list_task_ids, TASKS
from server.database import db

# ═══════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ═══════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=42, description="Random seed")
    episode_id: Optional[str] = Field(default=None, description="Custom episode ID")
    task_id: str = Field(default="easy", description="Task: easy | medium | hard")


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(..., description="Action dict with email_id, category, etc.")
    session_id: Optional[str] = Field(default=None, description="Session ID returned from /reset (optional)")


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False
    session_id: str = ""


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False
    session_id: str = ""


class StateResponse(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0


class HealthResponse(BaseModel):
    status: str = "healthy"


# ═══════════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Email Triage Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to triage "
        "emails: classify, prioritise, route, and respond. "
        "Persistent leaderboard and analytics powered by MongoDB."
    ),
    version="1.1.0",
)


@app.on_event("startup")
async def startup() -> None:
    """Connect to MongoDB on server start."""
    await db.connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    """Gracefully close MongoDB connection."""
    await db.close()

# CORS for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store: up to MAX_SESSIONS concurrent isolated environments ──
MAX_SESSIONS = 20
_sessions: OrderedDict[str, EmailTriageEnvironment] = OrderedDict()
_latest_session_id: Optional[str] = None  # track most recently reset session


def _get_session(session_id: Optional[str]) -> tuple[str, EmailTriageEnvironment]:
    """Return (session_id, env) for the given id, or the latest active session."""
    sid = session_id or _latest_session_id
    if sid is None or sid not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )
    return sid, _sessions[sid]


def _create_session(episode_id: Optional[str]) -> tuple[str, EmailTriageEnvironment]:
    """Create (or reuse) a session and return (session_id, env)."""
    global _latest_session_id
    sid = episode_id or str(uuid4())
    if sid not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.popitem(last=False)  # evict oldest
        _sessions[sid] = EmailTriageEnvironment()
    _latest_session_id = sid
    return sid, _sessions[sid]


# ─────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns 200 if server is running."""
    return HealthResponse(status="healthy")


@app.get("/")
async def root():
    """Root endpoint — basic info and global stats."""
    stats = await db.get_summary_stats()
    return {
        "name": "email_triage_env",
        "version": "1.1.0",
        "status": "healthy",
        "tasks": list_task_ids(),
        "stats": stats,
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment for a new episode. Returns a session_id for use with /step."""
    try:
        sid, env = _create_session(request.episode_id)
        obs = env.reset(
            seed=request.seed,
            episode_id=sid,
            task_id=request.task_id,
        )
        obs_dict = obs.model_dump()

        # Persist session to MongoDB (non-blocking, best-effort)
        await db.save_session(
            session_id=sid,
            task_id=request.task_id,
            seed=request.seed or 42,
        )

        return ResetResponse(
            observation=obs_dict,
            reward=obs.reward,
            done=obs.done,
            session_id=sid,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Execute an agent action. Pass session_id from /reset for multi-session support."""
    sid, env = _get_session(request.session_id)

    try:
        action = EmailAction(**request.action)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action: {e}",
        )

    obs = env.step(action)
    obs_dict = obs.model_dump()

    # On episode completion, persist final score to MongoDB
    if obs.done:
        grading = obs_dict.get("metadata", {}).get("grading", {})
        task_id = obs_dict.get("task_id", "unknown")
        _state = env.state
        await db.save_session(
            session_id=sid,
            task_id=task_id,
            seed=42,  # seed stored at reset time; reuse here
            completed=True,
            final_score=grading.get("final_score"),
            dimension_scores=grading.get("dimension_scores", {}),
            steps_taken=grading.get("steps_taken", _state.step_count),
            emails_processed=grading.get("emails_processed"),
            emails_total=grading.get("emails_total"),
        )

    return StepResponse(
        observation=obs_dict,
        reward=obs.reward,
        done=obs.done,
        session_id=sid,
    )


@app.get("/state", response_model=StateResponse)
async def get_state(session_id: Optional[str] = None):
    """Return the current environment state. Pass session_id for multi-session support."""
    sid, env = _get_session(session_id)
    s = env.state
    return StateResponse(
        episode_id=s.episode_id,
        step_count=s.step_count,
    )


@app.get("/schema")
async def schema():
    """Return JSON schemas for Action and Observation models."""
    return {
        "action": EmailAction.model_json_schema(),
        "observation": EmailObservation.model_json_schema(),
    }


@app.get("/tasks")
async def tasks():
    """List all available tasks with descriptions."""
    return {
        tid: {
            "name": t.name,
            "difficulty": t.difficulty,
            "num_emails": t.num_emails,
            "max_steps": t.max_steps,
            "description": t.description,
        }
        for tid, t in TASKS.items()
    }


@app.get("/leaderboard")
async def leaderboard(task_id: Optional[str] = None, limit: int = 10):
    """Return top sessions by final score. Filter by task_id (easy|medium|hard)."""
    if task_id and task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task_id must be easy, medium, or hard")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    entries = await db.get_leaderboard(task_id=task_id, limit=limit)

    # Serialize datetime objects
    for entry in entries:
        if "completed_at" in entry and hasattr(entry["completed_at"], "isoformat"):
            entry["completed_at"] = entry["completed_at"].isoformat()

    return {
        "task_id": task_id or "all",
        "count": len(entries),
        "storage": "mongodb" if db.online else "in-memory",
        "entries": entries,
    }


@app.get("/analytics")
async def analytics():
    """Return per-task aggregated statistics (avg/best/worst score, run count)."""
    stats = await db.get_task_analytics()
    summary = await db.get_summary_stats()
    return {
        "summary": summary,
        "by_task": stats,
        "storage": "mongodb" if db.online else "in-memory",
    }


@app.get("/runs")
async def inference_runs(limit: int = 20):
    """Return most recent inference.py runs stored in MongoDB."""
    runs = await db.get_inference_runs(limit=limit)
    for run in runs:
        if "timestamp" in run and hasattr(run["timestamp"], "isoformat"):
            run["timestamp"] = run["timestamp"].isoformat()
    return {"count": len(runs), "runs": runs}


# ─────────────────────────────────────────────────────────────────────────
#  Direct execution
# ─────────────────────────────────────────────────────────────────────────

def main():
    """Run the server directly."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
