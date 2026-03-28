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
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailAction, EmailObservation
from server.environment import EmailTriageEnvironment
from server.tasks import list_task_ids, TASKS

# ═══════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ═══════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=42, description="Random seed")
    episode_id: Optional[str] = Field(default=None, description="Custom episode ID")
    task_id: str = Field(default="easy", description="Task: easy | medium | hard")


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(..., description="Action dict with email_id, category, etc.")


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


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
        "emails: classify, prioritise, route, and respond."
    ),
    version="1.0.0",
)

# CORS for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent environment instance (stateful across requests)
env = EmailTriageEnvironment()


# ─────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns 200 if server is running."""
    return HealthResponse(status="healthy")


@app.get("/")
async def root():
    """Root endpoint — basic info."""
    return {
        "name": "email_triage_env",
        "version": "1.0.0",
        "status": "healthy",
        "tasks": list_task_ids(),
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment for a new episode."""
    try:
        obs = env.reset(
            seed=request.seed,
            episode_id=request.episode_id,
            task_id=request.task_id,
        )
        obs_dict = obs.model_dump()
        return ResetResponse(
            observation=obs_dict,
            reward=obs.reward,
            done=obs.done,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Execute an agent action and return the resulting observation."""
    try:
        action = EmailAction(**request.action)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action: {e}",
        )

    obs = env.step(action)
    obs_dict = obs.model_dump()
    return StepResponse(
        observation=obs_dict,
        reward=obs.reward,
        done=obs.done,
    )


@app.get("/state", response_model=StateResponse)
async def get_state():
    """Return the current environment state."""
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
