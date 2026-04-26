"""
FastAPI application for the Email Triage Environment.

OpenEnv-core endpoints (WebSocket, Meta-standard):
  WS   /ws/reset   — start a new episode (WebSocket)
  WS   /ws/step    — execute an agent action (WebSocket)
  WS   /ws/state   — retrieve episode state (WebSocket)
  GET  /web        — interactive browser UI (ENABLE_WEB_INTERFACE=true)

REST endpoints (backwards-compatible, hackathon validator):
  POST /reset      — start a new episode
  POST /step       — execute an agent action
  GET  /state      — retrieve current environment state
  GET  /health     — health check
  GET  /schema     — action/observation JSON schemas
  GET  /tasks      — list available tasks
  GET  /rubric     — reward rubric definitions (7 independent components)
  GET  /curriculum — task progression for curriculum learning
  GET  /leaderboard — top sessions by score
  GET  /analytics  — per-task aggregated statistics

Usage (local):
    uvicorn server.app:app --host 0.0.0.0 --port 7860

Usage with openenv-core client:
    from client import EmailTriageClient
    async with EmailTriageClient(base_url='http://localhost:7860') as env:
        result = await env.reset(task_id='easy')
        result = await env.step(action)

Usage with TRL GRPOTrainer:
    See notebooks/train_grpo.ipynb — uses openenv-core GenericEnvClient
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailAction, EmailObservation
from server.environment import EmailTriageEnvironment
from server.tasks import list_task_ids, get_curriculum_order, TASKS
from server.graders import get_rubric_definitions
from server.reward import REWARD_RUBRIC
from server.database import db

# ── openenv-core integration (WebSocket + web UI) ─────────────────────────
# Lazy import — only activated at runtime, not at module load time.
# This prevents Python 3.14 gradio/typer issues in local dev.
_HAS_OPENENV_CORE = False
_openenv_app = None

# ═══════════════════════════════════════════════════════════════════════════
#  Request / Response models
# ═══════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=42, description="Random seed")
    episode_id: Optional[str] = Field(default=None, description="Custom episode ID")
    task_id: str = Field(default="easy", description="Task: easy | medium | hard")


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(..., description="Action dict with email_id, category, etc.")
    session_id: Optional[str] = Field(default=None, description="Session ID from /reset")


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
    title="Email Triage RL Environment",
    description=(
        "An OpenEnv-compliant RL environment where AI agents learn to triage "
        "emails: classify, prioritise, route, and respond. "
        "Seven independent reward components prevent reward hacking. "
        "Three difficulty levels enable curriculum learning.\n\n"
        "Theme: World Modeling — Personalized Tasks (#3.2)\n\n"
        "Powered by [openenv-core](https://github.com/meta-pytorch/OpenEnv) — "
        "compatible with TRL, Unsloth, ART, and Oumi GRPO trainers via WebSocket."
    ),
    version="2.0.0",
)

# ── Mount openenv-core WebSocket app at /ws ────────────────────────────────
# Trainers using openenv-core EnvClient connect here over WebSocket.
# Example: EmailTriageClient(base_url="http://localhost:7860")
if _HAS_OPENENV_CORE:
    try:
        _openenv_app = create_fastapi_app(
            env=EmailTriageEnvironment,
            action_cls=EmailAction,
            observation_cls=EmailObservation,
            max_concurrent_envs=20,
        )
        app.mount("/ws", _openenv_app)
    except Exception:
        pass  # non-fatal — REST endpoints still work


@app.on_event("startup")
async def startup() -> None:
    await db.connect()
    # ── Try to activate openenv-core WebSocket protocol ──────────────────
    global _HAS_OPENENV_CORE, _openenv_app
    try:
        from openenv.core import create_fastapi_app
        _openenv_app = create_fastapi_app(
            env=EmailTriageEnvironment,
            action_cls=EmailAction,
            observation_cls=EmailObservation,
            max_concurrent_envs=20,
        )
        app.mount("/ws", _openenv_app)
        _HAS_OPENENV_CORE = True
        print("✅ openenv-core WebSocket protocol activated at /ws")
    except Exception as e:
        print(f"ℹ openenv-core WebSocket not available ({type(e).__name__}) — REST API only")

    # ── Mount Gradio demo at /demo (manual play + adapter vs baseline) ───
    if os.environ.get("DISABLE_GRADIO", "").lower() != "true":
        try:
            import gradio as gr  # noqa: F401
            from demo import build_ui
            blocks = build_ui()
            gr.mount_gradio_app(app, blocks, path="/demo")
            print("✅ Gradio demo mounted at /demo")
        except Exception as e:
            print(f"ℹ Gradio demo not mounted ({type(e).__name__}: {e})")


@app.on_event("shutdown")
async def shutdown() -> None:
    await db.close()


# ── Web interface endpoint ─────────────────────────────────────────────────
# Enable with: ENABLE_WEB_INTERFACE=true uvicorn server.app:app ...
# Then open: http://localhost:7860/web
@app.get("/web")
async def web_interface_info():
    """Browser-based interactive environment explorer."""
    if not _HAS_OPENENV_CORE:
        return {
            "status": "unavailable",
            "reason": "openenv-core not installed",
            "install": "pip install openenv-core>=0.2.0",
        }
    if os.environ.get("ENABLE_WEB_INTERFACE", "").lower() != "true":
        return {
            "status": "disabled",
            "enable": "Set ENABLE_WEB_INTERFACE=true and restart the server",
            "description": "Two-pane UI: send actions on left, see observations on right",
        }
    return {"status": "enabled", "url": "/ws/web"}

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
_latest_session_id: Optional[str] = None


def _get_session(session_id: Optional[str]) -> tuple[str, EmailTriageEnvironment]:
    sid = session_id or _latest_session_id
    if sid is None or sid not in _sessions:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /reset first.",
        )
    return sid, _sessions[sid]


def _create_session(episode_id: Optional[str]) -> tuple[str, EmailTriageEnvironment]:
    global _latest_session_id
    sid = episode_id or str(uuid4())
    if sid not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.popitem(last=False)  # evict oldest
        _sessions[sid] = EmailTriageEnvironment()
    _latest_session_id = sid
    return sid, _sessions[sid]


# ─────────────────────────────────────────────────────────────────────────
#  Core OpenEnv endpoints
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")


@app.get("/")
async def root():
    stats = await db.get_summary_stats()
    return {
        "name": "email_triage_env",
        "version": "2.0.0",
        "theme": "World Modeling — Personalized Tasks (#3.2)",
        "status": "healthy",
        "tasks": list_task_ids(),
        "reward_components": list(REWARD_RUBRIC.keys()),
        "stats": stats,
        "ui": {
            "gradio_demo": "/demo",
            "openenv_websocket": "/ws",
            "openapi_docs": "/docs",
        },
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment for a new episode. Returns a session_id for /step."""
    try:
        sid, env = _create_session(request.episode_id)
        obs = env.reset(
            seed=request.seed,
            episode_id=sid,
            task_id=request.task_id,
        )
        obs_dict = obs.model_dump()

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
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs = env.step(action)
    obs_dict = obs.model_dump()

    if obs.done:
        grading = obs_dict.get("metadata", {}).get("grading", {})
        task_id = obs_dict.get("task_id", "unknown")
        _state = env.state
        await db.save_session(
            session_id=sid,
            task_id=task_id,
            seed=42,
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
    sid, env = _get_session(session_id)
    s = env.state
    return StateResponse(episode_id=s.episode_id, step_count=s.step_count)


@app.get("/schema")
async def schema():
    """JSON schemas for Action and Observation models."""
    return {
        "action": EmailAction.model_json_schema(),
        "observation": EmailObservation.model_json_schema(),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Discovery endpoints
# ─────────────────────────────────────────────────────────────────────────

@app.get("/tasks")
async def tasks():
    """List all available tasks with descriptions and scoring weights."""
    return {
        tid: {
            "name": t.name,
            "difficulty": t.difficulty,
            "curriculum_level": t.curriculum_level,
            "num_emails": t.num_emails,
            "max_steps": t.max_steps,
            "optimal_steps": t.optimal_steps,
            "description": t.description,
            "scoring_weights": t.scoring_weights,
        }
        for tid, t in TASKS.items()
    }


@app.get("/rubric")
async def rubric():
    """Return all reward rubric definitions.

    Each reward component is independent — an agent cannot game one without
    independently satisfying the others.
    """
    step_rubric = REWARD_RUBRIC
    episode_rubric = get_rubric_definitions()
    return {
        "description": (
            "Seven independent per-step reward components + five episode-level graders. "
            "Independence prevents reward hacking: optimising one component does not "
            "automatically improve others."
        ),
        "per_step_rewards": step_rubric,
        "episode_graders": episode_rubric,
        "anti_hacking_design": [
            "Format compliance checked before any content reward is applied",
            "Re-processing same email returns -0.15 with no other reward",
            "Escalation is graded independently of routing",
            "Response quality uses hidden keyword sets (not shown to agent)",
            "Priority accuracy uses non-linear scoring (off-by-2+ gets 0, not partial)",
        ],
    }


@app.get("/curriculum")
async def curriculum():
    """Return the curriculum progression (easy → medium → hard)."""
    return {
        "description": (
            "Three tasks in increasing difficulty. Start with 'easy' to ensure "
            "non-zero reward before advancing. Curriculum training schedule: "
            "train easy until avg reward > 0.6, then medium until > 0.5, then hard."
        ),
        "progression": get_curriculum_order(),
        "advancement_thresholds": {
            "easy_to_medium": 0.60,
            "medium_to_hard": 0.50,
        },
    }


# ─────────────────────────────────────────────────────────────────────────
#  Leaderboard / Analytics
# ─────────────────────────────────────────────────────────────────────────

@app.get("/leaderboard")
async def leaderboard(task_id: Optional[str] = None, limit: int = 10):
    """Top sessions by final score. Filter by task_id (easy|medium|hard)."""
    if task_id and task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task_id must be easy, medium, or hard")
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    entries = await db.get_leaderboard(task_id=task_id, limit=limit)
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
    """Per-task aggregated statistics (avg/best/worst score, run count)."""
    stats = await db.get_task_analytics()
    summary = await db.get_summary_stats()
    return {
        "summary": summary,
        "by_task": stats,
        "storage": "mongodb" if db.online else "in-memory",
    }


@app.get("/runs")
async def inference_runs(limit: int = 20):
    """Most recent inference.py runs stored in the database."""
    runs = await db.get_inference_runs(limit=limit)
    for run in runs:
        if "timestamp" in run and hasattr(run["timestamp"], "isoformat"):
            run["timestamp"] = run["timestamp"].isoformat()
    return {"count": len(runs), "runs": runs}


# ─────────────────────────────────────────────────────────────────────────
#  Direct execution
# ─────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
