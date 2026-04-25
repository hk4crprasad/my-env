# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An OpenEnv-compliant RL environment where agents triage emails (classify, prioritize, route, respond). Deployed on Hugging Face Spaces at port 7860.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run server locally:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

**Run Docker:**
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 --env-file .env email-triage-env
```

**Run baseline inference benchmark:**
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b"
export HF_TOKEN="hf_xxx"
python inference.py
```

**Environment setup:** Copy `.env.example` to `.env` and fill `API_BASE_URL`, `HF_TOKEN`, `MODEL_NAME`, `MONGODB_URL`, `PORT`.

## Architecture

```
server/app.py             — FastAPI entry point; session management (max 20 concurrent, FIFO eviction)
server/environment.py     — EmailTriageEnvironment: reset()/step() OpenEnv interface
server/tasks.py           — 3 task configs: easy (5 emails), medium (10), hard (20)
server/email_generator.py — Seed-based deterministic synthetic email generation
server/graders.py         — Deterministic episode grading (keyword-based, no LLM)
server/reward.py          — Step-wise reward shaping (+0.20 classification, +0.15 priority, etc.)
server/database.py        — Motor (async MongoDB) with in-memory fallback
models.py                 — Pydantic models: EmailAction, EmailObservation, State, EmailData, EmailGroundTruth
inference.py              — Baseline OpenAI-client agent for benchmarking
```

## Key API Endpoints

- `POST /reset` → start episode, returns `session_id`
- `POST /step` → submit action (pass `session_id` for multi-session)
- `GET /state` → current inbox state
- `GET /leaderboard` → top scores (MongoDB-backed)
- `GET /schema` → Action/Observation JSON schemas

## Valid Action Values

- **categories**: `spam`, `billing`, `technical`, `general`, `urgent`
- **departments**: `engineering`, `billing`, `support`, `management`
- **priorities**: `1`–`5` (integer)

## Design Invariants

- Grading is fully deterministic (keyword matching in `graders.py` — never call an LLM for scoring)
- Email generation is seed-based for reproducibility (`email_generator.py`)
- Reward is shaped per-step, not just end-of-episode (`reward.py`)
- MongoDB is optional; `database.py` silently falls back to in-memory dicts
- OpenEnv imports are optional; `models.py` has fallback stubs for standalone use
