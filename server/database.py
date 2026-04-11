"""
MongoDB integration for the Email Triage Environment.

Provides persistent storage for:
  - Sessions        → tracks every env session (reset + steps + final score)
  - Leaderboard     → ranked top scores per task and overall
  - Inference runs  → full stdout logs from inference.py
  - Analytics       → per-task and per-dimension aggregated stats

Environment Variables:
  MONGODB_URL   MongoDB connection string (default: mongodb://localhost:27017)
                Use MongoDB Atlas for production: mongodb+srv://...
  MONGODB_DB    Database name (default: email_triage)

Falls back to in-memory store if MongoDB is unavailable, so the server
always starts cleanly even without a running database.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB  = os.getenv("MONGODB_DB",  "email_triage")

# ─────────────────────────────────────────────────────────────────────────────
#  In-memory fallback store
# ─────────────────────────────────────────────────────────────────────────────

class _InMemoryStore:
    """Thread-safe in-memory fallback when MongoDB is not available."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._runs: List[Dict[str, Any]] = []

    # Sessions
    def upsert_session(self, doc: Dict[str, Any]) -> None:
        self._sessions[doc["session_id"]] = doc

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def leaderboard(self, task_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        done = [s for s in self._sessions.values() if s.get("completed")]
        if task_id:
            done = [s for s in done if s.get("task_id") == task_id]
        done.sort(key=lambda s: s.get("final_score", 0.0), reverse=True)
        return done[:limit]

    def task_analytics(self) -> List[Dict[str, Any]]:
        from collections import defaultdict
        buckets: Dict[str, List[float]] = defaultdict(list)
        for s in self._sessions.values():
            if s.get("completed") and s.get("final_score") is not None:
                buckets[s["task_id"]].append(s["final_score"])
        result = []
        for tid, scores in buckets.items():
            result.append({
                "task_id": tid,
                "runs": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4),
                "best_score": round(max(scores), 4),
                "worst_score": round(min(scores), 4),
            })
        return sorted(result, key=lambda x: x["task_id"])

    # Inference runs
    def save_run(self, doc: Dict[str, Any]) -> None:
        self._runs.append(doc)

    def get_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(self._runs[-limit:]))

    def total_sessions(self) -> int:
        return len(self._sessions)

    def total_completed(self) -> int:
        return sum(1 for s in self._sessions.values() if s.get("completed"))


_fallback = _InMemoryStore()

# ─────────────────────────────────────────────────────────────────────────────
#  DatabaseManager
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Async MongoDB manager with transparent in-memory fallback.

    Usage:
        db = DatabaseManager()
        await db.connect()    # call once at app startup
        await db.save_session(...)
        await db.get_leaderboard()
        await db.close()
    """

    def __init__(self):
        self._client = None
        self._db = None
        self.online = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to MongoDB. Falls back to in-memory on failure."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            self._client = AsyncIOMotorClient(
                MONGODB_URL,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
            )
            # Force connection attempt
            await self._client.admin.command("ping")
            self._db = self._client[MONGODB_DB]

            # Create indexes
            await self._ensure_indexes()

            self.online = True
            logger.info("✅ MongoDB connected: %s / %s", MONGODB_URL, MONGODB_DB)

        except Exception as exc:
            logger.warning(
                "⚠️  MongoDB unavailable (%s) — using in-memory fallback.", exc
            )
            self._client = None
            self._db = None
            self.online = False

    async def _ensure_indexes(self) -> None:
        """Create required indexes (idempotent)."""
        sessions = self._db["sessions"]
        await sessions.create_index("session_id", unique=True)
        await sessions.create_index([("final_score", -1)])
        await sessions.create_index("task_id")
        await sessions.create_index("completed")

        runs = self._db["inference_runs"]
        await runs.create_index("timestamp")
        await runs.create_index("model_name")

    async def close(self) -> None:
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed.")

    # ── Sessions ─────────────────────────────────────────────────────────────

    async def save_session(
        self,
        session_id: str,
        task_id: str,
        seed: int,
        *,
        completed: bool = False,
        final_score: Optional[float] = None,
        dimension_scores: Optional[Dict[str, float]] = None,
        steps_taken: Optional[int] = None,
        emails_processed: Optional[int] = None,
        emails_total: Optional[int] = None,
    ) -> None:
        """Create or update a session document."""
        now = datetime.now(timezone.utc)
        doc: Dict[str, Any] = {
            "session_id": session_id,
            "task_id": task_id,
            "seed": seed,
            "completed": completed,
            "updated_at": now,
        }
        if completed:
            doc["final_score"] = final_score or 0.0
            doc["dimension_scores"] = dimension_scores or {}
            doc["steps_taken"] = steps_taken or 0
            doc["emails_processed"] = emails_processed or 0
            doc["emails_total"] = emails_total or 0
            doc["completed_at"] = now

        if not self.online:
            existing = _fallback.get_session(session_id) or {"created_at": now, "task_id": task_id, "seed": seed}
            existing.update(doc)
            _fallback.upsert_session(existing)
            return

        try:
            # Upsert: set on insert + always update the changed fields
            await self._db["sessions"].update_one(
                {"session_id": session_id},
                {
                    "$setOnInsert": {"created_at": now},
                    "$set": doc,
                },
                upsert=True,
            )
        except Exception as exc:
            logger.error("save_session error: %s", exc)
            _fallback.upsert_session(doc)

    # ── Leaderboard ──────────────────────────────────────────────────────────

    async def get_leaderboard(
        self, task_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return top sessions sorted by final_score descending."""
        if not self.online:
            return _fallback.leaderboard(task_id=task_id, limit=limit)

        try:
            query: Dict[str, Any] = {"completed": True}
            if task_id:
                query["task_id"] = task_id

            cursor = (
                self._db["sessions"]
                .find(
                    query,
                    {
                        "_id": 0,
                        "session_id": 1,
                        "task_id": 1,
                        "final_score": 1,
                        "dimension_scores": 1,
                        "steps_taken": 1,
                        "emails_processed": 1,
                        "emails_total": 1,
                        "completed_at": 1,
                    },
                )
                .sort("final_score", -1)
                .limit(limit)
            )
            return [doc async for doc in cursor]

        except Exception as exc:
            logger.error("get_leaderboard error: %s", exc)
            return _fallback.leaderboard(task_id=task_id, limit=limit)

    # ── Analytics ────────────────────────────────────────────────────────────

    async def get_task_analytics(self) -> List[Dict[str, Any]]:
        """Return per-task aggregated stats (avg, best, worst score, run count)."""
        if not self.online:
            return _fallback.task_analytics()

        try:
            pipeline = [
                {"$match": {"completed": True, "final_score": {"$exists": True}}},
                {
                    "$group": {
                        "_id": "$task_id",
                        "runs": {"$sum": 1},
                        "avg_score": {"$avg": "$final_score"},
                        "best_score": {"$max": "$final_score"},
                        "worst_score": {"$min": "$final_score"},
                    }
                },
                {"$sort": {"_id": 1}},
            ]
            cursor = self._db["sessions"].aggregate(pipeline)
            results = []
            async for doc in cursor:
                results.append({
                    "task_id": doc["_id"],
                    "runs": doc["runs"],
                    "avg_score": round(doc["avg_score"], 4),
                    "best_score": round(doc["best_score"], 4),
                    "worst_score": round(doc["worst_score"], 4),
                })
            return results

        except Exception as exc:
            logger.error("get_task_analytics error: %s", exc)
            return _fallback.task_analytics()

    async def get_summary_stats(self) -> Dict[str, Any]:
        """Return overall summary stats for the root/health endpoint."""
        if not self.online:
            return {
                "total_sessions": _fallback.total_sessions(),
                "total_completed": _fallback.total_completed(),
                "storage": "in-memory",
            }

        try:
            total = await self._db["sessions"].count_documents({})
            completed = await self._db["sessions"].count_documents({"completed": True})
            return {
                "total_sessions": total,
                "total_completed": completed,
                "storage": "mongodb",
            }
        except Exception as exc:
            logger.error("get_summary_stats error: %s", exc)
            return {"total_sessions": 0, "total_completed": 0, "storage": "error"}

    # ── Inference Runs ────────────────────────────────────────────────────────

    async def save_inference_run(
        self,
        model_name: str,
        results: Dict[str, Any],
        elapsed_s: float,
    ) -> None:
        """Persist a completed inference.py run to the database."""
        doc = {
            "model_name": model_name,
            "results": results,
            "elapsed_s": round(elapsed_s, 2),
            "timestamp": datetime.now(timezone.utc),
        }
        if not self.online:
            _fallback.save_run(doc)
            return
        try:
            await self._db["inference_runs"].insert_one(doc)
        except Exception as exc:
            logger.error("save_inference_run error: %s", exc)
            _fallback.save_run(doc)

    async def get_inference_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most recent inference runs."""
        if not self.online:
            return _fallback.get_runs(limit=limit)
        try:
            cursor = (
                self._db["inference_runs"]
                .find({}, {"_id": 0})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return [doc async for doc in cursor]
        except Exception as exc:
            logger.error("get_inference_runs error: %s", exc)
            return _fallback.get_runs(limit=limit)


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton instance (imported by app.py)
# ─────────────────────────────────────────────────────────────────────────────

db = DatabaseManager()
