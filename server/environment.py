"""
Email Triage Environment — Core Implementation.

Implements the OpenEnv interface: reset(), step(), state property.
Manages inbox state, processes agent actions, and computes rewards.
"""

from __future__ import annotations

import sys
import os
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    EmailAction,
    EmailData,
    EmailGroundTruth,
    EmailObservation,
    InboxStats,
    State,
    VALID_CATEGORIES,
    VALID_DEPARTMENTS,
    VALID_PRIORITIES,
)
from server.email_generator import generate_emails
from server.tasks import TaskDefinition, get_task, list_task_ids
from server.graders import grade_episode
from server.reward import compute_step_reward, REWARD_RUBRIC


class EmailTriageEnvironment:
    """OpenEnv-compliant email triage environment.

    The agent receives an inbox of emails and must classify, prioritise,
    route, and optionally respond to each one. Rewards come from seven
    independent rubric components, making reward hacking much harder than
    a single scalar signal.

    Usage:
        >>> env = EmailTriageEnvironment()
        >>> obs = env.reset(task_id="easy")
        >>> while not obs.done:
        ...     action = decide(obs)
        ...     obs = env.step(action)
        >>> print(obs.metadata["grading"])
    """

    def __init__(self):
        self._task: Optional[TaskDefinition] = None
        self._emails: List[EmailData] = []
        self._ground_truths: List[EmailGroundTruth] = []
        self._truth_map: Dict[str, EmailGroundTruth] = {}
        self._valid_email_ids: set[str] = set()

        self._processed_ids: set[str] = set()
        self._actions_taken: List[Dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._reward_breakdown: Dict[str, float] = {}  # per-rubric totals

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._grading_result: Optional[Dict[str, Any]] = None
        self._episode_start_time: float = 0.0

    # ─────────────────────────────────────────────────────────────────
    #  OpenEnv interface
    # ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        **kwargs: Any,
    ) -> EmailObservation:
        """Reset the environment for a new episode."""
        if seed is None:
            seed = 42

        self._task = get_task(task_id)
        self._emails, self._ground_truths = generate_emails(task_id, seed)
        self._truth_map = {gt.email_id: gt for gt in self._ground_truths}
        self._valid_email_ids = {e.email_id for e in self._emails}

        self._processed_ids = set()
        self._actions_taken = []
        self._cumulative_reward = 0.0
        self._reward_breakdown = {}
        self._done = False
        self._grading_result = None
        self._episode_start_time = time.time()

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return self._make_observation(
            step_reward=0.0,
            feedback="Environment reset. Start triaging your inbox!",
        )

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        """Execute one triage action."""
        if self._done:
            return self._make_observation(
                step_reward=0.0,
                feedback="Episode is already done. Call reset() to start a new one.",
            )

        if self._task is None:
            return self._make_observation(
                step_reward=0.0,
                feedback="No task loaded. Call reset(task_id=...) first.",
            )

        # ── Parse action ──────────────────────────────────────────────
        if isinstance(action, dict):
            action_dict = action
        elif hasattr(action, "model_dump"):
            action_dict = action.model_dump()
        elif hasattr(action, "__dict__"):
            action_dict = action.__dict__
        else:
            action_dict = dict(action)

        email_id = action_dict.get("email_id", "")
        already_processed = email_id in self._processed_ids

        # Check if email exists
        gt = self._truth_map.get(email_id)
        if gt is None:
            self._state.step_count += 1
            return self._make_observation(
                step_reward=-0.1,
                feedback=f"Unknown email_id: '{email_id}'. Check the inbox.",
            )

        # ── Compute reward (7 independent components) ─────────────────
        step_reward, feedback = compute_step_reward(
            action_dict,
            gt,
            requires_priority=self._task.requires_priority,
            requires_routing=self._task.requires_routing,
            requires_response=self._task.requires_response,
            already_processed=already_processed,
            valid_email_ids=self._valid_email_ids,
        )

        self._cumulative_reward += step_reward
        self._state.step_count += 1

        if not already_processed:
            self._processed_ids.add(email_id)
            self._actions_taken.append(action_dict)

        # ── Check termination ─────────────────────────────────────────
        all_processed = len(self._processed_ids) >= len(self._emails)
        out_of_steps = self._state.step_count >= self._task.max_steps

        if all_processed or out_of_steps:
            self._done = True
            self._grading_result = grade_episode(
                self._task,
                self._actions_taken,
                self._ground_truths,
                self._state.step_count,
            )
            elapsed = round(time.time() - self._episode_start_time, 2)
            self._grading_result["elapsed_s"] = elapsed
            feedback += f" | Episode complete! Final score: {self._grading_result['final_score']:.4f}"

        return self._make_observation(step_reward=step_reward, feedback=feedback)

    @property
    def state(self) -> State:
        return self._state

    @property
    def rubric(self) -> Dict[str, Any]:
        """Return rubric definitions for all reward components."""
        return REWARD_RUBRIC

    def close(self) -> None:
        pass

    # ─────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _unprocessed_emails(self) -> List[EmailData]:
        return [e for e in self._emails if e.email_id not in self._processed_ids]

    def _make_observation(
        self, step_reward: float = 0.0, feedback: Optional[str] = None
    ) -> EmailObservation:
        unprocessed = self._unprocessed_emails()
        stats = InboxStats(
            total=len(self._emails),
            unprocessed=len(unprocessed),
            processed=len(self._processed_ids),
        )

        steps_remaining = None
        if self._task:
            steps_remaining = max(0, self._task.max_steps - self._state.step_count)

        meta: Dict[str, Any] = {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
        }
        if self._grading_result:
            meta["grading"] = self._grading_result

        return EmailObservation(
            done=self._done,
            reward=step_reward,
            metadata=meta,
            emails=[e.model_dump() for e in unprocessed],
            inbox_stats=stats.model_dump(),
            task_id=self._task.task_id if self._task else "",
            task_description=self._task.description if self._task else "",
            action_feedback=feedback,
            step_reward=step_reward,
            cumulative_reward=round(self._cumulative_reward, 4),
            steps_remaining=steps_remaining,
        )
