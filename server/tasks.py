"""
Task definitions for the Email Triage Environment.

Each task specifies what the agent should accomplish, how many emails
to process, the maximum allowed steps, and a human-readable description.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable definition of a single task."""

    task_id: str
    name: str
    difficulty: str  # easy | medium | hard
    description: str
    num_emails: int
    max_steps: int
    requires_priority: bool
    requires_routing: bool
    requires_response: bool
    scoring_weights: Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────
#  Task catalogue
# ─────────────────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="easy",
        name="Basic Email Classification",
        difficulty="easy",
        description=(
            "You have 5 emails in your inbox. Classify each email into one of "
            "these categories: spam, billing, technical, general, urgent.\n\n"
            "For each email, provide:\n"
            "- email_id: the ID of the email\n"
            "- category: one of spam, billing, technical, general, urgent\n\n"
            "You do NOT need to set priority, department, or draft responses "
            "for this task. Focus only on correct classification."
        ),
        num_emails=5,
        max_steps=10,
        requires_priority=False,
        requires_routing=False,
        requires_response=False,
        scoring_weights={
            "classification": 1.0,
        },
    ),
    "medium": TaskDefinition(
        task_id="medium",
        name="Priority & Routing",
        difficulty="medium",
        description=(
            "You have 10 emails in your inbox. For each email you must:\n\n"
            "1. CLASSIFY into a category: spam, billing, technical, general, urgent\n"
            "2. Assign a PRIORITY from 1 (critical) to 5 (low)\n"
            "3. ROUTE to the correct department: engineering, billing, support, management\n\n"
            "Some emails are ambiguous — use your best judgement. Pay attention to "
            "context clues like sender domain, urgency language, and technical details.\n\n"
            "Emails requiring a response will earn bonus points if you draft one."
        ),
        num_emails=10,
        max_steps=25,
        requires_priority=True,
        requires_routing=True,
        requires_response=False,
        scoring_weights={
            "classification": 0.40,
            "priority": 0.30,
            "routing": 0.30,
        },
    ),
    "hard": TaskDefinition(
        task_id="hard",
        name="SLA Triage Under Pressure",
        difficulty="hard",
        description=(
            "You have 20 emails in your inbox with limited time. For each email:\n\n"
            "1. CLASSIFY: spam, billing, technical, general, urgent\n"
            "2. PRIORITIZE: 1 (critical) to 5 (low)\n"
            "3. ROUTE: engineering, billing, support, management\n"
            "4. RESPOND: Draft replies for critical/urgent emails (priority 1)\n"
            "5. ESCALATE: Flag emails that need management attention\n\n"
            "Watch out for:\n"
            "- Thread chains (check thread_id — related emails share context)\n"
            "- Red herrings (emails that LOOK urgent but aren't — check carefully)\n"
            "- Phishing/spam disguised as legitimate alerts\n"
            "- Time pressure: you have 40 steps for 20 emails\n\n"
            "Your score depends on classification accuracy, priority accuracy, "
            "routing accuracy, response quality, and time efficiency."
        ),
        num_emails=20,
        max_steps=40,
        requires_priority=True,
        requires_routing=True,
        requires_response=True,
        scoring_weights={
            "classification": 0.25,
            "priority": 0.20,
            "routing": 0.20,
            "response": 0.20,
            "efficiency": 0.15,
        },
    ),
}


def get_task(task_id: str) -> TaskDefinition:
    """Retrieve a task definition by ID.

    Raises:
        ValueError: If task_id is not recognised.
    """
    task = TASKS.get(task_id)
    if task is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}"
        )
    return task


def list_task_ids() -> List[str]:
    """Return all registered task IDs in order of difficulty."""
    return ["easy", "medium", "hard"]
