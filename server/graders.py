"""
Deterministic graders for the Email Triage Environment.

Each grader:
  - Operates on the completed episode data
  - Returns a score between 0.0 and 1.0
  - Is fully deterministic (same inputs → same output)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailGroundTruth, VALID_CATEGORIES, VALID_DEPARTMENTS
from server.tasks import TaskDefinition


def _classification_accuracy(
    actions: List[Dict[str, Any]], truths: Dict[str, EmailGroundTruth]
) -> float:
    """Fraction of emails correctly classified."""
    if not truths:
        return 0.0
    correct = 0
    for act in actions:
        eid = act.get("email_id", "")
        gt = truths.get(eid)
        if gt and (act.get("category") or "").lower().strip() == gt.category:
            correct += 1
    return correct / len(truths)


def _priority_accuracy(
    actions: List[Dict[str, Any]], truths: Dict[str, EmailGroundTruth]
) -> float:
    """Score priority assignments:
    - Exact match: 1.0 per email
    - Off by 1: 0.5 per email
    - Off by more: 0.0
    """
    if not truths:
        return 0.0
    total_score = 0.0
    for act in actions:
        eid = act.get("email_id", "")
        gt = truths.get(eid)
        if gt is None:
            continue
        agent_pri = act.get("priority")
        if agent_pri is None:
            continue
        try:
            agent_pri = int(agent_pri)
        except (ValueError, TypeError):
            continue
        diff = abs(agent_pri - gt.priority)
        if diff == 0:
            total_score += 1.0
        elif diff == 1:
            total_score += 0.5
    return total_score / len(truths)


def _routing_accuracy(
    actions: List[Dict[str, Any]], truths: Dict[str, EmailGroundTruth]
) -> float:
    """Fraction of emails routed to the correct department."""
    if not truths:
        return 0.0
    correct = 0
    for act in actions:
        eid = act.get("email_id", "")
        gt = truths.get(eid)
        if gt and (act.get("department") or "").lower().strip() == gt.department:
            correct += 1
    return correct / len(truths)


def _response_quality(
    actions: List[Dict[str, Any]], truths: Dict[str, EmailGroundTruth]
) -> float:
    """Score quality of drafted responses for emails that require them.

    Uses deterministic keyword matching (no LLM).
    """
    response_required = {
        eid: gt for eid, gt in truths.items() if gt.requires_response
    }
    if not response_required:
        return 1.0  # No responses needed → full score

    total_score = 0.0
    for act in actions:
        eid = act.get("email_id", "")
        gt = response_required.get(eid)
        if gt is None:
            continue
        draft = (act.get("response_draft") or "").strip().lower()
        if not draft:
            continue
        keywords = gt.expected_keywords
        if not keywords:
            total_score += 0.5  # Draft exists but no keywords to check
            continue
        matches = sum(1 for kw in keywords if kw.lower() in draft)
        coverage = matches / len(keywords)
        total_score += coverage

    return total_score / len(response_required)


def _time_efficiency(steps_taken: int, max_steps: int, num_emails: int) -> float:
    """Score based on how efficiently the agent used its steps.

    Optimal = one action per email. More steps = lower score.
    """
    optimal = num_emails
    if steps_taken <= optimal:
        return 1.0
    excess = steps_taken - optimal
    max_excess = max_steps - optimal
    if max_excess <= 0:
        return 1.0
    return max(0.0, 1.0 - (excess / max_excess))


# ═══════════════════════════════════════════════════════════════════════════
#  Public grading API
# ═══════════════════════════════════════════════════════════════════════════

def grade_episode(
    task: TaskDefinition,
    actions_taken: List[Dict[str, Any]],
    ground_truths: List[EmailGroundTruth],
    steps_taken: int,
) -> Dict[str, Any]:
    """Grade a completed episode.

    Args:
        task: The task definition
        actions_taken: List of action dicts (one per email, may have duplicates)
        ground_truths: Ground truth for each email
        steps_taken: Total steps the agent used

    Returns:
        Dict with 'final_score' (0.0–1.0) and per-dimension breakdowns.
    """
    # Deduplicate: keep the LAST action per email_id (agent may retry)
    latest_actions: Dict[str, Dict[str, Any]] = {}
    for act in actions_taken:
        eid = act.get("email_id", "")
        if eid:
            latest_actions[eid] = act
    deduped = list(latest_actions.values())

    truth_map = {gt.email_id: gt for gt in ground_truths}
    weights = task.scoring_weights

    scores: Dict[str, float] = {}

    # Classification (always scored)
    if "classification" in weights:
        scores["classification"] = _classification_accuracy(deduped, truth_map)

    # Priority (medium + hard)
    if "priority" in weights:
        scores["priority"] = _priority_accuracy(deduped, truth_map)

    # Routing (medium + hard)
    if "routing" in weights:
        scores["routing"] = _routing_accuracy(deduped, truth_map)

    # Response quality (hard)
    if "response" in weights:
        scores["response"] = _response_quality(deduped, truth_map)

    # Time efficiency (hard)
    if "efficiency" in weights:
        scores["efficiency"] = _time_efficiency(
            steps_taken, task.max_steps, task.num_emails
        )

    # Weighted final score
    final = sum(weights.get(k, 0) * v for k, v in scores.items())
    final = round(max(0.0, min(1.0, final)), 4)

    return {
        "final_score": final,
        "dimension_scores": {k: round(v, 4) for k, v in scores.items()},
        "emails_processed": len(deduped),
        "emails_total": len(ground_truths),
        "steps_taken": steps_taken,
        "max_steps": task.max_steps,
    }
