"""
Deterministic graders for the Email Triage Environment.

Each grader:
  - Operates on completed episode data
  - Returns a score between 0.0 and 1.0
  - Is fully deterministic (same inputs → same output, no LLM calls)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

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
    """Score priority assignments: exact=1.0, off-by-1=0.5, else=0.0."""
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
    """Score response draft quality using keyword matching (no LLM)."""
    response_required = {
        eid: gt for eid, gt in truths.items() if gt.requires_response
    }
    if not response_required:
        return 1.0

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
            total_score += 0.5
            continue
        matches = sum(1 for kw in keywords if kw.lower() in draft)
        coverage = matches / len(keywords)
        total_score += coverage

    return total_score / len(response_required)


def _time_efficiency(steps_taken: int, max_steps: int, num_emails: int) -> float:
    """Score based on how efficiently the agent used its steps.

    Optimal = one action per email.
    """
    optimal = num_emails
    if steps_taken <= optimal:
        return 1.0
    excess = steps_taken - optimal
    max_excess = max_steps - optimal
    if max_excess <= 0:
        return 1.0
    return max(0.0, 1.0 - (excess / max_excess))


def _escalation_accuracy(
    actions: List[Dict[str, Any]], truths: Dict[str, EmailGroundTruth]
) -> float:
    """Fraction of escalation decisions that are correct (TP + TN) / total."""
    if not truths:
        return 1.0
    correct = 0
    total = len(truths)
    truth_map = truths
    for act in actions:
        eid = act.get("email_id", "")
        gt = truth_map.get(eid)
        if gt is None:
            total -= 1
            continue
        should_escalate = gt.department == "management" or gt.priority == 1
        did_escalate = bool(act.get("escalate", False))
        if did_escalate == should_escalate:
            correct += 1
    if total <= 0:
        return 1.0
    return correct / total


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

    Returns a dict with 'final_score' (0.0–1.0) and per-dimension breakdowns.
    Deduplication: keeps the LAST action per email_id.
    """
    latest_actions: Dict[str, Dict[str, Any]] = {}
    for act in actions_taken:
        eid = act.get("email_id", "")
        if eid:
            latest_actions[eid] = act
    deduped = list(latest_actions.values())

    truth_map = {gt.email_id: gt for gt in ground_truths}
    weights = task.scoring_weights

    scores: Dict[str, float] = {}

    if "classification" in weights:
        scores["classification"] = _classification_accuracy(deduped, truth_map)

    if "priority" in weights:
        scores["priority"] = _priority_accuracy(deduped, truth_map)

    if "routing" in weights:
        scores["routing"] = _routing_accuracy(deduped, truth_map)

    if "response" in weights:
        scores["response"] = _response_quality(deduped, truth_map)

    if "efficiency" in weights:
        scores["efficiency"] = _time_efficiency(
            steps_taken, task.max_steps, task.num_emails
        )

    # Escalation accuracy is always tracked (unweighted bonus info)
    scores["escalation_accuracy"] = _escalation_accuracy(deduped, truth_map)

    final = sum(weights.get(k, 0) * v for k, v in scores.items() if k in weights)
    final = round(max(0.0, min(1.0, final)), 4)

    return {
        "final_score": final,
        "dimension_scores": {k: round(v, 4) for k, v in scores.items()},
        "emails_processed": len(deduped),
        "emails_total": len(ground_truths),
        "steps_taken": steps_taken,
        "max_steps": task.max_steps,
        "scoring_weights": weights,
    }


def get_rubric_definitions() -> Dict[str, Any]:
    """Return structured rubric definitions for all graded dimensions."""
    return {
        "classification": {
            "description": "Correctly identify email category (spam/billing/technical/general/urgent)",
            "weight_by_task": {"easy": 1.0, "medium": 0.40, "hard": 0.25},
            "scoring": "fraction correct",
            "anti_gaming": "5 independent categories; close pairs get partial credit only",
        },
        "priority": {
            "description": "Assign correct urgency level 1–5",
            "weight_by_task": {"easy": 0.0, "medium": 0.30, "hard": 0.20},
            "scoring": "exact=1.0, off-by-1=0.5, else=0.0 per email; averaged",
            "anti_gaming": "Penalises both over- and under-prioritisation",
        },
        "routing": {
            "description": "Route email to correct department",
            "weight_by_task": {"easy": 0.0, "medium": 0.30, "hard": 0.20},
            "scoring": "fraction correctly routed",
            "anti_gaming": "4 departments; wrong routing always penalised",
        },
        "response": {
            "description": "Draft quality response for emails requiring a reply",
            "weight_by_task": {"easy": 0.0, "medium": 0.0, "hard": 0.20},
            "scoring": "keyword coverage fraction (≥60%→1.0, ≥30%→0.5, else→0)",
            "anti_gaming": "Keywords are hidden from agent; coverage measured post-hoc",
        },
        "efficiency": {
            "description": "Process all emails without wasting steps",
            "weight_by_task": {"easy": 0.0, "medium": 0.0, "hard": 0.15},
            "scoring": "linear penalty above optimal (1 step/email)",
            "anti_gaming": "Encourages decisive single-pass processing",
        },
        "escalation_accuracy": {
            "description": "Correct escalation flag (bonus tracking, unweighted)",
            "weight_by_task": {"easy": 0.0, "medium": 0.0, "hard": 0.0},
            "scoring": "(TP + TN) / total emails",
            "anti_gaming": "Penalises both missing escalations and false alarms",
        },
    }
