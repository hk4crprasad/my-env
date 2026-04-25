"""
Reward shaping for the Email Triage Environment.

Each dimension is its own independent function, making reward hacking harder:
a model that games classification can't simultaneously game escalation.
Per-step rewards provide dense signal (not just binary end-of-episode).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from models import EmailGroundTruth, VALID_CATEGORIES, VALID_DEPARTMENTS


# ─────────────────────────────────────────────────────────────────────────────
#  Independent reward components
# ─────────────────────────────────────────────────────────────────────────────

def reward_classification(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for email category classification."""
    agent_cat = (action.get("category") or "").lower().strip()
    if agent_cat == gt.category:
        return 0.20
    if agent_cat not in VALID_CATEGORIES:
        return -0.10  # invalid value
    close_pairs = {
        frozenset({"urgent", "technical"}),
        frozenset({"billing", "general"}),
    }
    if frozenset({agent_cat, gt.category}) in close_pairs:
        return 0.08  # partial credit for adjacent categories
    return -0.05


def reward_priority(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for priority assignment."""
    agent_pri = action.get("priority")
    if agent_pri is None:
        return -0.05
    try:
        agent_pri = int(agent_pri)
    except (ValueError, TypeError):
        return -0.05
    diff = abs(agent_pri - gt.priority)
    if diff == 0:
        return 0.15
    if diff == 1:
        return 0.07  # off by one: partial credit
    return -0.05


def reward_routing(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for department routing."""
    agent_dept = (action.get("department") or "").lower().strip()
    if agent_dept == gt.department:
        return 0.15
    if agent_dept not in VALID_DEPARTMENTS:
        return -0.08
    return -0.05


def reward_response_quality(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for response draft quality (keyword-based, no LLM)."""
    if not gt.requires_response:
        return 0.0  # not required → neutral
    draft = (action.get("response_draft") or "").strip().lower()
    if not draft:
        return -0.10
    keywords = gt.expected_keywords
    if not keywords:
        return 0.10  # draft provided, no specific keywords expected
    matches = sum(1 for kw in keywords if kw.lower() in draft)
    coverage = matches / len(keywords)
    if coverage >= 0.6:
        return 0.30
    if coverage >= 0.3:
        return 0.15
    return 0.05


def reward_escalation(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for escalation decisions."""
    agent_escalate = bool(action.get("escalate", False))
    should_escalate = gt.department == "management" or gt.priority == 1
    if agent_escalate and should_escalate:
        return 0.05
    if agent_escalate and not should_escalate:
        return -0.10  # unnecessary escalation
    return 0.0


def reward_format_compliance(action: Dict[str, Any], valid_email_ids: Set[str]) -> float:
    """Independent reward for action format correctness.

    Catches hallucinated or missing required fields before any other reward is
    applied — an agent cannot exploit other reward functions with malformed output.
    """
    if not action.get("email_id"):
        return -0.15
    if action["email_id"] not in valid_email_ids:
        return -0.10  # hallucinated or wrong email ID
    if not action.get("category"):
        return -0.05
    return 0.05  # well-formed action bonus


def reward_reprocess_penalty(already_processed: bool) -> float:
    """Independent penalty for re-processing an already-handled email."""
    return -0.15 if already_processed else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Composite step reward (combines all active components)
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_reward(
    action: Dict[str, Any],
    ground_truth: EmailGroundTruth,
    *,
    requires_priority: bool = False,
    requires_routing: bool = False,
    requires_response: bool = False,
    already_processed: bool = False,
    valid_email_ids: Optional[Set[str]] = None,
) -> tuple[float, str]:
    """Compute reward for a single triage action.

    Aggregates all independent reward components and returns a human-readable
    feedback string alongside the total reward.

    Returns:
        (total_reward, feedback_message)
    """
    if valid_email_ids is None:
        valid_email_ids = set()

    reward = 0.0
    feedback_parts: list[str] = []

    # ── Anti-reprocessing (checked first — short-circuits everything else) ──
    r_reprocess = reward_reprocess_penalty(already_processed)
    if already_processed:
        reward += r_reprocess
        return round(reward, 4), "Penalty: this email was already processed."

    # ── Format compliance (independent check) ─────────────────────────────
    if valid_email_ids:
        r_fmt = reward_format_compliance(action, valid_email_ids)
        reward += r_fmt
        if r_fmt < 0:
            if action.get("email_id") not in valid_email_ids:
                return round(reward, 4), f"✗ Unknown email_id '{action.get('email_id')}'. Check the inbox."
            feedback_parts.append("✗ Malformed action (missing required fields)")
        else:
            pass  # silent success

    # ── Classification ─────────────────────────────────────────────────────
    r_cls = reward_classification(action, ground_truth)
    reward += r_cls
    agent_cat = (action.get("category") or "").lower().strip()
    if r_cls == 0.20:
        feedback_parts.append(f"✓ category={ground_truth.category}")
    elif r_cls == 0.08:
        feedback_parts.append(f"~ category: '{agent_cat}' (expected '{ground_truth.category}')")
    elif r_cls < 0:
        feedback_parts.append(f"✗ category: '{agent_cat}' (expected '{ground_truth.category}')")

    # ── Priority ───────────────────────────────────────────────────────────
    if requires_priority:
        r_pri = reward_priority(action, ground_truth)
        reward += r_pri
        agent_pri = action.get("priority", "?")
        if r_pri == 0.15:
            feedback_parts.append(f"✓ priority={ground_truth.priority}")
        elif r_pri == 0.07:
            feedback_parts.append(f"~ priority={agent_pri} (expected {ground_truth.priority})")
        else:
            feedback_parts.append(f"✗ priority={agent_pri} (expected {ground_truth.priority})")

    # ── Routing ────────────────────────────────────────────────────────────
    if requires_routing:
        r_rte = reward_routing(action, ground_truth)
        reward += r_rte
        agent_dept = (action.get("department") or "").lower().strip()
        if r_rte == 0.15:
            feedback_parts.append(f"✓ dept={ground_truth.department}")
        else:
            feedback_parts.append(f"✗ dept='{agent_dept}' (expected '{ground_truth.department}')")

    # ── Response quality ───────────────────────────────────────────────────
    if requires_response:
        r_resp = reward_response_quality(action, ground_truth)
        reward += r_resp
        if r_resp >= 0.30:
            kws = ground_truth.expected_keywords
            matches = sum(1 for kw in kws if kw.lower() in (action.get("response_draft") or "").lower())
            feedback_parts.append(f"✓ response ({matches}/{len(kws)} keywords)")
        elif r_resp > 0:
            feedback_parts.append("~ response (partial keywords)")
        elif r_resp < 0:
            feedback_parts.append("✗ response required but missing")

    # ── Escalation ─────────────────────────────────────────────────────────
    r_esc = reward_escalation(action, ground_truth)
    reward += r_esc
    if r_esc > 0:
        feedback_parts.append("✓ escalation correct")
    elif r_esc < 0:
        feedback_parts.append("✗ unnecessary escalation")

    feedback = " | ".join(feedback_parts) if feedback_parts else "Action recorded."
    return round(reward, 4), feedback


def compute_time_efficiency(steps_taken: int, max_steps: int, num_emails: int) -> float:
    """Score the agent's time efficiency (0.0–1.0).

    Optimal = 1 step per email. More steps = lower score.
    """
    optimal = num_emails
    if steps_taken <= optimal:
        return 1.0
    excess = steps_taken - optimal
    max_excess = max_steps - optimal
    if max_excess <= 0:
        return 1.0
    return max(0.0, 1.0 - (excess / max_excess))


# ─────────────────────────────────────────────────────────────────────────────
#  Rubric definitions (for /rubric endpoint and training logging)
# ─────────────────────────────────────────────────────────────────────────────

REWARD_RUBRIC = {
    "classification": {
        "description": "Correct email category (spam/billing/technical/general/urgent)",
        "max_reward": 0.20,
        "min_reward": -0.10,
        "independent": True,
    },
    "priority": {
        "description": "Correct priority 1–5 (exact +0.15, off-by-1 +0.07)",
        "max_reward": 0.15,
        "min_reward": -0.05,
        "independent": True,
    },
    "routing": {
        "description": "Correct department routing (engineering/billing/support/management)",
        "max_reward": 0.15,
        "min_reward": -0.08,
        "independent": True,
    },
    "response_quality": {
        "description": "Response draft keyword coverage (≥60% → +0.30, ≥30% → +0.15)",
        "max_reward": 0.30,
        "min_reward": -0.10,
        "independent": True,
    },
    "escalation": {
        "description": "Correct escalation flag (management-bound or priority-1 emails)",
        "max_reward": 0.05,
        "min_reward": -0.10,
        "independent": True,
    },
    "format_compliance": {
        "description": "Action is well-formed with valid email_id and required fields",
        "max_reward": 0.05,
        "min_reward": -0.15,
        "independent": True,
    },
    "anti_reprocessing": {
        "description": "Penalty for re-submitting an already-processed email",
        "max_reward": 0.0,
        "min_reward": -0.15,
        "independent": True,
    },
}
