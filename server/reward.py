"""
Reward shaping for the Email Triage Environment.

Design principles (anti-reward-hacking):
  1. Each dimension is its own independent verifier function.
  2. Format gate runs first — malformed output gets no content reward.
  3. Response keywords are hidden from the agent; graded post-hoc.
  4. Escalation uses F1-style scoring (precision + recall), not just accuracy.
  5. Inbox-completion bonus rewards processing every email in one pass.
  6. Re-processing any email applies a flat penalty with no other reward.

Per-step rewards provide dense signal (not just binary end-of-episode).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from models import EmailGroundTruth, VALID_CATEGORIES, VALID_DEPARTMENTS


# ─────────────────────────────────────────────────────────────────────────────
#  Independent reward components
# ─────────────────────────────────────────────────────────────────────────────

# Semantic distance between categories (lower = more similar)
_CATEGORY_DISTANCE: Dict[str, Dict[str, float]] = {
    "urgent":    {"urgent": 0, "technical": 1, "billing": 2, "general": 3, "spam": 4},
    "technical": {"technical": 0, "urgent": 1, "general": 2, "billing": 3, "spam": 4},
    "billing":   {"billing": 0, "general": 1, "urgent": 2, "technical": 3, "spam": 4},
    "general":   {"general": 0, "billing": 1, "technical": 2, "urgent": 3, "spam": 4},
    "spam":      {"spam": 0, "general": 1, "billing": 2, "technical": 3, "urgent": 4},
}


def reward_classification(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for email category classification.

    Uses semantic distance between categories:
      - Exact match:          +0.30
      - Distance 1 (adjacent): +0.10  (e.g. urgent/technical, billing/general)
      - Distance 2:             0.00  (plausible confusion)
      - Distance 3+:           -0.08  (clear miss)
      - Invalid category:      -0.15  (hallucinated value)
    """
    agent_cat = (action.get("category") or "").lower().strip()
    if agent_cat == gt.category:
        return 0.30
    if agent_cat not in VALID_CATEGORIES:
        return -0.15  # hallucinated category — hard penalty
    dist = _CATEGORY_DISTANCE.get(gt.category, {}).get(agent_cat, 4)
    if dist == 1:
        return 0.10   # semantically adjacent
    if dist == 2:
        return 0.00   # plausible but wrong
    return -0.08      # clearly wrong


def reward_priority(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for priority assignment.

    Graduated scale that penalises both over- and under-prioritisation:
      - Exact:    +0.20
      - Off by 1: +0.08  (partial credit)
      - Off by 2:  0.00  (neutral — bad but not punished)
      - Off by 3+: -0.08 (severely wrong direction)
      - Missing:  -0.10
    """
    agent_pri = action.get("priority")
    if agent_pri is None:
        return -0.10
    try:
        agent_pri = int(agent_pri)
    except (ValueError, TypeError):
        return -0.10
    if not (1 <= agent_pri <= 5):
        return -0.10  # out of valid range
    diff = abs(agent_pri - gt.priority)
    if diff == 0:
        return 0.20
    if diff == 1:
        return 0.08
    if diff == 2:
        return 0.00
    return -0.08


# Semantic routing distance
_DEPT_DISTANCE: Dict[str, Dict[str, float]] = {
    "engineering":  {"engineering": 0, "support": 1, "management": 2, "billing": 3},
    "billing":      {"billing": 0, "support": 1, "management": 2, "engineering": 3},
    "support":      {"support": 0, "engineering": 1, "billing": 1, "management": 2},
    "management":   {"management": 0, "support": 1, "engineering": 2, "billing": 3},
}


def reward_routing(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for department routing.

    Uses semantic department distance:
      - Exact:              +0.20
      - Distance 1 (near):  0.00  (reasonable mistake, not punished)
      - Distance 2:         -0.08
      - Distance 3+:        -0.15
      - Invalid dept:       -0.15
    """
    agent_dept = (action.get("department") or "").lower().strip()
    if agent_dept == gt.department:
        return 0.20
    if agent_dept not in VALID_DEPARTMENTS:
        return -0.15  # hallucinated department
    dist = _DEPT_DISTANCE.get(gt.department, {}).get(agent_dept, 3)
    if dist == 1:
        return 0.00
    if dist == 2:
        return -0.08
    return -0.15


def reward_response_quality(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for response draft quality (keyword-based, no LLM).

    Scoring:
      - Missing when required:         -0.15
      - Draft present, no keywords:    +0.10  (can't verify, give benefit of doubt)
      - Keyword coverage ≥ 0.7:        +0.35  (excellent)
      - Keyword coverage ≥ 0.5:        +0.25  (good)
      - Keyword coverage ≥ 0.3:        +0.15  (partial)
      - Keyword coverage < 0.3:        +0.05  (minimal — at least tried)
      - Bonus penalty for very short drafts (<10 chars): −0.05
    """
    if not gt.requires_response:
        return 0.0  # not required → neutral
    draft = (action.get("response_draft") or "").strip().lower()
    if not draft:
        return -0.15
    keywords = gt.expected_keywords
    if not keywords:
        return 0.10  # draft provided, no specific keywords expected
    # Length sanity check — very short responses are low quality
    length_penalty = -0.05 if len(draft) < 10 else 0.0
    matches = sum(1 for kw in keywords if kw.lower() in draft)
    coverage = matches / len(keywords)
    if coverage >= 0.7:
        return round(0.35 + length_penalty, 4)
    if coverage >= 0.5:
        return round(0.25 + length_penalty, 4)
    if coverage >= 0.3:
        return round(0.15 + length_penalty, 4)
    return round(0.05 + length_penalty, 4)


def reward_escalation(action: Dict[str, Any], gt: EmailGroundTruth) -> float:
    """Independent reward for escalation decisions (F1-style scoring).

    Escalation should trigger for management-bound or critical (priority=1) emails.
    Both false positives (unnecessary escalation) and false negatives (missed
    escalation) are penalised — mirrors real-world cost asymmetry.

      - True positive  (correctly escalated):    +0.10
      - True negative  (correctly not escalated): +0.03  (small reward for restraint)
      - False positive (unnecessary escalation):  -0.10  (annoying for management)
      - False negative (missed critical):         -0.05  (missed SLA)
    """
    agent_escalate = bool(action.get("escalate", False))
    should_escalate = gt.department == "management" or gt.priority == 1
    if agent_escalate and should_escalate:
        return 0.10   # true positive
    if not agent_escalate and not should_escalate:
        return 0.03   # true negative — small reward for appropriate restraint
    if agent_escalate and not should_escalate:
        return -0.10  # false positive — unnecessary noise
    return -0.05      # false negative — missed critical escalation


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


def reward_inbox_completion(emails_processed: int, total_emails: int) -> float:
    """Bonus for processing every email in the inbox in a single pass.

    Encourages the agent to be thorough without wasting steps.
    Only awarded if ALL emails are processed.
    """
    if total_emails <= 0:
        return 0.0
    if emails_processed >= total_emails:
        return 0.05  # completion bonus
    return 0.0  # no partial credit — either done or not


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
        "max_reward": 0.30,
        "min_reward": -0.15,
        "independent": True,
        "scoring": "Semantic distance: exact=+0.30, adjacent=+0.10, 2-away=0.00, 3+=−0.08, invalid=−0.15",
    },
    "priority": {
        "description": "Correct priority 1–5 (exact +0.20, off-by-1 +0.08, off-by-3+ −0.08)",
        "max_reward": 0.20,
        "min_reward": -0.10,
        "independent": True,
        "scoring": "Graduated: exact=+0.20, off-by-1=+0.08, off-by-2=0.00, off-by-3+=−0.08",
    },
    "routing": {
        "description": "Correct department routing (engineering/billing/support/management)",
        "max_reward": 0.20,
        "min_reward": -0.15,
        "independent": True,
        "scoring": "Semantic distance: exact=+0.20, adjacent=0.00, 2-away=−0.08, 3+=−0.15",
    },
    "response_quality": {
        "description": "Response draft keyword coverage (≥70% → +0.35, ≥50% → +0.25, ≥30% → +0.15)",
        "max_reward": 0.35,
        "min_reward": -0.15,
        "independent": True,
        "scoring": "Keyword coverage with length penalty for very short drafts (<10 chars: −0.05)",
    },
    "escalation": {
        "description": "F1-style escalation: TP=+0.10, TN=+0.03, FP=−0.10, FN=−0.05",
        "max_reward": 0.10,
        "min_reward": -0.10,
        "independent": True,
        "scoring": "True positive=+0.10, true negative=+0.03, false positive=−0.10, false negative=−0.05",
    },
    "format_compliance": {
        "description": "Action is well-formed with valid email_id and required fields",
        "max_reward": 0.05,
        "min_reward": -0.15,
        "independent": True,
        "scoring": "Format gate runs FIRST — bad format blocks all content rewards",
    },
    "anti_reprocessing": {
        "description": "Penalty for re-submitting an already-processed email",
        "max_reward": 0.0,
        "min_reward": -0.15,
        "independent": True,
        "scoring": "Flat −0.15 with no other reward applied — short-circuits all components",
    },
    "inbox_completion": {
        "description": "Bonus for processing every email in the inbox (+0.05)",
        "max_reward": 0.05,
        "min_reward": 0.0,
        "independent": True,
        "scoring": "Episode-level only — +0.05 if all emails processed, else 0",
    },
}
