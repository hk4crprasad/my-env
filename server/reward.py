"""
Reward shaping for the Email Triage Environment.

Provides meaningful per-step reward signal (not just binary end-of-episode).
Rewards partial correctness and penalises undesirable behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from models import EmailGroundTruth, VALID_CATEGORIES, VALID_DEPARTMENTS


def compute_step_reward(
    action: Dict[str, Any],
    ground_truth: EmailGroundTruth,
    *,
    requires_priority: bool = False,
    requires_routing: bool = False,
    requires_response: bool = False,
    already_processed: bool = False,
) -> tuple[float, str]:
    """Compute reward for a single triage action.

    Returns:
        (reward, feedback_message)
    """
    reward = 0.0
    feedback_parts: list[str] = []

    # ── Penalty: acting on an already-processed email ──────────────────
    if already_processed:
        return -0.15, "Penalty: this email was already processed."

    # ── Classification reward ──────────────────────────────────────────
    agent_cat = (action.get("category") or "").lower().strip()
    true_cat = ground_truth.category

    if agent_cat == true_cat:
        reward += 0.20
        feedback_parts.append(f"✓ Correct category: {true_cat}")
    elif agent_cat in VALID_CATEGORIES:
        # Partial: check for "close" categories
        close_pairs = {
            frozenset({"urgent", "technical"}),
            frozenset({"billing", "general"}),
        }
        if frozenset({agent_cat, true_cat}) in close_pairs:
            reward += 0.08
            feedback_parts.append(
                f"~ Partially correct category: you said '{agent_cat}', expected '{true_cat}'"
            )
        else:
            reward -= 0.05
            feedback_parts.append(
                f"✗ Wrong category: you said '{agent_cat}', expected '{true_cat}'"
            )
    else:
        reward -= 0.10
        feedback_parts.append(f"✗ Invalid category: '{agent_cat}'")

    # ── Priority reward ────────────────────────────────────────────────
    if requires_priority:
        agent_pri = action.get("priority")
        true_pri = ground_truth.priority

        if agent_pri is not None:
            try:
                agent_pri = int(agent_pri)
            except (ValueError, TypeError):
                agent_pri = None

        if agent_pri is not None:
            diff = abs(agent_pri - true_pri)
            if diff == 0:
                reward += 0.15
                feedback_parts.append(f"✓ Correct priority: {true_pri}")
            elif diff == 1:
                reward += 0.07
                feedback_parts.append(
                    f"~ Close priority: you said {agent_pri}, expected {true_pri}"
                )
            else:
                reward -= 0.05
                feedback_parts.append(
                    f"✗ Wrong priority: you said {agent_pri}, expected {true_pri}"
                )
        else:
            reward -= 0.05
            feedback_parts.append("✗ Priority not provided (required for this task)")

    # ── Routing reward ─────────────────────────────────────────────────
    if requires_routing:
        agent_dept = (action.get("department") or "").lower().strip()
        true_dept = ground_truth.department

        if agent_dept == true_dept:
            reward += 0.15
            feedback_parts.append(f"✓ Correct department: {true_dept}")
        elif agent_dept in VALID_DEPARTMENTS:
            reward -= 0.05
            feedback_parts.append(
                f"✗ Wrong department: you said '{agent_dept}', expected '{true_dept}'"
            )
        else:
            reward -= 0.08
            feedback_parts.append(f"✗ Invalid department: '{agent_dept}'")

    # ── Response draft reward ──────────────────────────────────────────
    if requires_response and ground_truth.requires_response:
        draft = (action.get("response_draft") or "").strip().lower()
        if draft:
            # Score based on keyword overlap (deterministic, no LLM)
            keywords = ground_truth.expected_keywords
            if keywords:
                matches = sum(1 for kw in keywords if kw.lower() in draft)
                coverage = matches / len(keywords)
                if coverage >= 0.6:
                    reward += 0.30
                    feedback_parts.append(
                        f"✓ Good response draft ({matches}/{len(keywords)} key points)"
                    )
                elif coverage >= 0.3:
                    reward += 0.15
                    feedback_parts.append(
                        f"~ Partial response draft ({matches}/{len(keywords)} key points)"
                    )
                else:
                    reward += 0.05
                    feedback_parts.append(
                        f"~ Weak response draft ({matches}/{len(keywords)} key points)"
                    )
            else:
                reward += 0.10
                feedback_parts.append("✓ Response provided (no specific keywords expected)")
        else:
            reward -= 0.10
            feedback_parts.append("✗ Response required but not drafted")

    # ── Escalation bonus/penalty ───────────────────────────────────────
    agent_escalate = action.get("escalate", False)
    should_escalate = (
        ground_truth.department == "management" or ground_truth.priority == 1
    )
    if agent_escalate and should_escalate:
        reward += 0.05
        feedback_parts.append("✓ Correct escalation")
    elif agent_escalate and not should_escalate:
        reward -= 0.10
        feedback_parts.append("✗ Unnecessary escalation")

    feedback = " | ".join(feedback_parts) if feedback_parts else "Action recorded."
    return round(reward, 4), feedback


def compute_time_efficiency(steps_taken: int, max_steps: int, num_emails: int) -> float:
    """Score the agent's time efficiency (0.0–1.0).

    Optimal = 1 step per email.  More steps = lower score.
    """
    optimal = num_emails
    if steps_taken <= optimal:
        return 1.0
    excess = steps_taken - optimal
    max_excess = max_steps - optimal
    if max_excess <= 0:
        return 1.0
    return max(0.0, 1.0 - (excess / max_excess))
