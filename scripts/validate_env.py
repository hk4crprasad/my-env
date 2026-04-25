"""
Environment validation script.

Run this to confirm the environment works end-to-end:
  - reset() returns a valid observation
  - step() handles correct, incorrect, malformed, and reprocess actions
  - Episode terminates on completion and produces a grading
  - Each independent reward component is exercised
  - All three difficulty levels run cleanly

Usage:
    python scripts/validate_env.py

Exit code: 0 = all checks pass, 1 = any check failed.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import EmailTriageEnvironment
from server.email_generator import generate_emails
from server.tasks import TASKS, list_task_ids
from server.reward import REWARD_RUBRIC
from server.graders import get_rubric_definitions


PASSED = 0
FAILED = 0


def check(name: str, fn) -> bool:
    """Run a check and report ✓/✗."""
    global PASSED, FAILED
    try:
        result = fn()
        if result is False:
            FAILED += 1
            print(f"  ✗ {name}")
            return False
        PASSED += 1
        print(f"  ✓ {name}")
        return True
    except Exception as e:
        FAILED += 1
        print(f"  ✗ {name}  ({type(e).__name__}: {e})")
        if "VERBOSE" in os.environ:
            traceback.print_exc()
        return False


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * (60 - len(title)))


def main():
    print("=" * 70)
    print("  EMAIL TRIAGE ENVIRONMENT — VALIDATION")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    section("Task catalogue")
    check("3 tasks registered", lambda: len(TASKS) == 3)
    check("curriculum order: easy → medium → hard",
           lambda: list_task_ids() == ["easy", "medium", "hard"])
    for tid in ["easy", "medium", "hard"]:
        t = TASKS[tid]
        check(f"  {tid}: {t.num_emails} emails, {t.max_steps} max steps",
               lambda: t.num_emails > 0 and t.max_steps >= t.num_emails)

    # ─────────────────────────────────────────────────────────────────────
    section("Email generator (deterministic)")
    for tid in ["easy", "medium", "hard"]:
        emails_a, _ = generate_emails(tid, 42)
        emails_b, _ = generate_emails(tid, 42)
        check(f"  {tid}: same seed → same emails",
               lambda ea=emails_a, eb=emails_b: [e.email_id for e in ea] == [e.email_id for e in eb])

    # ─────────────────────────────────────────────────────────────────────
    section("Reward rubric")
    expected = {"classification", "priority", "routing", "response_quality",
                 "escalation", "format_compliance", "anti_reprocessing", "inbox_completion"}
    check("8 independent reward components",
           lambda: set(REWARD_RUBRIC.keys()) == expected)
    check("all components marked independent",
           lambda: all(r["independent"] for r in REWARD_RUBRIC.values()))
    check("episode rubric has 6 dimensions",
           lambda: len(get_rubric_definitions()) == 6)

    # ─────────────────────────────────────────────────────────────────────
    section("Environment lifecycle (easy task)")
    env = EmailTriageEnvironment()
    obs = env.reset(task_id="easy", seed=42)
    check("reset returns observation with emails", lambda: len(obs.emails) == 5)
    check("reset resets cumulative reward", lambda: obs.cumulative_reward == 0.0)
    check("step_count starts at 0", lambda: env.state.step_count == 0)

    # Take a correct action
    first_email = obs.emails[0]
    correct = {"email_id": first_email["email_id"], "category": "spam",
                "priority": 5, "department": "support"}
    # Get ground truth
    _, gts = generate_emails("easy", 42)
    gt = next(g for g in gts if g.email_id == first_email["email_id"])
    correct["category"] = gt.category

    obs2 = env.step(correct)
    check("correct action gives positive reward", lambda: obs2.step_reward > 0)
    check("step_count incremented", lambda: env.state.step_count == 1)

    # Reprocessing penalty
    obs3 = env.step(correct)
    check("reprocessing returns negative reward", lambda: obs3.step_reward < 0)

    # Bad action
    obs4 = env.step({"email_id": "fake_id_does_not_exist", "category": "spam"})
    check("unknown email_id penalised", lambda: obs4.step_reward < 0)

    # ─────────────────────────────────────────────────────────────────────
    section("Episode termination & grading")
    env2 = EmailTriageEnvironment()
    obs = env2.reset(task_id="easy", seed=42)
    emails, gts = generate_emails("easy", 42)
    truth_map = {gt.email_id: gt for gt in gts}
    for email in emails:
        gt = truth_map[email.email_id]
        obs = env2.step({
            "email_id": email.email_id,
            "category": gt.category,
            "priority": gt.priority,
            "department": gt.department,
        })
    check("episode terminates after all emails processed", lambda: obs.done is True)
    grading = obs.metadata.get("grading", {})
    check("grading is produced", lambda: bool(grading))
    check("perfect actions → final_score == 1.0",
           lambda: grading.get("final_score") == 1.0)
    check("classification accuracy == 1.0",
           lambda: grading["dimension_scores"].get("classification") == 1.0)

    # ─────────────────────────────────────────────────────────────────────
    section("All difficulty levels run end-to-end")
    for tid in ["easy", "medium", "hard"]:
        env3 = EmailTriageEnvironment()
        obs = env3.reset(task_id=tid, seed=42)
        emails, gts = generate_emails(tid, 42)
        truth_map = {gt.email_id: gt for gt in gts}
        for email in emails:
            gt = truth_map[email.email_id]
            action = {
                "email_id": email.email_id,
                "category": gt.category,
                "priority": gt.priority,
                "department": gt.department,
            }
            if gt.requires_response and gt.expected_keywords:
                action["response_draft"] = " ".join(gt.expected_keywords)
            if gt.department == "management" or gt.priority == 1:
                action["escalate"] = True
            obs = env3.step(action)

        score = obs.metadata.get("grading", {}).get("final_score", 0.0)
        check(f"  {tid}: ground-truth actions → score={score:.2f}",
               lambda s=score: s >= 0.95)

    # ─────────────────────────────────────────────────────────────────────
    section("Out-of-steps termination")
    env4 = EmailTriageEnvironment()
    obs = env4.reset(task_id="easy", seed=42)
    # Burn all steps with bad actions
    for _ in range(15):
        obs = env4.step({"email_id": "fake", "category": "general"})
        if obs.done:
            break
    check("episode terminates when out of steps", lambda: obs.done is True)

    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print("=" * 70)

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
